#!/usr/bin/env python
import argparse
import json
import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from mcdo_clip import (
    ImageFolderDataset,
    attach_dropout_adapters,
    compute_embedding_covariance,
    enable_mc_dropout,
    load_clip_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare embedding covariance on clean vs blurred images")
    parser.add_argument("data_dir", type=str, help="Directory of images (recursively scanned)")
    parser.add_argument("out_dir", type=str, help="Where to store results")
    parser.add_argument("--model", default="ViT-B-32", type=str, help="CLIP model name")
    parser.add_argument("--pretrained", default="openai", type=str, help="Pretrained tag")
    parser.add_argument("--device", default="cpu", type=str, help="Device for inference")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader workers")
    parser.add_argument("--passes", default=8, type=int, help="MC passes for covariance estimation")
    parser.add_argument("--dropout", default=0.01, type=float, help="Dropout probability")
    parser.add_argument("--blur-kernel", default=11, type=int, help="Gaussian blur kernel size (odd integer)")
    parser.add_argument("--blur-sigma", default=2.0, type=float, help="Gaussian blur sigma")
    parser.add_argument("--no-l2", action="store_true", help="Disable L2 normalization on embeddings")
    parser.add_argument(
        "--adapter-targets",
        default="",
        type=str,
        help="Comma-separated module paths to wrap with dropout adapters (e.g., visual.transformer.resblocks.0,visual.proj)",
    )
    return parser.parse_args()


def build_loader(data_dir: str, transform, batch_size: int, num_workers: int) -> DataLoader:
    ds = ImageFolderDataset.from_dir(data_dir, transform=transform)
    if len(ds) == 0:
        raise ValueError(f"No images found under {data_dir}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, preprocess, _ = load_clip_model(args.model, args.pretrained, device=args.device)
    targets = [t for t in args.adapter_targets.split(",") if t]
    if targets:
        attach_dropout_adapters(model, targets, p=args.dropout)
        enable_mc_dropout(model, p=None)
    else:
        enable_mc_dropout(model, p=args.dropout)

    clean_loader = build_loader(args.data_dir, preprocess, args.batch_size, args.num_workers)

    kernel = args.blur_kernel
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("--blur-kernel must be a positive odd integer")
    blur_tf = transforms.Compose([
        transforms.GaussianBlur(kernel_size=kernel, sigma=args.blur_sigma),
        preprocess,
    ])
    blur_loader = build_loader(args.data_dir, blur_tf, args.batch_size, args.num_workers)

    clean_stats = compute_embedding_covariance(
        model,
        clean_loader,
        passes=args.passes,
        device=args.device,
        l2_normalize=not args.no_l2,
    )
    blur_stats = compute_embedding_covariance(
        model,
        blur_loader,
        passes=args.passes,
        device=args.device,
        l2_normalize=not args.no_l2,
    )

    clean_cov = clean_stats["cov"].numpy()
    blur_cov = blur_stats["cov"].numpy()
    clean_mean = clean_stats["mean"].numpy()
    blur_mean = blur_stats["mean"].numpy()
    if clean_stats["count"] != blur_stats["count"]:
        raise RuntimeError(f"Sample counts differ: clean={clean_stats['count']}, blur={blur_stats['count']}")

    diff = blur_cov - clean_cov
    norm_clean = float(np.linalg.norm(clean_cov, ord="fro"))
    frob_diff = float(np.linalg.norm(diff, ord="fro"))

    metrics = {
        "passes": args.passes,
        "dropout": args.dropout,
        "count": clean_stats["count"],
        "blur": {"kernel_size": kernel, "sigma": args.blur_sigma},
        "trace": {"clean": float(np.trace(clean_cov)), "blur": float(np.trace(blur_cov))},
        "frobenius_diff": frob_diff,
        "relative_frobenius": float(frob_diff / max(norm_clean, 1e-12)),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }

    np.savez(
        os.path.join(args.out_dir, "covariances.npz"),
        clean_cov=clean_cov,
        blur_cov=blur_cov,
        clean_mean=clean_mean,
        blur_mean=blur_mean,
    )
    with open(os.path.join(args.out_dir, "covariance_comparison.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(" -", os.path.join(args.out_dir, "covariances.npz"))
    print(" -", os.path.join(args.out_dir, "covariance_comparison.json"))


if __name__ == "__main__":
    main()
