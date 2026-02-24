#!/usr/bin/env python
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mcdo_clip import ImageFolderDataset, compute_complexity_metrics, correlate_metrics_to_variance
from mcdo_clip import enable_mc_dropout, load_clip_model, sample_mc_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complexity metrics vs MC variance")
    parser.add_argument("data_dir", type=str, help="Directory of images (recursively scanned)")
    parser.add_argument("out_dir", type=str, help="Where to store results")
    parser.add_argument("--model", default="ViT-B-32", type=str, help="CLIP model name")
    parser.add_argument("--pretrained", default="openai", type=str, help="Pretrained tag")
    parser.add_argument("--device", default="cpu", type=str, help="Device")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader workers")
    parser.add_argument("--dropout", default=0.01, type=float, help="Dropout probability")
    parser.add_argument("--passes", default=64, type=int, help="MC passes")
    parser.add_argument("--no-l2", action="store_true", help="Disable L2 normalization on embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, preprocess, _ = load_clip_model(args.model, args.pretrained, device=args.device)
    ds_embed = ImageFolderDataset.from_dir(args.data_dir, transform=preprocess)
    dl_embed = DataLoader(
        ds_embed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    enable_mc_dropout(model, p=args.dropout)
    stats = sample_mc_embeddings(
        model,
        dl_embed,
        passes=args.passes,
        device=args.device,
        l2_normalize=not args.no_l2,
    )
    l2_var = stats["l2_var"].numpy()

    # Complexity metrics (raw images)
    raw_ds = ImageFolderDataset.from_dir(args.data_dir, transform=None)
    metrics: Dict[str, List[float]] = {"entropy": [], "edge_density": [], "colorfulness": [], "jpeg_ratio": []}
    for img, _path in tqdm(raw_ds, desc="complexity"):
        vals = compute_complexity_metrics(img)
        for k, v in vals.items():
            metrics[k].append(v)

    correlations = correlate_metrics_to_variance(metrics, l2_var)
    np.savez(os.path.join(args.out_dir, "complexity_vs_variance.npz"), l2_var=l2_var, **metrics)
    with open(os.path.join(args.out_dir, "complexity_correlations.json"), "w") as f:
        json.dump({"dropout": args.dropout, "correlations": correlations}, f, indent=2)

    print("Saved:")
    print(" -", os.path.join(args.out_dir, "complexity_vs_variance.npz"))
    print(" -", os.path.join(args.out_dir, "complexity_correlations.json"))


if __name__ == "__main__":
    main()
