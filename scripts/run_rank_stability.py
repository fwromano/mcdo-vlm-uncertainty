#!/usr/bin/env python
import argparse
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mcdo_clip import (
    ImageFolderDataset,
    enable_mc_dropout,
    load_clip_model,
    sample_mc_embeddings,
    attach_dropout_adapters,
)
from mcdo_clip.metrics import rank_spearman_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC Dropout rank stability across dropout rates")
    parser.add_argument("data_dir", type=str, help="Directory of images (recursively scanned)")
    parser.add_argument("out_dir", type=str, help="Where to store results")
    parser.add_argument("--model", default="ViT-B-32", type=str, help="CLIP model name")
    parser.add_argument("--pretrained", default="openai", type=str, help="Pretrained tag")
    parser.add_argument("--device", default="cpu", type=str, help="Device for inference")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader workers")
    parser.add_argument(
        "--dropout-rates",
        default="0.001,0.005,0.01,0.02,0.05,0.1",
        type=str,
        help="Comma-separated dropout probabilities",
    )
    parser.add_argument("--passes", default=64, type=int, help="MC passes per rate")
    parser.add_argument("--no-l2", action="store_true", help="Disable L2 normalization on embeddings")
    parser.add_argument(
        "--adapter-targets",
        default="",
        type=str,
        help="Comma-separated module paths to wrap with dropout adapters (e.g., visual.transformer.resblocks.0,visual.proj)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rates: List[float] = [float(x) for x in args.dropout_rates.split(",")]

    model, preprocess, _ = load_clip_model(args.model, args.pretrained, device=args.device)
    dataset = ImageFolderDataset.from_dir(args.data_dir, transform=preprocess)
    if len(dataset) == 0:
        raise ValueError(f"No images found under {args.data_dir}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_vars = []
    targets = [t for t in args.adapter_targets.split(",") if t]
    for p in rates:
        if targets:
            attach_dropout_adapters(model, targets, p=p)
            enable_mc_dropout(model, p=None)
        else:
            enable_mc_dropout(model, p=p)
        stats = sample_mc_embeddings(
            model,
            dataloader,
            passes=args.passes,
            device=args.device,
            l2_normalize=not args.no_l2,
        )
        all_vars.append(stats["l2_var"].numpy())

    var_matrix = np.stack(all_vars, axis=0)
    corr = rank_spearman_matrix(var_matrix)

    np.savez(os.path.join(args.out_dir, "mc_variances.npz"), dropout_rates=rates, l2_var=var_matrix)
    with open(os.path.join(args.out_dir, "rank_correlation.json"), "w") as f:
        json.dump({"dropout_rates": rates, "spearman": corr.tolist()}, f, indent=2)

    print("Saved:")
    print(" -", os.path.join(args.out_dir, "mc_variances.npz"))
    print(" -", os.path.join(args.out_dir, "rank_correlation.json"))


if __name__ == "__main__":
    main()
