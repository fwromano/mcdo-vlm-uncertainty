#!/usr/bin/env python
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mcdo_clip import ImageFolderDataset, enable_mc_dropout, load_clip_model, sample_mc_embeddings
from mcdo_clip import attach_dropout_adapters


class InMemoryDataset(Dataset):
    def __init__(self, images: List[Image.Image], transform=None) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, f"mem_{idx}"


def make_solid(n: int, size: int) -> List[Image.Image]:
    imgs = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
        arr = np.ones((size, size, 3), dtype=np.uint8) * color.reshape(1, 1, 3)
        imgs.append(Image.fromarray(arr))
    return imgs


def make_noise(n: int, size: int) -> List[Image.Image]:
    rng = np.random.default_rng(123)
    imgs = []
    for _ in range(n):
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        imgs.append(Image.fromarray(arr))
    return imgs


def make_gradients(n: int, size: int) -> List[Image.Image]:
    imgs = []
    rng = np.random.default_rng(7)
    xs = np.linspace(0, 1, size, dtype=np.float32)
    base = np.tile(xs[None, :, None], (size, 1, 3))
    for _ in range(n):
        scales = rng.random(3, dtype=np.float32)
        arr = np.clip(base * scales, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr))
    return imgs


def build_loader(images: List[Image.Image], preprocess, batch_size: int, num_workers: int) -> DataLoader:
    ds = InMemoryDataset(images, transform=preprocess)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MC variance on trivial vs natural inputs")
    parser.add_argument("out_dir", type=str, help="Where to store results")
    parser.add_argument("--natural-dir", type=str, default=None, help="Optional directory of natural images")
    parser.add_argument("--natural-limit", type=int, default=100, help="Max natural images")
    parser.add_argument("--model", default="ViT-B-32", type=str)
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--dropout", default=0.01, type=float)
    parser.add_argument("--passes", default=64, type=int)
    parser.add_argument("--size", default=224, type=int, help="Synthetic image size")
    parser.add_argument("--count", default=100, type=int, help="Images per synthetic category")
    parser.add_argument(
        "--adapter-targets",
        default="",
        type=str,
        help="Comma-separated module paths to wrap with dropout adapters (e.g., visual.transformer.resblocks.0,visual.proj)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, preprocess, _ = load_clip_model(args.model, args.pretrained, device=args.device)
    if args.adapter_targets:
        targets = [t for t in args.adapter_targets.split(",") if t]
        attach_dropout_adapters(model, targets, p=args.dropout)
        enable_mc_dropout(model, p=None)
    else:
        enable_mc_dropout(model, p=args.dropout)

    results = {}

    categories = {
        "solid": make_solid(args.count, args.size),
        "gradient": make_gradients(args.count, args.size),
        "noise": make_noise(args.count, args.size),
    }

    for name, images in categories.items():
        loader = build_loader(images, preprocess, args.batch_size, args.num_workers)
        stats = sample_mc_embeddings(model, loader, passes=args.passes, device=args.device)
        results[name] = stats["l2_var"].numpy().tolist()

    if args.natural_dir:
        nat_ds = ImageFolderDataset.from_dir(args.natural_dir, transform=preprocess)
        if args.natural_limit and len(nat_ds) > args.natural_limit:
            nat_ds.paths = nat_ds.paths[: args.natural_limit]
        nat_dl = DataLoader(
            nat_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        nat_stats = sample_mc_embeddings(model, nat_dl, passes=args.passes, device=args.device)
        results["natural"] = nat_stats["l2_var"].numpy().tolist()

    with open(os.path.join(args.out_dir, "trivial_baseline.json"), "w") as f:
        json.dump({"dropout": args.dropout, "passes": args.passes, "l2_var": results}, f, indent=2)

    print("Saved:", os.path.join(args.out_dir, "trivial_baseline.json"))


if __name__ == "__main__":
    main()
