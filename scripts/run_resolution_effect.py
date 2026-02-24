#!/usr/bin/env python
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mcdo_clip import ImageFolderDataset, enable_mc_dropout, load_clip_model, sample_mc_embeddings


def build_transforms(base_size: int, variants: List[int]) -> Dict[str, transforms.Compose]:
    tfs: Dict[str, transforms.Compose] = {}
    for v in variants:
        if v == base_size:
            name = f"native_{base_size}"
            tfs[name] = transforms.Compose([
                transforms.Resize(base_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(base_size),
                transforms.ToTensor(),
            ])
        elif v > base_size:
            name = f"highres_{v}_to_{base_size}"
            tfs[name] = transforms.Compose([
                transforms.Resize(v, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(v),
                transforms.Resize(base_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            name = f"down_{v}_up_{base_size}"
            tfs[name] = transforms.Compose([
                transforms.Resize(v, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Resize(base_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    return tfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolution/upsampling effect on MC variance")
    parser.add_argument("data_dir", type=str, help="Directory of images")
    parser.add_argument("out_dir", type=str, help="Where to store results")
    parser.add_argument("--base-size", type=int, default=224, help="Target CLIP resolution")
    parser.add_argument(
        "--variants",
        type=str,
        default="224,112,56,28,448",
        help="Comma-separated sizes to test (will downsample/upsample to base size)",
    )
    parser.add_argument("--model", default="ViT-B-32", type=str)
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--dropout", default=0.01, type=float)
    parser.add_argument("--passes", default=64, type=int)
    parser.add_argument("--no-l2", action="store_true", help="Disable L2 normalization")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    variants = [int(v) for v in args.variants.split(",")]

    model, _preprocess, _ = load_clip_model(args.model, args.pretrained, device=args.device)
    enable_mc_dropout(model, p=args.dropout)

    results = {}
    tfs = build_transforms(args.base_size, variants)
    for name, tf in tfs.items():
        ds = ImageFolderDataset.from_dir(args.data_dir, transform=tf)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        stats = sample_mc_embeddings(
            model,
            dl,
            passes=args.passes,
            device=args.device,
            l2_normalize=not args.no_l2,
        )
        results[name] = stats["l2_var"].numpy().tolist()

    np.savez(os.path.join(args.out_dir, "resolution_variance.npz"), **results)
    with open(os.path.join(args.out_dir, "resolution_variance.json"), "w") as f:
        json.dump({"dropout": args.dropout, "passes": args.passes, "variants": results}, f, indent=2)

    print("Saved:", os.path.join(args.out_dir, "resolution_variance.json"))


if __name__ == "__main__":
    main()
