#!/usr/bin/env python
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from mcdo_clip import enable_mc_dropout, load_clip_model


def tokenize_prompts(tokenizer, classnames: List[str]) -> torch.Tensor:
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
    return tokenizer(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate MC variance with downstream error")
    parser.add_argument("data_dir", type=str, help="ImageFolder dataset root")
    parser.add_argument("out_dir", type=str, help="Where to store results")
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

    model, preprocess, tokenizer = load_clip_model(args.model, args.pretrained, device=args.device)
    enable_mc_dropout(model, p=args.dropout)

    dataset = datasets.ImageFolder(args.data_dir, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    classnames = dataset.classes
    text_tokens = tokenize_prompts(tokenizer, classnames).to(args.device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    logit_scale = model.logit_scale.exp()

    num_samples = len(dataset)
    sum_emb = None
    sumsq_emb = None
    sum_logits = None
    labels_arr = np.zeros(num_samples, dtype=np.int64)

    for t in range(args.passes):
        offset = 0
        for images, labels in tqdm(dataloader, desc=f"pass {t+1}/{args.passes}"):
            b = images.shape[0]
            images = images.to(args.device)
            with torch.no_grad():
                img_feat = model.encode_image(images)
                if not args.no_l2:
                    img_feat = F.normalize(img_feat, dim=-1)
                logits = logit_scale * img_feat @ text_features.T
            img_feat = img_feat.detach().cpu()
            logits = logits.detach().cpu()

            if sum_emb is None:
                d = img_feat.shape[1]
                c = logits.shape[1]
                sum_emb = torch.zeros((num_samples, d), dtype=img_feat.dtype)
                sumsq_emb = torch.zeros_like(sum_emb)
                sum_logits = torch.zeros((num_samples, c), dtype=logits.dtype)

            sum_emb[offset : offset + b] += img_feat
            sumsq_emb[offset : offset + b] += img_feat * img_feat
            sum_logits[offset : offset + b] += logits
            labels_arr[offset : offset + b] = labels.numpy()
            offset += b

    mean_emb = sum_emb / args.passes
    var_emb = sumsq_emb / args.passes - mean_emb * mean_emb
    l2_var = var_emb.sum(dim=-1).numpy()
    mean_logits = sum_logits / args.passes
    preds = mean_logits.argmax(dim=1).numpy()
    correct = (preds == labels_arr).astype(np.int32)

    # Calibration curve: bin by variance deciles
    bins = np.quantile(l2_var, np.linspace(0, 1, 11))
    bin_indices = np.digitize(l2_var, bins[1:-1])
    bin_acc = []
    for b in range(10):
        mask = bin_indices == b
        if mask.sum() == 0:
            bin_acc.append(None)
        else:
            bin_acc.append(float(correct[mask].mean()))

    np.savez(os.path.join(args.out_dir, "error_correlation.npz"), l2_var=l2_var, correct=correct, preds=preds, labels=labels_arr)
    with open(os.path.join(args.out_dir, "error_correlation.json"), "w") as f:
        json.dump(
            {
                "dropout": args.dropout,
                "passes": args.passes,
                "accuracy": float(correct.mean()),
                "variance_accuracy_corr": float(np.corrcoef(l2_var, correct)[0, 1]),
                "bin_edges": bins.tolist(),
                "bin_accuracy": bin_acc,
            },
            f,
            indent=2,
        )

    print("Saved:", os.path.join(args.out_dir, "error_correlation.json"))


if __name__ == "__main__":
    main()
