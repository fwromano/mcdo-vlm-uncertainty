#!/usr/bin/env python
"""
Image ablation uncertainty test.

For each image, create degraded versions (Gaussian blur, downsample+upsample)
and check that MC dropout uncertainty increases with degradation.

This is a paired test — each image is its own control.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.stats import wilcoxon

from phase_one.common import (
    build_loader, list_images, load_model, run_mc_trial,
    sample_paths, set_all_seeds, ImagePathDataset, pil_collate,
)
from torch.utils.data import DataLoader


DATA_DIR = "data/raw/imagenet_val"
DEVICE = "mps"
DROPOUT = 0.01
PASSES = 64
SEED = 42
N = 500
BATCH_SIZE = 32


class DegradedImageDataset(torch.utils.data.Dataset):
    """Wraps image paths, applying a degradation function."""
    def __init__(self, image_paths, degrade_fn):
        self.image_paths = [str(Path(p)) for p in image_paths]
        self.degrade_fn = degrade_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        degraded = self.degrade_fn(image)
        class_name = Path(path).parent.name
        return degraded, path, class_name


def blur_image(img, radius=5):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def heavy_blur(img, radius=15):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def downsample_image(img, factor=4):
    w, h = img.size
    small = img.resize((max(w // factor, 1), max(h // factor, 1)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def heavy_downsample(img, factor=8):
    w, h = img.size
    small = img.resize((max(w // factor, 1), max(h // factor, 1)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def run_test(model_key, sampled_paths):
    print(f"\n{'='*50}")
    print(f"Model: {model_key}")
    print(f"{'='*50}")

    set_all_seeds(SEED)
    vlm = load_model(model_key, device=DEVICE)
    vlm.ensure_uniform_dropout(DROPOUT)

    degradations = {
        "clean": None,
        "blur_r5": lambda img: blur_image(img, 5),
        "blur_r15": lambda img: heavy_blur(img, 15),
        "downsample_4x": lambda img: downsample_image(img, 4),
        "downsample_8x": lambda img: heavy_downsample(img, 8),
    }

    uncertainties = {}
    det_embeddings = {}

    for deg_name, deg_fn in degradations.items():
        print(f"\n  --- {deg_name} ---")

        if deg_fn is None:
            loader = build_loader(sampled_paths, batch_size=BATCH_SIZE, num_workers=0)
        else:
            ds = DegradedImageDataset(sampled_paths, deg_fn)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pil_collate)

        # Deterministic embedding
        vlm.disable_dropout()
        det_feats = []
        with torch.no_grad():
            for images, _, _ in loader:
                f = vlm.encode_images(images, normalize=True).detach().cpu()
                det_feats.append(f)
        det_emb = torch.cat(det_feats, dim=0)
        det_embeddings[deg_name] = det_emb

        # MC uncertainty
        vlm.ensure_uniform_dropout(DROPOUT)
        trial = run_mc_trial(
            vlm=vlm, loader=loader, passes=PASSES,
            collect_pass_features=False,
            progress=True, progress_desc=f"{model_key}/{deg_name}",
            cache_precomputed_pixels=False,
        )
        unc = trial["trace_pre"].numpy()
        uncertainties[deg_name] = unc
        print(f"    mean_unc = {unc.mean():.6f}, std = {unc.std():.6f}")

    # Paired comparisons: each degraded vs clean
    clean_unc = uncertainties["clean"]
    results = {"model": model_key, "N": N, "comparisons": {}}

    for deg_name in ["blur_r5", "blur_r15", "downsample_4x", "downsample_8x"]:
        deg_unc = uncertainties[deg_name]
        diff = deg_unc - clean_unc
        frac_increased = (diff > 0).mean()

        # Wilcoxon signed-rank test (paired non-parametric)
        stat, p_value = wilcoxon(deg_unc, clean_unc, alternative="greater")

        # Cosine similarity between degraded and clean deterministic embeddings
        cos_sim = F.cosine_similarity(det_embeddings["clean"], det_embeddings[deg_name], dim=1)
        mean_cos = cos_sim.mean().item()

        # MC mean vs deterministic (clean) — how far does MC mean of degraded drift from clean det?
        # This measures "semantic shift" from degradation

        comp = {
            "frac_unc_increased": float(frac_increased),
            "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            "wilcoxon_p": float(p_value),
            "effect_significant": p_value < 0.05,
            "mean_unc_clean": float(clean_unc.mean()),
            "mean_unc_degraded": float(deg_unc.mean()),
            "mean_cosine_clean_vs_degraded": float(mean_cos),
        }
        results["comparisons"][deg_name] = comp

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"\n  {deg_name} vs clean:")
        print(f"    {frac_increased:.1%} images have higher uncertainty when degraded")
        print(f"    mean_diff = {diff.mean():.6f}, Wilcoxon p = {p_value:.2e} ({sig})")
        print(f"    cosine(clean_det, degraded_det) = {mean_cos:.4f}")

    return results


def main():
    t0 = time.time()
    all_paths = list_images(DATA_DIR)
    sampled = sample_paths(all_paths, N, SEED)

    all_results = {}
    for model_key in ["clip_b32", "siglip2_b16"]:
        all_results[model_key] = run_test(model_key, sampled)
        torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed

    out_path = Path("outputs/prelim_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"Total elapsed: {elapsed/60:.1f} min")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
