#!/usr/bin/env python
"""Head-to-head: PE-Core blocks 7-9 (accidentally tested) vs blocks 9-11 (intended).

Resolves whether the sorting bug materially affected the PE-Core results.
Uses the FIXED get_mlp_output_projections() with natural sort.

Configs:
  - blocks 7-9 fc2, p=0.01  (reproduces what pe_core_sweep actually tested)
  - blocks 9-11 fc2, p=0.01 (what was intended — the true last 3)
  - all-12 fc2, p=0.01      (baseline for comparison)

N=500, T=64, weighted_trace_pre + trace_pre, blur_r5 + downsample_8x
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Dataset

from phase_one.common import (
    build_loader,
    detect_best_device,
    list_images,
    load_model,
    pil_collate,
    run_mc_trial,
    sample_paths,
    set_all_seeds,
)
from phase_two.metrics import weighted_trace_pre, topk_dim_trace
from phase_two.perturbation import (
    disable_all_perturbation,
    get_mlp_output_projections,
    perturb_modules,
)

DEGRADATIONS = {
    "blur_r5": lambda img: img.filter(ImageFilter.GaussianBlur(radius=5)),
    "downsample_8x": lambda img: img.resize(
        (max(img.width // 8, 1), max(img.height // 8, 1)), Image.BILINEAR
    ).resize((img.width, img.height), Image.BILINEAR),
}


class DegradedImageDataset(Dataset):
    def __init__(self, paths, degrade_fn):
        self.paths = [str(p) for p in paths]
        self.degrade_fn = degrade_fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            image = img.convert("RGB")
        return self.degrade_fn(image), self.paths[idx], Path(self.paths[idx]).parent.name


def run_mc_with_features(vlm, loader, perturbation_configs, passes, seed):
    root = vlm.vision_root
    set_all_seeds(seed)
    vlm.disable_dropout()
    disable_all_perturbation(root)

    with perturb_modules(root, perturbation_configs):
        trial = run_mc_trial(
            vlm=vlm, loader=loader, passes=passes,
            collect_pass_features=True,
            cache_precomputed_pixels=False,
        )

    pass_pre = trial["pass_pre"]
    tp = trial["trace_pre"].numpy()
    wtp = weighted_trace_pre(pass_pre)
    return tp, wtp


def paired_test(clean, degraded):
    diff = degraded - clean
    frac = float((diff > 0).mean())
    try:
        _, p = wilcoxon(degraded, clean, alternative="greater")
    except ValueError:
        p = 1.0
    return {"frac_increased": frac, "p_value": float(p)}


def main():
    t0 = time.time()
    device = detect_best_device()
    set_all_seeds(42)

    paths = sample_paths(list_images("data/raw/imagenet_val"), 500, seed=42)
    vlm = load_model("pe_core_b16", device=device)
    root = vlm.vision_root

    # With the fix, this now returns blocks in natural order: 0,1,2,...,11
    fc2_all = get_mlp_output_projections(root)
    n = len(fc2_all)

    print(f"PE-Core B/16: {n} MLP output projections (natural sort)")
    for i, p in enumerate(fc2_all):
        print(f"  [{i}] {p}")

    # Define the three configs
    blocks_7_9 = fc2_all[7:10]   # What was accidentally tested
    blocks_9_11 = fc2_all[9:12]  # What was intended (true last 3)

    print(f"\nBlocks 7-9: {[p.split('.')[-2] for p in blocks_7_9]}")
    print(f"  {blocks_7_9}")
    print(f"Blocks 9-11: {[p.split('.')[-2] for p in blocks_9_11]}")
    print(f"  {blocks_9_11}")

    configs = {
        "blocks_7_9": [(m, "dropout", 0.01) for m in blocks_7_9],
        "blocks_9_11": [(m, "dropout", 0.01) for m in blocks_9_11],
        "all_12": [(m, "dropout", 0.01) for m in fc2_all],
    }

    N, T = len(paths), 64
    print(f"\nN={N}, T={T}, {len(configs)} configs × 2 metrics × 2 degradations")
    print("=" * 80)

    all_results = {}

    for cfg_name, pcfg in configs.items():
        print(f"\n--- {cfg_name} ---")

        loader_clean = build_loader(paths, batch_size=32, num_workers=0)
        clean_tp, clean_wtp = run_mc_with_features(vlm, loader_clean, pcfg, T, 42)

        cfg_results = {}

        for deg_name, deg_fn in DEGRADATIONS.items():
            ds = DegradedImageDataset(paths, deg_fn)
            loader_deg = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                                    collate_fn=pil_collate)
            deg_tp, deg_wtp = run_mc_with_features(vlm, loader_deg, pcfg, T, 42)

            for mn, clean_arr, deg_arr in [
                ("trace_pre", clean_tp, deg_tp),
                ("weighted_trace_pre", clean_wtp, deg_wtp),
            ]:
                result = paired_test(clean_arr, deg_arr)
                key = f"{deg_name}__{mn}"
                cfg_results[key] = result

                frac = result["frac_increased"]
                status = "PASS" if frac >= 0.75 else "weak" if frac >= 0.60 else "FAIL"
                print(f"  {deg_name:>14s} | {mn:>22s}: {frac:5.1%}  [{status}]")

        all_results[cfg_name] = cfg_results

    # Summary
    print(f"\n{'=' * 80}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'':>18s}  {'blur trace':>12s}  {'blur weighted':>14s}  {'down trace':>12s}  {'down weighted':>14s}")
    for cfg_name in configs:
        r = all_results[cfg_name]
        bt = r["blur_r5__trace_pre"]["frac_increased"]
        bw = r["blur_r5__weighted_trace_pre"]["frac_increased"]
        dt = r["downsample_8x__trace_pre"]["frac_increased"]
        dw = r["downsample_8x__weighted_trace_pre"]["frac_increased"]
        print(f"  {cfg_name:>16s}  {bt:11.1%}  {bw:13.1%}  {dt:11.1%}  {dw:13.1%}")

    elapsed = (time.time() - t0) / 60
    print(f"\nElapsed: {elapsed:.1f} min")

    out_path = Path("outputs/pe_core_block_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
