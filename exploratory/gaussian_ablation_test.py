#!/usr/bin/env python
"""Ablation test: does Gaussian@c_proj uncertainty increase under image degradation?

The critical validity test: if degrading an image (blur, downsample) doesn't
reliably increase uncertainty, then we've built a precise ruler that measures
the wrong thing — high reliability but no validity.

Tests Gaussian noise on block 11 c_proj vs uniform dropout baseline.
Paired within-image comparison using Wilcoxon signed-rank test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from phase_two.ablation import DEGRADATIONS, DegradedImageDataset, paired_comparison
from phase_two.perturbation import disable_all_perturbation, perturb_modules


# ── Perturbation configs to test ─────────────────────────────────────────

CONFIGS = {
    "gaussian_block11": {
        "module": "transformer.resblocks.11.mlp.c_proj",
        "ptype": "gaussian",
        "magnitude": 0.05,
    },
    "gaussian_block9": {
        "module": "transformer.resblocks.9.mlp.c_proj",
        "ptype": "gaussian",
        "magnitude": 0.05,
    },
    "uniform_dropout": {
        "module": None,  # special: uses ensure_uniform_dropout
        "ptype": "dropout",
        "magnitude": 0.01,
    },
}


def run_mc_for_config(vlm, loader, config, passes):
    """Run MC trial with a specific perturbation config."""
    root = vlm.vision_root

    if config["module"] is None:
        # Uniform dropout baseline
        disable_all_perturbation(root)
        vlm.ensure_uniform_dropout(config["magnitude"])
        trial = run_mc_trial(
            vlm=vlm, loader=loader, passes=passes,
            collect_pass_features=False,
            cache_precomputed_pixels=False,
        )
    else:
        # Targeted perturbation
        vlm.disable_dropout()
        disable_all_perturbation(root)
        with perturb_modules(root, [(config["module"], config["ptype"], config["magnitude"])]):
            trial = run_mc_trial(
                vlm=vlm, loader=loader, passes=passes,
                collect_pass_features=False,
                cache_precomputed_pixels=False,
            )

    return trial["trace_pre"].numpy()


def main():
    t0 = time.time()
    device = detect_best_device()
    set_all_seeds(42)

    # ── Data ──
    data_dir = "data/raw/imagenet_val"
    all_paths = list_images(data_dir)
    sampled = sample_paths(all_paths, 500, seed=42)

    # ── Model ──
    vlm = load_model("clip_b32", device=device)
    N = len(sampled)
    T = 64

    sep = "=" * 90
    print(f"\nGaussian Ablation Test: N={N}, T={T}")
    print(sep)

    results = {}

    for cfg_name, config in CONFIGS.items():
        print(f"\n  Config: {cfg_name} ({config['ptype']}@{config['magnitude']})")
        print(f"  {'-' * 80}")

        # Clean images
        loader_clean = build_loader(sampled, batch_size=32, num_workers=0)
        set_all_seeds(42)
        clean_unc = run_mc_for_config(vlm, loader_clean, config, T)
        print(f"    clean:          mean_unc={clean_unc.mean():.6f}")

        cfg_results = {"config": config, "degradations": {}}

        for deg_name, deg_fn in DEGRADATIONS.items():
            ds = DegradedImageDataset(sampled, deg_fn)
            loader_deg = DataLoader(
                ds, batch_size=32, shuffle=False,
                num_workers=0, collate_fn=pil_collate,
            )
            set_all_seeds(42)
            deg_unc = run_mc_for_config(vlm, loader_deg, config, T)

            comp = paired_comparison(clean_unc, deg_unc)
            cfg_results["degradations"][deg_name] = comp

            sig = "***" if comp["wilcoxon_p_greater"] < 0.001 else \
                  "**" if comp["wilcoxon_p_greater"] < 0.01 else \
                  "*" if comp["wilcoxon_p_greater"] < 0.05 else "ns"
            status = "PASS" if comp["frac_increased"] >= 0.75 else \
                     "weak" if comp["frac_increased"] >= 0.60 else "FAIL"

            print(
                f"    {deg_name:>16s}:  {comp['frac_increased']:5.1%} increased  "
                f"Δ={comp['rel_change_pct']:+6.1f}%  p={comp['wilcoxon_p_greater']:.2e}  "
                f"{sig}  [{status}]"
            )

        results[cfg_name] = cfg_results

    # ── Summary ──
    print(f"\n{sep}")
    print("SUMMARY: % images with HIGHER uncertainty when degraded (trace_pre)")
    print(sep)
    header = f"  {'Config':>20s}"
    for deg in DEGRADATIONS:
        header += f"  {deg:>14s}"
    print(header)
    print(f"  {'-' * 80}")

    for cfg_name in results:
        row = f"  {cfg_name:>20s}"
        for deg in DEGRADATIONS:
            frac = results[cfg_name]["degradations"][deg]["frac_increased"]
            mark = " *" if frac >= 0.75 else ""
            row += f"  {frac:13.1%}{mark}"
        print(row)

    # ── Comparison with previous prelim_ablation results ──
    prev_path = Path("outputs/prelim_ablation.json")
    if prev_path.exists():
        with open(prev_path) as f:
            prev = json.load(f)
        if "clip_b32" in prev:
            print(f"\n  Previous (uniform dropout, prelim_ablation.py):")
            row = f"  {'prev_uniform_drop':>20s}"
            for deg in DEGRADATIONS:
                frac = prev["clip_b32"]["comparisons"].get(deg, {}).get("frac_unc_increased", float("nan"))
                mark = " *" if frac >= 0.75 else ""
                row += f"  {frac:13.1%}{mark}"
            print(row)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed / 60:.1f} min")

    out_path = Path("outputs/gaussian_ablation_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
