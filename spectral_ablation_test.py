#!/usr/bin/env python
"""
Spectral ablation test: do uncertainty metrics change under image degradation?

For each metric × degradation pair, tests:
  H0: metric(degraded) = metric(clean)  (no response to degradation)
  H1: metric(degraded) > metric(clean)  (metric increases with degradation)
       — or H1': metric changes in either direction

Paired within-image test using Wilcoxon signed-rank.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
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
from phase_two.ablation import DEGRADATIONS, DegradedImageDataset
from phase_two.metrics import compute_all_metrics


# ── Helpers ──────────────────────────────────────────────────────────────

def run_mc_and_metrics(vlm, loader, passes, desc="MC"):
    """Run MC dropout, collect per-pass features, compute all metrics."""
    trial = run_mc_trial(
        vlm=vlm, loader=loader, passes=passes,
        collect_pass_features=True, progress=True,
        progress_desc=desc, cache_precomputed_pixels=False,
    )
    metrics = compute_all_metrics(trial["pass_pre"], trial["pass_post"])
    return metrics


def paired_test(clean_vals, deg_vals, metric_name):
    """Paired comparison of metric values: degraded vs clean."""
    diff = deg_vals - clean_vals

    frac_increased = float((diff > 0).mean())
    frac_decreased = float((diff < 0).mean())
    mean_diff = float(diff.mean())
    median_diff = float(np.median(diff))

    # Relative effect size (avoid div-by-zero)
    clean_mean = clean_vals.mean()
    rel_change = float(mean_diff / clean_mean) if abs(clean_mean) > 1e-15 else 0.0

    # Two-sided Wilcoxon (does it change at all?)
    try:
        _, p_two = wilcoxon(deg_vals, clean_vals, alternative="two-sided")
    except ValueError:
        p_two = 1.0

    # One-sided Wilcoxon (does it increase?)
    try:
        _, p_greater = wilcoxon(deg_vals, clean_vals, alternative="greater")
    except ValueError:
        p_greater = 1.0

    # Determine dominant direction
    if frac_increased > frac_decreased:
        dominant_dir = "UP"
        dominant_frac = frac_increased
    elif frac_decreased > frac_increased:
        dominant_dir = "DOWN"
        dominant_frac = frac_decreased
    else:
        dominant_dir = "NONE"
        dominant_frac = 0.5

    return {
        "frac_increased": frac_increased,
        "frac_decreased": frac_decreased,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "rel_change_pct": rel_change * 100,
        "wilcoxon_p_two_sided": float(p_two),
        "wilcoxon_p_greater": float(p_greater),
        "dominant_direction": dominant_dir,
        "dominant_frac": dominant_frac,
        "clean_mean": float(clean_mean),
        "degraded_mean": float(deg_vals.mean()),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spectral ablation test")
    parser.add_argument("--data-dir", default="data/raw/imagenet_val")
    parser.add_argument("--model", default="clip_b32")
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/spectral_ablation.json")
    args = parser.parse_args()

    if args.device is None:
        args.device = detect_best_device()

    t0 = time.time()
    set_all_seeds(args.seed)

    # ── sample images ──
    all_paths = list_images(args.data_dir)
    sampled = sample_paths(all_paths, args.n_images, args.seed)

    # ── load model ──
    vlm = load_model(args.model, device=args.device)
    vlm.ensure_uniform_dropout(args.dropout)

    # ── clean baseline ──
    print(f"\n{'='*70}")
    print(f"Model: {args.model}  N={args.n_images}  T={args.passes}  p={args.dropout}")
    print(f"{'='*70}")

    loader_clean = build_loader(sampled, batch_size=args.batch_size, num_workers=0)
    print("\n  Computing CLEAN metrics ...")
    clean_metrics = run_mc_and_metrics(vlm, loader_clean, args.passes, f"{args.model}/clean")

    metric_names = list(clean_metrics.keys())

    # ── degraded conditions ──
    all_results = {
        "model": args.model,
        "N": args.n_images,
        "T": args.passes,
        "p": args.dropout,
        "degradations": {},
    }

    for deg_name, deg_fn in DEGRADATIONS.items():
        print(f"\n  Computing {deg_name} metrics ...")
        ds = DegradedImageDataset(sampled, deg_fn)
        loader_deg = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=pil_collate,
        )
        deg_metrics = run_mc_and_metrics(vlm, loader_deg, args.passes, f"{args.model}/{deg_name}")

        deg_results = {}
        for mname in metric_names:
            deg_results[mname] = paired_test(clean_metrics[mname], deg_metrics[mname], mname)

        all_results["degradations"][deg_name] = deg_results

    # ── print summary table ──
    print(f"\n{'='*100}")
    print(f"  H0: metric(degraded) = metric(clean)")
    print(f"  H1: metric(degraded) != metric(clean)  [two-sided]")
    print(f"{'='*100}")

    for deg_name in DEGRADATIONS:
        print(f"\n  ── {deg_name} ──")
        print(f"  {'Metric':>20s}  {'Dir':>4s}  {'%Dom':>5s}  {'Rel Δ%':>8s}  {'p(two)':>10s}  {'Verdict':>8s}")
        print(f"  {'-'*65}")

        for mname in metric_names:
            r = all_results["degradations"][deg_name][mname]
            p2 = r["wilcoxon_p_two_sided"]

            if p2 < 0.001:
                verdict = "***"
            elif p2 < 0.01:
                verdict = "**"
            elif p2 < 0.05:
                verdict = "*"
            else:
                verdict = "ns"

            print(
                f"  {mname:>20s}  {r['dominant_direction']:>4s}  "
                f"{r['dominant_frac']:5.1%}  {r['rel_change_pct']:+7.1f}%  "
                f"{p2:10.2e}  {verdict:>8s}"
            )

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed

    print(f"\n{'='*100}")
    print(f"Elapsed: {elapsed / 60:.1f} min")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
