#!/usr/bin/env python
"""PE-Core-B/16 uncertainty evaluation: ablation validity + reliability.

Tests whether Meta's Perception Encoder (a next-gen contrastive VLM) produces
valid MC dropout uncertainty using the same protocol as CLIP B/32 and L/14.

Configs:
  - all_fc2_dropout_p01: dropout on all 12 MLP output projections (fc2)
  - uniform_dropout_p01: dropout on all Linear modules (baseline)

Protocol:
  - Ablation: blur_r5, downsample_8x — paired Wilcoxon test
  - Reliability: K=3 independent trials, pairwise Spearman
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from phase_one.common import (
    build_loader,
    detect_best_device,
    list_images,
    load_model,
    run_mc_trial,
    sample_paths,
    set_all_seeds,
    spearman_safe,
)
from phase_two.ablation import run_ablation_test
from phase_two.perturbation import (
    disable_all_perturbation,
    get_mlp_output_projections,
    perturb_modules,
)


def main():
    t0 = time.time()
    device = detect_best_device()
    set_all_seeds(42)

    data_dir = "data/raw/imagenet_val"
    all_paths = list_images(data_dir)
    sampled = sample_paths(all_paths, 500, seed=42)

    vlm = load_model("pe_core_b16", device=device)
    root = vlm.vision_root
    N, T, K = len(sampled), 64, 3

    # Auto-detect MLP output projections
    fc2_paths = get_mlp_output_projections(root)
    print(f"PE-Core-B/16: {len(fc2_paths)} MLP output projections detected")
    print(f"  First: {fc2_paths[0]}")
    print(f"  Last:  {fc2_paths[-1]}")

    fc2_configs = [(p, "dropout", 0.01) for p in fc2_paths]

    sep = "=" * 70
    print(f"\nPE-Core-B/16: N={N}, T={T}, K={K}")
    print(sep)

    results = {}

    for cfg_name, cfg_setup in [
        ("all_fc2_dropout_p01", "perturb"),
        ("uniform_dropout_p01", "uniform"),
    ]:
        print(f"\n  Config: {cfg_name}")
        print(f"  {'-' * 60}")

        # ── Ablation validity ──
        if cfg_setup == "perturb":
            ablation = run_ablation_test(
                vlm, sampled, fc2_configs,
                passes=T, batch_size=32, seed=42,
                degradation_names=["blur_r5", "downsample_8x"],
            )
        else:
            vlm.disable_dropout()
            disable_all_perturbation(root)
            vlm.ensure_uniform_dropout(0.01)
            ablation = run_ablation_test(
                vlm, sampled, None,
                passes=T, batch_size=32, seed=42,
                degradation_names=["blur_r5", "downsample_8x"],
            )

        for deg_name in ["blur_r5", "downsample_8x"]:
            comp = ablation[deg_name]
            status = "PASS" if comp["frac_increased"] >= 0.75 else \
                     "weak" if comp["frac_increased"] >= 0.60 else "FAIL"
            print(
                f"    {deg_name:>14s}: {comp['frac_increased']:5.1%} increased  "
                f"Δ={comp['rel_change_pct']:+6.1f}%  p={comp['wilcoxon_p_greater']:.2e}  [{status}]"
            )

        # ── Reliability ──
        trials = []
        for k in range(K):
            loader = build_loader(sampled, batch_size=32, num_workers=0)
            set_all_seeds(42 + k * 1000)

            if cfg_setup == "perturb":
                vlm.disable_dropout()
                disable_all_perturbation(root)
                with perturb_modules(root, fc2_configs):
                    trial = run_mc_trial(
                        vlm=vlm, loader=loader, passes=T,
                        collect_pass_features=False,
                        cache_precomputed_pixels=False,
                    )
            else:
                vlm.disable_dropout()
                disable_all_perturbation(root)
                vlm.ensure_uniform_dropout(0.01)
                trial = run_mc_trial(
                    vlm=vlm, loader=loader, passes=T,
                    collect_pass_features=False,
                    cache_precomputed_pixels=False,
                )
            trials.append(trial["trace_pre"].numpy())

        rhos = [
            spearman_safe(trials[i], trials[j])
            for i in range(K) for j in range(i + 1, K)
        ]
        median_rho = float(np.median(rhos))
        print(f"    reliability: Spearman={median_rho:.4f} (range {min(rhos):.3f}-{max(rhos):.3f})")

        results[cfg_name] = {
            "ablation": ablation,
            "reliability_spearman": median_rho,
            "reliability_rhos": [float(r) for r in rhos],
        }

    # ── Summary ──
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    for cfg_name, res in results.items():
        print(f"  {cfg_name}:")
        print(f"    reliability: Spearman={res['reliability_spearman']:.4f}")
        for deg_name in ["blur_r5", "downsample_8x"]:
            frac = res["ablation"][deg_name]["frac_increased"]
            status = "PASS" if frac >= 0.75 else "weak" if frac >= 0.60 else "FAIL"
            print(f"    {deg_name:>14s}: {frac:5.1%} [{status}]")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed / 60:.1f} min")

    out_path = Path("outputs/pe_core_exp.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
