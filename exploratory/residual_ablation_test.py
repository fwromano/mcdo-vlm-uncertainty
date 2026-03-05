#!/usr/bin/env python
"""Residual stream perturbation: inject Gaussian noise between blocks.

Instead of perturbing inside a module (c_proj, attention), perturb the
residual stream BETWEEN blocks. After block N's output (which is the
accumulated representation), inject noise before block N+1's layernorm.

This tests: given everything computed so far, how sensitive is the
remaining computation to perturbation of the accumulated representation?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
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


# ── Residual stream perturbation via hooks ────────────────────────────────

class ResidualNoiseHook:
    """Injects Gaussian noise into a block's output (the residual stream).

    noise = N(0, (mag * x.std())^2)  — scaled to the stream's magnitude.
    Only active when enabled=True (toggled per forward pass).
    """
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
        self.enabled = True
        self._handle = None

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        # output shape: [seq_len, batch, dim] for CLIP ViT
        noise = torch.randn_like(output) * (self.magnitude * output.std())
        return output + noise

    def register(self, module):
        self._handle = module.register_forward_hook(self)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class ResidualPerturbation:
    """Context manager: inject Gaussian noise into residual stream after specified blocks."""

    def __init__(self, vision_root, block_indices, magnitude=0.05):
        self.vision_root = vision_root
        self.block_indices = block_indices
        self.magnitude = magnitude
        self.hooks = []

    def __enter__(self):
        resblocks = self.vision_root.transformer.resblocks
        for idx in self.block_indices:
            hook = ResidualNoiseHook(self.magnitude)
            hook.register(resblocks[idx])
            self.hooks.append(hook)
        # Put model in train mode so run_mc_trial doesn't skip stochasticity
        self.vision_root.train()
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.vision_root.eval()


# ── Helpers ───────────────────────────────────────────────────────────────

def run_mc_residual(vlm, loader, block_indices, magnitude, passes):
    """Run MC trial with residual stream perturbation."""
    vlm.disable_dropout()
    root = vlm.vision_root
    with ResidualPerturbation(root, block_indices, magnitude):
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

    data_dir = "data/raw/imagenet_val"
    all_paths = list_images(data_dir)
    sampled = sample_paths(all_paths, 500, seed=42)

    vlm = load_model("clip_b32", device=device)
    N, T = len(sampled), 64

    # Configs: which blocks to inject noise after
    configs = {
        "after_block11": [11],
        "after_block9": [9],
        "after_block6": [6],
        "after_block0": [0],
        "after_blocks_9_10_11": [9, 10, 11],
        "after_blocks_6_to_11": [6, 7, 8, 9, 10, 11],
        "after_all_blocks": list(range(12)),
    }

    sep = "=" * 90
    print(f"\nResidual Stream Perturbation Test: N={N}, T={T}, mag=0.05")
    print(sep)

    results = {}

    for cfg_name, block_ids in configs.items():
        print(f"\n  {cfg_name} (blocks {block_ids}):")

        # Clean
        loader_clean = build_loader(sampled, batch_size=32, num_workers=0)
        set_all_seeds(42)
        clean_unc = run_mc_residual(vlm, loader_clean, block_ids, 0.05, T)
        print(f"    clean: mean_unc={clean_unc.mean():.6f}")

        cfg_results = {"blocks": block_ids, "degradations": {}}

        for deg_name, deg_fn in DEGRADATIONS.items():
            ds = DegradedImageDataset(sampled, deg_fn)
            loader_deg = DataLoader(
                ds, batch_size=32, shuffle=False,
                num_workers=0, collate_fn=pil_collate,
            )
            set_all_seeds(42)
            deg_unc = run_mc_residual(vlm, loader_deg, block_ids, 0.05, T)

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

    # ── Reliability quick check on best configs ──
    # Run K=3 independent trials to measure pairwise Spearman for top configs
    from phase_one.common import spearman_safe
    print(f"\n{sep}")
    print("RELIABILITY CHECK (K=3 trials, trace_pre pairwise Spearman):")
    print(sep)

    reliability_configs = ["after_block11", "after_blocks_9_10_11", "after_all_blocks"]
    for cfg_name in reliability_configs:
        block_ids = configs[cfg_name]
        trials = []
        loader_clean = build_loader(sampled, batch_size=32, num_workers=0)
        for k in range(3):
            set_all_seeds(42 + k * 1000)
            unc = run_mc_residual(vlm, loader_clean, block_ids, 0.05, T)
            trials.append(unc)

        # Pairwise Spearman
        rhos = []
        for i in range(3):
            for j in range(i + 1, 3):
                rhos.append(spearman_safe(trials[i], trials[j]))
        median_rho = float(np.median(rhos))
        print(f"  {cfg_name:>25s}:  Spearman={median_rho:.4f}")

    # ── Summary ──
    print(f"\n{sep}")
    print("SUMMARY: % images with HIGHER uncertainty when degraded")
    print(sep)
    header = f"  {'Config':>25s}  {'blur_r5':>10s}  {'down_8x':>10s}"
    print(header)
    print(f"  {'-' * 55}")
    for cfg_name in results:
        row = f"  {cfg_name:>25s}"
        for deg in DEGRADATIONS:
            frac = results[cfg_name]["degradations"][deg]["frac_increased"]
            mark = " *" if frac >= 0.75 else ""
            row += f"  {frac:9.1%}{mark}"
        print(row)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed / 60:.1f} min")

    out_path = Path("outputs/residual_ablation_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
