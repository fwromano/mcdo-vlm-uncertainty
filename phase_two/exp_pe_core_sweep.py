#!/usr/bin/env python
"""PE-Core-B/16 proper sweep: multiple dropout rates, module subsets, metrics.

Gives PE-Core a fair shake — the initial 2-config test was too narrow to be conclusive.

Configs tested:
  - p = {0.001, 0.005, 0.01}
  - Modules: {late-3-fc2 (blocks 9-11 only), all-12-fc2}
  = 6 configs total

Metrics: trace_pre AND weighted_trace_pre (our new best metric)
Degradations: blur_r5, downsample_8x
N=500, T=64
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


# ── Degradation (inline to avoid circular import issues) ──────────
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


# ── Core runner ───────────────────────────────────────────────────

def run_mc_with_features(
    vlm, loader, perturbation_configs, passes, seed
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MC trial, return (trace_pre, weighted_trace_pre, topk64_trace_pre)."""
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

    pass_pre = trial["pass_pre"]  # (T, N, D)
    tp = trial["trace_pre"].numpy()
    wtp = weighted_trace_pre(pass_pre)
    tk64 = topk_dim_trace(pass_pre, k=64)
    return tp, wtp, tk64


def paired_test(clean: np.ndarray, degraded: np.ndarray) -> Dict:
    diff = degraded - clean
    frac = float((diff > 0).mean())
    try:
        _, p = wilcoxon(degraded, clean, alternative="greater")
    except ValueError:
        p = 1.0
    rel = (degraded.mean() - clean.mean()) / clean.mean() * 100 if abs(clean.mean()) > 1e-15 else 0.0
    return {"frac_increased": frac, "p_value": float(p), "rel_change_pct": rel}


def main():
    t0 = time.time()
    device = detect_best_device()
    set_all_seeds(42)

    paths = sample_paths(list_images("data/raw/imagenet_val"), 500, seed=42)
    vlm = load_model("pe_core_b16", device=device)
    root = vlm.vision_root

    fc2_all = get_mlp_output_projections(root)
    n_blocks = len(fc2_all)
    # Late-3: last 3 blocks (like blocks 9-11 in 12-block model)
    fc2_late3 = fc2_all[n_blocks - 3:]

    print(f"PE-Core-B/16: {n_blocks} MLP output projections")
    print(f"  All:    {fc2_all[0]} ... {fc2_all[-1]}")
    print(f"  Late-3: {fc2_late3[0]} ... {fc2_late3[-1]}")

    N, T = len(paths), 64
    p_values = [0.001, 0.005, 0.01]
    module_sets = {
        "late3_fc2": fc2_late3,
        "all_fc2": fc2_all,
    }

    configs = []
    for mod_name, modules in module_sets.items():
        for p in p_values:
            cfg_name = f"{mod_name}_p{str(p).replace('.', '')}"
            perturbation_configs = [(m, "dropout", p) for m in modules]
            configs.append((cfg_name, perturbation_configs))

    metric_names = ["trace_pre", "weighted_trace_pre", "topk64_trace_pre"]
    deg_names = ["blur_r5", "downsample_8x"]

    print(f"\nN={N}, T={T}, {len(configs)} configs × {len(metric_names)} metrics × {len(deg_names)} degradations")
    print("=" * 90)

    all_results = {}

    for ci, (cfg_name, pcfg) in enumerate(configs):
        print(f"\n[{ci+1}/{len(configs)}] {cfg_name}")
        print("-" * 70)

        # Clean run
        loader_clean = build_loader(paths, batch_size=32, num_workers=0)
        clean_tp, clean_wtp, clean_tk64 = run_mc_with_features(vlm, loader_clean, pcfg, T, 42)
        clean = {"trace_pre": clean_tp, "weighted_trace_pre": clean_wtp, "topk64_trace_pre": clean_tk64}

        cfg_results = {}

        for deg_name in deg_names:
            ds = DegradedImageDataset(paths, DEGRADATIONS[deg_name])
            loader_deg = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=pil_collate)
            deg_tp, deg_wtp, deg_tk64 = run_mc_with_features(vlm, loader_deg, pcfg, T, 42)
            deg = {"trace_pre": deg_tp, "weighted_trace_pre": deg_wtp, "topk64_trace_pre": deg_tk64}

            for mn in metric_names:
                result = paired_test(clean[mn], deg[mn])
                key = f"{deg_name}__{mn}"
                cfg_results[key] = result

                frac = result["frac_increased"]
                status = "PASS" if frac >= 0.75 else "weak" if frac >= 0.60 else "FAIL"
                print(f"  {deg_name:>14s} | {mn:>22s}: {frac:5.1%}  [{status}]  p={result['p_value']:.2e}")

        all_results[cfg_name] = cfg_results

    # ── Summary table ──
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")
    header = f"{'Config':>22s}"
    for deg in deg_names:
        for mn in metric_names:
            short = mn.replace("_trace_pre", "").replace("trace_pre", "trace")
            header += f"  {deg[:4]}_{short:>8s}"
    print(header)
    print("-" * len(header))

    best_frac = 0.0
    best_cfg = ""
    best_detail = ""

    for cfg_name, cfg_results in all_results.items():
        row = f"{cfg_name:>22s}"
        for deg in deg_names:
            for mn in metric_names:
                key = f"{deg}__{mn}"
                frac = cfg_results[key]["frac_increased"]
                status = "+" if frac >= 0.75 else "~" if frac >= 0.60 else "-"
                row += f"  {frac:5.1%}{status:>2s}"
                if frac > best_frac:
                    best_frac = frac
                    best_cfg = cfg_name
                    best_detail = f"{deg}/{mn}"
        print(row)

    print(f"\nBest: {best_cfg} @ {best_detail} → {best_frac:.1%}")
    elapsed = (time.time() - t0) / 60
    print(f"Elapsed: {elapsed:.1f} min")

    out_path = Path("outputs/pe_core_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
