#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from phase_one.common import (
    build_loader,
    list_images,
    load_manifest,
    load_model,
    reliability_from_trials,
    run_mc_trial,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 Exp 4 subset: CLIP-B/32 vs SigLIP2-B/16")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="", help="Optional existing manifest JSON")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def status(metrics: Dict[str, float]) -> str:
    if metrics["pairwise_spearman_median"] >= 0.8 and metrics["snr"] >= 2.0 and metrics["icc"] >= 0.75:
        return "usable"
    if metrics["pairwise_spearman_median"] < 0.6 or metrics["snr"] < 1.0:
        return "failed"
    return "marginal"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        all_paths = list_images(args.data_dir)
        sampled_paths = sample_paths(all_paths, num_images=args.num_images, seed=args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase1_exp4_manifest.json"))

    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    all_results: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp4-subset] Loading model: {model_key}")
        vlm = load_model(model_key, device=args.device)
        replaced = vlm.ensure_uniform_dropout(args.dropout)

        trials_trace: List[np.ndarray] = []
        trials_angular: List[np.ndarray] = []

        for trial_idx in range(args.trials):
            seed = args.seed + trial_idx
            set_all_seeds(seed)
            vlm.ensure_uniform_dropout(args.dropout)
            trial = run_mc_trial(
                vlm=vlm,
                loader=loader,
                passes=args.passes,
                collect_pass_features=True,
                compute_angular=True,
            )
            trials_trace.append(trial["trace_pre"].numpy())
            trials_angular.append(trial["angular_var"].numpy())

        arr_trace = np.stack(trials_trace, axis=0)
        arr_angular = np.stack(trials_angular, axis=0)

        trace_metrics = reliability_from_trials(arr_trace)
        angular_metrics = reliability_from_trials(arr_angular)
        trace_metrics["status"] = status(trace_metrics)
        angular_metrics["status"] = status(angular_metrics)

        np.savez_compressed(
            out_dir / f"exp4_subset_{model_key}_trials.npz",
            paths=np.asarray(sampled_paths),
            trace_pre=arr_trace,
            angular_var=arr_angular,
        )

        all_results[model_key] = {
            "injected_linear_dropout_wrappers": replaced,
            "trace_pre": trace_metrics,
            "angular_var": angular_metrics,
            "trace_mean": float(arr_trace.mean()),
            "angular_mean": float(arr_angular.mean()),
        }

    comparison = {}
    if len(model_keys) == 2:
        m0, m1 = model_keys
        comparison = {
            "trace_mean_delta": all_results[m1]["trace_mean"] - all_results[m0]["trace_mean"],
            "angular_mean_delta": all_results[m1]["angular_mean"] - all_results[m0]["angular_mean"],
        }

    save_json(
        {
            "experiment": "exp4_subset_recipe",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "trials": args.trials,
            "passes": args.passes,
            "dropout": args.dropout,
            "results": all_results,
            "comparison": comparison,
        },
        str(out_dir / "exp4_subset_summary.json"),
    )

    print(f"[Exp4-subset] Complete. Results: {out_dir}")


if __name__ == "__main__":
    main()
