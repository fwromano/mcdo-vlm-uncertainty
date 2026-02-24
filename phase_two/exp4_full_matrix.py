#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

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
    should_save_checkpoint,
)
from phase_two.dropout_types import configure_dropout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Exp 4 full: cross-model matrix")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_b16,siglip2_so400m,clip_l14,siglip2_g16")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def spearman_matrix(rows: np.ndarray) -> np.ndarray:
    k = rows.shape[0]
    out = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            rho, _ = spearmanr(rows[i], rows[j])
            out[i, j] = out[j, i] = 0.0 if np.isnan(rho) else float(rho)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        sampled_paths = sample_paths(list_images(args.data_dir), args.num_images, args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase2_exp4_manifest.json"))

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    results: Dict[str, Dict[str, object]] = {}
    valid_models: List[str] = []
    mean_trace_rows: List[np.ndarray] = []
    mean_angular_rows: List[np.ndarray] = []

    for model_key in model_keys:
        print(f"[Exp4-full] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            results[model_key] = {"error": str(exc)}
            save_json(results[model_key], str(model_out / "exp4_error.json"))
            continue

        trace_trials: List[np.ndarray] = []
        angular_trials: List[np.ndarray] = []
        wrapped = 0

        for trial_idx in range(args.trials):
            set_all_seeds(args.seed + trial_idx)
            cfg = configure_dropout(vlm, dropout_type="E", p=args.dropout)
            wrapped = max(wrapped, cfg.wrapped_modules)
            trial = run_mc_trial(
                vlm=vlm,
                loader=loader,
                passes=args.passes,
                collect_pass_features=True,
                compute_angular=True,
            )
            trace_trials.append(trial["trace_pre"].numpy())
            angular_trials.append(trial["angular_var"].numpy())

            completed = trial_idx + 1
            if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                np.savez_compressed(
                    model_out / "exp4_full_trials_partial.npz",
                    paths=np.asarray(sampled_paths),
                    trace_pre=np.stack(trace_trials, axis=0),
                    angular_var=np.stack(angular_trials, axis=0),
                    completed_trials=np.asarray([completed], dtype=np.int64),
                    total_trials=np.asarray([args.trials], dtype=np.int64),
                )
                save_json(
                    {
                        "experiment": "exp4_full_matrix",
                        "model": model_key,
                        "completed_trials": completed,
                        "total_trials": args.trials,
                    },
                    str(model_out / "exp4_full_progress.json"),
                )

        arr_trace = np.stack(trace_trials, axis=0)
        arr_angular = np.stack(angular_trials, axis=0)

        np.savez_compressed(
            model_out / "exp4_full_trials.npz",
            paths=np.asarray(sampled_paths),
            trace_pre=arr_trace,
            angular_var=arr_angular,
        )

        summary = {
            "experiment": "exp4_full_matrix",
            "model": model_key,
            "num_images": len(sampled_paths),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "wrapped_modules": wrapped,
            "trace_pre": reliability_from_trials(arr_trace),
            "angular_var": reliability_from_trials(arr_angular),
            "trace_mean": float(arr_trace.mean()),
            "angular_mean": float(arr_angular.mean()),
        }

        save_json(summary, str(model_out / "exp4_full_summary.json"))
        results[model_key] = summary
        valid_models.append(model_key)
        mean_trace_rows.append(arr_trace.mean(axis=0))
        mean_angular_rows.append(arr_angular.mean(axis=0))

    if valid_models:
        trace_matrix = spearman_matrix(np.stack(mean_trace_rows, axis=0))
        angular_matrix = spearman_matrix(np.stack(mean_angular_rows, axis=0))
    else:
        trace_matrix = np.zeros((0, 0), dtype=np.float64)
        angular_matrix = np.zeros((0, 0), dtype=np.float64)

    save_json(
        {
            "experiment": "exp4_full_matrix",
            "models_requested": model_keys,
            "models_completed": valid_models,
            "num_images": len(sampled_paths),
            "results_by_model": results,
            "cross_model_spearman": {
                "trace_pre": trace_matrix.tolist(),
                "angular_var": angular_matrix.tolist(),
            },
        },
        str(out_dir / "exp4_full_overall_summary.json"),
    )

    print(f"[Exp4-full] Complete: {out_dir}")


if __name__ == "__main__":
    main()
