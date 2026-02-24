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
    parser = argparse.ArgumentParser(description="Phase 2 Exp 1: rank stability across dropout rates p")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--p-values", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
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
        save_manifest(sampled_paths, str(out_dir / "phase2_exp1_manifest.json"))

    p_values = [float(x.strip()) for x in args.p_values.split(",") if x.strip()]
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]

    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp1] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            overall[model_key] = {"error": str(exc)}
            save_json(overall[model_key], str(model_out / "exp1_error.json"))
            continue

        p_mean_unc: List[np.ndarray] = []
        p_trial_arrays: Dict[str, np.ndarray] = {}
        p_reliability: Dict[str, Dict[str, float]] = {}
        p_wrapped: Dict[str, int] = {}

        for p in p_values:
            trial_unc: List[np.ndarray] = []
            wrapped_count = 0
            for trial_idx in range(args.trials):
                set_all_seeds(args.seed + int(p * 1e6) + trial_idx)
                cfg = configure_dropout(vlm, dropout_type="E", p=p)
                wrapped_count = max(wrapped_count, cfg.wrapped_modules)
                trial = run_mc_trial(vlm=vlm, loader=loader, passes=args.passes, collect_pass_features=False)
                trial_unc.append(trial["trace_pre"].numpy())

                completed = trial_idx + 1
                if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                    partial = np.stack(trial_unc, axis=0)
                    np.savez_compressed(
                        model_out / f"exp1_p_{str(p).replace('.', '_')}_partial.npz",
                        paths=np.asarray(sampled_paths),
                        p_value=np.asarray([p], dtype=np.float64),
                        uncertainty_trials=partial,
                        completed_trials=np.asarray([completed], dtype=np.int64),
                        total_trials=np.asarray([args.trials], dtype=np.int64),
                    )
                    save_json(
                        {
                            "experiment": "exp1_rank_p",
                            "model": model_key,
                            "p_value": p,
                            "completed_trials": completed,
                            "total_trials": args.trials,
                        },
                        str(model_out / f"exp1_p_{str(p).replace('.', '_')}_progress.json"),
                    )
            arr = np.stack(trial_unc, axis=0)
            p_mean_unc.append(arr.mean(axis=0))
            p_trial_arrays[str(p)] = arr
            p_reliability[str(p)] = reliability_from_trials(arr)
            p_wrapped[str(p)] = wrapped_count

        mean_matrix = np.stack(p_mean_unc, axis=0)
        rho_matrix = spearman_matrix(mean_matrix)

        np.savez_compressed(
            model_out / "exp1_rank_p_trials.npz",
            paths=np.asarray(sampled_paths),
            p_values=np.asarray(p_values, dtype=np.float64),
            mean_uncertainty=mean_matrix,
            spearman_matrix=rho_matrix,
            **{f"trials_p_{str(p).replace('.', '_')}": arr for p, arr in p_trial_arrays.items()},
        )

        summary = {
            "experiment": "exp1_rank_p",
            "model": model_key,
            "num_images": len(sampled_paths),
            "passes": args.passes,
            "trials": args.trials,
            "p_values": p_values,
            "reliability_by_p": p_reliability,
            "wrapped_modules_by_p": p_wrapped,
            "spearman_matrix": rho_matrix.tolist(),
        }
        save_json(summary, str(model_out / "exp1_summary.json"))
        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp1_rank_p",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "p_values": p_values,
            "results": overall,
        },
        str(out_dir / "exp1_overall_summary.json"),
    )

    print(f"[Exp1] Complete: {out_dir}")


if __name__ == "__main__":
    main()
