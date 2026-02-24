#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
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
from phase_two.dropout_types import DROPOUT_TYPES, configure_dropout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Exp 3: dropout type ablation")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32")
    parser.add_argument("--dropout-types", type=str, default="A,B,C,D,E")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def anisotropy_from_pass_features(pass_pre: torch.Tensor) -> np.ndarray:
    t, n, _ = pass_pre.shape
    denom = max(t - 1, 1)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x = pass_pre[:, i, :].to(torch.float64)
        xc = x - x.mean(dim=0, keepdim=True)
        var_d = (xc.pow(2).sum(dim=0) / float(denom)).clamp_min(0.0)
        trace_total = float(var_d.sum().item())
        if trace_total <= 1e-12:
            out[i] = 0.0
            continue
        gram = (xc @ xc.T) / float(denom)
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        eigmax = float(eigvals.max().item())
        out[i] = eigmax / trace_total

    return out


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
        save_manifest(sampled_paths, str(out_dir / "phase2_exp3_manifest.json"))

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    dropout_types = [d.strip().upper() for d in args.dropout_types.split(",") if d.strip()]
    invalid = [d for d in dropout_types if d not in DROPOUT_TYPES]
    if invalid:
        known = ", ".join(sorted(DROPOUT_TYPES))
        raise ValueError(f"Unsupported dropout types: {invalid}. Known: {known}")

    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp3] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        per_type: Dict[str, Dict[str, object]] = {}
        type_mean_trace: List[np.ndarray] = []
        valid_types: List[str] = []

        for d_type in dropout_types:
            print(f"[Exp3]   type={d_type}")
            try:
                vlm = load_model(model_key, device=args.device)
            except Exception as exc:  # noqa: BLE001
                per_type[d_type] = {"error": str(exc)}
                continue

            trial_trace: List[np.ndarray] = []
            trial_angular: List[np.ndarray] = []
            trial_aniso: List[np.ndarray] = []
            wrapped = 0
            selected_paths: List[str] = []
            notes = ""

            for trial_idx in range(args.trials):
                set_all_seeds(args.seed + 10_000 * trial_idx + ord(d_type))
                cfg = configure_dropout(vlm, dropout_type=d_type, p=args.dropout)
                wrapped = max(wrapped, cfg.wrapped_modules)
                selected_paths = cfg.selected_paths
                notes = cfg.notes

                trial = run_mc_trial(
                    vlm=vlm,
                    loader=loader,
                    passes=args.passes,
                    collect_pass_features=True,
                    compute_angular=True,
                )
                trial_trace.append(trial["trace_pre"].numpy())
                trial_angular.append(trial["angular_var"].numpy())
                trial_aniso.append(anisotropy_from_pass_features(trial["pass_pre"]))

                completed = trial_idx + 1
                if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                    np.savez_compressed(
                        model_out / f"exp3_type_{d_type}_partial.npz",
                        paths=np.asarray(sampled_paths),
                        trace_pre=np.stack(trial_trace, axis=0),
                        angular_var=np.stack(trial_angular, axis=0),
                        anisotropy=np.stack(trial_aniso, axis=0),
                        completed_trials=np.asarray([completed], dtype=np.int64),
                        total_trials=np.asarray([args.trials], dtype=np.int64),
                    )
                    save_json(
                        {
                            "experiment": "exp3_dropout_type",
                            "model": model_key,
                            "dropout_type": d_type,
                            "completed_trials": completed,
                            "total_trials": args.trials,
                        },
                        str(model_out / f"exp3_type_{d_type}_progress.json"),
                    )

            arr_trace = np.stack(trial_trace, axis=0)
            arr_angular = np.stack(trial_angular, axis=0)
            arr_aniso = np.stack(trial_aniso, axis=0)

            np.savez_compressed(
                model_out / f"exp3_type_{d_type}_trials.npz",
                paths=np.asarray(sampled_paths),
                trace_pre=arr_trace,
                angular_var=arr_angular,
                anisotropy=arr_aniso,
            )

            per_type[d_type] = {
                "wrapped_modules": wrapped,
                "selected_paths": selected_paths,
                "notes": notes,
                "trace_pre": reliability_from_trials(arr_trace),
                "angular_var": reliability_from_trials(arr_angular),
                "anisotropy": reliability_from_trials(arr_aniso),
                "trace_mean": float(arr_trace.mean()),
                "angular_mean": float(arr_angular.mean()),
                "anisotropy_mean": float(arr_aniso.mean()),
            }
            valid_types.append(d_type)
            type_mean_trace.append(arr_trace.mean(axis=0))

        if type_mean_trace:
            trace_mat = np.stack(type_mean_trace, axis=0)
            cross_type_rho = spearman_matrix(trace_mat)
            cross_payload = {"types": valid_types, "trace_spearman_matrix": cross_type_rho.tolist()}
        else:
            cross_payload = {"types": [], "trace_spearman_matrix": []}

        summary = {
            "experiment": "exp3_dropout_type",
            "model": model_key,
            "num_images": len(sampled_paths),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "dropout_types": dropout_types,
            "results_by_type": per_type,
            "cross_type": cross_payload,
        }
        save_json(summary, str(model_out / "exp3_summary.json"))
        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp3_dropout_type",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "dropout_types": dropout_types,
            "results": overall,
        },
        str(out_dir / "exp3_overall_summary.json"),
    )

    print(f"[Exp3] Complete: {out_dir}")


if __name__ == "__main__":
    main()
