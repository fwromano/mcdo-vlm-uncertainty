#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from phase_one.common import (
    build_loader,
    list_images,
    load_manifest,
    load_model,
    run_mc_trial,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
    should_save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 Exp 0b: pre/post norm covariance geometry")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="", help="Optional existing manifest JSON")
    parser.add_argument("--model", type=str, default="clip_b32")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def geometry_from_pass_features(pass_features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return trace/d, off-diagonal mass ratio, top-10 eigvals per image.

    `pass_features` shape: (T, N, D)
    """
    t, n, _ = pass_features.shape
    trace = np.zeros(n, dtype=np.float64)
    offdiag_ratio = np.zeros(n, dtype=np.float64)
    eig_top10 = np.zeros((n, 10), dtype=np.float64)

    denom = max(t - 1, 1)

    for idx in range(n):
        x = pass_features[:, idx, :].to(torch.float64)  # T x D
        xc = x - x.mean(dim=0, keepdim=True)

        var_d = (xc.pow(2).sum(dim=0) / float(denom)).clamp_min(0.0)
        trace[idx] = float(var_d.mean().item())

        # Non-zero covariance spectrum is captured by T x T Gram matrix.
        gram = (xc @ xc.T) / float(denom)
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)

        k = min(10, eigvals.numel())
        if k > 0:
            top = torch.flip(eigvals[-k:], dims=[0])
            eig_top10[idx, :k] = top.cpu().numpy()

        fro_sq = float((eigvals**2).sum().item())
        diag_sq = float((var_d**2).sum().item())
        offdiag_sq = max(fro_sq - diag_sq, 0.0)
        denom_fro = max(fro_sq**0.5, 1e-12)
        offdiag_ratio[idx] = (offdiag_sq**0.5) / denom_fro

    return trace, offdiag_ratio, eig_top10


def summarize(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25.0)),
        "p75": float(np.percentile(arr, 75.0)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        all_paths = list_images(args.data_dir)
        sampled_paths = sample_paths(all_paths, num_images=args.num_images, seed=args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase1_exp0b_manifest.json"))

    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    vlm = load_model(args.model, device=args.device)
    replaced = vlm.ensure_uniform_dropout(args.dropout)

    trial_trace_pre: List[np.ndarray] = []
    trial_trace_post: List[np.ndarray] = []
    trial_angular: List[np.ndarray] = []
    trial_offdiag_pre: List[np.ndarray] = []
    trial_offdiag_post: List[np.ndarray] = []
    trial_eigs_pre: List[np.ndarray] = []
    trial_eigs_post: List[np.ndarray] = []

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

        trace_pre, offdiag_pre, eigs_pre = geometry_from_pass_features(trial["pass_pre"])
        trace_post, offdiag_post, eigs_post = geometry_from_pass_features(trial["pass_post"])

        trial_trace_pre.append(trace_pre)
        trial_trace_post.append(trace_post)
        trial_angular.append(trial["angular_var"].numpy())
        trial_offdiag_pre.append(offdiag_pre)
        trial_offdiag_post.append(offdiag_post)
        trial_eigs_pre.append(eigs_pre)
        trial_eigs_post.append(eigs_post)

        completed = trial_idx + 1
        if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
            np.savez_compressed(
                out_dir / "exp0b_geometry_trials_partial.npz",
                paths=np.asarray(sampled_paths),
                trace_pre=np.stack(trial_trace_pre, axis=0),
                trace_post=np.stack(trial_trace_post, axis=0),
                angular_var=np.stack(trial_angular, axis=0),
                offdiag_pre=np.stack(trial_offdiag_pre, axis=0),
                offdiag_post=np.stack(trial_offdiag_post, axis=0),
                eigs_pre=np.stack(trial_eigs_pre, axis=0),
                eigs_post=np.stack(trial_eigs_post, axis=0),
                completed_trials=np.asarray([completed], dtype=np.int64),
                total_trials=np.asarray([args.trials], dtype=np.int64),
            )
            save_json(
                {
                    "experiment": "exp0b_norm_geometry",
                    "model": args.model,
                    "completed_trials": completed,
                    "total_trials": args.trials,
                },
                str(out_dir / "exp0b_progress.json"),
            )

    arr_trace_pre = np.stack(trial_trace_pre, axis=0)
    arr_trace_post = np.stack(trial_trace_post, axis=0)
    arr_angular = np.stack(trial_angular, axis=0)
    arr_offdiag_pre = np.stack(trial_offdiag_pre, axis=0)
    arr_offdiag_post = np.stack(trial_offdiag_post, axis=0)
    arr_eigs_pre = np.stack(trial_eigs_pre, axis=0)
    arr_eigs_post = np.stack(trial_eigs_post, axis=0)

    np.savez_compressed(
        out_dir / "exp0b_geometry_trials.npz",
        paths=np.asarray(sampled_paths),
        trace_pre=arr_trace_pre,
        trace_post=arr_trace_post,
        angular_var=arr_angular,
        offdiag_pre=arr_offdiag_pre,
        offdiag_post=arr_offdiag_post,
        eigs_pre=arr_eigs_pre,
        eigs_post=arr_eigs_post,
    )

    summary = {
        "experiment": "exp0b_norm_geometry",
        "model": args.model,
        "num_images": len(sampled_paths),
        "trials": args.trials,
        "passes": args.passes,
        "dropout": args.dropout,
        "injected_linear_dropout_wrappers": replaced,
        "pre_norm": {
            "trace_per_dim": summarize(arr_trace_pre.mean(axis=0)),
            "offdiag_mass_ratio": summarize(arr_offdiag_pre.mean(axis=0)),
            "eigs_top10_mean": arr_eigs_pre.mean(axis=(0, 1)).tolist(),
        },
        "post_norm": {
            "trace_per_dim": summarize(arr_trace_post.mean(axis=0)),
            "offdiag_mass_ratio": summarize(arr_offdiag_post.mean(axis=0)),
            "eigs_top10_mean": arr_eigs_post.mean(axis=(0, 1)).tolist(),
        },
        "angular": {
            "angular_var": summarize(arr_angular.mean(axis=0)),
        },
    }
    save_json(summary, str(out_dir / "exp0b_summary.json"))

    print(f"[Exp0b] Complete. Results: {out_dir}")


if __name__ == "__main__":
    main()
