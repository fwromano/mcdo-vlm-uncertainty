#!/usr/bin/env python
"""Phase 2 Exp 6: MC Mean Convergence to Deterministic Embedding.

Tests whether the mean of T MC dropout forward passes converges to the
deterministic (no-dropout) embedding as T increases.  If MC dropout is
unbiased, the relative L2 distance should decrease as ~1/sqrt(T).

For each model and each snapshot T in {4,8,16,32,64,...}, we compute:
    mean_mc(T) = (1/T) sum_{t=1}^{T} f(x; mask_t)
    det        = f(x; no dropout)
    rel_dist   = ||mean_mc - det|| / ||det||

We report per-image relative distances and their population statistics
at each T value.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from phase_one.common import (
    VisionLanguageModel,
    build_loader,
    list_images,
    load_manifest,
    load_model,
    run_mc_trial,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
    set_dropout_mode,
    should_save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 Exp 6: MC mean convergence to deterministic embedding"
    )
    parser.add_argument("data_dir", type=str, help="Image directory (recursive)")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--manifest", type=str, default="", help="Optional manifest JSON")
    parser.add_argument(
        "--models",
        type=str,
        default="siglip2_b16,siglip2_so400m",
        help="Comma-separated model keys",
    )
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--trials", type=int, default=3, help="Independent trials per model")
    parser.add_argument(
        "--passes",
        type=str,
        default="4,8,16,32,64",
        help="Comma-separated T snapshot values (nested extraction from T_max)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def deterministic_features(
    vlm: VisionLanguageModel,
    loader: Any,
    *,
    use_precomputed: bool = True,
) -> torch.Tensor:
    """Single forward pass with dropout disabled -> (N, D) float64 features."""
    from phase_one.common import precompute_pixel_values

    vlm.disable_dropout()

    if use_precomputed:
        pixel_values, _ = precompute_pixel_values(vlm, loader, to_device=True)
        batch_size = int(loader.batch_size) if isinstance(loader.batch_size, int) and loader.batch_size > 0 else len(loader.dataset)
        parts = []
        for offset in range(0, pixel_values.shape[0], batch_size):
            batch = pixel_values[offset : offset + batch_size]
            feats = vlm.encode_pixel_values(batch, normalize=False).detach().cpu().to(torch.float64)
            parts.append(feats)
        return torch.cat(parts, dim=0)

    parts = []
    for images, _, _ in loader:
        feats = vlm.encode_images(images, normalize=False).detach().cpu().to(torch.float64)
        parts.append(feats)
    return torch.cat(parts, dim=0)


def nested_mc_means(
    vlm: VisionLanguageModel,
    loader: Any,
    T_max: int,
    snapshot_Ts: List[int],
    dropout_p: float,
    seed: int,
    progress: bool = False,
    progress_desc: str = "",
) -> Dict[int, torch.Tensor]:
    """Run T_max MC passes, snapshot the running mean at each T in snapshot_Ts.

    Returns dict mapping T -> mean features tensor (N, D) in float64.
    """
    from phase_one.common import precompute_pixel_values

    set_all_seeds(seed)
    vlm.ensure_uniform_dropout(dropout_p)

    sorted_Ts = sorted(snapshot_Ts)
    snap_idx = 0
    snapshots: Dict[int, torch.Tensor] = {}

    sum_pre: Optional[torch.Tensor] = None

    pixel_values, _ = precompute_pixel_values(vlm, loader, to_device=True)
    batch_size = int(loader.batch_size) if isinstance(loader.batch_size, int) and loader.batch_size > 0 else len(loader.dataset)

    pass_iter: Any = range(T_max)
    if progress:
        try:
            from tqdm.auto import tqdm
            pass_iter = tqdm(pass_iter, total=T_max, desc=progress_desc, leave=False)
        except Exception:
            pass

    for pass_idx in pass_iter:
        parts = []
        for offset in range(0, pixel_values.shape[0], batch_size):
            batch = pixel_values[offset : offset + batch_size]
            feats = vlm.encode_pixel_values(batch, normalize=False).detach().cpu().to(torch.float64)
            parts.append(feats)
        pre = torch.cat(parts, dim=0)

        if sum_pre is None:
            sum_pre = torch.zeros_like(pre)

        sum_pre += pre

        T_done = pass_idx + 1
        if snap_idx < len(sorted_Ts) and T_done == sorted_Ts[snap_idx]:
            snapshots[T_done] = sum_pre / T_done
            snap_idx += 1

    if progress and hasattr(pass_iter, "close"):
        pass_iter.close()

    return snapshots


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        all_paths = list_images(args.data_dir)
        sampled_paths = sample_paths(all_paths, num_images=args.num_images, seed=args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase2_exp6_manifest.json"))

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    passes_list = sorted(int(x.strip()) for x in args.passes.split(",") if x.strip())
    T_max = max(passes_list)

    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    overall: Dict[str, Any] = {}

    for model_key in model_keys:
        print(f"[Exp6] Loading model: {model_key}")
        vlm = load_model(model_key, device=args.device)
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        # Deterministic reference embedding (no dropout)
        print(f"[Exp6] {model_key}: computing deterministic reference embedding")
        det_feats = deterministic_features(vlm, loader)  # (N, D)
        det_norms = det_feats.norm(dim=1)  # (N,)

        trial_results: Dict[int, List[np.ndarray]] = {T: [] for T in passes_list}

        for trial_idx in range(args.trials):
            print(f"[Exp6] {model_key} | trial {trial_idx + 1}/{args.trials}")
            seed = args.seed + 100_000 * trial_idx

            snapshots = nested_mc_means(
                vlm, loader, T_max, passes_list, args.dropout, seed,
                progress=show_progress,
                progress_desc=f"Exp6 {model_key} trial {trial_idx + 1}/{args.trials}",
            )

            for T in passes_list:
                mc_mean = snapshots[T]  # (N, D)
                diff = mc_mean - det_feats
                abs_dist = diff.norm(dim=1)  # (N,)
                rel_dist = (abs_dist / det_norms.clamp(min=1e-12)).numpy()
                trial_results[T].append(rel_dist)

            completed = trial_idx + 1
            if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                for T in passes_list:
                    arr = np.stack(trial_results[T])
                    np.savez_compressed(
                        model_out / f"exp6_convergence_T{T}_partial.npz",
                        paths=np.asarray(sampled_paths),
                        rel_dist=arr,
                        completed_trials=np.asarray([completed], dtype=np.int64),
                    )

        # Aggregate per-T statistics across trials
        model_summary: Dict[str, Any] = {}
        for T in passes_list:
            arr = np.stack(trial_results[T])  # (K, N)
            # Average over trials, then compute population stats
            mean_over_trials = arr.mean(axis=0)  # (N,)
            model_summary[f"T={T}"] = {
                "num_trials": args.trials,
                "num_images": len(sampled_paths),
                "rel_dist_mean": float(mean_over_trials.mean()),
                "rel_dist_median": float(np.median(mean_over_trials)),
                "rel_dist_std": float(mean_over_trials.std()),
                "rel_dist_max": float(mean_over_trials.max()),
                "rel_dist_q95": float(np.percentile(mean_over_trials, 95)),
            }
            np.savez_compressed(
                model_out / f"exp6_convergence_T{T}.npz",
                paths=np.asarray(sampled_paths),
                rel_dist=arr,
            )

        # Check 1/sqrt(T) scaling: fit log(rel_dist_mean) vs log(T)
        log_T = np.log(np.array(passes_list, dtype=np.float64))
        log_dist = np.log(np.array([model_summary[f"T={T}"]["rel_dist_mean"] for T in passes_list]))
        if len(passes_list) >= 2:
            slope, intercept = np.polyfit(log_T, log_dist, 1)
        else:
            slope, intercept = float("nan"), float("nan")

        model_summary["convergence_slope"] = float(slope)
        model_summary["expected_slope"] = -0.5
        model_summary["slope_interpretation"] = (
            "slope ~ -0.5 confirms 1/sqrt(T) convergence (unbiased MC estimator)"
        )

        overall[model_key] = model_summary
        save_json(
            {
                "experiment": "exp6_mean_convergence",
                "model": model_key,
                "dropout": args.dropout,
                "trials": args.trials,
                "passes": passes_list,
                "num_images": len(sampled_paths),
                "results": model_summary,
            },
            str(model_out / "exp6_summary.json"),
        )

    save_json(
        {
            "experiment": "exp6_mean_convergence",
            "models": model_keys,
            "passes": passes_list,
            "dropout": args.dropout,
            "trials": args.trials,
            "num_images": len(sampled_paths),
            "results": overall,
        },
        str(out_dir / "exp6_overall_summary.json"),
    )

    print(f"[Exp6] Complete. Results: {out_dir}")


if __name__ == "__main__":
    main()
