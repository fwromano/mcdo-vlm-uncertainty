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
    parser = argparse.ArgumentParser(description="Phase 1 Exp 0: nested MC estimator validation")
    parser.add_argument("data_dir", type=str, help="Image directory (recursive)")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--manifest", type=str, default="", help="Optional existing manifest JSON")
    parser.add_argument(
        "--models",
        type=str,
        default="clip_b32,siglip2_b16,siglip2_so400m",
        help="Comma-separated model keys",
    )
    parser.add_argument("--num-images", type=int, default=500, help="Number of images")
    parser.add_argument("--dropout", type=float, default=0.01, help="Uniform linear-layer dropout p")
    parser.add_argument("--trials", type=int, default=10, help="K independent trials")
    parser.add_argument("--passes", type=str, default="4,16,64", help="Comma-separated MC passes T")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def status_from_metrics(metrics: Dict[str, float]) -> str:
    rho = metrics["pairwise_spearman_median"]
    snr = metrics["snr"]
    icc = metrics["icc"]
    if rho >= 0.8 and snr >= 2.0 and icc >= 0.75:
        return "usable"
    if rho < 0.6 or snr < 1.0:
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
        save_manifest(sampled_paths, str(out_dir / "phase1_exp0_manifest.json"))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    passes_list = [int(x.strip()) for x in args.passes.split(",") if x.strip()]

    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    overall: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_key in models:
        print(f"[Exp0] Loading model: {model_key}")
        vlm = load_model(model_key, device=args.device)
        replaced = vlm.ensure_uniform_dropout(args.dropout)
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        model_summary: Dict[str, Dict[str, float]] = {}

        for passes in passes_list:
            print(f"[Exp0] {model_key} | T={passes} | K={args.trials}")
            trial_pre: List[np.ndarray] = []
            trial_post: List[np.ndarray] = []

            for trial_idx in range(args.trials):
                seed = args.seed + 100_000 * passes + trial_idx
                set_all_seeds(seed)
                vlm.ensure_uniform_dropout(args.dropout)
                trial = run_mc_trial(vlm=vlm, loader=loader, passes=passes, collect_pass_features=False)
                trial_pre.append(trial["trace_pre"].numpy())
                trial_post.append(trial["trace_post"].numpy())

            pre_arr = np.stack(trial_pre, axis=0)
            post_arr = np.stack(trial_post, axis=0)

            pre_metrics = reliability_from_trials(pre_arr)
            post_metrics = reliability_from_trials(post_arr)
            pre_metrics["status"] = status_from_metrics(pre_metrics)
            post_metrics["status"] = status_from_metrics(post_metrics)

            model_summary[f"T={passes}"] = {
                "pre_norm": pre_metrics,
                "post_norm": post_metrics,
            }

            np.savez_compressed(
                model_out / f"exp0_trials_T{passes}.npz",
                paths=np.asarray(sampled_paths),
                trial_pre=pre_arr,
                trial_post=post_arr,
            )

        overall[model_key] = model_summary
        save_json(
            {
                "experiment": "exp0_nested_mc",
                "model": model_key,
                "dropout": args.dropout,
                "trials": args.trials,
                "passes": passes_list,
                "num_images": len(sampled_paths),
                "injected_linear_dropout_wrappers": replaced,
                "results": model_summary,
            },
            str(model_out / "exp0_summary.json"),
        )

    save_json(
        {
            "experiment": "exp0_nested_mc",
            "models": models,
            "passes": passes_list,
            "dropout": args.dropout,
            "trials": args.trials,
            "num_images": len(sampled_paths),
            "results": overall,
        },
        str(out_dir / "exp0_overall_summary.json"),
    )

    print(f"[Exp0] Complete. Results: {out_dir}")


if __name__ == "__main__":
    main()
