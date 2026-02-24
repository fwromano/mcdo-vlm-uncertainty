#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from phase_one.common import (
    build_loader,
    list_images,
    load_model,
    run_mc_trial,
    sample_paths,
    save_json,
    set_all_seeds,
    should_save_checkpoint,
)
from phase_two.dropout_types import configure_dropout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Exp 2: synthetic + natural baselines")
    parser.add_argument("data_dir", type=str, help="Natural images root")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--num-natural", type=int, default=10)
    parser.add_argument("--num-each-synth", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def _solid(size: int, color: Tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    return Image.fromarray(arr)


def _gradient(size: int, horizontal: bool, invert: bool) -> Image.Image:
    x = np.linspace(0, 255, size, dtype=np.float32)
    grad = np.tile(x[None, :], (size, 1)) if horizontal else np.tile(x[:, None], (1, size))
    if invert:
        grad = 255.0 - grad
    arr = np.stack([grad, np.roll(grad, shift=size // 3, axis=1), np.roll(grad, shift=size // 5, axis=0)], axis=-1)
    return Image.fromarray(arr.astype(np.uint8))


def _noise(size: int, rng: np.random.Generator) -> Image.Image:
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def build_synthetic_set(out_dir: Path, size: int, n_each: int, seed: int) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    synth_root = out_dir / "synthetic_inputs"
    categories = ["solid", "gradient", "noise"]
    for cat in categories:
        (synth_root / cat).mkdir(parents=True, exist_ok=True)

    paths: Dict[str, List[str]] = {k: [] for k in categories}

    for i in range(n_each):
        color = tuple(int(x) for x in rng.integers(0, 256, size=3))
        img = _solid(size, color)
        p = synth_root / "solid" / f"solid_{i:03d}.png"
        img.save(p)
        paths["solid"].append(str(p))

    for i in range(n_each):
        img = _gradient(size, horizontal=(i % 2 == 0), invert=(i % 3 == 0))
        p = synth_root / "gradient" / f"gradient_{i:03d}.png"
        img.save(p)
        paths["gradient"].append(str(p))

    for i in range(n_each):
        img = _noise(size, rng)
        p = synth_root / "noise" / f"noise_{i:03d}.png"
        img.save(p)
        paths["noise"].append(str(p))

    return paths


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

    synth_paths = build_synthetic_set(out_dir, size=args.image_size, n_each=args.num_each_synth, seed=args.seed)

    natural = sample_paths(list_images(args.data_dir), args.num_natural, args.seed)
    all_categories: Dict[str, List[str]] = {
        "solid": synth_paths["solid"],
        "gradient": synth_paths["gradient"],
        "noise": synth_paths["noise"],
        "natural": natural,
    }

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    results: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp2] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            results[model_key] = {"error": str(exc)}
            save_json(results[model_key], str(model_out / "exp2_error.json"))
            continue

        cat_uncertainty: Dict[str, np.ndarray] = {}
        wrapped = 0

        for category, paths in all_categories.items():
            loader = build_loader(paths, batch_size=args.batch_size, num_workers=args.num_workers)
            trial_values: List[np.ndarray] = []
            for trial_idx in range(args.trials):
                set_all_seeds(args.seed + 1_000 * trial_idx + len(category))
                cfg = configure_dropout(vlm, dropout_type="E", p=args.dropout)
                wrapped = max(wrapped, cfg.wrapped_modules)
                trial = run_mc_trial(vlm=vlm, loader=loader, passes=args.passes, collect_pass_features=False)
                trial_values.append(trial["trace_pre"].numpy())

                completed = trial_idx + 1
                if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                    partial = np.stack(trial_values, axis=0)
                    np.savez_compressed(
                        model_out / f"exp2_{category}_partial.npz",
                        paths=np.asarray(paths),
                        uncertainty_trials=partial,
                        completed_trials=np.asarray([completed], dtype=np.int64),
                        total_trials=np.asarray([args.trials], dtype=np.int64),
                    )
                    save_json(
                        {
                            "experiment": "exp2_synthetic_natural",
                            "model": model_key,
                            "category": category,
                            "completed_trials": completed,
                            "total_trials": args.trials,
                        },
                        str(model_out / f"exp2_{category}_progress.json"),
                    )
            arr = np.stack(trial_values, axis=0).mean(axis=0)
            cat_uncertainty[category] = arr

        ranking = sorted(cat_uncertainty.keys(), key=lambda c: float(cat_uncertainty[c].mean()))
        summary = {
            "experiment": "exp2_synthetic_natural",
            "model": model_key,
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "num_each_synth": args.num_each_synth,
            "num_natural": args.num_natural,
            "wrapped_modules": wrapped,
            "category_stats": {cat: summarize(vals) for cat, vals in cat_uncertainty.items()},
            "ascending_uncertainty_order": ranking,
        }

        np.savez_compressed(model_out / "exp2_uncertainty_by_category.npz", **cat_uncertainty)
        save_json(summary, str(model_out / "exp2_summary.json"))
        results[model_key] = summary

    save_json(
        {
            "experiment": "exp2_synthetic_natural",
            "models": model_keys,
            "results": results,
        },
        str(out_dir / "exp2_overall_summary.json"),
    )

    print(f"[Exp2] Complete: {out_dir}")


if __name__ == "__main__":
    main()
