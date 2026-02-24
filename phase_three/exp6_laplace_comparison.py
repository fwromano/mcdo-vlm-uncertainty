#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from phase_one.common import (
    auroc_from_scores,
    build_loader,
    discover_class_names,
    list_images,
    load_manifest,
    load_model,
    parse_templates,
    reliability_from_trials,
    run_mc_trial,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
    should_save_checkpoint,
    spearman_safe,
)
from phase_two.dropout_types import configure_dropout, set_dropout_train_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Exp 6: MCDO vs projection-Laplace comparison")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--class-map", type=str, default="")
    parser.add_argument("--templates", type=str, default="a photo of a {}|a {}|an image of a {}")
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--laplace-lambda", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def _classification_metrics(
    vlm,
    loader,
    class_names: List[str],
    templates: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    text_features_by_template: List[torch.Tensor] = []
    for template in templates:
        prompts = [template.format(name) for name in class_names]
        text_features_by_template.append(vlm.encode_texts(prompts, normalize=True))

    margins: List[np.ndarray] = []
    entropies: List[np.ndarray] = []
    prompt_vars: List[np.ndarray] = []

    with torch.no_grad():
        for images, _paths, _ in loader:
            image_features = vlm.encode_images(images, normalize=True)
            logits_main = vlm.similarity_logits(image_features, text_features_by_template[0])

            top2 = torch.topk(logits_main, k=2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()

            probs_main = F.softmax(logits_main, dim=-1)
            entropy = (-(probs_main * torch.log(probs_main.clamp_min(1e-12))).sum(dim=-1)).cpu().numpy()

            max_probs = []
            for text_features in text_features_by_template:
                logits = vlm.similarity_logits(image_features, text_features)
                probs = F.softmax(logits, dim=-1)
                max_probs.append(probs.max(dim=-1).values.cpu().numpy())
            max_probs_arr = np.stack(max_probs, axis=0)
            if max_probs_arr.shape[0] > 1:
                prompt_var = np.var(max_probs_arr, axis=0, ddof=1)
            else:
                prompt_var = np.zeros(max_probs_arr.shape[1], dtype=np.float64)

            margins.append(margin)
            entropies.append(entropy)
            prompt_vars.append(prompt_var)

    return (
        np.concatenate(margins),
        np.concatenate(entropies),
        np.concatenate(prompt_vars),
        text_features_by_template[0],
    )


def _mcdo_uncertainty(
    vlm,
    loader,
    dropout: float,
    passes: int,
    trials: int,
    seed: int,
    save_every: int = 0,
    partial_npz_path: str = "",
    progress_json_path: str = "",
    progress_meta: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    trials_out: List[np.ndarray] = []
    for trial_idx in range(trials):
        set_all_seeds(seed + trial_idx)
        configure_dropout(vlm, dropout_type="E", p=dropout)
        trial = run_mc_trial(vlm=vlm, loader=loader, passes=passes, collect_pass_features=False)
        trials_out.append(trial["trace_pre"].numpy())

        completed = trial_idx + 1
        if partial_npz_path and should_save_checkpoint(completed=completed, total=trials, every=save_every):
            np.savez_compressed(
                partial_npz_path,
                mcdo_uncertainty_trials=np.stack(trials_out, axis=0),
                completed_trials=np.asarray([completed], dtype=np.int64),
                total_trials=np.asarray([trials], dtype=np.int64),
            )
            if progress_json_path:
                payload: Dict[str, object] = {
                    "completed_trials": completed,
                    "total_trials": trials,
                }
                if progress_meta:
                    payload.update(progress_meta)
                save_json(payload, progress_json_path)
    return np.stack(trials_out, axis=0)


def _collect_features_and_logits(vlm, loader, text_main: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []

    with torch.no_grad():
        for images, _paths, _ in loader:
            image_features = vlm.encode_images(images, normalize=True)
            logits = vlm.similarity_logits(image_features, text_main)
            features.append(image_features.detach().cpu().numpy().astype(np.float64, copy=False))
            logits_all.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.concatenate(features, axis=0), np.concatenate(logits_all, axis=0)


def _projection_laplace_uncertainty(
    x: np.ndarray,
    pseudo_labels: np.ndarray,
    num_classes: int,
    ridge_lambda: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n, d = x.shape
    eye = np.eye(d, dtype=np.float64)
    xtx = x.T @ x
    a = xtx + ridge_lambda * eye
    a_inv = np.linalg.pinv(a)

    leverage = np.einsum("nd,dd,nd->n", x, a_inv, x)
    leverage = np.clip(leverage, 0.0, None)

    xty = np.zeros((d, num_classes), dtype=np.float64)
    for class_idx in range(num_classes):
        idx = pseudo_labels == class_idx
        if np.any(idx):
            xty[:, class_idx] = x[idx].sum(axis=0)

    w = a_inv @ xty
    w_selected = w[:, pseudo_labels].T
    pred_assigned = np.sum(x * w_selected, axis=1)
    residual = 1.0 - pred_assigned
    sigma2 = float(np.mean(residual * residual))

    uncertainty = sigma2 * leverage
    return uncertainty, {
        "sigma2": sigma2,
        "leverage_mean": float(np.mean(leverage)),
        "leverage_std": float(np.std(leverage)),
        "ridge_lambda": float(ridge_lambda),
        "num_samples": float(n),
        "feature_dim": float(d),
    }


def _method_metrics(unc: np.ndarray, margins: np.ndarray, entropies: np.ndarray, prompt_sens: np.ndarray, quantile: float) -> Dict[str, float]:
    low_margin_thr = np.quantile(margins, quantile)
    high_entropy_thr = np.quantile(entropies, 1.0 - quantile)
    low_margin = (margins <= low_margin_thr).astype(np.int64)
    high_entropy = (entropies >= high_entropy_thr).astype(np.int64)

    return {
        "rho_uncertainty_vs_negative_margin": spearman_safe(unc, -margins),
        "rho_uncertainty_vs_entropy": spearman_safe(unc, entropies),
        "rho_uncertainty_vs_prompt_sensitivity": spearman_safe(unc, prompt_sens),
        "auroc_low_margin": auroc_from_scores(unc, low_margin),
        "auroc_high_entropy": auroc_from_scores(unc, high_entropy),
        "low_margin_threshold": float(low_margin_thr),
        "high_entropy_threshold": float(high_entropy_thr),
    }


def _reliability_or_none(values: np.ndarray) -> Optional[Dict[str, float]]:
    if values.shape[0] < 2:
        return None
    return reliability_from_trials(values)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        sampled_paths = sample_paths(list_images(args.data_dir), args.num_images, args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase3_exp6_manifest.json"))

    class_names = discover_class_names(args.data_dir, mapping_path=args.class_map or None)
    templates = parse_templates(args.templates)
    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp6] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            payload: Dict[str, object] = {"error": str(exc)}
            overall[model_key] = payload
            save_json(payload, str(model_out / "exp6_error.json"))
            continue

        margins, entropies, prompt_sens, text_main = _classification_metrics(
            vlm=vlm,
            loader=loader,
            class_names=class_names,
            templates=templates,
        )

        t0 = time.perf_counter()
        mcdo_trials = _mcdo_uncertainty(
            vlm=vlm,
            loader=loader,
            dropout=args.dropout,
            passes=args.passes,
            trials=args.trials,
            seed=args.seed,
            save_every=args.save_every,
            partial_npz_path=str(model_out / "exp6_mcdo_trials_partial.npz"),
            progress_json_path=str(model_out / "exp6_mcdo_progress.json"),
            progress_meta={"experiment": "exp6_laplace_comparison", "model": model_key},
        )
        mcdo_unc = mcdo_trials.mean(axis=0)
        mcdo_seconds = float(time.perf_counter() - t0)

        set_dropout_train_mode(vlm.vision_root, enabled=False)

        t1 = time.perf_counter()
        features, logits = _collect_features_and_logits(vlm=vlm, loader=loader, text_main=text_main)
        pseudo = np.argmax(logits, axis=1)
        laplace_unc, laplace_diag = _projection_laplace_uncertainty(
            x=features,
            pseudo_labels=pseudo,
            num_classes=logits.shape[1],
            ridge_lambda=args.laplace_lambda,
        )
        laplace_seconds = float(time.perf_counter() - t1)

        mcdo_metrics = _method_metrics(mcdo_unc, margins, entropies, prompt_sens, quantile=args.quantile)
        laplace_metrics = _method_metrics(laplace_unc, margins, entropies, prompt_sens, quantile=args.quantile)

        summary = {
            "experiment": "exp6_laplace_comparison",
            "model": model_key,
            "num_images": len(sampled_paths),
            "num_classes": len(class_names),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "laplace_lambda": args.laplace_lambda,
            "mcdo": {
                "metrics": mcdo_metrics,
                "runtime_seconds": mcdo_seconds,
                "uncertainty_mean": float(np.mean(mcdo_unc)),
                "uncertainty_std": float(np.std(mcdo_unc)),
                "reliability": _reliability_or_none(mcdo_trials),
            },
            "projection_laplace": {
                "metrics": laplace_metrics,
                "runtime_seconds": laplace_seconds,
                "uncertainty_mean": float(np.mean(laplace_unc)),
                "uncertainty_std": float(np.std(laplace_unc)),
                "diagnostics": laplace_diag,
            },
            "cross_method": {
                "rho_mcdo_vs_laplace": spearman_safe(mcdo_unc, laplace_unc),
                "mcdo_runtime_per_image_ms": (1000.0 * mcdo_seconds / max(len(sampled_paths), 1)),
                "laplace_runtime_per_image_ms": (1000.0 * laplace_seconds / max(len(sampled_paths), 1)),
            },
            "templates": templates,
            "quantile": args.quantile,
        }

        np.savez_compressed(
            model_out / "exp6_outputs.npz",
            paths=np.asarray(sampled_paths),
            margin=margins,
            entropy=entropies,
            prompt_sensitivity=prompt_sens,
            mcdo_uncertainty_trials=mcdo_trials,
            mcdo_uncertainty=mcdo_unc,
            laplace_uncertainty=laplace_unc,
            pseudo_labels=pseudo,
        )

        save_json(summary, str(model_out / "exp6_summary.json"))
        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp6_laplace_comparison",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "results": overall,
        },
        str(out_dir / "exp6_overall_summary.json"),
    )

    print(f"[Exp6] Complete: {out_dir}")


if __name__ == "__main__":
    main()
