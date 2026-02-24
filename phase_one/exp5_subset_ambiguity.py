#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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
    run_mc_trial,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
    should_save_checkpoint,
    spearman_safe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 Exp 5 subset: ambiguity prediction on ImageNet subset")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="", help="Optional existing manifest JSON")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--class-map", type=str, default="", help="Optional JSON or TSV folder->label map")
    parser.add_argument(
        "--templates",
        type=str,
        default="a photo of a {}|a {}|an image of a {}",
        help="Prompt templates separated by '|', each containing {}",
    )
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=1, help="Uncertainty trials K")
    parser.add_argument("--quantile", type=float, default=0.10, help="Tail quantile for AUROC labels")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        all_paths = list_images(args.data_dir)
        sampled_paths = sample_paths(all_paths, num_images=args.num_images, seed=args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase1_exp5_manifest.json"))

    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    templates = parse_templates(args.templates)
    class_names = discover_class_names(args.data_dir, mapping_path=args.class_map or None)

    print(f"[Exp5-subset] Using {len(class_names)} class prompts across {len(templates)} templates")

    all_summaries: Dict[str, Dict[str, float]] = {}

    for model_key in model_keys:
        print(f"[Exp5-subset] Loading model: {model_key}")
        vlm = load_model(model_key, device=args.device)
        injected = vlm.ensure_uniform_dropout(args.dropout)
        vlm.disable_dropout()

        # Deterministic ambiguity metrics.
        text_features_by_template: List[torch.Tensor] = []
        for template in templates:
            prompts = [template.format(name) for name in class_names]
            text_features = vlm.encode_texts(prompts, normalize=True)
            text_features_by_template.append(text_features)

        margins: List[np.ndarray] = []
        entropies: List[np.ndarray] = []
        prompt_var_parts: List[np.ndarray] = []

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
                prompt_var_parts.append(prompt_var)

        margin_arr = np.concatenate(margins, axis=0)
        entropy_arr = np.concatenate(entropies, axis=0)
        prompt_var_arr = np.concatenate(prompt_var_parts, axis=0)

        # Uncertainty from MCDO trace/d on pre-norm features.
        uncertainty_trials = []
        for trial_idx in range(args.trials):
            seed = args.seed + trial_idx
            set_all_seeds(seed)
            vlm.ensure_uniform_dropout(args.dropout)
            trial = run_mc_trial(vlm=vlm, loader=loader, passes=args.passes, collect_pass_features=False)
            uncertainty_trials.append(trial["trace_pre"].numpy())

            completed = trial_idx + 1
            if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                unc_partial = np.stack(uncertainty_trials, axis=0)
                np.savez_compressed(
                    out_dir / f"exp5_subset_{model_key}_partial.npz",
                    paths=np.asarray(sampled_paths),
                    uncertainty_trials=unc_partial,
                    margin=margin_arr,
                    entropy=entropy_arr,
                    prompt_sensitivity=prompt_var_arr,
                    completed_trials=np.asarray([completed], dtype=np.int64),
                    total_trials=np.asarray([args.trials], dtype=np.int64),
                )
                save_json(
                    {
                        "experiment": "exp5_subset_ambiguity",
                        "model": model_key,
                        "completed_trials": completed,
                        "total_trials": args.trials,
                    },
                    str(out_dir / f"exp5_subset_{model_key}_progress.json"),
                )

        unc_trials_arr = np.stack(uncertainty_trials, axis=0)
        uncertainty = unc_trials_arr.mean(axis=0)

        low_margin_thr = np.quantile(margin_arr, args.quantile)
        high_entropy_thr = np.quantile(entropy_arr, 1.0 - args.quantile)
        low_margin = (margin_arr <= low_margin_thr).astype(np.int64)
        high_entropy = (entropy_arr >= high_entropy_thr).astype(np.int64)

        metrics = {
            "rho_uncertainty_vs_negative_margin": spearman_safe(uncertainty, -margin_arr),
            "rho_uncertainty_vs_entropy": spearman_safe(uncertainty, entropy_arr),
            "rho_uncertainty_vs_prompt_sensitivity": spearman_safe(uncertainty, prompt_var_arr),
            "auroc_low_margin": auroc_from_scores(uncertainty, low_margin),
            "auroc_high_entropy": auroc_from_scores(uncertainty, high_entropy),
        }

        np.savez_compressed(
            out_dir / f"exp5_subset_{model_key}.npz",
            paths=np.asarray(sampled_paths),
            uncertainty_trials=unc_trials_arr,
            uncertainty=uncertainty,
            margin=margin_arr,
            entropy=entropy_arr,
            prompt_sensitivity=prompt_var_arr,
            low_margin=low_margin,
            high_entropy=high_entropy,
        )

        summary = {
            "experiment": "exp5_subset_ambiguity",
            "model": model_key,
            "num_images": len(sampled_paths),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "injected_linear_dropout_wrappers": injected,
            "num_classes": len(class_names),
            "templates": templates,
            "quantile": args.quantile,
            "thresholds": {
                "low_margin": float(low_margin_thr),
                "high_entropy": float(high_entropy_thr),
            },
            "metrics": metrics,
        }

        save_json(summary, str(out_dir / f"exp5_subset_{model_key}_summary.json"))
        all_summaries[model_key] = metrics

    save_json(
        {
            "experiment": "exp5_subset_ambiguity",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "templates": templates,
            "num_classes": len(class_names),
            "results": all_summaries,
        },
        str(out_dir / "exp5_subset_overall_summary.json"),
    )

    print(f"[Exp5-subset] Complete. Results: {out_dir}")


if __name__ == "__main__":
    main()
