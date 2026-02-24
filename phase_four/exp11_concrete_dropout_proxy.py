#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from phase_four.layerwise_dropout import (
    build_vision_layer_groups,
    configure_groupwise_vision_dropout,
    flattened_group_paths,
    group_schedule_to_dict,
)
from phase_two.dropout_types import set_dropout_train_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 Exp 11: concrete-style layerwise dropout-rate search (proxy)")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--num-images", type=int, default=2000)
    parser.add_argument("--class-map", type=str, default="")
    parser.add_argument("--templates", type=str, default="a photo of a {}|a {}|an image of a {}")
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--base-p", type=float, default=0.01)
    parser.add_argument("--min-p", type=float, default=0.001)
    parser.add_argument("--max-p", type=float, default=0.1)
    parser.add_argument("--num-candidates", type=int, default=24)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def _classification_metrics(vlm, loader, class_names: Sequence[str], templates: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            margins.append((top2[:, 0] - top2[:, 1]).cpu().numpy())

            probs_main = F.softmax(logits_main, dim=-1)
            entropies.append((-(probs_main * torch.log(probs_main.clamp_min(1e-12))).sum(dim=-1)).cpu().numpy())

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
            prompt_vars.append(prompt_var)

    return np.concatenate(margins), np.concatenate(entropies), np.concatenate(prompt_vars)


def _run_schedule(
    vlm,
    loader,
    groups,
    p_values: Sequence[float],
    passes: int,
    trials: int,
    seed: int,
    save_every: int = 0,
    partial_npz_path: str = "",
    progress_json_path: str = "",
    progress_meta: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    p_by_group = group_schedule_to_dict(groups, p_values)
    out: List[np.ndarray] = []

    for trial_idx in range(trials):
        set_all_seeds(seed + trial_idx)
        configure_groupwise_vision_dropout(vlm=vlm, groups=groups, p_by_group=p_by_group)
        trial = run_mc_trial(vlm=vlm, loader=loader, passes=passes, collect_pass_features=False)
        out.append(trial["trace_pre"].numpy())

        completed = trial_idx + 1
        if partial_npz_path and should_save_checkpoint(completed=completed, total=trials, every=save_every):
            np.savez_compressed(
                partial_npz_path,
                uncertainty_trials=np.stack(out, axis=0),
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

    return np.stack(out, axis=0)


def _objective_and_metrics(
    unc: np.ndarray,
    margins: np.ndarray,
    entropies: np.ndarray,
    prompt_sens: np.ndarray,
    quantile: float,
) -> Tuple[float, Dict[str, float]]:
    low_margin_thr = np.quantile(margins, quantile)
    high_entropy_thr = np.quantile(entropies, 1.0 - quantile)
    low_margin = (margins <= low_margin_thr).astype(np.int64)
    high_entropy = (entropies >= high_entropy_thr).astype(np.int64)

    rho_neg_margin = spearman_safe(unc, -margins)
    rho_entropy = spearman_safe(unc, entropies)
    rho_prompt = spearman_safe(unc, prompt_sens)

    metrics = {
        "rho_uncertainty_vs_negative_margin": rho_neg_margin,
        "rho_uncertainty_vs_entropy": rho_entropy,
        "rho_uncertainty_vs_prompt_sensitivity": rho_prompt,
        "auroc_low_margin": auroc_from_scores(unc, low_margin),
        "auroc_high_entropy": auroc_from_scores(unc, high_entropy),
        "low_margin_threshold": float(low_margin_thr),
        "high_entropy_threshold": float(high_entropy_thr),
    }

    objective = float(rho_neg_margin + rho_entropy + 0.5 * rho_prompt)
    return objective, metrics


def _reliability_or_none(values: np.ndarray) -> Optional[Dict[str, float]]:
    if values.shape[0] < 2:
        return None
    return reliability_from_trials(values)


def _log_uniform_candidates(
    num_candidates: int,
    num_groups: int,
    min_p: float,
    max_p: float,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    lo = np.log(max(min_p, 1e-8))
    hi = np.log(max(max_p, min_p + 1e-8))
    out: List[np.ndarray] = []
    for _ in range(num_candidates):
        vec = np.exp(rng.uniform(lo, hi, size=num_groups))
        out.append(vec.astype(np.float64))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        sampled_paths = sample_paths(list_images(args.data_dir), args.num_images, args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase4_exp11_manifest.json"))

    class_names = discover_class_names(args.data_dir, mapping_path=args.class_map or None)
    templates = parse_templates(args.templates)
    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp11] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            payload: Dict[str, object] = {"error": str(exc)}
            overall[model_key] = payload
            save_json(payload, str(model_out / "exp11_error.json"))
            continue

        set_dropout_train_mode(vlm.vision_root, enabled=False)
        margins, entropies, prompt_sens = _classification_metrics(
            vlm=vlm,
            loader=loader,
            class_names=class_names,
            templates=templates,
        )

        groups = build_vision_layer_groups(vlm=vlm, num_groups=args.num_groups)
        if not groups:
            payload = {"error": "No linear layers found in vision tower"}
            overall[model_key] = payload
            save_json(payload, str(model_out / "exp11_error.json"))
            continue

        schedules: List[np.ndarray] = [np.full(len(groups), float(args.base_p), dtype=np.float64)]
        schedules.extend(
            _log_uniform_candidates(
                num_candidates=args.num_candidates,
                num_groups=len(groups),
                min_p=args.min_p,
                max_p=args.max_p,
                seed=args.seed,
            )
        )

        candidate_summaries: List[Dict[str, object]] = []
        best_idx = -1
        best_score = -float("inf")
        best_trials: Optional[np.ndarray] = None
        baseline_trials: Optional[np.ndarray] = None

        for cand_idx, p_values in enumerate(schedules):
            cand_seed = args.seed + 10_000 * cand_idx
            unc_trials = _run_schedule(
                vlm=vlm,
                loader=loader,
                groups=groups,
                p_values=p_values,
                passes=args.passes,
                trials=args.trials,
                seed=cand_seed,
                save_every=args.save_every,
                partial_npz_path=str(model_out / f"exp11_candidate_{cand_idx:03d}_partial.npz"),
                progress_json_path=str(model_out / f"exp11_candidate_{cand_idx:03d}_progress.json"),
                progress_meta={"experiment": "exp11_concrete_dropout_proxy", "model": model_key, "candidate_index": cand_idx},
            )
            unc = unc_trials.mean(axis=0)

            score, metrics = _objective_and_metrics(
                unc=unc,
                margins=margins,
                entropies=entropies,
                prompt_sens=prompt_sens,
                quantile=args.quantile,
            )

            reliability = _reliability_or_none(unc_trials)
            if reliability is not None:
                score += 0.10 * float(reliability.get("icc", 0.0))

            summary = {
                "candidate_index": cand_idx,
                "p_values": [float(x) for x in p_values.tolist()],
                "objective": float(score),
                "metrics": metrics,
                "uncertainty_mean": float(np.mean(unc)),
                "uncertainty_std": float(np.std(unc)),
                "reliability": reliability,
            }
            candidate_summaries.append(summary)

            if cand_idx == 0:
                baseline_trials = unc_trials

            if score > best_score:
                best_score = score
                best_idx = cand_idx
                best_trials = unc_trials

            save_json(
                {
                    "experiment": "exp11_concrete_dropout_proxy",
                    "model": model_key,
                    "completed_candidates": cand_idx + 1,
                    "total_candidates": len(schedules),
                    "best_candidate_index_so_far": best_idx,
                    "best_objective_so_far": best_score,
                },
                str(model_out / "exp11_search_progress.json"),
            )

        candidate_summaries.sort(key=lambda x: float(x["objective"]), reverse=True)
        top_candidates = candidate_summaries[: min(10, len(candidate_summaries))]

        best_raw = next(x for x in candidate_summaries if int(x["candidate_index"]) == best_idx)
        baseline_raw = next(x for x in candidate_summaries if int(x["candidate_index"]) == 0)

        summary = {
            "experiment": "exp11_concrete_dropout_proxy",
            "model": model_key,
            "num_images": len(sampled_paths),
            "num_classes": len(class_names),
            "passes": args.passes,
            "trials": args.trials,
            "num_groups": len(groups),
            "num_candidates_random": args.num_candidates,
            "search_space": {
                "min_p": args.min_p,
                "max_p": args.max_p,
                "base_p": args.base_p,
            },
            "best": best_raw,
            "baseline": baseline_raw,
            "improvement_over_baseline": {
                "objective_delta": float(best_raw["objective"]) - float(baseline_raw["objective"]),
                "rho_negative_margin_delta": float(best_raw["metrics"]["rho_uncertainty_vs_negative_margin"]) - float(baseline_raw["metrics"]["rho_uncertainty_vs_negative_margin"]),
                "rho_entropy_delta": float(best_raw["metrics"]["rho_uncertainty_vs_entropy"]) - float(baseline_raw["metrics"]["rho_uncertainty_vs_entropy"]),
                "rho_prompt_delta": float(best_raw["metrics"]["rho_uncertainty_vs_prompt_sensitivity"]) - float(baseline_raw["metrics"]["rho_uncertainty_vs_prompt_sensitivity"]),
                "auroc_low_margin_delta": float(best_raw["metrics"]["auroc_low_margin"]) - float(baseline_raw["metrics"]["auroc_low_margin"]),
                "auroc_high_entropy_delta": float(best_raw["metrics"]["auroc_high_entropy"]) - float(baseline_raw["metrics"]["auroc_high_entropy"]),
            },
            "group_sizes": {str(g.group_id): len(g.paths) for g in groups},
            "top_candidates": top_candidates,
        }

        npz_payload: Dict[str, np.ndarray] = {
            "paths": np.asarray(sampled_paths),
            "margin": margins,
            "entropy": entropies,
            "prompt_sensitivity": prompt_sens,
            "candidate_objectives": np.asarray([float(c["objective"]) for c in candidate_summaries], dtype=np.float64),
        }
        if baseline_trials is not None:
            npz_payload["baseline_uncertainty_trials"] = baseline_trials
            npz_payload["baseline_uncertainty"] = baseline_trials.mean(axis=0)
        if best_trials is not None:
            npz_payload["best_uncertainty_trials"] = best_trials
            npz_payload["best_uncertainty"] = best_trials.mean(axis=0)

        np.savez_compressed(model_out / "exp11_outputs.npz", **npz_payload)
        save_json(summary, str(model_out / "exp11_summary.json"))

        save_json(
            {
                "experiment": "exp11_concrete_dropout_proxy",
                "model": model_key,
                "group_paths": {
                    str(group.group_id): group.paths for group in groups
                },
                "path_to_group": flattened_group_paths(groups),
            },
            str(model_out / "exp11_groups.json"),
        )

        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp11_concrete_dropout_proxy",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "results": overall,
        },
        str(out_dir / "exp11_overall_summary.json"),
    )

    print(f"[Exp11] Complete: {out_dir}")


if __name__ == "__main__":
    main()
