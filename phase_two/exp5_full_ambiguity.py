#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
from phase_two.dropout_types import configure_dropout, set_dropout_train_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Exp 5 full: ambiguity prediction")
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-json", type=str, default="", help="Optional retrieval eval JSON")
    parser.add_argument("--num-retrieval", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def _resolve_image_path(image_path: str, retrieval_json_path: str, data_dir: str) -> Optional[str]:
    raw = Path(image_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path(retrieval_json_path).parent / raw)
        candidates.append(Path(data_dir) / raw)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return None


def _load_retrieval_pairs(path: str, limit: int, data_dir: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("retrieval json must be a list")
    pairs: List[Dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        captions = item.get("captions")
        if not isinstance(image_path, str) or not isinstance(captions, list) or len(captions) < 2:
            continue
        clean_captions = [str(c).strip() for c in captions if str(c).strip()]
        if len(clean_captions) < 2:
            continue
        correct_index = item.get("correct_index", 0)
        try:
            correct_index = int(correct_index)
        except Exception:  # noqa: BLE001
            correct_index = 0
        correct_index = max(0, min(correct_index, len(clean_captions) - 1))

        resolved = _resolve_image_path(image_path=image_path, retrieval_json_path=path, data_dir=data_dir)
        if resolved is None:
            continue

        pairs.append({"image_path": resolved, "captions": clean_captions, "correct_index": correct_index})
        if len(pairs) >= limit:
            break
    return pairs


def _classification_metrics(vlm, loader, class_names: List[str], templates: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            prompt_var = np.var(max_probs_arr, axis=0, ddof=1) if max_probs_arr.shape[0] > 1 else np.zeros(max_probs_arr.shape[1])

            margins.append(margin)
            entropies.append(entropy)
            prompt_vars.append(prompt_var)

    return np.concatenate(margins), np.concatenate(entropies), np.concatenate(prompt_vars)


def _uncertainty_from_trials(
    vlm,
    loader,
    dropout: float,
    passes: int,
    trials: int,
    seed: int,
    save_every: int = 0,
    partial_npz_path: str = "",
    progress_json_path: str = "",
    extra_meta: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    arrs: List[np.ndarray] = []
    for trial_idx in range(trials):
        set_all_seeds(seed + trial_idx)
        configure_dropout(vlm, dropout_type="E", p=dropout)
        trial = run_mc_trial(vlm=vlm, loader=loader, passes=passes, collect_pass_features=False)
        arrs.append(trial["trace_pre"].numpy())

        completed = trial_idx + 1
        if partial_npz_path and should_save_checkpoint(completed=completed, total=trials, every=save_every):
            np.savez_compressed(
                partial_npz_path,
                uncertainty_trials=np.stack(arrs, axis=0),
                completed_trials=np.asarray([completed], dtype=np.int64),
                total_trials=np.asarray([trials], dtype=np.int64),
            )
            if progress_json_path:
                payload: Dict[str, object] = {
                    "completed_trials": completed,
                    "total_trials": trials,
                }
                if extra_meta:
                    payload.update(extra_meta)
                save_json(payload, progress_json_path)
    return np.stack(arrs, axis=0)


def _retrieval_deterministic_metrics(vlm, pairs: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    retrieval_gap: List[float] = []
    correct_rank: List[float] = []

    with torch.no_grad():
        for pair in pairs:
            image_path = str(pair["image_path"])
            captions = [str(x) for x in pair["captions"]]
            gt_index = int(pair["correct_index"])

            with Image.open(image_path) as img:
                image = img.convert("RGB")

            image_feat = vlm.encode_images([image], normalize=True)
            text_feat = vlm.encode_texts(captions, normalize=True)
            sims = vlm.similarity_logits(image_feat, text_feat)[0].detach().cpu().numpy()

            top2 = np.partition(sims, -2)[-2:]
            retrieval_gap.append(float(top2.max() - top2.min()))

            rank_desc = np.argsort(-sims)
            pos = int(np.where(rank_desc == gt_index)[0][0]) + 1
            correct_rank.append(float(pos))

    return np.asarray(retrieval_gap, dtype=np.float64), np.asarray(correct_rank, dtype=np.float64)


def _retrieval_uncertainty(
    vlm,
    pairs: List[Dict[str, object]],
    dropout: float,
    passes: int,
    trials: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    save_every: int = 0,
    partial_npz_path: str = "",
    progress_json_path: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    unique_paths: List[str] = []
    seen = set()
    for pair in pairs:
        path = str(pair["image_path"])
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    loader = build_loader(unique_paths, batch_size=batch_size, num_workers=num_workers)
    unc_trials = _uncertainty_from_trials(
        vlm=vlm,
        loader=loader,
        dropout=dropout,
        passes=passes,
        trials=trials,
        seed=seed,
        save_every=save_every,
        partial_npz_path=partial_npz_path,
        progress_json_path=progress_json_path,
        extra_meta={"split": "retrieval"},
    )
    unc_mean = unc_trials.mean(axis=0)
    by_path = {path: float(unc_mean[idx]) for idx, path in enumerate(unique_paths)}
    pair_uncertainty = np.asarray([by_path[str(pair["image_path"])] for pair in pairs], dtype=np.float64)
    return pair_uncertainty, unc_trials, unique_paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        sampled_paths = sample_paths(list_images(args.data_dir), args.num_images, args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase2_exp5_manifest.json"))

    class_names = discover_class_names(args.data_dir, mapping_path=args.class_map or None)
    templates = parse_templates(args.templates)
    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    retrieval_pairs: List[Dict[str, object]] = []
    if args.retrieval_json:
        retrieval_pairs = _load_retrieval_pairs(args.retrieval_json, limit=args.num_retrieval, data_dir=args.data_dir)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp5-full] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            overall[model_key] = {"error": str(exc)}
            save_json(overall[model_key], str(model_out / "exp5_error.json"))
            continue

        # Deterministic ambiguity targets.
        margins, entropies, prompt_sens = _classification_metrics(vlm, loader, class_names=class_names, templates=templates)

        # Uncertainty.
        unc_trials = _uncertainty_from_trials(
            vlm=vlm,
            loader=loader,
            dropout=args.dropout,
            passes=args.passes,
            trials=args.trials,
            seed=args.seed,
            save_every=args.save_every,
            partial_npz_path=str(model_out / "exp5_full_uncertainty_partial.npz"),
            progress_json_path=str(model_out / "exp5_full_uncertainty_progress.json"),
            extra_meta={"experiment": "exp5_full_ambiguity", "model": model_key, "split": "classification"},
        )
        uncertainty = unc_trials.mean(axis=0)

        low_margin_thr = np.quantile(margins, args.quantile)
        high_entropy_thr = np.quantile(entropies, 1.0 - args.quantile)
        low_margin = (margins <= low_margin_thr).astype(np.int64)
        high_entropy = (entropies >= high_entropy_thr).astype(np.int64)

        metrics = {
            "rho_uncertainty_vs_negative_margin": spearman_safe(uncertainty, -margins),
            "rho_uncertainty_vs_entropy": spearman_safe(uncertainty, entropies),
            "rho_uncertainty_vs_prompt_sensitivity": spearman_safe(uncertainty, prompt_sens),
            "auroc_low_margin": auroc_from_scores(uncertainty, low_margin),
            "auroc_high_entropy": auroc_from_scores(uncertainty, high_entropy),
        }

        retrieval_summary: Dict[str, object] = {"enabled": False}
        retrieval_gap: Optional[np.ndarray] = None
        retrieval_rank: Optional[np.ndarray] = None
        retrieval_uncertainty: Optional[np.ndarray] = None
        retrieval_unc_trials: Optional[np.ndarray] = None
        retrieval_pair_paths: Optional[np.ndarray] = None
        if retrieval_pairs:
            set_dropout_train_mode(vlm.vision_root, enabled=False)

            retrieval_gap, retrieval_rank = _retrieval_deterministic_metrics(vlm, retrieval_pairs)
            retrieval_uncertainty, retrieval_unc_trials, retrieval_unique_paths = _retrieval_uncertainty(
                vlm=vlm,
                pairs=retrieval_pairs,
                dropout=args.dropout,
                passes=args.passes,
                trials=args.trials,
                seed=args.seed + 100_000,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                save_every=args.save_every,
                partial_npz_path=str(model_out / "exp5_full_retrieval_uncertainty_partial.npz"),
                progress_json_path=str(model_out / "exp5_full_retrieval_uncertainty_progress.json"),
            )
            retrieval_pair_paths = np.asarray([str(pair["image_path"]) for pair in retrieval_pairs])

            low_gap_thr = np.quantile(retrieval_gap, args.quantile)
            high_rank_thr = np.quantile(retrieval_rank, 1.0 - args.quantile)
            low_gap = (retrieval_gap <= low_gap_thr).astype(np.int64)
            high_rank = (retrieval_rank >= high_rank_thr).astype(np.int64)

            retrieval_summary = {
                "enabled": True,
                "num_pairs": len(retrieval_pairs),
                "num_unique_images": len(retrieval_unique_paths),
                "quantile": args.quantile,
                "thresholds": {
                    "low_gap": float(low_gap_thr),
                    "high_rank": float(high_rank_thr),
                },
                "metrics": {
                    "rho_uncertainty_vs_negative_gap": spearman_safe(retrieval_uncertainty, -retrieval_gap),
                    "rho_uncertainty_vs_rank": spearman_safe(retrieval_uncertainty, retrieval_rank),
                    "auroc_low_gap": auroc_from_scores(retrieval_uncertainty, low_gap),
                    "auroc_high_rank": auroc_from_scores(retrieval_uncertainty, high_rank),
                },
            }

        npz_payload: Dict[str, np.ndarray] = {
            "paths": np.asarray(sampled_paths),
            "uncertainty_trials": unc_trials,
            "uncertainty": uncertainty,
            "margin": margins,
            "entropy": entropies,
            "prompt_sensitivity": prompt_sens,
            "low_margin": low_margin,
            "high_entropy": high_entropy,
        }
        if retrieval_gap is not None and retrieval_rank is not None and retrieval_uncertainty is not None:
            npz_payload["retrieval_pair_paths"] = retrieval_pair_paths if retrieval_pair_paths is not None else np.asarray([])
            npz_payload["retrieval_gap"] = retrieval_gap
            npz_payload["retrieval_rank"] = retrieval_rank
            npz_payload["retrieval_uncertainty"] = retrieval_uncertainty
            if retrieval_unc_trials is not None:
                npz_payload["retrieval_uncertainty_trials"] = retrieval_unc_trials

        np.savez_compressed(model_out / "exp5_full_outputs.npz", **npz_payload)

        summary = {
            "experiment": "exp5_full_ambiguity",
            "model": model_key,
            "num_images": len(sampled_paths),
            "num_classes": len(class_names),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "quantile": args.quantile,
            "templates": templates,
            "thresholds": {
                "low_margin": float(low_margin_thr),
                "high_entropy": float(high_entropy_thr),
            },
            "metrics": metrics,
            "retrieval": retrieval_summary,
        }
        save_json(summary, str(model_out / "exp5_full_summary.json"))
        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp5_full_ambiguity",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "results": overall,
        },
        str(out_dir / "exp5_full_overall_summary.json"),
    )

    print(f"[Exp5-full] Complete: {out_dir}")


if __name__ == "__main__":
    main()
