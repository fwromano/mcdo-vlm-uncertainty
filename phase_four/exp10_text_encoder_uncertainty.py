#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from phase_one.common import (
    auroc_from_scores,
    discover_class_names,
    load_model,
    reliability_from_trials,
    save_json,
    set_all_seeds,
    should_save_checkpoint,
    spearman_safe,
)
from phase_four.text_dropout import configure_text_dropout, disable_text_dropout


GENERIC_PROMPTS = [
    "an object",
    "a thing",
    "something",
    "an unknown item",
    "a blurry object",
    "a scene",
    "a visual concept",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 Exp 10: text-encoder uncertainty")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--class-map", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--specificity-quantile", type=float, default=0.20)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _specificity_score(prompt: str) -> float:
    tokens = _tokenize_words(prompt)
    tok_count = len(tokens)
    tok_term = min(tok_count, 20) / 20.0

    positive = {
        "detailed",
        "close",
        "closeup",
        "close-up",
        "high",
        "resolution",
        "wearing",
        "with",
        "next",
        "under",
        "inside",
        "outdoor",
        "indoor",
        "lighting",
        "texture",
        "color",
    }
    negative = {
        "something",
        "object",
        "thing",
        "item",
        "unknown",
        "unclear",
        "maybe",
        "possibly",
        "blurry",
        "stuff",
    }

    pos_hits = sum(1 for t in tokens if t in positive)
    neg_hits = sum(1 for t in tokens if t in negative)

    raw = 0.6 * tok_term + 0.08 * pos_hits - 0.10 * neg_hits
    score = (raw + 0.25) / 1.25
    return float(np.clip(score, 0.0, 1.0))


def _load_prompts_from_file(path: str) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    styles: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                style, prompt = line.split("\t", maxsplit=1)
                style = style.strip() or "file"
                prompt = prompt.strip()
            else:
                style = "file"
                prompt = line
            if prompt:
                prompts.append(prompt)
                styles.append(style)

    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts, styles


def _build_prompt_bank(class_names: Sequence[str]) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    styles: List[str] = []

    for g in GENERIC_PROMPTS:
        prompts.append(g)
        styles.append("generic")

    for name in class_names:
        clean = name.strip()
        if not clean:
            continue
        prompts.extend(
            [
                f"a {clean}",
                f"a photo of a {clean}",
                f"a detailed close-up photo of a {clean} in natural lighting",
                f"something that might be a {clean}",
            ]
        )
        styles.extend(["simple", "photo", "specific", "ambiguous"])

    return prompts, styles


def _sample_prompts(prompts: Sequence[str], styles: Sequence[str], num_prompts: int, seed: int) -> Tuple[List[str], List[str]]:
    if len(prompts) != len(styles):
        raise ValueError("prompts/styles length mismatch")

    prompts_list = list(prompts)
    styles_list = list(styles)

    if num_prompts <= 0:
        return prompts_list, styles_list

    n = len(prompts_list)
    if n == 0:
        raise ValueError("No prompts available")

    rng = np.random.default_rng(seed)
    if num_prompts <= n:
        idx = np.sort(rng.choice(n, size=num_prompts, replace=False))
        return [prompts_list[int(i)] for i in idx], [styles_list[int(i)] for i in idx]

    idx = rng.choice(n, size=num_prompts, replace=True)
    return [prompts_list[int(i)] for i in idx], [styles_list[int(i)] for i in idx]


def _encode_texts_batched(vlm, prompts: Sequence[str], normalize: bool, batch_size: int) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        part = prompts[start : start + batch_size]
        feats = vlm.encode_texts(part, normalize=normalize)
        chunks.append(feats.detach().cpu().to(torch.float64))
    return torch.cat(chunks, dim=0)


def _mc_text_uncertainty(
    vlm,
    prompts: Sequence[str],
    dropout: float,
    passes: int,
    trials: int,
    seed: int,
    batch_size: int,
    save_every: int = 0,
    partial_npz_path: str = "",
    progress_json_path: str = "",
    progress_meta: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    n = len(prompts)
    if n == 0:
        raise ValueError("No prompts provided")

    trials_out: List[np.ndarray] = []
    for trial_idx in range(trials):
        set_all_seeds(seed + trial_idx)
        configure_text_dropout(vlm, p=dropout)

        sum_feats: Optional[torch.Tensor] = None
        sq_feats: Optional[torch.Tensor] = None

        for _pass in range(passes):
            feats = _encode_texts_batched(vlm, prompts=prompts, normalize=False, batch_size=batch_size)
            if sum_feats is None:
                sum_feats = torch.zeros_like(feats)
                sq_feats = torch.zeros_like(feats)
            sum_feats += feats
            sq_feats += feats * feats

        if sum_feats is None or sq_feats is None:
            raise RuntimeError("No text features collected during MC passes")

        mean = sum_feats / float(passes)
        var = sq_feats / float(passes) - mean * mean
        dim = var.shape[1]
        trace = (var.sum(dim=1) / float(dim)).numpy()
        trials_out.append(trace)

        completed = trial_idx + 1
        if partial_npz_path and should_save_checkpoint(completed=completed, total=trials, every=save_every):
            np.savez_compressed(
                partial_npz_path,
                uncertainty_trials=np.stack(trials_out, axis=0),
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


def _style_stats(styles: Sequence[str], values: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    unique = sorted(set(styles))
    for style in unique:
        idx = [i for i, s in enumerate(styles) if s == style]
        if not idx:
            continue
        arr = values[np.asarray(idx, dtype=np.int64)]
        out[style] = {
            "count": float(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
        }
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = discover_class_names(args.data_dir, mapping_path=args.class_map or None)

    if args.prompt_file:
        prompts, styles = _load_prompts_from_file(args.prompt_file)
    else:
        prompts, styles = _build_prompt_bank(class_names)

    prompts, styles = _sample_prompts(prompts, styles, num_prompts=args.num_prompts, seed=args.seed)

    token_counts = np.asarray([len(_tokenize_words(p)) for p in prompts], dtype=np.float64)
    specificity = np.asarray([_specificity_score(p) for p in prompts], dtype=np.float64)

    q = float(np.clip(args.specificity_quantile, 0.0, 1.0))
    low_specificity_thr = float(np.quantile(specificity, q))
    low_specificity = (specificity <= low_specificity_thr).astype(np.int64)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp10] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            payload: Dict[str, object] = {"error": str(exc)}
            overall[model_key] = payload
            save_json(payload, str(model_out / "exp10_error.json"))
            continue

        unc_trials = _mc_text_uncertainty(
            vlm=vlm,
            prompts=prompts,
            dropout=args.dropout,
            passes=args.passes,
            trials=args.trials,
            seed=args.seed,
            batch_size=args.text_batch_size,
            save_every=args.save_every,
            partial_npz_path=str(model_out / "exp10_trials_partial.npz"),
            progress_json_path=str(model_out / "exp10_progress.json"),
            progress_meta={"experiment": "exp10_text_encoder_uncertainty", "model": model_key},
        )
        disable_text_dropout(vlm)

        unc = unc_trials.mean(axis=0)

        reliability: Optional[Dict[str, float]]
        if unc_trials.shape[0] >= 2:
            reliability = reliability_from_trials(unc_trials)
        else:
            reliability = None

        summary = {
            "experiment": "exp10_text_encoder_uncertainty",
            "model": model_key,
            "num_prompts": len(prompts),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "specificity_quantile": q,
            "thresholds": {
                "low_specificity": low_specificity_thr,
            },
            "metrics": {
                "rho_uncertainty_vs_specificity": spearman_safe(unc, specificity),
                "rho_uncertainty_vs_negative_specificity": spearman_safe(unc, -specificity),
                "rho_uncertainty_vs_token_count": spearman_safe(unc, token_counts),
                "auroc_low_specificity": auroc_from_scores(unc, low_specificity),
            },
            "uncertainty_stats": {
                "mean": float(np.mean(unc)),
                "std": float(np.std(unc)),
                "median": float(np.median(unc)),
            },
            "specificity_stats": {
                "mean": float(np.mean(specificity)),
                "std": float(np.std(specificity)),
                "median": float(np.median(specificity)),
            },
            "style_uncertainty": _style_stats(styles=styles, values=unc),
            "style_specificity": _style_stats(styles=styles, values=specificity),
            "reliability": reliability,
        }

        np.savez_compressed(
            model_out / "exp10_outputs.npz",
            prompts=np.asarray(prompts),
            styles=np.asarray(styles),
            token_count=token_counts,
            specificity=specificity,
            low_specificity=low_specificity,
            uncertainty_trials=unc_trials,
            uncertainty=unc,
        )
        save_json(summary, str(model_out / "exp10_summary.json"))
        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp10_text_encoder_uncertainty",
            "models": model_keys,
            "num_prompts": len(prompts),
            "results": overall,
        },
        str(out_dir / "exp10_overall_summary.json"),
    )

    save_json(
        {
            "num_prompts": len(prompts),
            "prompt_source": ("file" if args.prompt_file else "generated"),
            "prompt_file": args.prompt_file,
            "styles": sorted(set(styles)),
            "prompts": prompts,
        },
        str(out_dir / "exp10_prompt_manifest.json"),
    )

    print(f"[Exp10] Complete: {out_dir}")


if __name__ == "__main__":
    main()
