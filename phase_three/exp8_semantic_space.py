#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from phase_one.common import (
    auroc_from_scores,
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
    spearman_safe,
)
from phase_two.dropout_types import configure_dropout

DEFAULT_PROMPTS = [
    "a dog",
    "a cat",
    "a person",
    "a car",
    "a truck",
    "a bicycle",
    "a motorcycle",
    "an airplane",
    "a boat",
    "a bird",
    "a tree",
    "a flower",
    "a building",
    "a street scene",
    "an indoor room",
    "a plate of food",
    "a laptop computer",
    "a phone",
    "a sports ball",
    "a backpack",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Exp 8: semantic-space uncertainty diagnostics")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--prompts", type=str, default="")
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def _parse_prompts(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return list(DEFAULT_PROMPTS)
    prompts = [p.strip() for p in raw.split("|") if p.strip()]
    if len(prompts) < 2:
        return list(DEFAULT_PROMPTS)
    return prompts


def _semantic_basis(text_features: np.ndarray) -> np.ndarray:
    centered = text_features - text_features.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    keep = s > 1e-8
    if not np.any(keep):
        return np.zeros((text_features.shape[1], 0), dtype=np.float64)
    basis = vt[keep].T
    return basis.astype(np.float64, copy=False)


def _anisotropy(pass_pre: np.ndarray) -> np.ndarray:
    t, n, _ = pass_pre.shape
    denom = max(t - 1, 1)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x = pass_pre[:, i, :].astype(np.float64, copy=False)
        xc = x - x.mean(axis=0, keepdims=True)
        per_dim_var = np.sum(xc * xc, axis=0) / float(denom)
        trace_total = float(np.sum(per_dim_var))
        if trace_total <= 1e-12:
            out[i] = 0.0
            continue
        gram = (xc @ xc.T) / float(denom)
        eigvals = np.linalg.eigvalsh(gram)
        eigmax = float(np.max(np.clip(eigvals, 0.0, None)))
        out[i] = eigmax / trace_total

    return out


def _directional_metrics(pass_pre: np.ndarray, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, n, d = pass_pre.shape
    centered = pass_pre - pass_pre.mean(axis=0, keepdims=True)

    if basis.shape[1] == 0:
        parallel = np.zeros(n, dtype=np.float64)
        orthogonal = np.mean(np.sum(centered * centered, axis=2), axis=0) / float(d)
        frac = np.zeros(n, dtype=np.float64)
        return parallel, orthogonal, frac

    proj = np.einsum("tnd,dr->tnr", centered, basis)
    rec = np.einsum("tnr,dr->tnd", proj, basis)
    orth = centered - rec

    parallel = np.mean(np.sum(proj * proj, axis=2), axis=0) / float(d)
    orthogonal = np.mean(np.sum(orth * orth, axis=2), axis=0) / float(d)
    frac = parallel / np.maximum(parallel + orthogonal, 1e-12)
    return parallel, orthogonal, frac


def _similarity_variance_and_prompt_gap(pass_post: np.ndarray, text_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sims = np.einsum("tnd,pd->tnp", pass_post, text_features)
    sim_var_by_prompt = np.var(sims, axis=0, ddof=1) if sims.shape[0] > 1 else np.zeros((sims.shape[1], sims.shape[2]))
    sim_var_mean = np.mean(sim_var_by_prompt, axis=1)

    mean_post = np.mean(pass_post, axis=0)
    logits = mean_post @ text_features.T
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    top2 = np.sort(top2, axis=1)
    prompt_gap = top2[:, 1] - top2[:, 0]

    logits_shift = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shift)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    prompt_entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)

    return sim_var_mean, prompt_gap, prompt_entropy


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
        save_manifest(sampled_paths, str(out_dir / "phase3_exp8_manifest.json"))

    prompts = _parse_prompts(args.prompts)
    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp8] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            payload: Dict[str, object] = {"error": str(exc)}
            overall[model_key] = payload
            save_json(payload, str(model_out / "exp8_error.json"))
            continue

        text_features = vlm.encode_texts(prompts, normalize=True).detach().cpu().numpy().astype(np.float64, copy=False)
        basis = _semantic_basis(text_features)

        trial_trace: List[np.ndarray] = []
        trial_parallel: List[np.ndarray] = []
        trial_orthogonal: List[np.ndarray] = []
        trial_parallel_frac: List[np.ndarray] = []
        trial_anisotropy: List[np.ndarray] = []
        trial_sim_var: List[np.ndarray] = []
        trial_prompt_gap: List[np.ndarray] = []
        trial_prompt_entropy: List[np.ndarray] = []

        for trial_idx in range(args.trials):
            set_all_seeds(args.seed + trial_idx)
            configure_dropout(vlm, dropout_type="E", p=args.dropout)
            trial = run_mc_trial(
                vlm=vlm,
                loader=loader,
                passes=args.passes,
                collect_pass_features=True,
                compute_angular=False,
            )

            trace = trial["trace_pre"].numpy().astype(np.float64, copy=False)
            pass_pre = trial["pass_pre"].numpy().astype(np.float64, copy=False)
            pass_post = trial["pass_post"].numpy().astype(np.float64, copy=False)

            parallel, orthogonal, frac = _directional_metrics(pass_pre=pass_pre, basis=basis)
            anisotropy = _anisotropy(pass_pre)
            sim_var_mean, prompt_gap, prompt_entropy = _similarity_variance_and_prompt_gap(
                pass_post=pass_post,
                text_features=text_features,
            )

            trial_trace.append(trace)
            trial_parallel.append(parallel)
            trial_orthogonal.append(orthogonal)
            trial_parallel_frac.append(frac)
            trial_anisotropy.append(anisotropy)
            trial_sim_var.append(sim_var_mean)
            trial_prompt_gap.append(prompt_gap)
            trial_prompt_entropy.append(prompt_entropy)

            completed = trial_idx + 1
            if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                np.savez_compressed(
                    model_out / "exp8_trials_partial.npz",
                    paths=np.asarray(sampled_paths),
                    trace_pre_trials=np.stack(trial_trace, axis=0),
                    parallel_var_trials=np.stack(trial_parallel, axis=0),
                    orthogonal_var_trials=np.stack(trial_orthogonal, axis=0),
                    parallel_fraction_trials=np.stack(trial_parallel_frac, axis=0),
                    anisotropy_trials=np.stack(trial_anisotropy, axis=0),
                    similarity_variance_trials=np.stack(trial_sim_var, axis=0),
                    prompt_gap_trials=np.stack(trial_prompt_gap, axis=0),
                    prompt_entropy_trials=np.stack(trial_prompt_entropy, axis=0),
                    completed_trials=np.asarray([completed], dtype=np.int64),
                    total_trials=np.asarray([args.trials], dtype=np.int64),
                )
                save_json(
                    {
                        "experiment": "exp8_semantic_space",
                        "model": model_key,
                        "completed_trials": completed,
                        "total_trials": args.trials,
                    },
                    str(model_out / "exp8_progress.json"),
                )

        arr_trace = np.stack(trial_trace, axis=0)
        arr_parallel = np.stack(trial_parallel, axis=0)
        arr_orthogonal = np.stack(trial_orthogonal, axis=0)
        arr_parallel_frac = np.stack(trial_parallel_frac, axis=0)
        arr_anisotropy = np.stack(trial_anisotropy, axis=0)
        arr_sim_var = np.stack(trial_sim_var, axis=0)
        arr_prompt_gap = np.stack(trial_prompt_gap, axis=0)
        arr_prompt_entropy = np.stack(trial_prompt_entropy, axis=0)

        trace_mean = arr_trace.mean(axis=0)
        parallel_mean = arr_parallel.mean(axis=0)
        orthogonal_mean = arr_orthogonal.mean(axis=0)
        frac_mean = arr_parallel_frac.mean(axis=0)
        anisotropy_mean = arr_anisotropy.mean(axis=0)
        sim_var_mean = arr_sim_var.mean(axis=0)
        prompt_gap_mean = arr_prompt_gap.mean(axis=0)
        prompt_entropy_mean = arr_prompt_entropy.mean(axis=0)

        low_gap_thr = np.quantile(prompt_gap_mean, args.quantile)
        high_entropy_thr = np.quantile(prompt_entropy_mean, 1.0 - args.quantile)
        low_gap = (prompt_gap_mean <= low_gap_thr).astype(np.int64)
        high_entropy = (prompt_entropy_mean >= high_entropy_thr).astype(np.int64)

        top_idx = np.argsort(trace_mean)[-20:][::-1]
        top_uncertain = [
            {
                "path": sampled_paths[int(i)],
                "trace_pre": float(trace_mean[int(i)]),
                "prompt_gap": float(prompt_gap_mean[int(i)]),
                "parallel_fraction": float(frac_mean[int(i)]),
            }
            for i in top_idx
        ]

        summary = {
            "experiment": "exp8_semantic_space",
            "model": model_key,
            "num_images": len(sampled_paths),
            "num_prompts": len(prompts),
            "semantic_basis_rank": int(basis.shape[1]),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "corr": {
                "rho_trace_vs_parallel_var": spearman_safe(trace_mean, parallel_mean),
                "rho_trace_vs_orthogonal_var": spearman_safe(trace_mean, orthogonal_mean),
                "rho_trace_vs_parallel_fraction": spearman_safe(trace_mean, frac_mean),
                "rho_trace_vs_anisotropy": spearman_safe(trace_mean, anisotropy_mean),
                "rho_trace_vs_similarity_var": spearman_safe(trace_mean, sim_var_mean),
                "rho_trace_vs_negative_prompt_gap": spearman_safe(trace_mean, -prompt_gap_mean),
                "rho_trace_vs_prompt_entropy": spearman_safe(trace_mean, prompt_entropy_mean),
            },
            "auroc": {
                "low_prompt_gap": auroc_from_scores(trace_mean, low_gap),
                "high_prompt_entropy": auroc_from_scores(trace_mean, high_entropy),
            },
            "means": {
                "trace_pre": float(np.mean(trace_mean)),
                "parallel_var": float(np.mean(parallel_mean)),
                "orthogonal_var": float(np.mean(orthogonal_mean)),
                "parallel_fraction": float(np.mean(frac_mean)),
                "anisotropy": float(np.mean(anisotropy_mean)),
                "similarity_variance": float(np.mean(sim_var_mean)),
            },
            "reliability": {
                "trace_pre": _reliability_or_none(arr_trace),
                "parallel_var": _reliability_or_none(arr_parallel),
                "orthogonal_var": _reliability_or_none(arr_orthogonal),
                "parallel_fraction": _reliability_or_none(arr_parallel_frac),
                "anisotropy": _reliability_or_none(arr_anisotropy),
                "similarity_variance": _reliability_or_none(arr_sim_var),
            },
            "quantile": args.quantile,
            "thresholds": {
                "low_prompt_gap": float(low_gap_thr),
                "high_prompt_entropy": float(high_entropy_thr),
            },
            "prompts": prompts,
            "top_uncertain_examples": top_uncertain,
        }

        np.savez_compressed(
            model_out / "exp8_outputs.npz",
            paths=np.asarray(sampled_paths),
            trace_pre_trials=arr_trace,
            parallel_var_trials=arr_parallel,
            orthogonal_var_trials=arr_orthogonal,
            parallel_fraction_trials=arr_parallel_frac,
            anisotropy_trials=arr_anisotropy,
            similarity_variance_trials=arr_sim_var,
            prompt_gap_trials=arr_prompt_gap,
            prompt_entropy_trials=arr_prompt_entropy,
            trace_pre=trace_mean,
            parallel_var=parallel_mean,
            orthogonal_var=orthogonal_mean,
            parallel_fraction=frac_mean,
            anisotropy=anisotropy_mean,
            similarity_variance=sim_var_mean,
            prompt_gap=prompt_gap_mean,
            prompt_entropy=prompt_entropy_mean,
        )

        save_json(summary, str(model_out / "exp8_summary.json"))
        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp8_semantic_space",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "num_prompts": len(prompts),
            "results": overall,
        },
        str(out_dir / "exp8_overall_summary.json"),
    )

    print(f"[Exp8] Complete: {out_dir}")


if __name__ == "__main__":
    main()
