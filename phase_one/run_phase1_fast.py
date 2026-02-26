#!/usr/bin/env python
"""Mac-optimized Phase 1 runner for Apple Silicon unified memory.

Speedups over run_phase1.py:
  1. All models + pixel tensors pre-loaded into memory at startup
  2. fp16 forward passes (~2x on MPS)
  3. Nested T extraction: one T_max=64 run yields T=4,16,64 for free
  4. Zero disk I/O during experiment loops — write only final results
  5. Full-dataset single-batch forward (no batch loop overhead)

Usage:
    python -m phase_one.run_phase1_fast data/raw/imagenet_val outputs/fast \\
        --device mps
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from phase_one.common import (
    VisionLanguageModel,
    build_loader,
    discover_class_names,
    list_images,
    load_model,
    parse_templates,
    reliability_from_trials,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
    spearman_safe,
    auroc_from_scores,
)


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("data_dir", type=str)
    p.add_argument("out_dir", type=str)
    p.add_argument("--only", type=str, default="exp0,exp0b,exp4,exp5")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-half", dest="half", action="store_false", default=True,
        help="Disable fp16 (use fp32 forward passes)",
    )

    # Exp 0
    p.add_argument("--exp0-num-images", type=int, default=500)
    p.add_argument("--exp0-models", type=str, default="clip_b32,siglip2_b16,siglip2_so400m")
    p.add_argument("--exp0-trials", type=int, default=10)
    p.add_argument("--exp0-passes", type=str, default="4,16,64")

    # Exp 0b
    p.add_argument("--exp0b-num-images", type=int, default=500)
    p.add_argument("--exp0b-model", type=str, default="clip_b32")
    p.add_argument("--exp0b-trials", type=int, default=5)
    p.add_argument("--exp0b-passes", type=int, default=64)

    # Exp 4
    p.add_argument("--exp4-num-images", type=int, default=500)
    p.add_argument("--exp4-models", type=str, default="clip_b32,siglip2_b16")
    p.add_argument("--exp4-trials", type=int, default=10)
    p.add_argument("--exp4-passes", type=int, default=64)

    # Exp 5
    p.add_argument("--exp5-num-images", type=int, default=5000)
    p.add_argument("--exp5-models", type=str, default="clip_b32,siglip2_b16")
    p.add_argument("--exp5-trials", type=int, default=1)
    p.add_argument("--exp5-passes", type=int, default=64)
    p.add_argument("--exp5-class-map", type=str, default="")
    p.add_argument("--exp5-templates", type=str, default="a photo of a {}|a {}|an image of a {}")

    return p.parse_args()


# ── Preloading ──────────────────────────────────────────────────────────────


def load_pil_images(paths: List[str]) -> List[Image.Image]:
    images = []
    for p in paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))
    return images


def precompute_pixels(
    vlm: VisionLanguageModel,
    pil_images: List[Image.Image],
    half: bool,
    chunk: int = 64,
) -> torch.Tensor:
    parts = []
    for i in range(0, len(pil_images), chunk):
        pv = vlm._pixel_values_from_pil(pil_images[i : i + chunk])
        if half:
            pv = pv.half()
        parts.append(pv)
    return torch.cat(parts, dim=0).to(vlm.device, non_blocking=True)


# ── Core MC ─────────────────────────────────────────────────────────────────


@torch.inference_mode()
def _mc_pass(vlm: VisionLanguageModel, pixels: torch.Tensor) -> torch.Tensor:
    """One MC dropout forward pass → pre-norm features as CPU float64."""
    return vlm.encode_pixel_values(pixels, normalize=False).cpu().to(torch.float64)


def run_nested_trial(
    vlm: VisionLanguageModel,
    pixels: torch.Tensor,
    T_max: int,
    snapshot_Ts: List[int],
    dropout_p: float,
    seed: int,
    progress_desc: str = "",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Run T_max passes, snapshot trace_pre/trace_post at each T in snapshot_Ts."""
    set_all_seeds(seed)
    vlm.ensure_uniform_dropout(dropout_p)

    sorted_Ts = sorted(snapshot_Ts)
    snap_idx = 0
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}

    sum_pre: Optional[torch.Tensor] = None
    sq_pre: Optional[torch.Tensor] = None
    sum_post: Optional[torch.Tensor] = None
    sq_post: Optional[torch.Tensor] = None

    pass_iter: Any = range(T_max)
    if progress_desc:
        pass_iter = tqdm(pass_iter, total=T_max, desc=progress_desc, leave=False)

    for pass_idx in pass_iter:
        pre = _mc_pass(vlm, pixels)
        post = F.normalize(pre, dim=-1)

        if sum_pre is None:
            N, D = pre.shape
            sum_pre = torch.zeros(N, D, dtype=torch.float64)
            sq_pre = torch.zeros_like(sum_pre)
            sum_post = torch.zeros_like(sum_pre)
            sq_post = torch.zeros_like(sum_pre)

        sum_pre += pre
        sq_pre += pre * pre
        sum_post += post
        sq_post += post * post

        T_done = pass_idx + 1
        if snap_idx < len(sorted_Ts) and T_done == sorted_Ts[snap_idx]:
            D = pre.shape[1]
            var_p = sq_pre / T_done - (sum_pre / T_done) ** 2
            var_q = sq_post / T_done - (sum_post / T_done) ** 2
            snapshots[T_done] = {
                "trace_pre": (var_p.sum(1) / D).float(),
                "trace_post": (var_q.sum(1) / D).float(),
            }
            snap_idx += 1

    if progress_desc and hasattr(pass_iter, "close"):
        pass_iter.close()

    return snapshots


def run_trial_with_features(
    vlm: VisionLanguageModel,
    pixels: torch.Tensor,
    T: int,
    dropout_p: float,
    seed: int,
    progress_desc: str = "",
) -> Dict[str, Any]:
    """Run T passes collecting all per-pass features for geometry analysis."""
    set_all_seeds(seed)
    vlm.ensure_uniform_dropout(dropout_p)

    all_pre: List[torch.Tensor] = []
    all_post: List[torch.Tensor] = []

    pass_iter: Any = range(T)
    if progress_desc:
        pass_iter = tqdm(pass_iter, total=T, desc=progress_desc, leave=False)

    for _ in pass_iter:
        pre = _mc_pass(vlm, pixels)
        post = F.normalize(pre, dim=-1)
        all_pre.append(pre.float())
        all_post.append(post.float())

    if progress_desc and hasattr(pass_iter, "close"):
        pass_iter.close()

    pass_pre = torch.stack(all_pre, dim=0)   # (T, N, D)
    pass_post = torch.stack(all_post, dim=0)
    D = pass_pre.shape[2]

    var_pre = pass_pre.to(torch.float64).var(dim=0)
    var_post = pass_post.to(torch.float64).var(dim=0)

    # Angular variance
    mean_dir = F.normalize(pass_post.mean(dim=0), dim=-1)
    dots = (pass_post * mean_dir.unsqueeze(0)).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    angular_var = torch.acos(dots).var(dim=0, unbiased=True).float()

    # Vectorised covariance diagnostics
    centered_pre = (pass_pre - pass_pre.mean(0, keepdim=True)).to(torch.float64)
    centered_post = (pass_post - pass_post.mean(0, keepdim=True)).to(torch.float64)
    cov_pre = torch.einsum("tnd,tne->nde", centered_pre, centered_pre) / max(T - 1, 1)
    cov_post = torch.einsum("tnd,tne->nde", centered_post, centered_post) / max(T - 1, 1)

    trace_pre_per_d = torch.diagonal(cov_pre, dim1=1, dim2=2).sum(1) / D
    trace_post_per_d = torch.diagonal(cov_post, dim1=1, dim2=2).sum(1) / D
    diag_abs_pre = torch.diagonal(cov_pre, dim1=1, dim2=2).abs().sum(1)
    total_abs_pre = cov_pre.abs().sum(dim=(1, 2))
    offdiag_pre = (total_abs_pre - diag_abs_pre) / total_abs_pre.clamp(min=1e-12)
    diag_abs_post = torch.diagonal(cov_post, dim1=1, dim2=2).abs().sum(1)
    total_abs_post = cov_post.abs().sum(dim=(1, 2))
    offdiag_post = (total_abs_post - diag_abs_post) / total_abs_post.clamp(min=1e-12)

    return {
        "trace_pre": (var_pre.sum(1) / D).float(),
        "trace_post": (var_post.sum(1) / D).float(),
        "angular_var": angular_var,
        "trace_pre_per_d": trace_pre_per_d.float(),
        "trace_post_per_d": trace_post_per_d.float(),
        "offdiag_pre": offdiag_pre.float(),
        "offdiag_post": offdiag_post.float(),
        "pass_pre": pass_pre,
        "pass_post": pass_post,
    }


def run_simple_trial(
    vlm: VisionLanguageModel,
    pixels: torch.Tensor,
    T: int,
    dropout_p: float,
    seed: int,
    progress_desc: str = "",
) -> Dict[str, torch.Tensor]:
    """Run T passes, return trace only (for Exp 4/5)."""
    set_all_seeds(seed)
    vlm.ensure_uniform_dropout(dropout_p)

    sum_pre: Optional[torch.Tensor] = None
    sq_pre: Optional[torch.Tensor] = None

    pass_iter: Any = range(T)
    if progress_desc:
        pass_iter = tqdm(pass_iter, total=T, desc=progress_desc, leave=False)

    for _ in pass_iter:
        pre = _mc_pass(vlm, pixels)
        if sum_pre is None:
            N, D = pre.shape
            sum_pre = torch.zeros(N, D, dtype=torch.float64)
            sq_pre = torch.zeros_like(sum_pre)
        sum_pre += pre
        sq_pre += pre * pre

    if progress_desc and hasattr(pass_iter, "close"):
        pass_iter.close()

    D = sum_pre.shape[1]
    var_pre = sq_pre / T - (sum_pre / T) ** 2
    return {"trace_pre": (var_pre.sum(1) / D).float()}


# ── Experiments ─────────────────────────────────────────────────────────────


def status_from_metrics(m: Dict[str, float]) -> str:
    if np.isnan(m["pairwise_spearman_median"]) or np.isnan(m["snr"]) or np.isnan(m["icc"]):
        return "insufficient_trials"
    rho, snr, icc = m["pairwise_spearman_median"], m["snr"], m["icc"]
    if rho >= 0.8 and snr >= 2.0 and icc >= 0.75:
        return "usable"
    if rho < 0.6 or snr < 1.0:
        return "failed"
    return "marginal"


def reliability_or_placeholder(trials: np.ndarray) -> Dict[str, float]:
    if trials.shape[0] >= 2:
        return reliability_from_trials(trials)
    return {
        "between_var": float("nan"),
        "within_var": float("nan"),
        "snr": float("nan"),
        "icc": float("nan"),
        "pairwise_spearman_median": float("nan"),
        "pairwise_spearman_q25": float("nan"),
        "pairwise_spearman_q75": float("nan"),
    }


def do_exp0(
    models: Dict[str, VisionLanguageModel],
    pixel_cache: Dict[str, torch.Tensor],
    model_keys: List[str],
    passes_list: List[int],
    K: int,
    dropout_p: float,
    seed: int,
    out_dir: Path,
    paths: List[str],
) -> Dict:
    T_max = max(passes_list)
    overall: Dict[str, Any] = {}

    for mkey in model_keys:
        vlm = models[mkey]
        pix = pixel_cache[mkey]
        mdir = out_dir / mkey
        mdir.mkdir(parents=True, exist_ok=True)

        traces: Dict[int, Dict[str, List[np.ndarray]]] = {
            T: {"pre": [], "post": []} for T in passes_list
        }

        for k in range(K):
            snaps = run_nested_trial(
                vlm,
                pix,
                T_max,
                passes_list,
                dropout_p,
                seed + k,
                progress_desc=f"Exp0 {mkey} trial {k + 1}/{K}",
            )
            for T in passes_list:
                traces[T]["pre"].append(snaps[T]["trace_pre"].detach().numpy())
                traces[T]["post"].append(snaps[T]["trace_post"].detach().numpy())

        model_summary: Dict[str, Any] = {}
        for T in passes_list:
            pre_arr = np.stack(traces[T]["pre"])
            post_arr = np.stack(traces[T]["post"])
            pre_m = reliability_or_placeholder(pre_arr)
            post_m = reliability_or_placeholder(post_arr)
            pre_m["status"] = status_from_metrics(pre_m)
            post_m["status"] = status_from_metrics(post_m)
            model_summary[f"T={T}"] = {"pre_norm": pre_m, "post_norm": post_m}
            np.savez_compressed(
                mdir / f"exp0_trials_T{T}.npz",
                paths=np.asarray(paths), trial_pre=pre_arr, trial_post=post_arr,
            )

        overall[mkey] = model_summary
        save_json({
            "experiment": "exp0_nested_mc", "model": mkey,
            "dropout": dropout_p, "trials": K, "passes": passes_list,
            "num_images": len(paths), "nested_extraction": True,
            "results": model_summary,
        }, str(mdir / "exp0_summary.json"))

    save_json({
        "experiment": "exp0_nested_mc", "models": model_keys,
        "passes": passes_list, "dropout": dropout_p, "trials": K,
        "num_images": len(paths), "results": overall,
    }, str(out_dir / "exp0_overall_summary.json"))
    return overall


def do_exp0b(
    models: Dict[str, VisionLanguageModel],
    pixel_cache: Dict[str, torch.Tensor],
    model_key: str,
    T: int,
    K: int,
    dropout_p: float,
    seed: int,
    out_dir: Path,
    paths: List[str],
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vlm = models[model_key]
    pix = pixel_cache[model_key]

    trial_data: List[Dict[str, Any]] = []
    for k in range(K):
        trial_data.append(
            run_trial_with_features(
                vlm,
                pix,
                T,
                dropout_p,
                seed + k,
                progress_desc=f"Exp0b {model_key} trial {k + 1}/{K}",
            )
        )

    # Aggregate geometry diagnostics
    trace_pre_d = np.mean([t["trace_pre_per_d"].mean().item() for t in trial_data])
    trace_post_d = np.mean([t["trace_post_per_d"].mean().item() for t in trial_data])
    offdiag_pre = np.mean([t["offdiag_pre"].mean().item() for t in trial_data])
    offdiag_post = np.mean([t["offdiag_post"].mean().item() for t in trial_data])
    angular_mean = np.mean([t["angular_var"].mean().item() for t in trial_data])

    # Correlations (from last trial, which has the most data)
    last = trial_data[-1]
    tp = last["trace_pre"].detach().numpy()
    tq = last["trace_post"].detach().numpy()
    av = last["angular_var"].detach().numpy()
    from scipy.stats import spearmanr
    corr_pre_ang = float(spearmanr(tp, av).statistic)
    corr_post_ang = float(spearmanr(tq, av).statistic)
    corr_pre_post = float(spearmanr(tp, tq).statistic)

    summary = {
        "experiment": "exp0b_norm_geometry", "model": model_key,
        "passes": T, "trials": K, "num_images": len(paths),
        "trace_pre_per_d_mean": trace_pre_d,
        "trace_post_per_d_mean": trace_post_d,
        "offdiag_pre_mean": offdiag_pre,
        "offdiag_post_mean": offdiag_post,
        "angular_var_mean": angular_mean,
        "corr_trace_pre_angular": corr_pre_ang,
        "corr_trace_post_angular": corr_post_ang,
        "corr_trace_pre_post": corr_pre_post,
    }

    # Save arrays from last trial
    np.savez_compressed(
        out_dir / "exp0b_geometry_trials.npz",
        paths=np.asarray(paths),
        trace_pre=tp, trace_post=tq, angular_var=av,
    )
    save_json(summary, str(out_dir / "exp0b_summary.json"))
    return summary


def do_exp4(
    models: Dict[str, VisionLanguageModel],
    pixel_cache: Dict[str, torch.Tensor],
    model_keys: List[str],
    T: int,
    K: int,
    dropout_p: float,
    seed: int,
    out_dir: Path,
    paths: List[str],
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model: Dict[str, Any] = {}

    for mkey in model_keys:
        vlm = models[mkey]
        pix = pixel_cache[mkey]
        traces_pre: List[np.ndarray] = []
        traces_ang: List[np.ndarray] = []

        for k in range(K):
            result = run_trial_with_features(
                vlm,
                pix,
                T,
                dropout_p,
                seed + k,
                progress_desc=f"Exp4 {mkey} trial {k + 1}/{K}",
            )
            traces_pre.append(result["trace_pre"].detach().numpy())
            traces_ang.append(result["angular_var"].detach().numpy())

        pre_arr = np.stack(traces_pre)
        ang_arr = np.stack(traces_ang)
        pre_m = reliability_or_placeholder(pre_arr)
        ang_m = reliability_or_placeholder(ang_arr)
        pre_m["status"] = status_from_metrics(pre_m)
        ang_m["status"] = status_from_metrics(ang_m)

        per_model[mkey] = {
            "trace_pre": pre_m, "angular_var": ang_m,
            "trace_mean": float(pre_arr.mean()),
            "angular_mean": float(ang_arr.mean()),
        }

        np.savez_compressed(
            out_dir / f"exp4_{mkey}.npz",
            paths=np.asarray(paths), trial_pre=pre_arr, trial_angular=ang_arr,
        )

    summary = {
        "experiment": "exp4_subset_recipe",
        "models": model_keys, "passes": T, "trials": K,
        "num_images": len(paths), "results": per_model,
    }
    save_json(summary, str(out_dir / "exp4_subset_summary.json"))
    return summary


def do_exp5(
    models: Dict[str, VisionLanguageModel],
    pixel_cache: Dict[str, torch.Tensor],
    model_keys: List[str],
    T: int,
    K: int,
    dropout_p: float,
    seed: int,
    out_dir: Path,
    paths: List[str],
    data_dir: str,
    class_map: str,
    templates_raw: str,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = parse_templates(templates_raw)
    class_names = discover_class_names(data_dir, mapping_path=class_map or None)
    print(f"  [Exp5] {len(class_names)} classes × {len(templates)} templates")

    all_summaries: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}

    for mkey in model_keys:
        try:
            vlm = models[mkey]
            pix = pixel_cache[mkey]

            # Deterministic ambiguity metrics (no dropout)
            vlm.disable_dropout()
            text_feats = [
                vlm.encode_texts([t.format(c) for c in class_names], normalize=True)
                for t in templates
            ]

            # Compute margins/entropies in one pass using precomputed pixels
            image_feats = vlm.encode_pixel_values(pix, normalize=True)
            logits_main = vlm.similarity_logits(image_feats, text_feats[0])
            top2 = torch.topk(logits_main, k=2, dim=-1).values
            margin_arr = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()
            probs_main = F.softmax(logits_main, dim=-1)
            entropy_arr = (-(probs_main * probs_main.clamp_min(1e-12).log()).sum(-1)).detach().cpu().numpy()

            # Prompt sensitivity
            max_probs = []
            for tf in text_feats:
                logits = vlm.similarity_logits(image_feats, tf)
                max_probs.append(F.softmax(logits, dim=-1).max(-1).values.detach().cpu().numpy())
            prompt_var_arr = np.var(np.stack(max_probs), axis=0, ddof=1) if len(max_probs) > 1 else np.zeros(len(paths))

            # MC Dropout uncertainty
            unc_trials = []
            for k in range(K):
                trial = run_simple_trial(
                    vlm,
                    pix,
                    T,
                    dropout_p,
                    seed + k,
                    progress_desc=f"Exp5 {mkey} trial {k + 1}/{K}",
                )
                unc_trials.append(trial["trace_pre"].detach().numpy())

            unc_arr = np.stack(unc_trials).mean(axis=0)

            quantile = 0.10
            low_margin_thr = np.quantile(margin_arr, quantile)
            high_entropy_thr = np.quantile(entropy_arr, 1.0 - quantile)
            low_margin = (margin_arr <= low_margin_thr).astype(np.int64)
            high_entropy = (entropy_arr >= high_entropy_thr).astype(np.int64)

            metrics = {
                "rho_uncertainty_vs_negative_margin": spearman_safe(unc_arr, -margin_arr),
                "rho_uncertainty_vs_entropy": spearman_safe(unc_arr, entropy_arr),
                "rho_uncertainty_vs_prompt_sensitivity": spearman_safe(unc_arr, prompt_var_arr),
                "auroc_low_margin": auroc_from_scores(unc_arr, low_margin),
                "auroc_high_entropy": auroc_from_scores(unc_arr, high_entropy),
            }

            # Ground-truth labels from folder structure & top-1 predictions
            # class_names is ordered by sorted(folder_names), so build
            # a folder->index map to get the correct label per image.
            data_root = Path(data_dir)
            folder_to_idx = {
                name: i
                for i, name in enumerate(sorted(d.name for d in data_root.iterdir() if d.is_dir()))
            }
            gt_labels = np.array(
                [folder_to_idx.get(Path(p).parent.name, -1) for p in paths],
                dtype=np.int64,
            )
            logits_np = logits_main.detach().cpu().float().numpy()
            pred_labels = logits_np.argmax(axis=-1)

            np.savez_compressed(
                out_dir / f"exp5_subset_{mkey}.npz",
                paths=np.asarray(paths), uncertainty=unc_arr,
                margin=margin_arr, entropy=entropy_arr,
                prompt_sensitivity=prompt_var_arr,
                logits=logits_np, gt_labels=gt_labels,
                pred_labels=pred_labels,
            )
            save_json({
                "experiment": "exp5_subset_ambiguity", "model": mkey,
                "num_images": len(paths), "dropout": dropout_p,
                "passes": T, "trials": K, "num_classes": len(class_names),
                "templates": templates, "metrics": metrics,
            }, str(out_dir / f"exp5_subset_{mkey}_summary.json"))
            all_summaries[mkey] = metrics
        except Exception as exc:  # noqa: BLE001
            skipped[mkey] = str(exc)
            save_json(
                {
                    "experiment": "exp5_subset_ambiguity",
                    "model": mkey,
                    "status": "skipped",
                    "reason": str(exc),
                },
                str(out_dir / f"exp5_subset_{mkey}_error.json"),
            )
            print(f"  [Exp5] Skipping {mkey}: {exc}")

    save_json({
        "experiment": "exp5_subset_ambiguity", "models": model_keys,
        "num_images": len(paths), "results": all_summaries, "skipped": skipped,
    }, str(out_dir / "exp5_subset_overall_summary.json"))
    if not all_summaries:
        raise RuntimeError(f"Exp5 failed for all models: {skipped}")
    return all_summaries


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    selected = {x.strip() for x in args.only.split(",") if x.strip()}

    # ── Discover which models we need ───────────────────────────────────
    needed_models = set()
    exp0_models = [m.strip() for m in args.exp0_models.split(",")]
    exp4_models = [m.strip() for m in args.exp4_models.split(",")]
    exp5_models = [m.strip() for m in args.exp5_models.split(",")]
    if "exp0" in selected:
        needed_models.update(exp0_models)
    if "exp0b" in selected:
        needed_models.add(args.exp0b_model)
    if "exp4" in selected:
        needed_models.update(exp4_models)
    if "exp5" in selected:
        needed_models.update(exp5_models)

    # ── Determine max image count needed ────────────────────────────────
    max_images = 0
    if "exp0" in selected:
        max_images = max(max_images, args.exp0_num_images)
    if "exp0b" in selected:
        max_images = max(max_images, args.exp0b_num_images)
    if "exp4" in selected:
        max_images = max(max_images, args.exp4_num_images)
    if "exp5" in selected:
        max_images = max(max_images, args.exp5_num_images)

    # ── Sample and load images ──────────────────────────────────────────
    all_paths = list_images(args.data_dir)
    max_sampled = sample_paths(all_paths, max_images, args.seed)
    save_manifest(max_sampled, str(out_root / "manifest_all.json"))

    print(f"\n{'='*60}")
    print(f"  PHASE 1 FAST RUNNER — {'fp16' if args.half else 'fp32'} on {args.device}")
    print(f"  Models: {sorted(needed_models)}")
    print(f"  Max images: {len(max_sampled)}")
    print(f"{'='*60}\n")

    print("[PRELOAD] Loading PIL images into memory...")
    t0 = time.perf_counter()
    pil_images_full = load_pil_images(max_sampled)
    print(f"  {len(pil_images_full)} images loaded in {time.perf_counter()-t0:.1f}s")

    # ── Load all models ─────────────────────────────────────────────────
    print("[PRELOAD] Loading models...")
    models: Dict[str, VisionLanguageModel] = {}
    for mkey in sorted(needed_models):
        t0 = time.perf_counter()
        try:
            vlm = load_model(mkey, args.device)
            vlm.ensure_uniform_dropout(args.dropout)
            if args.half:
                vlm.model.half()
            n = sum(p.numel() for p in vlm.model.parameters())
            print(f"  {mkey}: {n/1e6:.0f}M params ({time.perf_counter()-t0:.1f}s)")
            models[mkey] = vlm
        except Exception as e:
            print(f"  {mkey}: FAILED — {e}")

    # ── Precompute pixel tensors per model ──────────────────────────────
    # We keep separate caches for different image counts
    def get_pixels(mkey: str, count: int) -> Tuple[torch.Tensor, List[str]]:
        sub_paths = max_sampled[:count]
        sub_pil = pil_images_full[:count]
        pix = precompute_pixels(models[mkey], sub_pil, half=args.half)
        return pix, sub_paths

    pixel_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, List[str]]] = {}

    def pixels_for(mkey: str, count: int) -> Tuple[torch.Tensor, List[str]]:
        key = (mkey, count)
        if key not in pixel_cache:
            print(f"[PRELOAD] Precomputing pixels: {mkey} × {count} images")
            t0 = time.perf_counter()
            pix, paths = get_pixels(mkey, count)
            mb = pix.nbytes / 1e6
            print(f"  {pix.shape} {pix.dtype} — {mb:.0f}MB ({time.perf_counter()-t0:.1f}s)")
            pixel_cache[key] = (pix, paths)
        return pixel_cache[key]

    # ── Run experiments ─────────────────────────────────────────────────
    passes_list = [int(x) for x in args.exp0_passes.split(",")]

    if "exp0" in selected:
        avail = [m for m in exp0_models if m in models]
        if avail:
            print(f"\n[EXP0] models={avail}, K={args.exp0_trials}, T={passes_list} (nested)")
            # Precompute pixels for all exp0 models
            for m in avail:
                pixels_for(m, args.exp0_num_images)
            pcache = {m: pixels_for(m, args.exp0_num_images)[0] for m in avail}
            paths0 = pixels_for(avail[0], args.exp0_num_images)[1]
            do_exp0(
                {m: models[m] for m in avail}, pcache, avail, passes_list,
                args.exp0_trials, args.dropout, args.seed, out_root / "exp0_nested_mc", paths0,
            )

    if "exp0b" in selected and args.exp0b_model in models:
        mkey = args.exp0b_model
        print(f"\n[EXP0B] model={mkey}, K={args.exp0b_trials}, T={args.exp0b_passes}")
        pix, paths_0b = pixels_for(mkey, args.exp0b_num_images)
        do_exp0b(
            models, {mkey: pix}, mkey, args.exp0b_passes, args.exp0b_trials,
            args.dropout, args.seed, out_root / "exp0b_norm_geometry", paths_0b,
        )

    if "exp4" in selected:
        avail = [m for m in exp4_models if m in models]
        if avail:
            print(f"\n[EXP4] models={avail}, K={args.exp4_trials}, T={args.exp4_passes}")
            for m in avail:
                pixels_for(m, args.exp4_num_images)
            pcache = {m: pixels_for(m, args.exp4_num_images)[0] for m in avail}
            paths4 = pixels_for(avail[0], args.exp4_num_images)[1]
            do_exp4(
                {m: models[m] for m in avail}, pcache, avail, args.exp4_passes,
                args.exp4_trials, args.dropout, args.seed, out_root / "exp4_subset_recipe", paths4,
            )

    if "exp5" in selected:
        avail = [m for m in exp5_models if m in models]
        if avail:
            print(f"\n[EXP5] models={avail}, K={args.exp5_trials}, T={args.exp5_passes}")
            for m in avail:
                pixels_for(m, args.exp5_num_images)
            pcache = {m: pixels_for(m, args.exp5_num_images)[0] for m in avail}
            paths5 = pixels_for(avail[0], args.exp5_num_images)[1]
            do_exp5(
                {m: models[m] for m in avail}, pcache, avail, args.exp5_passes,
                args.exp5_trials, args.dropout, args.seed, out_root / "exp5_subset_ambiguity",
                paths5, args.data_dir, args.exp5_class_map, args.exp5_templates,
            )

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Phase 1 fast run complete in {elapsed/60:.1f} minutes")
    print(f"  Outputs: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
