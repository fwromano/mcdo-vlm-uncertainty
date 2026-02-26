#!/usr/bin/env python
"""Conformal prediction coverage & sharpness evaluation for MC Dropout uncertainty.

Given exp5 outputs (per-image logits, uncertainty scores, ground-truth labels),
this experiment evaluates whether MC Dropout uncertainty enables valid prediction
sets with guaranteed coverage.

Metrics:
  - Coverage: fraction of prediction sets containing the true class
  - Sharpness: average prediction set size (smaller = better)
  - Conditional coverage: coverage stratified by uncertainty quantile
  - Uncertainty-adaptive efficiency: do low-uncertainty images get smaller sets?

Method: Split-conformal prediction (Vovk et al., 2005)
  1. Split data into calibration (cal) and test sets
  2. Compute nonconformity scores on cal set
  3. Find threshold at desired coverage level alpha
  4. Build prediction sets on test set
  5. Evaluate empirical coverage and sharpness

Usage:
    python -m phase_three.exp_conformal \
        outputs/phase1/exp5_subset_ambiguity \
        outputs/phase1/exp_conformal \
        --alpha 0.9 --models siglip2_b16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr


# ── Nonconformity scores ───────────────────────────────────────────────────


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def score_aps(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Adaptive Prediction Sets (APS) nonconformity score.

    For each sample, sort classes by descending probability, accumulate
    until the true class is included. The score is the cumulative
    probability at that point (plus a random tie-breaker for exact coverage).
    """
    n = probs.shape[0]
    sorted_idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
    cumsum = np.cumsum(sorted_probs, axis=-1)

    # Find where the true label sits in the sorted order
    scores = np.zeros(n)
    for i in range(n):
        rank = np.where(sorted_idx[i] == labels[i])[0][0]
        # APS score: cumulative probability up to and including true class
        scores[i] = cumsum[i, rank]
    return scores


def score_simple(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Simple nonconformity: 1 - p(true class)."""
    return 1.0 - probs[np.arange(len(labels)), labels]


# ── Conformal prediction sets ──────────────────────────────────────────────


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Compute conformal threshold for desired coverage level alpha.

    Returns q such that P(score <= q) >= alpha on the calibration set.
    Uses the finite-sample correction: ceil((n+1)*alpha) / n quantile.
    """
    n = len(cal_scores)
    level = np.ceil((n + 1) * alpha) / n
    level = min(level, 1.0)
    return float(np.quantile(cal_scores, level))


def build_prediction_sets_aps(
    probs: np.ndarray, threshold: float
) -> List[np.ndarray]:
    """Build APS prediction sets: include classes in decreasing probability
    order until cumulative probability exceeds threshold."""
    sets = []
    sorted_idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
    cumsum = np.cumsum(sorted_probs, axis=-1)

    for i in range(probs.shape[0]):
        # Include all classes until cumsum exceeds threshold
        k = np.searchsorted(cumsum[i], threshold, side="right") + 1
        k = min(k, probs.shape[1])
        sets.append(sorted_idx[i, :k])
    return sets


def build_prediction_sets_simple(
    probs: np.ndarray, threshold: float
) -> List[np.ndarray]:
    """Simple prediction sets: include all classes with 1-p(class) <= threshold,
    i.e., p(class) >= 1-threshold."""
    sets = []
    for i in range(probs.shape[0]):
        included = np.where(probs[i] >= 1.0 - threshold)[0]
        if len(included) == 0:
            included = np.array([probs[i].argmax()])
        sets.append(included)
    return sets


# ── Evaluation metrics ─────────────────────────────────────────────────────


def evaluate_sets(
    pred_sets: List[np.ndarray],
    labels: np.ndarray,
    uncertainty: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """Compute coverage, sharpness, and uncertainty-stratified metrics."""
    n = len(labels)
    covered = np.array([labels[i] in pred_sets[i] for i in range(n)])
    sizes = np.array([len(s) for s in pred_sets], dtype=np.float64)

    coverage = float(covered.mean())
    avg_size = float(sizes.mean())
    median_size = float(np.median(sizes))

    # Sharpness: normalized by number of classes (0 = singleton, 1 = all classes)
    sharpness = float(sizes.mean() / num_classes)

    # Uncertainty-stratified coverage (quartiles)
    quartiles = np.quantile(uncertainty, [0.25, 0.5, 0.75])
    q_labels = np.digitize(uncertainty, quartiles)  # 0,1,2,3

    stratified = {}
    quartile_names = ["Q1_low_unc", "Q2", "Q3", "Q4_high_unc"]
    for q in range(4):
        mask = q_labels == q
        if mask.sum() == 0:
            continue
        stratified[quartile_names[q]] = {
            "n": int(mask.sum()),
            "coverage": float(covered[mask].mean()),
            "avg_set_size": float(sizes[mask].mean()),
            "median_set_size": float(np.median(sizes[mask])),
        }

    # Correlation: uncertainty vs set size (should be positive if uncertainty is useful)
    rho_unc_size, _ = spearmanr(uncertainty, sizes)
    if np.isnan(rho_unc_size):
        rho_unc_size = 0.0

    # Correct vs incorrect: does uncertainty predict correctness?
    top1_correct = np.array(
        [labels[i] == pred_sets[i][0] if len(pred_sets[i]) > 0 else False for i in range(n)]
    )
    # Use top-1 prediction from the sorted probabilities, not the set
    rho_unc_correct, _ = spearmanr(uncertainty, top1_correct.astype(float))
    if np.isnan(rho_unc_correct):
        rho_unc_correct = 0.0

    return {
        "coverage": coverage,
        "avg_set_size": avg_size,
        "median_set_size": median_size,
        "sharpness_normalized": sharpness,
        "spearman_uncertainty_vs_set_size": float(rho_unc_size),
        "spearman_uncertainty_vs_top1_correct": float(rho_unc_correct),
        "stratified_by_uncertainty": stratified,
    }


# ── Main experiment ────────────────────────────────────────────────────────


def run_conformal(
    logits: np.ndarray,
    gt_labels: np.ndarray,
    uncertainty: np.ndarray,
    alphas: List[float],
    cal_fraction: float = 0.5,
    seed: int = 42,
    num_splits: int = 20,
    score_fn: str = "aps",
) -> Dict[str, Any]:
    """Run split-conformal prediction with multiple random cal/test splits.

    Args:
        logits: (N, C) raw logit matrix
        gt_labels: (N,) integer ground-truth class indices
        uncertainty: (N,) MC Dropout uncertainty per image
        alphas: target coverage levels, e.g. [0.8, 0.9, 0.95]
        cal_fraction: fraction of data for calibration
        seed: random seed
        num_splits: number of random cal/test splits to average over
        score_fn: "aps" or "simple"

    Returns:
        Dictionary with results per alpha level, averaged over splits.
    """
    # Filter out images with unknown labels
    valid = gt_labels >= 0
    if valid.sum() < len(gt_labels):
        print(f"  [Conformal] Dropping {(~valid).sum()} images with unknown labels")
    logits = logits[valid]
    gt_labels = gt_labels[valid]
    uncertainty = uncertainty[valid]

    n = len(gt_labels)
    num_classes = logits.shape[1]
    probs = softmax(logits)

    rng = np.random.RandomState(seed)
    results_by_alpha: Dict[str, Any] = {}

    for alpha in alphas:
        split_results = []

        for split_i in range(num_splits):
            # Random cal/test split
            perm = rng.permutation(n)
            n_cal = int(n * cal_fraction)
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal:]

            cal_probs = probs[cal_idx]
            cal_labels = gt_labels[cal_idx]
            test_probs = probs[test_idx]
            test_labels = gt_labels[test_idx]
            test_unc = uncertainty[test_idx]

            # Compute nonconformity scores on calibration set
            if score_fn == "aps":
                cal_scores = score_aps(cal_probs, cal_labels)
                threshold = conformal_threshold(cal_scores, alpha)
                pred_sets = build_prediction_sets_aps(test_probs, threshold)
            else:
                cal_scores = score_simple(cal_probs, cal_labels)
                threshold = conformal_threshold(cal_scores, alpha)
                pred_sets = build_prediction_sets_simple(test_probs, threshold)

            metrics = evaluate_sets(pred_sets, test_labels, test_unc, num_classes)
            metrics["threshold"] = threshold
            split_results.append(metrics)

        # Average over splits
        avg_metrics = {}
        scalar_keys = [
            "coverage", "avg_set_size", "median_set_size",
            "sharpness_normalized", "spearman_uncertainty_vs_set_size",
            "spearman_uncertainty_vs_top1_correct", "threshold",
        ]
        for key in scalar_keys:
            vals = [r[key] for r in split_results]
            avg_metrics[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        # Aggregate stratified results
        strat_agg: Dict[str, Dict[str, list]] = {}
        for r in split_results:
            for qname, qdata in r["stratified_by_uncertainty"].items():
                if qname not in strat_agg:
                    strat_agg[qname] = {"coverage": [], "avg_set_size": []}
                strat_agg[qname]["coverage"].append(qdata["coverage"])
                strat_agg[qname]["avg_set_size"].append(qdata["avg_set_size"])

        avg_strat = {}
        for qname, lists in strat_agg.items():
            avg_strat[qname] = {
                "coverage_mean": float(np.mean(lists["coverage"])),
                "coverage_std": float(np.std(lists["coverage"])),
                "avg_set_size_mean": float(np.mean(lists["avg_set_size"])),
                "avg_set_size_std": float(np.std(lists["avg_set_size"])),
            }
        avg_metrics["stratified_by_uncertainty"] = avg_strat

        results_by_alpha[f"alpha_{alpha}"] = avg_metrics

    # Top-1 accuracy for context
    top1_pred = probs.argmax(axis=-1)
    top1_acc = float((top1_pred == gt_labels).mean())

    return {
        "num_images": int(n),
        "num_classes": num_classes,
        "cal_fraction": cal_fraction,
        "num_splits": num_splits,
        "score_function": score_fn,
        "top1_accuracy": top1_acc,
        "results": results_by_alpha,
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("exp5_dir", type=str, help="Directory containing exp5 .npz outputs")
    p.add_argument("out_dir", type=str, help="Output directory for conformal results")
    p.add_argument("--models", type=str, default="",
                   help="Comma-separated model keys (default: all found in exp5_dir)")
    p.add_argument("--alpha", type=str, default="0.8,0.9,0.95",
                   help="Comma-separated target coverage levels")
    p.add_argument("--score-fn", type=str, default="aps", choices=["aps", "simple"],
                   help="Nonconformity score function")
    p.add_argument("--cal-fraction", type=float, default=0.5)
    p.add_argument("--num-splits", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp5_dir = Path(args.exp5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in args.alpha.split(",")]

    # Find available models
    npz_files = sorted(exp5_dir.glob("exp5_subset_*.npz"))
    if not npz_files:
        print(f"ERROR: No exp5_subset_*.npz files found in {exp5_dir}")
        sys.exit(1)

    available = {}
    for f in npz_files:
        mkey = f.stem.replace("exp5_subset_", "")
        if mkey.endswith("_error"):
            continue
        available[mkey] = f

    if args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
    else:
        model_keys = list(available.keys())

    print(f"Conformal prediction evaluation")
    print(f"  Models: {model_keys}")
    print(f"  Alpha levels: {alphas}")
    print(f"  Score function: {args.score_fn}")
    print(f"  Splits: {args.num_splits}, cal fraction: {args.cal_fraction}")

    all_results: Dict[str, Any] = {}

    for mkey in model_keys:
        if mkey not in available:
            print(f"  [{mkey}] SKIP — no exp5 output found")
            continue

        data = np.load(available[mkey], allow_pickle=True)

        # Check that logits were saved (need the extended exp5 output)
        if "logits" not in data or "gt_labels" not in data:
            print(f"  [{mkey}] SKIP — exp5 output missing 'logits' or 'gt_labels'.")
            print(f"           Re-run exp5 with the updated code that saves logits.")
            continue

        logits = data["logits"]
        gt_labels = data["gt_labels"]
        uncertainty = data["uncertainty"]

        print(f"\n  [{mkey}] {logits.shape[0]} images, {logits.shape[1]} classes")
        print(f"  [{mkey}] Top-1 accuracy: {(logits.argmax(1) == gt_labels).mean():.3f}")

        result = run_conformal(
            logits=logits,
            gt_labels=gt_labels,
            uncertainty=uncertainty,
            alphas=alphas,
            cal_fraction=args.cal_fraction,
            seed=args.seed,
            num_splits=args.num_splits,
            score_fn=args.score_fn,
        )

        # Print summary
        for akey, ametrics in result["results"].items():
            cov = ametrics["coverage"]
            sz = ametrics["avg_set_size"]
            sharp = ametrics["sharpness_normalized"]
            rho_sz = ametrics["spearman_uncertainty_vs_set_size"]
            print(f"  [{mkey}] {akey}: coverage={cov['mean']:.3f}±{cov['std']:.3f}, "
                  f"avg_size={sz['mean']:.1f}±{sz['std']:.1f}, "
                  f"sharpness={sharp['mean']:.4f}, "
                  f"ρ(unc,size)={rho_sz['mean']:.3f}")

            strat = ametrics.get("stratified_by_uncertainty", {})
            for qname, qdata in strat.items():
                print(f"    {qname}: coverage={qdata['coverage_mean']:.3f}, "
                      f"avg_size={qdata['avg_set_size_mean']:.1f}")

        all_results[mkey] = result

        # Save per-model result
        with open(out_dir / f"conformal_{mkey}.json", "w") as f:
            json.dump(result, f, indent=2)

    # Save overall summary
    summary = {
        "experiment": "conformal_prediction",
        "alphas": alphas,
        "score_function": args.score_fn,
        "cal_fraction": args.cal_fraction,
        "num_splits": args.num_splits,
        "models": model_keys,
        "results": all_results,
    }
    with open(out_dir / "conformal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
