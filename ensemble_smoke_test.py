#!/usr/bin/env python
"""
Ensemble smoke test: can we combine weak uncertainty signals into a stronger one?

Tests:
  1. Inter-metric correlations (are they measuring different things?)
  2. Naive average of z-scored metrics
  3. Sign-corrected weighted average
  4. PCA of metric matrix
  5. Cross-validated logistic regression (AUROC on held-out folds)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, rankdata

from phase_one.common import (
    auroc_from_scores,
    build_loader,
    detect_best_device,
    discover_class_names,
    list_images,
    load_model,
    run_mc_trial,
    sample_paths,
    set_all_seeds,
    spearman_safe,
)
from spectral_smoke_test import compute_all_metrics, bootstrap_ci, permutation_pvalue

TEMPLATES = ["a photo of a {}", "a {}", "an image of a {}"]


def auroc_safe(scores, labels):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return auroc_from_scores(np.asarray(scores, dtype=np.float64), labels)


def main():
    parser = argparse.ArgumentParser(description="Ensemble uncertainty metrics")
    parser.add_argument("--data-dir", default="data/raw/imagenet_val")
    parser.add_argument("--class-map", default="data/imagenet_class_map.json")
    parser.add_argument("--model", default="clip_b32")
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/ensemble_smoke.json")
    args = parser.parse_args()

    if args.device is None:
        args.device = detect_best_device()

    t0 = time.time()
    set_all_seeds(args.seed)

    # ── data ──
    data_dir = Path(args.data_dir)
    all_paths = list_images(str(data_dir))
    sampled = sample_paths(all_paths, args.n_images, args.seed)
    class_names = discover_class_names(str(data_dir), mapping_path=args.class_map)
    folders = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    folder_to_idx = {f: i for i, f in enumerate(folders)}
    loader = build_loader(sampled, batch_size=args.batch_size, num_workers=0)

    # ── model ──
    vlm = load_model(args.model, device=args.device)

    # ── classification signals ──
    vlm.disable_dropout()
    prompts = [TEMPLATES[0].format(n) for n in class_names]
    text_feat = vlm.encode_texts(prompts, normalize=True)

    all_entropy, all_margin, all_pred, all_gt = [], [], [], []
    with torch.no_grad():
        for images, paths, folder_names in loader:
            img_feat = vlm.encode_images(images, normalize=True)
            logits = vlm.similarity_logits(img_feat, text_feat)
            top2 = torch.topk(logits, k=2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            probs = F.softmax(logits, dim=-1)
            entropy = (-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)).cpu().numpy()
            pred = logits.argmax(-1).cpu().numpy()
            gt = [folder_to_idx.get(fn, -1) for fn in folder_names]
            all_entropy.append(entropy)
            all_margin.append(margin)
            all_pred.append(pred)
            all_gt.extend(gt)

    entropy = np.concatenate(all_entropy)
    pred = np.concatenate(all_pred)
    gt = np.array(all_gt)
    correct = (pred == gt).astype(int)
    error = 1 - correct
    accuracy = correct.mean()
    print(f"Model: {args.model}  N={args.n_images}  Accuracy: {accuracy:.1%}")

    # ── MC dropout ──
    vlm.ensure_uniform_dropout(args.dropout)
    print(f"Running MC dropout: T={args.passes}, p={args.dropout} ...")
    trial = run_mc_trial(
        vlm=vlm, loader=loader, passes=args.passes,
        collect_pass_features=True, progress=True,
        progress_desc=f"{args.model} MC",
    )
    metrics = compute_all_metrics(trial["pass_pre"], trial["pass_post"])
    metric_names = list(metrics.keys())
    N = args.n_images

    # ── build metric matrix ──
    M = np.column_stack([metrics[m] for m in metric_names])  # (N, K)
    K = len(metric_names)

    # ═══════════════════════════════════════════════════════════════════
    # 1. INTER-METRIC CORRELATIONS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("1. INTER-METRIC SPEARMAN CORRELATIONS")
    print(f"{'='*80}")

    corr_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            corr_matrix[i, j] = spearmanr(M[:, i], M[:, j]).statistic

    # Print condensed
    short_names = [m[:12] for m in metric_names]
    header = "            " + "  ".join(f"{s:>8s}" for s in short_names)
    print(header)
    for i, name in enumerate(short_names):
        row = f"  {name:>10s}"
        for j in range(K):
            row += f"  {corr_matrix[i,j]:+.3f}   "
        print(row)

    # ═══════════════════════════════════════════════════════════════════
    # 2. INDIVIDUAL BASELINES (for comparison)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("2. INDIVIDUAL METRIC BASELINES")
    print(f"{'='*80}")

    rho_fn = lambda x, y: spearmanr(x, y).statistic
    individual_rhos = {}

    print(f"  {'Metric':>20s}  {'rho(ent)':>10s}  {'AUROC(err)':>10s}")
    print(f"  {'-'*45}")
    for name in metric_names:
        rho = spearman_safe(metrics[name], entropy)
        auroc = auroc_safe(metrics[name], error)
        individual_rhos[name] = rho
        print(f"  {name:>20s}  {rho:+.4f}      {auroc:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. ENSEMBLE METHODS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("3. ENSEMBLE METHODS")
    print(f"{'='*80}")

    # z-score each column
    M_z = (M - M.mean(axis=0)) / M.std(axis=0, ddof=1).clip(1e-15)

    ensembles = {}

    # ── 3a. Naive average of all z-scored metrics ──
    avg_all = M_z.mean(axis=1)
    ensembles["avg_all"] = avg_all

    # ── 3b. Average of SIGNAL metrics only (trace_pre, trace_post, mean_cosine_dev) ──
    signal_idx = [i for i, n in enumerate(metric_names) if n in ("trace_pre", "trace_post", "mean_cosine_dev")]
    avg_signal = M_z[:, signal_idx].mean(axis=1)
    ensembles["avg_signal_only"] = avg_signal

    # ── 3c. Sign-corrected weighted average (weight = |rho|, sign = sign(rho)) ──
    weights = np.array([individual_rhos[m] for m in metric_names])
    # Flip sign for metrics that anti-correlate
    signed_M_z = M_z * np.sign(weights)
    abs_weights = np.abs(weights)
    abs_weights /= abs_weights.sum()  # normalize
    weighted_avg = (signed_M_z * abs_weights).sum(axis=1)
    ensembles["weighted_signed"] = weighted_avg

    # ── 3d. Rank average (rank each metric, average ranks) ──
    ranks = np.column_stack([rankdata(metrics[m]) for m in metric_names])
    # Flip ranks for anti-correlated metrics
    for i, m in enumerate(metric_names):
        if individual_rhos[m] < 0:
            ranks[:, i] = N + 1 - ranks[:, i]
    avg_rank = ranks.mean(axis=1)
    ensembles["rank_avg"] = avg_rank

    # ── 3e. PCA: first K components via SVD ──
    U, S, Vt = np.linalg.svd(M_z, full_matrices=False)
    n_pcs = min(K, 5)
    M_pca = U[:, :n_pcs] * S[:n_pcs]  # project
    pca_var_ratio = (S**2) / (S**2).sum()
    for pc in range(n_pcs):
        ensembles[f"PC{pc+1}"] = M_pca[:, pc]

    # ── 3f. Cross-validated ridge regression (predict error from metrics) ──
    # Manual 5-fold stratified CV with closed-form ridge
    rng = np.random.default_rng(42)
    n_folds = 5
    ridge_alpha = 1.0
    cv_scores = np.full(N, np.nan)
    # Stratified split: separate error=0 and error=1 indices, shuffle, assign folds
    idx_0 = np.where(error == 0)[0]
    idx_1 = np.where(error == 1)[0]
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)
    folds = np.zeros(N, dtype=int)
    for i, idx in enumerate(idx_0):
        folds[idx] = i % n_folds
    for i, idx in enumerate(idx_1):
        folds[idx] = i % n_folds

    for fold in range(n_folds):
        train_mask = folds != fold
        test_mask = folds == fold
        X_tr = M_z[train_mask]
        y_tr = error[train_mask].astype(float)
        X_te = M_z[test_mask]
        # Ridge: w = (X^T X + alpha I)^-1 X^T y
        XtX = X_tr.T @ X_tr + ridge_alpha * np.eye(K)
        w = np.linalg.solve(XtX, X_tr.T @ y_tr)
        cv_scores[test_mask] = X_te @ w
    ensembles["ridge_cv"] = cv_scores

    # ── 3g. Ridge CV, signal metrics only ──
    M_signal = M_z[:, signal_idx]
    K_sig = len(signal_idx)
    cv_scores_sig = np.full(N, np.nan)
    for fold in range(n_folds):
        train_mask = folds != fold
        test_mask = folds == fold
        X_tr = M_signal[train_mask]
        y_tr = error[train_mask].astype(float)
        X_te = M_signal[test_mask]
        XtX = X_tr.T @ X_tr + ridge_alpha * np.eye(K_sig)
        w = np.linalg.solve(XtX, X_tr.T @ y_tr)
        cv_scores_sig[test_mask] = X_te @ w
    ensembles["ridge_cv_signal"] = cv_scores_sig

    # ── Evaluate all ensembles ──
    print(f"\n  {'Method':>25s}  {'rho(ent)':>10s}  {'95% CI':>17s}  {'AUROC':>7s}  {'vs trace_pre':>13s}")
    print(f"  {'-'*78}")

    # Reference: trace_pre alone
    ref_rho = individual_rhos["trace_pre"]
    ref_auroc = auroc_safe(metrics["trace_pre"], error)

    results = {
        "model": args.model,
        "N": args.n_images,
        "T": args.passes,
        "p": args.dropout,
        "accuracy": float(accuracy),
        "individual_rhos": {m: float(individual_rhos[m]) for m in metric_names},
        "inter_metric_correlations": corr_matrix.tolist(),
        "metric_names": metric_names,
        "ensembles": {},
        "pca_explained_variance": pca_var_ratio[:n_pcs].tolist(),
    }

    # Print trace_pre baseline first
    ref_ci = bootstrap_ci(metrics["trace_pre"], entropy, rho_fn, n_boot=2000)
    print(
        f"  {'trace_pre (baseline)':>25s}  {ref_rho:+.4f}      "
        f"[{ref_ci[0]:+.3f}, {ref_ci[1]:+.3f}]  "
        f"{ref_auroc:.4f}   {'---':>13s}"
    )

    for ens_name, ens_vals in ensembles.items():
        rho = spearman_safe(ens_vals, entropy)
        auroc = auroc_safe(ens_vals, error)
        ci_lo, ci_hi = bootstrap_ci(ens_vals, entropy, rho_fn, n_boot=2000)
        delta = rho - ref_rho

        results["ensembles"][ens_name] = {
            "rho": float(rho),
            "rho_ci_lo": ci_lo,
            "rho_ci_hi": ci_hi,
            "auroc": float(auroc),
            "delta_vs_trace_pre": float(delta),
        }

        marker = "^" if delta > 0.02 else "v" if delta < -0.02 else "="
        print(
            f"  {ens_name:>25s}  {rho:+.4f}      "
            f"[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
            f"{auroc:.4f}   {delta:+.4f} {marker}"
        )

    # ── PCA explained variance ──
    print(f"\n  PCA explained variance ratios: {[f'{v:.3f}' for v in pca_var_ratio[:n_pcs]]}")

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    print(f"\n{'='*80}")
    print(f"Elapsed: {elapsed / 60:.1f} min")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
