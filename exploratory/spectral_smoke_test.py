#!/usr/bin/env python
"""
Smoke test: spectral & standard uncertainty metrics for validity.

For each metric, tests:
  H0: rho(metric, classification_entropy) = 0  (no validity signal)
  H1: rho(metric, classification_entropy) != 0  (metric tracks difficulty)

Reports Spearman rho, bootstrap 95% CI, permutation p-value, AUROC(error).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

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
from phase_two.metrics import compute_all_metrics

TEMPLATES = ["a photo of a {}", "a {}", "an image of a {}"]


# ── Statistical helpers ──────────────────────────────────────────────────

def bootstrap_ci(x, y, stat_fn, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a bivariate statistic."""
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(x[idx], y[idx])
    alpha = (1 - ci) / 2
    return float(np.nanpercentile(stats, 100 * alpha)), float(np.nanpercentile(stats, 100 * (1 - alpha)))


def permutation_pvalue(x, y, stat_fn, n_perm=5000, seed=42):
    """Two-sided permutation test p-value for H0: stat(x,y) = 0."""
    rng = np.random.default_rng(seed)
    observed = stat_fn(x, y)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if abs(stat_fn(x, y_perm)) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)  # +1 for continuity correction


def auroc_safe(scores, labels):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return auroc_from_scores(np.asarray(scores, dtype=np.float64), labels)


def auroc_fn(scores, labels):
    """Wrapper compatible with bootstrap_ci."""
    return auroc_safe(scores, labels)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spectral metrics smoke test")
    parser.add_argument("--data-dir", default="data/raw/imagenet_val")
    parser.add_argument("--class-map", default="data/imagenet_class_map.json")
    parser.add_argument("--model", default="clip_b32")
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--output", default="outputs/spectral_smoke.json")
    args = parser.parse_args()

    if args.device is None:
        args.device = detect_best_device()

    t0 = time.time()
    set_all_seeds(args.seed)

    # ── load data ──
    data_dir = Path(args.data_dir)
    all_paths = list_images(str(data_dir))
    sampled = sample_paths(all_paths, args.n_images, args.seed)
    class_names = discover_class_names(str(data_dir), mapping_path=args.class_map)
    folders = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    folder_to_idx = {f: i for i, f in enumerate(folders)}
    loader = build_loader(sampled, batch_size=args.batch_size, num_workers=0)

    # ── load model ──
    vlm = load_model(args.model, device=args.device)

    # ── classification signals (deterministic) ──
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
    margin = np.concatenate(all_margin)
    pred = np.concatenate(all_pred)
    gt = np.array(all_gt)
    correct = (pred == gt).astype(int)
    error = 1 - correct
    accuracy = correct.mean()
    print(f"Model: {args.model}  N={args.n_images}  Accuracy: {accuracy:.1%}")

    # ── MC dropout features ──
    vlm.ensure_uniform_dropout(args.dropout)
    print(f"Running MC dropout: T={args.passes}, p={args.dropout} ...")
    trial = run_mc_trial(
        vlm=vlm, loader=loader, passes=args.passes,
        collect_pass_features=True, progress=True,
        progress_desc=f"{args.model} MC",
    )
    pass_pre = trial["pass_pre"]
    pass_post = trial["pass_post"]

    # ── compute all metrics ──
    print("Computing metrics (eigendecomposition per image) ...")
    metrics = compute_all_metrics(pass_pre, pass_post)

    # ── evaluate each metric ──
    rho_fn = lambda x, y: spearmanr(x, y).statistic

    header = (
        f"{'Metric':>20s}  {'rho':>7s}  {'95% CI':>15s}  {'perm-p':>8s}  "
        f"{'AUROC':>6s}  {'AUROC 95% CI':>15s}  {'Verdict':>10s}"
    )
    print(f"\n{'='*90}")
    print(f"  H0: rho(metric, classification_entropy) = 0")
    print(f"  H1: rho(metric, classification_entropy) != 0")
    print(f"{'='*90}")
    print(header)
    print("-" * 90)

    results = {
        "model": args.model,
        "N": args.n_images,
        "T": args.passes,
        "p": args.dropout,
        "accuracy": float(accuracy),
        "metrics": {},
    }

    for name, vals in metrics.items():
        vals = np.asarray(vals, dtype=np.float64)
        rho = rho_fn(vals, entropy)
        rho_lo, rho_hi = bootstrap_ci(vals, entropy, rho_fn, n_boot=args.n_boot)
        p_val = permutation_pvalue(vals, entropy, rho_fn, n_perm=args.n_perm)
        auroc = auroc_safe(vals, error)
        auroc_lo, auroc_hi = bootstrap_ci(vals, error, auroc_fn, n_boot=args.n_boot)

        # Verdict
        if p_val < 0.01 and rho_lo > 0:
            verdict = "SIGNAL"
        elif p_val < 0.05:
            verdict = "WEAK"
        else:
            verdict = "NULL"

        results["metrics"][name] = {
            "rho": float(rho),
            "rho_ci_lo": rho_lo,
            "rho_ci_hi": rho_hi,
            "perm_p": float(p_val),
            "auroc": float(auroc),
            "auroc_ci_lo": auroc_lo,
            "auroc_ci_hi": auroc_hi,
            "verdict": verdict,
        }

        print(
            f"{name:>20s}  {rho:+.4f}  [{rho_lo:+.3f}, {rho_hi:+.3f}]  "
            f"{p_val:.1e}  {auroc:.4f}  [{auroc_lo:.3f}, {auroc_hi:.3f}]  "
            f"{verdict:>10s}"
        )

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    print(f"\n{'='*90}")
    print(f"Elapsed: {elapsed/60:.1f} min")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
