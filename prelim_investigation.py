#!/usr/bin/env python
"""
3-angle preliminary investigation of MC dropout uncertainty signal.

Angle 1: Alternative uncertainty metrics (both models, N=500)
Angle 2: clip_b32 at larger N (N=2000) to confirm signal
Angle 3: What does siglip2_b16 uncertainty actually correlate with?
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from phase_one.common import (
    auroc_from_scores, build_loader, discover_class_names, list_images,
    load_model, run_mc_trial, sample_paths, set_all_seeds, spearman_safe,
)

DATA_DIR = "data/raw/imagenet_val"
CLASS_MAP = "data/imagenet_class_map.json"
TEMPLATES = ["a photo of a {}", "a {}", "an image of a {}"]
DROPOUT = 0.01
PASSES = 64
DEVICE = "mps"
SEED = 42
BATCH_SIZE = 32

def auroc_safe(scores, labels):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return auroc_from_scores(np.asarray(scores, dtype=np.float64), labels)

def compute_alt_metrics(pass_features_pre, pass_features_post):
    """Compute alternative uncertainty metrics from per-pass features.
    pass_features: shape (T, N, D) in float32
    Returns dict of metric_name -> (N,) array
    """
    T, N, D = pass_features_pre.shape
    metrics = {}

    # 1. trace_pre (standard - baseline)
    var_pre = pass_features_pre.var(dim=0)  # (N, D)
    metrics["trace_pre"] = (var_pre.sum(dim=1) / D).numpy()

    # 2. trace_post (post L2-norm)
    var_post = pass_features_post.var(dim=0)
    metrics["trace_post"] = (var_post.sum(dim=1) / D).numpy()

    # 3. max_dim_var (max variance across dimensions, pre-norm)
    metrics["max_dim_var"] = var_pre.max(dim=1).values.numpy()

    # 4. top_eigenvalue (approximate via power iteration on covariance)
    centered = pass_features_pre - pass_features_pre.mean(dim=0, keepdim=True)  # (T, N, D)
    # For each image, compute top eigenvalue of (T x D) centered matrix
    top_eigs = []
    for i in range(N):
        # cov is (D, D), but T << D so use (T, T) trick
        X = centered[:, i, :]  # (T, D)
        gram = X @ X.T / (T - 1)  # (T, T)
        eig = torch.linalg.eigvalsh(gram)
        top_eigs.append(eig[-1].item())
    metrics["top_eigenvalue"] = np.array(top_eigs)

    # 5. norm_var (variance of the L2 norm across passes)
    norms = pass_features_pre.norm(dim=2)  # (T, N)
    metrics["norm_var"] = norms.var(dim=0).numpy()

    # 6. cosine_var (1 - mean pairwise cosine sim across passes)
    mean_dir = F.normalize(pass_features_post.mean(dim=0), dim=-1)  # (N, D)
    cos_sims = (pass_features_post * mean_dir.unsqueeze(0)).sum(dim=-1)  # (T, N)
    metrics["mean_cosine_dev"] = (1.0 - cos_sims.mean(dim=0)).numpy()

    return metrics


def get_classification_signals(vlm, loader, class_names, data_dir, templates):
    """Get margin, entropy, predictions, correctness."""
    text_features = []
    for template in templates:
        prompts = [template.format(name) for name in class_names]
        tf = vlm.encode_texts(prompts, normalize=True)
        text_features.append(tf)

    # Build folder→index mapping from sorted directory names (synset IDs).
    # discover_class_names returns class_names in sorted-folder order,
    # so folder index i corresponds to class_names[i].
    folders = sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])
    folder_to_idx = {folder: i for i, folder in enumerate(folders)}

    margins, entropies, preds, gt_labels = [], [], [], []

    with torch.no_grad():
        for images, paths, folder_names in loader:
            img_feats = vlm.encode_images(images, normalize=True)
            logits = vlm.similarity_logits(img_feats, text_features[0])
            top2 = torch.topk(logits, k=2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()

            probs = F.softmax(logits, dim=-1)
            entropy = (-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).detach().cpu().numpy()
            pred = logits.argmax(dim=-1).detach().cpu().numpy()

            gt = [folder_to_idx.get(fn, -1) for fn in folder_names]

            margins.append(margin)
            entropies.append(entropy)
            preds.append(pred)
            gt_labels.extend(gt)

    return {
        "margin": np.concatenate(margins),
        "entropy": np.concatenate(entropies),
        "pred": np.concatenate(preds),
        "gt": np.array(gt_labels),
    }


def run_angle1(results):
    """Alternative uncertainty metrics for both models, N=500."""
    print("\n" + "="*60)
    print("ANGLE 1: Alternative Uncertainty Metrics (N=500)")
    print("="*60)

    all_paths = list_images(DATA_DIR)
    sampled = sample_paths(all_paths, 500, SEED)
    class_names = discover_class_names(DATA_DIR, mapping_path=CLASS_MAP)
    angle1 = {}

    for model_key in ["clip_b32", "siglip2_b16"]:
        print(f"\n--- {model_key} ---")
        set_all_seeds(SEED)
        vlm = load_model(model_key, device=DEVICE)
        vlm.ensure_uniform_dropout(DROPOUT)
        loader = build_loader(sampled, batch_size=BATCH_SIZE, num_workers=0)

        # Get classification signals
        vlm.disable_dropout()
        signals = get_classification_signals(vlm, loader, class_names, DATA_DIR, TEMPLATES)
        correct = (signals["pred"] == signals["gt"]).astype(int)
        accuracy = correct.mean()
        print(f"  Accuracy: {accuracy:.1%} ({correct.sum()}/{len(correct)})")

        # Run MC with per-pass features
        vlm.ensure_uniform_dropout(DROPOUT)
        trial = run_mc_trial(
            vlm=vlm, loader=loader, passes=PASSES,
            collect_pass_features=True, progress=True, progress_desc=f"{model_key} MC"
        )

        alt_metrics = compute_alt_metrics(trial["pass_pre"], trial["pass_post"])
        model_results = {"accuracy": float(accuracy)}

        for metric_name, unc_arr in alt_metrics.items():
            rho_entropy = spearman_safe(unc_arr, signals["entropy"])
            rho_neg_margin = spearman_safe(unc_arr, -signals["margin"])
            auroc_err = auroc_safe(unc_arr, 1 - correct)

            model_results[metric_name] = {
                "rho_entropy": rho_entropy,
                "rho_neg_margin": rho_neg_margin,
                "auroc_error": auroc_err,
            }
            print(f"  {metric_name:20s}  rho(ent)={rho_entropy:+.4f}  rho(-mrg)={rho_neg_margin:+.4f}  AUROC(err)={auroc_err:.4f}")

        angle1[model_key] = model_results
        del vlm
        torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

    results["angle1_alt_metrics"] = angle1


def run_angle2(results):
    """clip_b32 at N=2000 to confirm signal strength."""
    print("\n" + "="*60)
    print("ANGLE 2: clip_b32 at N=2000")
    print("="*60)

    all_paths = list_images(DATA_DIR)
    sampled = sample_paths(all_paths, 2000, SEED)
    class_names = discover_class_names(DATA_DIR, mapping_path=CLASS_MAP)

    set_all_seeds(SEED)
    vlm = load_model("clip_b32", device=DEVICE)
    vlm.ensure_uniform_dropout(DROPOUT)
    loader = build_loader(sampled, batch_size=BATCH_SIZE, num_workers=0)

    # Classification
    vlm.disable_dropout()
    signals = get_classification_signals(vlm, loader, class_names, DATA_DIR, TEMPLATES)
    correct = (signals["pred"] == signals["gt"]).astype(int)
    accuracy = correct.mean()
    print(f"  Accuracy: {accuracy:.1%} ({correct.sum()}/{len(correct)})")

    # MC with per-pass features
    vlm.ensure_uniform_dropout(DROPOUT)
    trial = run_mc_trial(
        vlm=vlm, loader=loader, passes=PASSES,
        collect_pass_features=True, progress=True, progress_desc="clip_b32 N=2000 MC"
    )

    alt_metrics = compute_alt_metrics(trial["pass_pre"], trial["pass_post"])
    angle2 = {"accuracy": float(accuracy), "N": 2000}

    for metric_name, unc_arr in alt_metrics.items():
        rho_entropy = spearman_safe(unc_arr, signals["entropy"])
        rho_neg_margin = spearman_safe(unc_arr, -signals["margin"])
        auroc_err = auroc_safe(unc_arr, 1 - correct)

        angle2[metric_name] = {
            "rho_entropy": rho_entropy,
            "rho_neg_margin": rho_neg_margin,
            "auroc_error": auroc_err,
        }
        print(f"  {metric_name:20s}  rho(ent)={rho_entropy:+.4f}  rho(-mrg)={rho_neg_margin:+.4f}  AUROC(err)={auroc_err:.4f}")

    results["angle2_clip_b32_n2000"] = angle2
    del vlm
    torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None


def run_angle3(results):
    """What does siglip2_b16 uncertainty correlate with?"""
    print("\n" + "="*60)
    print("ANGLE 3: What does siglip2_b16 uncertainty correlate with? (N=1000)")
    print("="*60)

    all_paths = list_images(DATA_DIR)
    sampled = sample_paths(all_paths, 1000, SEED)
    class_names = discover_class_names(DATA_DIR, mapping_path=CLASS_MAP)

    set_all_seeds(SEED)
    vlm = load_model("siglip2_b16", device=DEVICE)
    vlm.ensure_uniform_dropout(DROPOUT)
    loader = build_loader(sampled, batch_size=BATCH_SIZE, num_workers=0)

    # Classification
    vlm.disable_dropout()
    signals = get_classification_signals(vlm, loader, class_names, DATA_DIR, TEMPLATES)
    correct = (signals["pred"] == signals["gt"]).astype(int)
    accuracy = correct.mean()
    print(f"  Accuracy: {accuracy:.1%} ({correct.sum()}/{len(correct)})")

    # Deterministic features
    det_feats_pre, det_feats_post = [], []
    with torch.no_grad():
        for images, _, _ in loader:
            f_pre = vlm.encode_images(images, normalize=False).detach().cpu()
            f_post = F.normalize(f_pre, dim=-1)
            det_feats_pre.append(f_pre)
            det_feats_post.append(f_post)
    det_pre = torch.cat(det_feats_pre, dim=0)
    det_post = torch.cat(det_feats_post, dim=0)

    # Feature properties
    feat_norms = det_pre.norm(dim=1).numpy()
    centroid = det_post.mean(dim=0)
    centroid_dist = (1.0 - (det_post @ centroid.unsqueeze(1)).squeeze()).numpy()

    # MC uncertainty
    vlm.ensure_uniform_dropout(DROPOUT)
    trial = run_mc_trial(
        vlm=vlm, loader=loader, passes=PASSES,
        collect_pass_features=True, progress=True, progress_desc="siglip2_b16 N=1000 MC"
    )
    unc_trace = trial["trace_pre"].numpy()

    # Text encoder info
    text_features = []
    for t in TEMPLATES:
        prompts = [t.format(name) for name in class_names]
        tf = vlm.encode_texts(prompts, normalize=True)
        text_features.append(tf)

    # Max logit (deterministic)
    with torch.no_grad():
        logits = vlm.similarity_logits(det_post.to(DEVICE), text_features[0])
        max_logit = logits.max(dim=-1).values.detach().cpu().numpy()

    angle3 = {"accuracy": float(accuracy), "N": 1000}
    correlates = {
        "feat_norm": feat_norms,
        "centroid_distance": centroid_dist,
        "max_logit": max_logit,
        "classification_entropy": signals["entropy"],
        "negative_margin": -signals["margin"],
        "correctness": correct.astype(float),
    }

    print("\n  Correlation of trace_pre uncertainty with:")
    for name, arr in correlates.items():
        rho = spearman_safe(unc_trace, arr)
        angle3[f"rho_vs_{name}"] = rho
        print(f"    {name:25s}  rho = {rho:+.4f}")

    # Correct vs incorrect uncertainty comparison
    if correct.sum() > 0 and (1 - correct).sum() > 0:
        unc_correct = unc_trace[correct == 1]
        unc_wrong = unc_trace[correct == 0]
        angle3["mean_unc_correct"] = float(unc_correct.mean())
        angle3["mean_unc_wrong"] = float(unc_wrong.mean())
        angle3["unc_ratio_wrong_over_correct"] = float(unc_wrong.mean() / unc_correct.mean())
        print(f"\n  Mean uncertainty (correct):   {unc_correct.mean():.6f}")
        print(f"  Mean uncertainty (incorrect): {unc_wrong.mean():.6f}")
        print(f"  Ratio (wrong/correct):        {unc_wrong.mean() / unc_correct.mean():.3f}")

    # Quintile analysis
    quintile_idx = np.argsort(unc_trace)
    q_size = len(unc_trace) // 5
    print("\n  Accuracy by uncertainty quintile:")
    for q in range(5):
        start = q * q_size
        end = start + q_size if q < 4 else len(unc_trace)
        idx = quintile_idx[start:end]
        q_acc = correct[idx].mean()
        q_unc = unc_trace[idx].mean()
        angle3[f"q{q+1}_accuracy"] = float(q_acc)
        angle3[f"q{q+1}_mean_unc"] = float(q_unc)
        print(f"    Q{q+1} (n={len(idx)}, mean_unc={q_unc:.6f}): acc={q_acc:.3f}")

    results["angle3_siglip2_correlates"] = angle3
    del vlm
    torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None


def main():
    t0 = time.time()
    results = {}

    run_angle1(results)
    run_angle2(results)
    run_angle3(results)

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    print(f"\n{'='*60}")
    print(f"Total elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    out_path = Path("outputs/prelim_investigation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
