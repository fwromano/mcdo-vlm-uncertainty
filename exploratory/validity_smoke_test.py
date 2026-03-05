#!/usr/bin/env python
"""Validity smoke test: does the Gaussian-based uncertainty actually mean anything?

Tests whether each perturbation config's uncertainty ranking correlates with
meaningful proxies:
  - Classification entropy (semantic uncertainty)
  - Classification margin (confidence gap)
  - Classification error (practical utility)

High reliability (Spearman across trials) without validity (correlation with
these proxies) would mean we built a very consistent but meaningless metric.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from phase_two.ablation import DEGRADATIONS, DegradedImageDataset
from phase_two.metrics import compute_all_metrics
from phase_two.perturbation import disable_all_perturbation, perturb_modules

TEMPLATES = ["a photo of a {}", "a {}", "an image of a {}"]


def auroc_safe(scores, labels):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return auroc_from_scores(np.asarray(scores, dtype=np.float64), labels)


def main():
    t0 = time.time()
    device = detect_best_device()
    vlm = load_model("clip_b32", device=device)

    # ── Data ──
    data_dir = "data/raw/imagenet_val"
    all_paths = list_images(data_dir)
    sampled = sample_paths(all_paths, 500, seed=42)
    class_names = discover_class_names(data_dir, mapping_path="data/imagenet_class_map.json")
    folders = sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])
    folder_to_idx = {f: i for i, f in enumerate(folders)}
    loader = build_loader(sampled, batch_size=32, num_workers=0)

    # ── Classification ground truth ──
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
    print(f"Accuracy: {accuracy:.1%}  N={len(sampled)}")
    print()

    # ── Configurations to test ──
    configs = {
        "gaussian_block11": ("transformer.resblocks.11.mlp.c_proj", "gaussian", 0.05),
        "gaussian_block10": ("transformer.resblocks.10.mlp.c_proj", "gaussian", 0.05),
        "gaussian_block9": ("transformer.resblocks.9.mlp.c_proj", "gaussian", 0.05),
        "gaussian_block0": ("transformer.resblocks.0.mlp.c_proj", "gaussian", 0.05),
        "dropout_block11": ("transformer.resblocks.11.mlp.c_proj", "dropout", 0.05),
        "dropout_block9": ("transformer.resblocks.9.mlp.c_proj", "dropout", 0.01),
        "scale_block11": ("transformer.resblocks.11.mlp.c_proj", "scale", 0.05),
    }

    print("Running MC trials (T=64, K=1 each)...")
    sep = "=" * 90
    print(sep)

    results = {}
    root = vlm.vision_root
    metric_names = None

    for label, (module, ptype, mag) in configs.items():
        disable_all_perturbation(root)
        set_all_seeds(42)
        with perturb_modules(root, [(module, ptype, mag)]):
            trial = run_mc_trial(
                vlm=vlm, loader=loader, passes=64, collect_pass_features=True,
            )

        metrics = compute_all_metrics(trial["pass_pre"], trial["pass_post"])
        if metric_names is None:
            metric_names = list(metrics.keys())

        short_mod = module.split(".")[-3]
        print(f"\n  {label} ({ptype}@{mag} on {short_mod}):")
        print(f"  {'Metric':>25s}  {'rho(entropy)':>12s}  {'rho(margin)':>12s}  {'AUROC(err)':>10s}")
        print(f"  {'-' * 65}")

        entry = {"module": module, "ptype": ptype, "magnitude": mag, "metrics": {}}
        for m in metric_names:
            rho_ent = spearman_safe(metrics[m], entropy)
            rho_mar = spearman_safe(metrics[m], margin)
            auroc = auroc_safe(metrics[m], error)
            entry["metrics"][m] = {
                "rho_entropy": float(rho_ent),
                "rho_margin": float(rho_mar),
                "auroc_error": float(auroc),
            }
            flag = ""
            if abs(rho_ent) > 0.15:
                flag = " <-- signal"
            if abs(rho_ent) > 0.3:
                flag = " <-- STRONG"
            print(
                f"  {m:>25s}  {rho_ent:+.4f}        {rho_mar:+.4f}        {auroc:.4f}{flag}"
            )
        results[label] = entry

    # ── Uniform dropout baseline ──
    print(f"\n  uniform_dropout_E (dropout@0.01 on all 36 modules):")
    print(f"  {'Metric':>25s}  {'rho(entropy)':>12s}  {'rho(margin)':>12s}  {'AUROC(err)':>10s}")
    print(f"  {'-' * 65}")
    disable_all_perturbation(root)
    set_all_seeds(42)
    vlm.ensure_uniform_dropout(0.01)
    trial = run_mc_trial(
        vlm=vlm, loader=loader, passes=64, collect_pass_features=True,
    )
    metrics = compute_all_metrics(trial["pass_pre"], trial["pass_post"])

    entry = {"module": "all", "ptype": "dropout", "magnitude": 0.01, "metrics": {}}
    for m in metric_names:
        rho_ent = spearman_safe(metrics[m], entropy)
        rho_mar = spearman_safe(metrics[m], margin)
        auroc = auroc_safe(metrics[m], error)
        entry["metrics"][m] = {
            "rho_entropy": float(rho_ent),
            "rho_margin": float(rho_mar),
            "auroc_error": float(auroc),
        }
        flag = ""
        if abs(rho_ent) > 0.15:
            flag = " <-- signal"
        if abs(rho_ent) > 0.3:
            flag = " <-- STRONG"
        print(
            f"  {m:>25s}  {rho_ent:+.4f}        {rho_mar:+.4f}        {auroc:.4f}{flag}"
        )
    results["uniform_dropout_E"] = entry

    # ── Summary comparison (trace_pre only) ──
    print(f"\n{sep}")
    print("SUMMARY: trace_pre across all configs")
    print(sep)
    print(f"  {'Config':>25s}  {'rho(ent)':>10s}  {'rho(mar)':>10s}  {'AUROC':>7s}")
    print(f"  {'-' * 60}")
    for label, entry in results.items():
        tp = entry["metrics"]["trace_pre"]
        print(
            f"  {label:>25s}  {tp['rho_entropy']:+.4f}      {tp['rho_margin']:+.4f}      {tp['auroc_error']:.4f}"
        )

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed / 60:.1f} min")

    out_path = Path("outputs/validity_smoke_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
