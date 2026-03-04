"""Uncertainty metrics computed from per-pass MC dropout features.

All metrics take per-pass feature tensors and return per-image scalar scores.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_all_metrics(
    pass_pre: torch.Tensor, pass_post: torch.Tensor
) -> dict[str, np.ndarray]:
    """Compute all candidate uncertainty metrics from per-pass features.

    Args:
        pass_pre:  (T, N, D) raw features per pass
        pass_post: (T, N, D) L2-normalised features per pass

    Returns:
        dict  metric_name -> (N,) numpy array
    """
    T, N, D = pass_pre.shape
    centered = pass_pre - pass_pre.mean(dim=0, keepdim=True)
    var_pre = pass_pre.var(dim=0)  # (N, D)

    metrics: dict[str, np.ndarray] = {}

    # ── baselines ──
    metrics["trace_pre"] = (var_pre.sum(dim=1) / D).numpy()

    var_post = pass_post.var(dim=0)
    metrics["trace_post"] = (var_post.sum(dim=1) / D).numpy()

    metrics["max_dim_var"] = var_pre.max(dim=1).values.numpy()
    metrics["norm_var"] = pass_pre.norm(dim=2).var(dim=0).numpy()

    mean_dir = F.normalize(pass_post.mean(dim=0), dim=-1)
    cos_sims = (pass_post * mean_dir.unsqueeze(0)).sum(dim=-1)
    metrics["mean_cosine_dev"] = (1.0 - cos_sims.mean(dim=0)).numpy()

    # ── eigenvalue-based (via T×T Gram trick) ──
    eff_rank = np.zeros(N)
    spec_entropy = np.zeros(N)
    top1_ratio = np.zeros(N)
    top_eig = np.zeros(N)

    for i in range(N):
        X = centered[:, i, :]       # (T, D)
        G = X @ X.T / (T - 1)       # (T, T)
        eigs = torch.linalg.eigvalsh(G).clamp_min(0)  # ascending

        total = eigs.sum().item()
        if total < 1e-15:
            continue

        top_eig[i] = eigs[-1].item()

        sum_sq = (eigs**2).sum().item()
        eff_rank[i] = (total**2) / sum_sq

        p = eigs / total
        p = p[p > 1e-12]
        spec_entropy[i] = -(p * torch.log(p)).sum().item()

        top1_ratio[i] = eigs[-1].item() / total

    metrics["top_eigenvalue"] = top_eig
    metrics["effective_rank"] = eff_rank
    metrics["spectral_entropy"] = spec_entropy
    metrics["top1_ratio"] = top1_ratio

    # ── metric engineering ──
    metrics["weighted_trace_pre"] = weighted_trace_pre(pass_pre)
    metrics["topk64_trace_pre"] = topk_dim_trace(pass_pre, k=64)

    return metrics


def weighted_trace_pre(pass_pre: torch.Tensor) -> np.ndarray:
    """Trace weighted by per-dimension discriminative power.

    weight_d = Var_images(mean_d) / sum(Var_images(mean_d))
    uncertainty_i = sum_d(weight_d * Var_passes(feat_id))

    Dimensions that vary more across images (discriminative) get higher weight.
    """
    T, N, D = pass_pre.shape
    # Per-image mean over passes: (N, D)
    img_means = pass_pre.mean(dim=0)
    # Per-dim variance across images: (D,) — discriminative power
    disc_power = img_means.var(dim=0).clamp_min(0)
    total = disc_power.sum()
    if total < 1e-12:
        weights = torch.ones(D, dtype=pass_pre.dtype) / D
    else:
        weights = disc_power / total  # (D,) sums to 1

    # Per-pass per-image variance: (N, D)
    var_pre = pass_pre.var(dim=0)
    # Weighted sum per image: (N,)
    return (var_pre * weights.unsqueeze(0)).sum(dim=1).numpy()


def topk_dim_trace(pass_pre: torch.Tensor, k: int = 64) -> np.ndarray:
    """trace_pre restricted to top-K dimensions by across-image discriminative power."""
    T, N, D = pass_pre.shape
    k = min(k, D)
    img_means = pass_pre.mean(dim=0)
    disc_power = img_means.var(dim=0)
    top_dims = disc_power.topk(k).indices  # (K,)
    var_pre = pass_pre.var(dim=0)           # (N, D)
    return (var_pre[:, top_dims].sum(dim=1) / k).numpy()
