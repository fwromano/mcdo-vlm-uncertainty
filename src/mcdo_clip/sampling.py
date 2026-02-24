from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import encode_images


@torch.no_grad()
def sample_mc_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    passes: int = 64,
    device: str = "cpu",
    l2_normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Run MC sampling and return mean/variance per sample.

    Returns dict with keys: mean (N,D), var (N,D), l2_var (N).
    """

    model.to(device)
    num_samples = len(dataloader.dataset)
    mean = None
    sq_mean = None

    for t in range(passes):
        offset = 0
        for images, _paths in dataloader:
            b = images.shape[0]
            images = images.to(device)
            emb = encode_images(model, images)
            if l2_normalize:
                emb = F.normalize(emb, dim=-1)
            emb = emb.detach().cpu()

            if mean is None:
                d = emb.shape[1]
                mean = torch.zeros((num_samples, d), dtype=emb.dtype)
                sq_mean = torch.zeros_like(mean)
            mean[offset : offset + b] += emb
            sq_mean[offset : offset + b] += emb * emb
            offset += b

    assert mean is not None and sq_mean is not None
    mean = mean / passes
    var = sq_mean / passes - mean * mean
    l2_var = var.sum(dim=-1)
    return {"mean": mean, "var": var, "l2_var": l2_var}


def compute_mc_stats(var: torch.Tensor) -> Dict[str, float]:
    """Aggregate scalar stats from per-sample variance (sum over dims)."""
    l2_var = var.sum(dim=-1).numpy()
    return {
        "mean": float(np.mean(l2_var)),
        "std": float(np.std(l2_var)),
        "min": float(np.min(l2_var)),
        "max": float(np.max(l2_var)),
    }


@torch.no_grad()
def compute_embedding_covariance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    passes: int = 8,
    device: str = "cpu",
    l2_normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute dataset-level covariance over all embeddings across MC passes."""

    model.to(device)
    total = 0
    mean = None
    second = None

    for _ in range(passes):
        for images, _paths in dataloader:
            images = images.to(device)
            emb = encode_images(model, images)
            if l2_normalize:
                emb = F.normalize(emb, dim=-1)
            emb = emb.detach().cpu().to(torch.float64)

            if mean is None:
                d = emb.shape[1]
                mean = torch.zeros(d, dtype=torch.float64)
                second = torch.zeros((d, d), dtype=torch.float64)

            mean += emb.sum(dim=0)
            second += emb.T @ emb
            total += emb.shape[0]

    if mean is None or second is None or total == 0:
        raise ValueError("No embeddings were collected; check dataloader contents.")

    mean = mean / float(total)
    cov = second / float(total) - torch.outer(mean, mean)
    return {"mean": mean, "cov": cov, "count": total}
