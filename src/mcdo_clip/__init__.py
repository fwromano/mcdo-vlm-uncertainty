"""Utilities for MC Dropout uncertainty experiments with CLIP."""

from .sampling import sample_mc_embeddings, compute_mc_stats, compute_embedding_covariance
from .metrics import (
    compute_pixel_entropy,
    compute_edge_density,
    compute_colorfulness,
    compute_jpeg_compressibility,
    compute_complexity_metrics,
)
from .models import load_clip_model, enable_mc_dropout, attach_dropout_adapters
from .data import ImageFolderDataset

__all__ = [
    "sample_mc_embeddings",
    "compute_mc_stats",
    "compute_embedding_covariance",
    "compute_pixel_entropy",
    "compute_edge_density",
    "compute_colorfulness",
    "compute_jpeg_compressibility",
    "compute_complexity_metrics",
    "load_clip_model",
    "enable_mc_dropout",
    "attach_dropout_adapters",
    "ImageFolderDataset",
]
