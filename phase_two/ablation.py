"""Shared ablation test utilities for image degradation validity testing.

The ablation test checks whether degraded images (blur, downsample) produce
higher uncertainty than their clean counterparts. This is a paired within-image
comparison using the Wilcoxon signed-rank test.

Pass threshold: >= 75% of images show higher uncertainty when degraded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Dataset

from phase_one.common import (
    VisionLanguageModel,
    build_loader,
    pil_collate,
    run_mc_trial,
    set_all_seeds,
)
from phase_two.perturbation import disable_all_perturbation, perturb_modules


# ── Degradation functions ──────────────────────────────────────────

DEGRADATIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "blur_r5": lambda img: img.filter(ImageFilter.GaussianBlur(radius=5)),
    "blur_r15": lambda img: img.filter(ImageFilter.GaussianBlur(radius=15)),
    "downsample_4x": lambda img: img.resize(
        (max(img.width // 4, 1), max(img.height // 4, 1)), Image.BILINEAR
    ).resize((img.width, img.height), Image.BILINEAR),
    "downsample_8x": lambda img: img.resize(
        (max(img.width // 8, 1), max(img.height // 8, 1)), Image.BILINEAR
    ).resize((img.width, img.height), Image.BILINEAR),
}


class DegradedImageDataset(Dataset):
    """PIL image dataset that applies a degradation function on load."""

    def __init__(
        self, image_paths: Sequence[str], degrade_fn: Callable[[Image.Image], Image.Image]
    ) -> None:
        self.image_paths = [str(Path(p)) for p in image_paths]
        self.degrade_fn = degrade_fn

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        degraded = self.degrade_fn(image)
        class_name = Path(path).parent.name
        return degraded, path, class_name


# ── Statistical comparison ──────────────────────────────────────────

def paired_comparison(
    clean_unc: np.ndarray, deg_unc: np.ndarray
) -> Dict[str, float]:
    """Paired Wilcoxon test: does degradation increase uncertainty?

    Returns dict with frac_increased, rel_change_pct, wilcoxon_p_greater,
    wilcoxon_p_two_sided, clean_mean, degraded_mean.
    """
    diff = deg_unc - clean_unc
    frac_increased = float((diff > 0).mean())
    mean_diff = float(diff.mean())
    median_diff = float(np.median(diff))

    try:
        _, p_greater = wilcoxon(deg_unc, clean_unc, alternative="greater")
    except ValueError:
        p_greater = 1.0

    try:
        _, p_two = wilcoxon(deg_unc, clean_unc, alternative="two-sided")
    except ValueError:
        p_two = 1.0

    clean_mean = float(clean_unc.mean())
    deg_mean = float(deg_unc.mean())
    rel_change = (deg_mean - clean_mean) / clean_mean * 100 if abs(clean_mean) > 1e-15 else 0.0

    return {
        "frac_increased": frac_increased,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "rel_change_pct": rel_change,
        "wilcoxon_p_greater": float(p_greater),
        "wilcoxon_p_two_sided": float(p_two),
        "clean_mean": clean_mean,
        "degraded_mean": deg_mean,
    }


# ── High-level ablation runner ──────────────────────────────────────


def _run_mc_with_perturbation(
    vlm: VisionLanguageModel,
    loader: DataLoader,
    perturbation_configs: Optional[List[tuple]],
    passes: int,
    seed: int,
) -> np.ndarray:
    """Run MC trial and return trace_pre uncertainty values.

    Args:
        perturbation_configs: List of (path, ptype, magnitude) for perturb_modules.
            If None, uses vlm's current dropout state (caller should set up uniform dropout).
    """
    root = vlm.vision_root
    set_all_seeds(seed)

    if perturbation_configs is not None:
        vlm.disable_dropout()
        disable_all_perturbation(root)
        with perturb_modules(root, perturbation_configs):
            trial = run_mc_trial(
                vlm=vlm, loader=loader, passes=passes,
                collect_pass_features=False,
                cache_precomputed_pixels=False,
            )
    else:
        trial = run_mc_trial(
            vlm=vlm, loader=loader, passes=passes,
            collect_pass_features=False,
            cache_precomputed_pixels=False,
        )
    return trial["trace_pre"].numpy()


def run_ablation_test(
    vlm: VisionLanguageModel,
    image_paths: Sequence[str],
    perturbation_configs: Optional[List[tuple]],
    passes: int = 64,
    batch_size: int = 32,
    seed: int = 42,
    degradation_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full ablation validity test: clean vs degraded images.

    Args:
        vlm: Loaded VisionLanguageModel
        image_paths: Paths to clean images
        perturbation_configs: List of (path, ptype, mag) for perturb_modules,
            or None for uniform dropout (caller must set up via ensure_uniform_dropout first).
        passes: Number of MC forward passes
        batch_size: Batch size for data loading
        seed: Random seed
        degradation_names: Which degradations to test (default: all)

    Returns:
        Dict with 'clean_mean' and per-degradation paired_comparison results.
    """
    degs = degradation_names or list(DEGRADATIONS.keys())

    # Clean images
    loader_clean = build_loader(image_paths, batch_size=batch_size, num_workers=0)
    clean_unc = _run_mc_with_perturbation(vlm, loader_clean, perturbation_configs, passes, seed)

    results: Dict[str, Any] = {"clean_mean": float(clean_unc.mean())}

    for deg_name in degs:
        if deg_name not in DEGRADATIONS:
            raise ValueError(f"Unknown degradation '{deg_name}'; choose from {list(DEGRADATIONS)}")

        ds = DegradedImageDataset(image_paths, DEGRADATIONS[deg_name])
        loader_deg = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=pil_collate,
        )
        deg_unc = _run_mc_with_perturbation(vlm, loader_deg, perturbation_configs, passes, seed)
        results[deg_name] = paired_comparison(clean_unc, deg_unc)

    return results
