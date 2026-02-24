from __future__ import annotations

import io
from typing import Dict, Iterable

import numpy as np
from PIL import Image
from scipy.stats import spearmanr


def _to_gray_array(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return arr


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_pixel_entropy(image: Image.Image) -> float:
    hist = np.asarray(image.convert("L").histogram(), dtype=np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob)).sum())


def compute_edge_density(image: Image.Image, threshold: float = 0.2) -> float:
    gray = _to_gray_array(image)
    gx, gy = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.size == 0:
        return 0.0
    edge_thresh = np.quantile(mag, 1.0 - threshold)
    return float((mag >= edge_thresh).mean())


def compute_colorfulness(image: Image.Image) -> float:
    rgb = _to_rgb_array(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def compute_jpeg_compressibility(image: Image.Image, quality: int = 75) -> float:
    rgb = image.convert("RGB")
    raw_size = np.asarray(rgb, dtype=np.uint8).nbytes
    with io.BytesIO() as buf:
        rgb.save(buf, format="JPEG", quality=quality)
        jpeg_size = buf.tell()
    if raw_size == 0:
        return 0.0
    return float(jpeg_size / raw_size)


def compute_complexity_metrics(image: Image.Image) -> Dict[str, float]:
    return {
        "entropy": compute_pixel_entropy(image),
        "edge_density": compute_edge_density(image),
        "colorfulness": compute_colorfulness(image),
        "jpeg_ratio": compute_jpeg_compressibility(image),
    }


def rank_spearman_matrix(values: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation matrix across rows of values (shape K×N)."""
    k, _ = values.shape
    corr = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            rho, _ = spearmanr(values[i], values[j])
            corr[i, j] = corr[j, i] = rho
    return corr


def correlate_metrics_to_variance(metrics: Dict[str, Iterable[float]], mc_var: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return Pearson and Spearman correlations of each metric vs MC variance."""
    results: Dict[str, Dict[str, float]] = {}
    mc_var = np.asarray(mc_var)
    for name, vals in metrics.items():
        arr = np.asarray(list(vals))
        if arr.shape[0] != mc_var.shape[0]:
            raise ValueError(f"Metric {name} length {arr.shape[0]} != variance length {mc_var.shape[0]}")
        pearson = float(np.corrcoef(arr, mc_var)[0, 1]) if arr.std() > 0 else 0.0
        rho, _ = spearmanr(arr, mc_var)
        results[name] = {"pearson": pearson, "spearman": float(rho)}
    return results
