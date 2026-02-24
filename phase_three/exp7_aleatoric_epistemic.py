#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from phase_one.common import (
    auroc_from_scores,
    build_loader,
    discover_class_names,
    list_images,
    load_manifest,
    load_model,
    parse_templates,
    reliability_from_trials,
    run_mc_trial,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
    should_save_checkpoint,
    spearman_safe,
)
from phase_two.dropout_types import configure_dropout, set_dropout_train_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Exp 7: aleatoric vs epistemic diagnostics")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--class-map", type=str, default="")
    parser.add_argument("--templates", type=str, default="a photo of a {}|a {}|an image of a {}")
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--jpeg-qualities", type=str, default="100,80,60,40,20")
    parser.add_argument("--blur-sigmas", type=str, default="0,1,2,4")
    parser.add_argument("--occlusion-ratios", type=str, default="0,0.1,0.25")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    return parser.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), optimize=False)
    buf.seek(0)
    with Image.open(buf) as im:
        out = im.convert("RGB")
    buf.close()
    return out


def _apply_occlusion(image: Image.Image, ratio: float, rng: np.random.Generator) -> Image.Image:
    if ratio <= 0.0:
        return image.copy()
    arr = np.asarray(image, dtype=np.uint8).copy()
    h, w = arr.shape[0], arr.shape[1]
    occ_area = max(1, int(ratio * h * w))
    side = max(1, int(np.sqrt(occ_area)))
    side_h = min(side, h)
    side_w = min(side, w)
    top = int(rng.integers(0, max(1, h - side_h + 1)))
    left = int(rng.integers(0, max(1, w - side_w + 1)))
    arr[top : top + side_h, left : left + side_w] = 0
    return Image.fromarray(arr)


def _degradation_variants(
    image: Image.Image,
    jpeg_qualities: List[int],
    blur_sigmas: List[float],
    occlusion_ratios: List[float],
    seed: int,
) -> Tuple[List[str], List[Image.Image]]:
    names: List[str] = ["clean"]
    variants: List[Image.Image] = [image.copy()]

    for q in jpeg_qualities:
        names.append(f"jpeg_{q}")
        variants.append(_jpeg_roundtrip(image, quality=q))

    for sigma in blur_sigmas:
        names.append(f"blur_{sigma}")
        if sigma <= 0.0:
            variants.append(image.copy())
        else:
            variants.append(image.filter(ImageFilter.GaussianBlur(radius=float(sigma))))

    rng = np.random.default_rng(seed)
    for ratio in occlusion_ratios:
        names.append(f"occlusion_{ratio}")
        variants.append(_apply_occlusion(image=image, ratio=float(ratio), rng=rng))

    return names, variants


def _classification_metrics(vlm, loader, class_names: List[str], templates: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_features_by_template: List[torch.Tensor] = []
    for template in templates:
        prompts = [template.format(name) for name in class_names]
        text_features_by_template.append(vlm.encode_texts(prompts, normalize=True))

    margins: List[np.ndarray] = []
    entropies: List[np.ndarray] = []
    prompt_vars: List[np.ndarray] = []

    with torch.no_grad():
        for images, _paths, _ in loader:
            image_features = vlm.encode_images(images, normalize=True)
            logits_main = vlm.similarity_logits(image_features, text_features_by_template[0])

            top2 = torch.topk(logits_main, k=2, dim=-1).values
            margins.append((top2[:, 0] - top2[:, 1]).cpu().numpy())

            probs_main = F.softmax(logits_main, dim=-1)
            entropies.append((-(probs_main * torch.log(probs_main.clamp_min(1e-12))).sum(dim=-1)).cpu().numpy())

            max_probs = []
            for text_features in text_features_by_template:
                logits = vlm.similarity_logits(image_features, text_features)
                probs = F.softmax(logits, dim=-1)
                max_probs.append(probs.max(dim=-1).values.cpu().numpy())
            max_probs_arr = np.stack(max_probs, axis=0)
            if max_probs_arr.shape[0] > 1:
                prompt_var = np.var(max_probs_arr, axis=0, ddof=1)
            else:
                prompt_var = np.zeros(max_probs_arr.shape[1], dtype=np.float64)
            prompt_vars.append(prompt_var)

    return np.concatenate(margins), np.concatenate(entropies), np.concatenate(prompt_vars)


def _epistemic_uncertainty(
    vlm,
    loader,
    dropout: float,
    passes: int,
    trials: int,
    seed: int,
    save_every: int = 0,
    partial_npz_path: str = "",
    progress_json_path: str = "",
    progress_meta: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    out: List[np.ndarray] = []
    for trial_idx in range(trials):
        set_all_seeds(seed + trial_idx)
        configure_dropout(vlm, dropout_type="E", p=dropout)
        trial = run_mc_trial(vlm=vlm, loader=loader, passes=passes, collect_pass_features=False)
        out.append(trial["trace_pre"].numpy())

        completed = trial_idx + 1
        if partial_npz_path and should_save_checkpoint(completed=completed, total=trials, every=save_every):
            np.savez_compressed(
                partial_npz_path,
                epistemic_uncertainty_trials=np.stack(out, axis=0),
                completed_trials=np.asarray([completed], dtype=np.int64),
                total_trials=np.asarray([trials], dtype=np.int64),
            )
            if progress_json_path:
                payload: Dict[str, object] = {
                    "completed_trials": completed,
                    "total_trials": trials,
                }
                if progress_meta:
                    payload.update(progress_meta)
                save_json(payload, progress_json_path)
    return np.stack(out, axis=0)


def _aleatoric_uncertainty(
    vlm,
    sampled_paths: List[str],
    jpeg_qualities: List[int],
    blur_sigmas: List[float],
    occlusion_ratios: List[float],
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    scores = np.zeros(len(sampled_paths), dtype=np.float64)
    variant_names: List[str] = []

    with torch.no_grad():
        for idx, path in enumerate(sampled_paths):
            with Image.open(path) as img:
                image = img.convert("RGB")

            names, variants = _degradation_variants(
                image=image,
                jpeg_qualities=jpeg_qualities,
                blur_sigmas=blur_sigmas,
                occlusion_ratios=occlusion_ratios,
                seed=seed + idx,
            )
            if not variant_names:
                variant_names = names

            feats = vlm.encode_images(variants, normalize=False).detach().cpu().numpy().astype(np.float64, copy=False)
            if feats.shape[0] < 2:
                scores[idx] = 0.0
                continue

            per_dim_var = np.var(feats, axis=0, ddof=1)
            scores[idx] = float(np.mean(per_dim_var))

    return scores, variant_names


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std <= 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std


def _method_metrics(unc: np.ndarray, margins: np.ndarray, entropies: np.ndarray, quantile: float) -> Dict[str, float]:
    low_margin_thr = np.quantile(margins, quantile)
    high_entropy_thr = np.quantile(entropies, 1.0 - quantile)
    low_margin = (margins <= low_margin_thr).astype(np.int64)
    high_entropy = (entropies >= high_entropy_thr).astype(np.int64)

    return {
        "rho_uncertainty_vs_negative_margin": spearman_safe(unc, -margins),
        "rho_uncertainty_vs_entropy": spearman_safe(unc, entropies),
        "auroc_low_margin": auroc_from_scores(unc, low_margin),
        "auroc_high_entropy": auroc_from_scores(unc, high_entropy),
        "low_margin_threshold": float(low_margin_thr),
        "high_entropy_threshold": float(high_entropy_thr),
    }


def _reliability_or_none(values: np.ndarray) -> Optional[Dict[str, float]]:
    if values.shape[0] < 2:
        return None
    return reliability_from_trials(values)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        sampled_paths = load_manifest(args.manifest)
    else:
        sampled_paths = sample_paths(list_images(args.data_dir), args.num_images, args.seed)
        save_manifest(sampled_paths, str(out_dir / "phase3_exp7_manifest.json"))

    jpeg_qualities = _parse_int_list(args.jpeg_qualities)
    blur_sigmas = _parse_float_list(args.blur_sigmas)
    occlusion_ratios = _parse_float_list(args.occlusion_ratios)

    class_names = discover_class_names(args.data_dir, mapping_path=args.class_map or None)
    templates = parse_templates(args.templates)
    loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    overall: Dict[str, Dict[str, object]] = {}

    for model_key in model_keys:
        print(f"[Exp7] model={model_key}")
        model_out = out_dir / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        try:
            vlm = load_model(model_key, device=args.device)
        except Exception as exc:  # noqa: BLE001
            payload: Dict[str, object] = {"error": str(exc)}
            overall[model_key] = payload
            save_json(payload, str(model_out / "exp7_error.json"))
            continue

        margins, entropies, prompt_sens = _classification_metrics(
            vlm=vlm,
            loader=loader,
            class_names=class_names,
            templates=templates,
        )

        epi_trials = _epistemic_uncertainty(
            vlm=vlm,
            loader=loader,
            dropout=args.dropout,
            passes=args.passes,
            trials=args.trials,
            seed=args.seed,
            save_every=args.save_every,
            partial_npz_path=str(model_out / "exp7_epistemic_partial.npz"),
            progress_json_path=str(model_out / "exp7_epistemic_progress.json"),
            progress_meta={"experiment": "exp7_aleatoric_epistemic", "model": model_key},
        )
        epi = epi_trials.mean(axis=0)

        set_dropout_train_mode(vlm.vision_root, enabled=False)
        ale, variant_names = _aleatoric_uncertainty(
            vlm=vlm,
            sampled_paths=sampled_paths,
            jpeg_qualities=jpeg_qualities,
            blur_sigmas=blur_sigmas,
            occlusion_ratios=occlusion_ratios,
            seed=args.seed,
        )

        ale_z = _zscore(ale)
        epi_z = _zscore(epi)
        combined_sum = ale_z + epi_z
        combined_product = ale_z * epi_z

        summary = {
            "experiment": "exp7_aleatoric_epistemic",
            "model": model_key,
            "num_images": len(sampled_paths),
            "dropout": args.dropout,
            "passes": args.passes,
            "trials": args.trials,
            "jpeg_qualities": jpeg_qualities,
            "blur_sigmas": blur_sigmas,
            "occlusion_ratios": occlusion_ratios,
            "num_degradation_variants": len(variant_names),
            "variant_names": variant_names,
            "corr": {
                "rho_aleatoric_vs_epistemic": spearman_safe(ale, epi),
                "rho_aleatoric_vs_negative_margin": spearman_safe(ale, -margins),
                "rho_epistemic_vs_negative_margin": spearman_safe(epi, -margins),
                "rho_aleatoric_vs_entropy": spearman_safe(ale, entropies),
                "rho_epistemic_vs_entropy": spearman_safe(epi, entropies),
                "rho_aleatoric_vs_prompt_sensitivity": spearman_safe(ale, prompt_sens),
                "rho_epistemic_vs_prompt_sensitivity": spearman_safe(epi, prompt_sens),
                "rho_combined_sum_vs_negative_margin": spearman_safe(combined_sum, -margins),
                "rho_combined_sum_vs_entropy": spearman_safe(combined_sum, entropies),
                "rho_combined_product_vs_negative_margin": spearman_safe(combined_product, -margins),
                "rho_combined_product_vs_entropy": spearman_safe(combined_product, entropies),
            },
            "methods": {
                "aleatoric": _method_metrics(ale, margins, entropies, quantile=args.quantile),
                "epistemic": _method_metrics(epi, margins, entropies, quantile=args.quantile),
                "combined_sum": _method_metrics(combined_sum, margins, entropies, quantile=args.quantile),
                "combined_product": _method_metrics(combined_product, margins, entropies, quantile=args.quantile),
            },
            "epistemic_reliability": _reliability_or_none(epi_trials),
        }

        np.savez_compressed(
            model_out / "exp7_outputs.npz",
            paths=np.asarray(sampled_paths),
            margin=margins,
            entropy=entropies,
            prompt_sensitivity=prompt_sens,
            aleatoric_uncertainty=ale,
            epistemic_uncertainty_trials=epi_trials,
            epistemic_uncertainty=epi,
            combined_sum=combined_sum,
            combined_product=combined_product,
        )

        save_json(summary, str(model_out / "exp7_summary.json"))
        overall[model_key] = summary

    save_json(
        {
            "experiment": "exp7_aleatoric_epistemic",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "results": overall,
        },
        str(out_dir / "exp7_overall_summary.json"),
    )

    print(f"[Exp7] Complete: {out_dir}")


if __name__ == "__main__":
    main()
