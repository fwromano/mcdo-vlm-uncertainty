from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader, Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a {}",
    "an image of a {}",
]
_PRECOMPUTED_PIXEL_CACHE: Dict[Tuple[int, int, str, bool], Tuple[torch.Tensor, List[str]]] = {}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    backend: str
    open_clip_model: Optional[str] = None
    open_clip_pretrained: Optional[str] = None
    hf_model_id: Optional[str] = None


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "clip_b32": ModelSpec(
        key="clip_b32",
        backend="open_clip",
        open_clip_model="ViT-B-32",
        open_clip_pretrained="openai",
    ),
    "clip_l14": ModelSpec(
        key="clip_l14",
        backend="open_clip",
        open_clip_model="ViT-L-14",
        open_clip_pretrained="openai",
    ),
    "siglip2_b16": ModelSpec(
        key="siglip2_b16",
        backend="siglip2",
        hf_model_id="google/siglip2-base-patch16-224",
    ),
    "siglip2_so400m": ModelSpec(
        key="siglip2_so400m",
        backend="siglip2",
        hf_model_id="google/siglip2-so400m-patch14-384",
    ),
    "siglip2_g16": ModelSpec(
        key="siglip2_g16",
        backend="siglip2",
        hf_model_id="google/siglip2-giant-patch16-384",
    ),
}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str]) -> None:
        self.image_paths = [str(Path(p)) for p in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str]:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        class_name = Path(path).parent.name
        return image, path, class_name


def pil_collate(batch: Sequence[Tuple[Image.Image, str, str]]) -> Tuple[List[Image.Image], List[str], List[str]]:
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    class_names = [item[2] for item in batch]
    return images, paths, class_names


def list_images(data_dir: str) -> List[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    paths.sort()
    if not paths:
        raise ValueError(f"No image files found under {data_dir}")
    return paths


def sample_paths(paths: Sequence[str], num_images: int, seed: int) -> List[str]:
    path_list = list(paths)
    if num_images <= 0 or num_images >= len(path_list):
        return path_list
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(path_list), size=num_images, replace=False))
    return [path_list[i] for i in idx]


def save_manifest(paths: Sequence[str], out_path: str) -> None:
    payload = {"num_images": len(paths), "paths": list(paths)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    paths = payload.get("paths", [])
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"Manifest {manifest_path} does not contain a non-empty `paths` list")
    return [str(Path(p)) for p in paths]


def should_save_checkpoint(completed: int, total: int, every: int) -> bool:
    if every <= 0:
        return False
    return completed % every == 0 or completed >= total


def build_loader(image_paths: Sequence[str], batch_size: int, num_workers: int) -> DataLoader:
    dataset = ImagePathDataset(image_paths)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pil_collate,
    )


class LinearDropoutWrapper(nn.Module):
    def __init__(self, linear: nn.Linear, p: float) -> None:
        super().__init__()
        self.linear = linear
        self.dropout = nn.Dropout(p)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.linear(*args, **kwargs)
        return self.dropout(out)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features


def _replace_module(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    parent_path, _, leaf = module_path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new_module)


def inject_uniform_linear_dropout(root: nn.Module, p: float) -> int:
    replacements = 0
    for name, module in list(root.named_modules()):
        if not name:
            continue
        if isinstance(module, nn.Linear):
            parent_path, _, _ = name.rpartition(".")
            parent = root.get_submodule(parent_path) if parent_path else root
            if isinstance(parent, LinearDropoutWrapper):
                continue
            _replace_module(root, name, LinearDropoutWrapper(module, p))
            replacements += 1
    return replacements


def set_dropout_mode(root: nn.Module, enabled: bool, p: Optional[float] = None) -> None:
    for module in root.modules():
        if isinstance(module, LinearDropoutWrapper):
            if p is not None:
                module.dropout.p = p
            module.dropout.train(enabled)
        elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            if p is not None:
                module.p = p
            module.train(enabled)


def _as_feature_tensor(feats: Any, source: str) -> torch.Tensor:
    if isinstance(feats, torch.Tensor):
        return feats
    if hasattr(feats, "pooler_output") and getattr(feats, "pooler_output") is not None:
        return feats.pooler_output
    if isinstance(feats, (tuple, list)) and feats and isinstance(feats[0], torch.Tensor):
        return feats[0]
    raise TypeError(f"{source} returned unsupported feature type: {type(feats)}")


class VisionLanguageModel:
    def __init__(
        self,
        spec: ModelSpec,
        model: nn.Module,
        image_processor: Any,
        text_processor: Any,
        device: str,
    ) -> None:
        self.spec = spec
        self.model = model
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.device = device
        self.model.eval()

        if spec.backend == "open_clip":
            self.vision_root = self.model.visual
        elif hasattr(self.model, "vision_model"):
            self.vision_root = self.model.vision_model
        else:
            self.vision_root = self.model

        self._dropout_injected = False

    @property
    def name(self) -> str:
        return self.spec.key

    def ensure_uniform_dropout(self, p: float) -> int:
        if not self._dropout_injected:
            replaced = inject_uniform_linear_dropout(self.vision_root, p)
            self._dropout_injected = True
        else:
            replaced = 0
        set_dropout_mode(self.vision_root, enabled=True, p=p)
        return replaced

    def disable_dropout(self) -> None:
        self.model.eval()
        set_dropout_mode(self.vision_root, enabled=False)

    def _pixel_values_from_pil(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if self.spec.backend == "open_clip":
            tensors = [self.image_processor(img) for img in images]
            pixel_values = torch.stack(tensors, dim=0)
            return pixel_values

        encoded = self.image_processor(images=list(images), return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        return pixel_values

    @torch.inference_mode()
    def encode_pixel_values(self, pixel_values: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, non_blocking=True)

        if self.spec.backend == "open_clip":
            feats = self.model.encode_image(pixel_values, normalize=normalize)
            return feats.float()

        feats = _as_feature_tensor(
            self.model.get_image_features(pixel_values=pixel_values),
            source=f"{self.spec.hf_model_id}.get_image_features",
        )
        feats = feats.float()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image], normalize: bool = False) -> torch.Tensor:
        pixel_values = self._pixel_values_from_pil(images)
        return self.encode_pixel_values(pixel_values, normalize=normalize)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], normalize: bool = True) -> torch.Tensor:
        if self.spec.backend == "open_clip":
            tokens = self.text_processor(list(texts)).to(self.device)
            feats = self.model.encode_text(tokens, normalize=normalize)
            return feats.float()

        if self.text_processor is None:
            raise RuntimeError(
                f"Text processor is unavailable for `{self.spec.key}`. "
                "Install `protobuf` and `sentencepiece` to enable text encoding."
            )
        try:
            encoded = self.text_processor(text=list(texts), padding=True, return_tensors="pt")
        except TypeError:
            encoded = self.text_processor(list(texts), padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        feats = _as_feature_tensor(
            self.model.get_text_features(**encoded),
            source=f"{self.spec.hf_model_id}.get_text_features",
        )
        feats = feats.float()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    def similarity_logits(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        logits = image_features @ text_features.T

        if hasattr(self.model, "logit_scale"):
            scale = self.model.logit_scale
            if isinstance(scale, torch.Tensor):
                scale = scale.to(logits.device)
                logits = logits * scale.exp()
            else:
                logits = logits * float(np.exp(scale))

        if hasattr(self.model, "logit_bias"):
            bias = self.model.logit_bias
            if bias is None:
                pass
            elif isinstance(bias, torch.Tensor):
                logits = logits + bias.to(logits.device)
            else:
                logits = logits + float(bias)

        return logits


def load_model(spec_key: str, device: str) -> VisionLanguageModel:
    if spec_key not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model key `{spec_key}`. Known: {known}")
    spec = MODEL_REGISTRY[spec_key]

    if spec.backend == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.open_clip_model,
            pretrained=spec.open_clip_pretrained,
            device=device,
        )
        tokenizer = open_clip.get_tokenizer(spec.open_clip_model)
        model.eval()
        return VisionLanguageModel(
            spec=spec,
            model=model,
            image_processor=preprocess,
            text_processor=tokenizer,
            device=device,
        )

    from transformers import AutoModel

    text_processor: Any = None
    processor_error: Optional[Exception] = None
    hf_kwargs: Dict[str, Any] = {}
    if os.environ.get("MCDO_HF_LOCAL_ONLY", "").strip().lower() in {"1", "true", "yes"}:
        hf_kwargs["local_files_only"] = True

    # AutoProcessor can fail when tokenizer dependencies are missing.
    # Fall back to loading image/tokenizer separately so image-only experiments
    # (Exp 0/0b/4) still run.
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            spec.hf_model_id,
            use_fast=False,
            **hf_kwargs,
        )
        image_processor = processor
        text_processor = processor
    except Exception as exc:  # noqa: BLE001
        processor_error = exc
        from transformers import AutoImageProcessor, AutoTokenizer

        image_processor = AutoImageProcessor.from_pretrained(
            spec.hf_model_id,
            use_fast=False,
            **hf_kwargs,
        )
        try:
            text_processor = AutoTokenizer.from_pretrained(
                spec.hf_model_id,
                use_fast=False,
                **hf_kwargs,
            )
        except Exception as tok_exc:  # noqa: BLE001
            warnings.warn(
                "Tokenizer load failed for "
                f"`{spec.hf_model_id}` ({tok_exc}). Text-dependent experiments "
                "require `protobuf` and `sentencepiece` "
                "(install: `pip install protobuf sentencepiece`). "
                f"Original AutoProcessor error: {processor_error}",
                RuntimeWarning,
            )
            text_processor = None

    model = AutoModel.from_pretrained(spec.hf_model_id, **hf_kwargs)
    model.to(device)
    model.eval()
    return VisionLanguageModel(
        spec=spec,
        model=model,
        image_processor=image_processor,
        text_processor=text_processor,
        device=device,
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_precomputed_pixel_cache() -> None:
    _PRECOMPUTED_PIXEL_CACHE.clear()


def precompute_pixel_values(
    vlm: VisionLanguageModel,
    loader: DataLoader,
    to_device: bool = True,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, List[str]]:
    cache_key = (id(vlm), id(loader), vlm.device, bool(to_device))
    if use_cache and cache_key in _PRECOMPUTED_PIXEL_CACHE:
        pixel_values, paths = _PRECOMPUTED_PIXEL_CACHE[cache_key]
        return pixel_values, list(paths)

    all_pixels: List[torch.Tensor] = []
    all_paths: List[str] = []
    for images, paths, _ in loader:
        all_pixels.append(vlm._pixel_values_from_pil(images))
        all_paths.extend(paths)

    if not all_pixels:
        raise RuntimeError("No images found while precomputing pixel tensors.")

    pixel_values = torch.cat(all_pixels, dim=0)
    if to_device:
        pixel_values = pixel_values.to(vlm.device, non_blocking=True)

    if use_cache:
        _PRECOMPUTED_PIXEL_CACHE[cache_key] = (pixel_values, list(all_paths))
    return pixel_values, all_paths


def run_mc_trial(
    vlm: VisionLanguageModel,
    loader: DataLoader,
    passes: int,
    collect_pass_features: bool = False,
    compute_angular: bool = False,
    progress: bool = False,
    progress_desc: str = "",
    progress_position: int = 0,
    progress_leave: bool = False,
    use_precomputed_pixels: bool = True,
    cache_precomputed_pixels: bool = True,
    precompute_to_device: bool = True,
) -> Dict[str, Any]:
    num_samples = len(loader.dataset)
    batch_size = int(loader.batch_size) if isinstance(loader.batch_size, int) and loader.batch_size > 0 else num_samples

    sum_pre = None
    sq_pre = None
    sum_post = None
    sq_post = None

    pass_pre = None
    pass_post = None
    path_order: List[str] = []
    pixel_values: Optional[torch.Tensor] = None

    if use_precomputed_pixels:
        try:
            pixel_values, path_order = precompute_pixel_values(
                vlm=vlm,
                loader=loader,
                to_device=precompute_to_device,
                use_cache=cache_precomputed_pixels,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            oom_like = "out of memory" in message or " oom" in message or message.endswith("oom")
            if precompute_to_device and oom_like:
                if torch.cuda.is_available() and vlm.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                pixel_values, path_order = precompute_pixel_values(
                    vlm=vlm,
                    loader=loader,
                    to_device=False,
                    use_cache=cache_precomputed_pixels,
                )
            else:
                raise

    pass_iter: Any = range(passes)
    if progress:
        try:
            from tqdm.auto import tqdm

            pass_iter = tqdm(
                pass_iter,
                total=passes,
                desc=(progress_desc or "MC passes"),
                position=progress_position,
                leave=progress_leave,
                dynamic_ncols=True,
            )
        except Exception:  # noqa: BLE001
            pass_iter = range(passes)

    for pass_idx in pass_iter:
        if pixel_values is not None:
            for offset in range(0, num_samples, batch_size):
                batch = pixel_values[offset : offset + batch_size]
                pre = vlm.encode_pixel_values(batch, normalize=False).detach().cpu().to(torch.float64)
                post = F.normalize(pre, dim=-1)

                bsz = pre.shape[0]
                if sum_pre is None:
                    dim = pre.shape[1]
                    sum_pre = torch.zeros((num_samples, dim), dtype=torch.float64)
                    sq_pre = torch.zeros_like(sum_pre)
                    sum_post = torch.zeros_like(sum_pre)
                    sq_post = torch.zeros_like(sum_pre)
                    if collect_pass_features:
                        pass_pre = torch.zeros((passes, num_samples, dim), dtype=torch.float32)
                        pass_post = torch.zeros((passes, num_samples, dim), dtype=torch.float32)

                sum_pre[offset : offset + bsz] += pre
                sq_pre[offset : offset + bsz] += pre * pre
                sum_post[offset : offset + bsz] += post
                sq_post[offset : offset + bsz] += post * post

                if collect_pass_features:
                    pass_pre[pass_idx, offset : offset + bsz] = pre.float()
                    pass_post[pass_idx, offset : offset + bsz] = post.float()
            continue

        offset = 0
        current_paths: List[str] = []
        for images, paths, _ in loader:
            pre = vlm.encode_images(images, normalize=False).detach().cpu().to(torch.float64)
            post = F.normalize(pre, dim=-1)

            bsz = pre.shape[0]
            current_paths.extend(paths)

            if sum_pre is None:
                dim = pre.shape[1]
                sum_pre = torch.zeros((num_samples, dim), dtype=torch.float64)
                sq_pre = torch.zeros_like(sum_pre)
                sum_post = torch.zeros_like(sum_pre)
                sq_post = torch.zeros_like(sum_pre)
                if collect_pass_features:
                    pass_pre = torch.zeros((passes, num_samples, dim), dtype=torch.float32)
                    pass_post = torch.zeros((passes, num_samples, dim), dtype=torch.float32)

            sum_pre[offset : offset + bsz] += pre
            sq_pre[offset : offset + bsz] += pre * pre
            sum_post[offset : offset + bsz] += post
            sq_post[offset : offset + bsz] += post * post

            if collect_pass_features:
                pass_pre[pass_idx, offset : offset + bsz] = pre.float()
                pass_post[pass_idx, offset : offset + bsz] = post.float()

            offset += bsz

        if pass_idx == 0:
            path_order = current_paths
        elif current_paths != path_order:
            raise RuntimeError("DataLoader order changed between MC passes. Use shuffle=False.")

    if progress and hasattr(pass_iter, "close"):
        pass_iter.close()

    if sum_pre is None or sq_pre is None or sum_post is None or sq_post is None:
        raise RuntimeError("No features were collected. Check dataloader/input paths.")

    mean_pre = sum_pre / float(passes)
    mean_post = sum_post / float(passes)
    var_pre = sq_pre / float(passes) - mean_pre * mean_pre
    var_post = sq_post / float(passes) - mean_post * mean_post

    dim = var_pre.shape[1]
    trace_pre = (var_pre.sum(dim=1) / float(dim)).float()
    trace_post = (var_post.sum(dim=1) / float(dim)).float()

    out: Dict[str, Any] = {
        "paths": path_order,
        "trace_pre": trace_pre,
        "trace_post": trace_post,
    }

    if collect_pass_features:
        out["pass_pre"] = pass_pre
        out["pass_post"] = pass_post

    if compute_angular:
        if pass_post is None:
            raise RuntimeError("Angular variance requested but pass features were not collected.")
        mean_dir = F.normalize(pass_post.mean(dim=0), dim=-1)
        dots = (pass_post * mean_dir.unsqueeze(0)).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angles = torch.acos(dots)
        angular_var = angles.var(dim=0, unbiased=True)
        out["angular_var"] = angular_var.float()

    return out


def reliability_from_trials(values: np.ndarray) -> Dict[str, float]:
    if values.ndim != 2:
        raise ValueError(f"Expected values with shape (K, N), got {values.shape}")

    num_trials, num_images = values.shape
    if num_trials < 2:
        raise ValueError("Need at least 2 trials for reliability metrics")

    by_image = values.T  # N x K
    image_means = by_image.mean(axis=1)
    within_var = by_image.var(axis=1, ddof=1)

    signal = float(np.var(image_means, ddof=1))
    noise = float(np.mean(within_var))
    snr = float(signal / noise) if noise > 0 else float("inf")

    grand = float(by_image.mean())
    ss_between = float(num_trials * np.sum((image_means - grand) ** 2))
    ss_within = float(np.sum((by_image - image_means[:, None]) ** 2))

    ms_between = ss_between / max(num_images - 1, 1)
    ms_within = ss_within / max(num_images * (num_trials - 1), 1)
    denom = ms_between + (num_trials - 1) * ms_within
    icc = float((ms_between - ms_within) / denom) if denom > 0 else 0.0

    pairwise_rho: List[float] = []
    for i in range(num_trials):
        for j in range(i + 1, num_trials):
            rho, _ = spearmanr(values[i], values[j])
            if np.isnan(rho):
                rho = 0.0
            pairwise_rho.append(float(rho))

    pairwise_arr = np.asarray(pairwise_rho, dtype=np.float64)
    q25, q75 = np.percentile(pairwise_arr, [25.0, 75.0])

    return {
        "num_trials": float(num_trials),
        "num_images": float(num_images),
        "signal": signal,
        "noise": noise,
        "snr": snr,
        "icc": icc,
        "pairwise_spearman_median": float(np.median(pairwise_arr)),
        "pairwise_spearman_iqr": float(q75 - q25),
        "pairwise_spearman_q25": float(q25),
        "pairwise_spearman_q75": float(q75),
    }


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    rho, _ = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def auroc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def parse_templates(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return list(DEFAULT_PROMPT_TEMPLATES)
    templates = [t.strip() for t in raw.split("|") if t.strip()]
    if not templates:
        return list(DEFAULT_PROMPT_TEMPLATES)
    for template in templates:
        if "{}" not in template:
            raise ValueError(f"Prompt template `{template}` must include {{}} placeholder")
    return templates


def discover_class_names(data_dir: str, mapping_path: Optional[str] = None) -> List[str]:
    root = Path(data_dir)
    folder_names = sorted([p.name for p in root.iterdir() if p.is_dir()])

    mapping: Dict[str, str] = {}
    if mapping_path:
        path = Path(mapping_path)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t", maxsplit=1)
                    if len(parts) != 2:
                        continue
                    mapping[parts[0]] = parts[1]

    if folder_names:
        return [mapping.get(name, name.replace("_", " ")) for name in folder_names]

    paths = list_images(data_dir)
    fallback = sorted({Path(p).parent.name for p in paths})
    return [mapping.get(name, name.replace("_", " ")) for name in fallback]


def save_json(payload: Dict[str, Any], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
