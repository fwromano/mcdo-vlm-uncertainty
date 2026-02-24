from __future__ import annotations

import importlib
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class DropoutAdapter(nn.Module):
    """Wrap a module with dropout applied to its output."""

    def __init__(self, module: nn.Module, p: float) -> None:
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return self.dropout(out)


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
    precision: str = "fp32",
):
    """Load CLIP via open_clip.create_model_and_transforms."""
    open_clip = importlib.import_module("open_clip")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, precision=precision
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def _iter_dropout_modules(model: nn.Module) -> Iterable[nn.Module]:
    return (m for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))


def set_dropout_rate(model: nn.Module, p: float) -> None:
    for m in _iter_dropout_modules(model):
        m.p = p


def enable_mc_dropout(model: nn.Module, p: Optional[float] = None) -> None:
    """Enable dropout layers during eval; optionally override rate."""
    model.eval()
    for m in _iter_dropout_modules(model):
        m.train()
        if p is not None:
            m.p = p


def _resolve_parent(model: nn.Module, module_path: str) -> Tuple[nn.Module, str]:
    parts = module_path.split(".")
    parent = model
    for name in parts[:-1]:
        parent = getattr(parent, name)
    return parent, parts[-1]


def attach_dropout_adapters(model: nn.Module, module_paths: Sequence[str], p: float) -> None:
    """Wrap specified modules with DropoutAdapter to inject dropout where absent."""
    for path in module_paths:
        parent, attr = _resolve_parent(model, path)
        target = getattr(parent, attr)
        setattr(parent, attr, DropoutAdapter(target, p))


def encode_images(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):
        return model.encode_image(images)
    if hasattr(model, "forward"):
        return model(images)
    raise AttributeError("Model lacks encode_image or forward")
