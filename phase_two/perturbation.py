"""General perturbation framework for uncertainty estimation.

Extends beyond dropout to support multiple stochastic perturbation types,
enabling a principled search over the space of possible perturbation strategies.

Perturbation types:
  dropout     — Standard Bernoulli dropout: zero neurons with probability p.
  gaussian    — Additive Gaussian noise: out + N(0, (mag * out.std())^2).
                Scaled relative to layer output so magnitude is comparable
                across layers with different activation scales.
  scale       — Multiplicative noise: out * (1 + N(0, mag^2)).
                Tests whether the model is sensitive to relative magnitudes.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_one.common import LinearDropoutWrapper

PERTURBATION_TYPES = {"dropout", "gaussian", "scale"}


class PerturbationWrapper(nn.Module):
    """Wraps a linear layer with a configurable stochastic perturbation."""

    def __init__(self, linear: nn.Linear, ptype: str, magnitude: float) -> None:
        if ptype not in PERTURBATION_TYPES:
            raise ValueError(f"Unknown perturbation type '{ptype}'; choose from {sorted(PERTURBATION_TYPES)}")
        super().__init__()
        self.linear = linear
        self.ptype = ptype
        self.magnitude = magnitude

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.linear(*args, **kwargs)
        if not self.training or self.magnitude <= 0:
            return out

        if self.ptype == "dropout":
            return F.dropout(out, p=self.magnitude, training=True)
        elif self.ptype == "gaussian":
            noise_std = self.magnitude * out.detach().std().clamp(min=1e-12)
            return out + torch.randn_like(out) * noise_std
        elif self.ptype == "scale":
            return out * (1.0 + torch.randn_like(out) * self.magnitude)

        raise RuntimeError(f"Unhandled perturbation type: {self.ptype}")

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


# ── Module manipulation helpers ──────────────────────────────────────


def _replace_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new)


def _unwrap_linear(module: nn.Module) -> nn.Linear:
    """Get the underlying nn.Linear from any wrapper."""
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, (PerturbationWrapper, LinearDropoutWrapper)):
        return _unwrap_linear(module.linear)
    raise TypeError(f"Cannot unwrap {type(module).__name__} to nn.Linear")


def named_linears(root: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """List all linear modules (unwrapping wrappers) with their paths."""
    out: List[Tuple[str, nn.Linear]] = []
    for name, module in root.named_modules():
        if not name:
            continue
        if isinstance(module, (nn.Linear, PerturbationWrapper, LinearDropoutWrapper)):
            out.append((name, _unwrap_linear(module)))
    return out


def get_mlp_output_projections(root: nn.Module) -> List[str]:
    """Return module paths for every MLP output projection in a ViT vision encoder.

    Works across naming conventions:
      CLIP (open_clip):  transformer.resblocks.N.mlp.c_proj  (3072 -> 768)
      timm / PE-Core:    trunk.blocks.N.mlp.fc2              (3072 -> 768)

    Strategy: group Linear children by parent, find pairs where the first
    expands (in < out) and the second contracts (in > out). The contracting
    one is the output projection. Only considers parents inside transformer
    block lists (paths containing 'resblocks' or 'blocks').
    """
    from collections import defaultdict

    parent_linears: dict[str, list[tuple[str, nn.Linear]]] = defaultdict(list)
    for name, module in root.named_modules():
        if isinstance(module, (nn.Linear, PerturbationWrapper, LinearDropoutWrapper)):
            linear = _unwrap_linear(module)
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            parent_linears[parent].append((name, linear))

    def _natural_sort_key(item: tuple[str, list]) -> list:
        """Sort 'blocks.10' after 'blocks.9', not before 'blocks.2'."""
        return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", item[0])]

    out_projs: List[str] = []
    for parent, linears in sorted(parent_linears.items(), key=_natural_sort_key):
        # Only transformer block MLPs, not attn_pool, head, etc.
        if "resblocks" not in parent and "blocks" not in parent:
            continue
        # Skip attention parents (contain 'attn' but not 'mlp')
        if "attn" in parent and "mlp" not in parent:
            continue
        if len(linears) != 2:
            continue
        (_, m1), (n2, m2) = linears
        # First linear expands, second contracts -> second is output projection
        if m1.out_features > m1.in_features and m2.in_features > m2.out_features:
            out_projs.append(n2)

    return out_projs


def disable_all_perturbation(root: nn.Module) -> None:
    """Set all perturbation/dropout wrappers to eval mode (pass-through)."""
    for module in root.modules():
        if isinstance(module, (PerturbationWrapper, LinearDropoutWrapper)):
            module.eval()
        elif isinstance(module, nn.Dropout):
            module.eval()


@contextmanager
def perturb_modules(
    root: nn.Module,
    configs: List[Tuple[str, str, float]],
) -> Iterator[List[PerturbationWrapper]]:
    """Context manager: temporarily apply perturbation to specific modules.

    Args:
        root: Model root module (e.g., vlm.vision_root)
        configs: List of (module_path, perturbation_type, magnitude)

    Yields:
        List of PerturbationWrapper instances (in training mode)

    On exit, restores the original modules.
    """
    originals: List[Tuple[str, nn.Module]] = []
    wrappers: List[PerturbationWrapper] = []

    try:
        for path, ptype, mag in configs:
            module = root.get_submodule(path)
            originals.append((path, module))
            linear = _unwrap_linear(module)
            wrapper = PerturbationWrapper(linear, ptype, mag)
            wrapper.train(True)
            _replace_module(root, path, wrapper)
            wrappers.append(wrapper)
        yield wrappers
    finally:
        for path, orig in originals:
            _replace_module(root, path, orig)
