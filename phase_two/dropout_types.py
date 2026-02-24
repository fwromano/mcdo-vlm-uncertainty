from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from phase_one.common import LinearDropoutWrapper

DROPOUT_TYPES = {"A", "B", "C", "D", "E"}


@dataclass
class DropoutConfigResult:
    dropout_type: str
    wrapped_modules: int
    selected_paths: List[str]
    notes: str = ""


def _replace_module(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    parent_path, _, leaf = module_path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new_module)


def _named_linears(root: nn.Module) -> List[Tuple[str, nn.Linear]]:
    out: List[Tuple[str, nn.Linear]] = []
    for name, module in root.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            out.append((name, module))
    return out


def _attn_paths(root: nn.Module) -> List[str]:
    tokens = ("attn", "attention", "self_attn", "out_proj", "q_proj", "k_proj", "v_proj")
    return [name for name, _ in _named_linears(root) if any(tok in name.lower() for tok in tokens)]


def _mlp_paths(root: nn.Module) -> List[str]:
    tokens = ("mlp", "ffn", "feed_forward", "c_fc", "fc1", "fc2", "intermediate", "output.dense")
    return [name for name, _ in _named_linears(root) if any(tok in name.lower() for tok in tokens)]


def _projection_only_paths(root: nn.Module) -> List[str]:
    all_paths = [name for name, _ in _named_linears(root)]
    if not all_paths:
        return []

    candidates = [
        p
        for p in all_paths
        if ("proj" in p.lower() or "projection" in p.lower() or "head" in p.lower()) and "attn" not in p.lower()
    ]
    if candidates:
        return [sorted(candidates)[-1]]
    return [all_paths[-1]]


def _inject_linear_wrappers(root: nn.Module, paths: Sequence[str], p: float) -> int:
    wrapped = 0
    for path in paths:
        module = root.get_submodule(path)
        if isinstance(module, LinearDropoutWrapper):
            module.dropout.p = p
            wrapped += 1
            continue
        if not isinstance(module, nn.Linear):
            continue
        _replace_module(root, path, LinearDropoutWrapper(module, p))
        wrapped += 1
    return wrapped


class StochasticDepthBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module, p: float) -> None:
        super().__init__()
        self.block = block
        self.p = p

    def forward(self, *args, **kwargs):
        if self.training and self.p > 0.0:
            if torch.rand(1).item() < self.p:
                if args:
                    return args[0]
                if "hidden_states" in kwargs:
                    return kwargs["hidden_states"]
        return self.block(*args, **kwargs)


def _candidate_block_paths(root: nn.Module) -> List[str]:
    paths: List[str] = []
    for name, module in root.named_modules():
        if not name:
            continue
        if isinstance(module, StochasticDepthBlockWrapper):
            paths.append(name)
            continue
        class_name = module.__class__.__name__.lower()
        if "residualattentionblock" in class_name or "encoderlayer" in class_name:
            paths.append(name)
            continue
        if re.search(r"(resblocks|layers)\.\d+$", name):
            paths.append(name)
    # Keep only leaf-most paths to avoid wrapping parents and children simultaneously.
    dedup: List[str] = []
    for p in sorted(set(paths)):
        if any(other != p and other.startswith(p + ".") for other in paths):
            continue
        dedup.append(p)
    return dedup


def _inject_stochastic_depth(root: nn.Module, p: float) -> int:
    block_paths = _candidate_block_paths(root)
    wrapped = 0
    for path in block_paths:
        module = root.get_submodule(path)
        if isinstance(module, StochasticDepthBlockWrapper):
            module.p = p
            wrapped += 1
            continue
        _replace_module(root, path, StochasticDepthBlockWrapper(module, p))
        wrapped += 1
    return wrapped


def set_dropout_train_mode(root: nn.Module, enabled: bool, p: Optional[float] = None) -> None:
    for module in root.modules():
        if isinstance(module, LinearDropoutWrapper):
            if p is not None:
                module.dropout.p = p
            module.dropout.train(enabled)
        elif isinstance(module, StochasticDepthBlockWrapper):
            if p is not None:
                module.p = p
            module.train(enabled)
        elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            if p is not None:
                module.p = p
            module.train(enabled)


def configure_dropout(vlm, dropout_type: str, p: float) -> DropoutConfigResult:
    if dropout_type not in DROPOUT_TYPES:
        known = ", ".join(sorted(DROPOUT_TYPES))
        raise ValueError(f"Unknown dropout type `{dropout_type}`; expected one of: {known}")

    root = vlm.vision_root

    if dropout_type == "C":
        wrapped = _inject_stochastic_depth(root, p)
        set_dropout_train_mode(root, enabled=True, p=p)
        paths = _candidate_block_paths(root)
        return DropoutConfigResult(
            dropout_type=dropout_type,
            wrapped_modules=wrapped,
            selected_paths=paths,
            notes="Stochastic-depth style residual block skipping",
        )

    if dropout_type == "A":
        paths = _attn_paths(root)
        notes = "Attention-associated linear modules"
    elif dropout_type == "B":
        paths = _mlp_paths(root)
        notes = "MLP/FFN-associated linear modules"
    elif dropout_type == "D":
        paths = _projection_only_paths(root)
        notes = "Single projection-like linear module"
    else:
        paths = [name for name, _ in _named_linears(root)]
        notes = "Uniform linear-layer dropout"

    wrapped = _inject_linear_wrappers(root, paths, p)
    set_dropout_train_mode(root, enabled=True, p=p)
    return DropoutConfigResult(dropout_type=dropout_type, wrapped_modules=wrapped, selected_paths=list(paths), notes=notes)
