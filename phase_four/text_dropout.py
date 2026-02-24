from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch.nn as nn

from phase_one.common import LinearDropoutWrapper


@dataclass
class TextDropoutConfigResult:
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


def _text_root(vlm) -> nn.Module:
    model = vlm.model
    if getattr(vlm.spec, "backend", "") == "open_clip":
        if hasattr(model, "transformer"):
            return model.transformer
        return model
    if hasattr(model, "text_model"):
        return model.text_model
    return model


def _set_dropout_train_mode(root: nn.Module, enabled: bool, p: Optional[float] = None) -> None:
    for module in root.modules():
        if isinstance(module, LinearDropoutWrapper):
            if p is not None:
                module.dropout.p = p
            module.dropout.train(enabled)
        elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            if p is not None:
                module.p = p
            module.train(enabled)


def configure_text_dropout(vlm, p: float, paths: Optional[Sequence[str]] = None) -> TextDropoutConfigResult:
    root = _text_root(vlm)
    selected = list(paths) if paths is not None else [name for name, _ in _named_linears(root)]

    wrapped = 0
    for path in selected:
        module = root.get_submodule(path)
        if isinstance(module, LinearDropoutWrapper):
            module.dropout.p = p
            wrapped += 1
            continue
        if not isinstance(module, nn.Linear):
            continue
        _replace_module(root, path, LinearDropoutWrapper(module, p))
        wrapped += 1

    _set_dropout_train_mode(root, enabled=True, p=p)
    return TextDropoutConfigResult(
        wrapped_modules=wrapped,
        selected_paths=selected,
        notes="Uniform text-tower linear dropout",
    )


def disable_text_dropout(vlm) -> None:
    _set_dropout_train_mode(_text_root(vlm), enabled=False)
