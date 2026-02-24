from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch.nn as nn

from phase_one.common import LinearDropoutWrapper
from phase_two.dropout_types import set_dropout_train_mode


@dataclass
class LayerGroupConfig:
    group_id: int
    paths: List[str]


def _replace_module(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    parent_path, _, leaf = module_path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new_module)


def _named_linears(root: nn.Module) -> List[str]:
    out: List[str] = []
    for name, module in root.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            out.append(name)
    return out


def _depth_key(path: str) -> Tuple[int, int, str]:
    digits = [int(x) for x in re.findall(r"\d+", path)]
    depth_hint = digits[0] if digits else -1
    return (depth_hint, len(path.split(".")), path)


def build_vision_layer_groups(vlm, num_groups: int) -> List[LayerGroupConfig]:
    if num_groups <= 0:
        raise ValueError("num_groups must be >= 1")

    paths = sorted(_named_linears(vlm.vision_root), key=_depth_key)
    if not paths:
        return []

    group_size = int(math.ceil(len(paths) / float(num_groups)))
    groups: List[LayerGroupConfig] = []
    for idx in range(num_groups):
        start = idx * group_size
        end = min(len(paths), (idx + 1) * group_size)
        if start >= end:
            break
        groups.append(LayerGroupConfig(group_id=idx, paths=paths[start:end]))
    return groups


def configure_groupwise_vision_dropout(
    vlm,
    groups: Sequence[LayerGroupConfig],
    p_by_group: Dict[int, float],
) -> int:
    root = vlm.vision_root
    wrapped = 0

    for group in groups:
        p = float(p_by_group[group.group_id])
        for path in group.paths:
            module = root.get_submodule(path)
            if isinstance(module, LinearDropoutWrapper):
                module.dropout.p = p
                wrapped += 1
                continue
            if not isinstance(module, nn.Linear):
                continue
            _replace_module(root, path, LinearDropoutWrapper(module, p))
            wrapped += 1

    set_dropout_train_mode(root, enabled=True)
    return wrapped


def group_schedule_to_dict(groups: Sequence[LayerGroupConfig], p_values: Sequence[float]) -> Dict[int, float]:
    if len(groups) != len(p_values):
        raise ValueError("p_values length must match number of groups")
    out: Dict[int, float] = {}
    for group, p in zip(groups, p_values):
        out[group.group_id] = float(p)
    return out


def flattened_group_paths(groups: Sequence[LayerGroupConfig]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for group in groups:
        for path in group.paths:
            out[path] = group.group_id
    return out
