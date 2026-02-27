from __future__ import annotations

import os
import sys
import types
import unittest
import warnings
from unittest import mock

import torch
import torch.nn as nn

import phase_one.common as common


class _DummyOpenClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Identity()


class _DummyHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Identity()

    def get_image_features(self, pixel_values):  # noqa: ARG002
        return torch.zeros((1, 4), dtype=torch.float32)

    def get_text_features(self, **encoded):  # noqa: ARG002
        return torch.zeros((1, 4), dtype=torch.float32)


class _DummyProcessor:
    def __call__(self, *args, **kwargs):  # noqa: ARG002
        return {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        }


def _fake_transformers_module(call_log: dict[str, list]) -> types.ModuleType:
    module = types.ModuleType("transformers")

    class AutoModel:  # noqa: D401
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            call_log["auto_model"].append((model_id, kwargs))
            return _DummyHFModel()

    class AutoProcessor:  # noqa: D401
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            call_log["auto_processor"].append((model_id, kwargs))
            return _DummyProcessor()

    class AutoImageProcessor:  # noqa: D401
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            call_log["auto_image_processor"].append((model_id, kwargs))
            return _DummyProcessor()

    class AutoTokenizer:  # noqa: D401
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            call_log["auto_tokenizer"].append((model_id, kwargs))
            return _DummyProcessor()

    module.AutoModel = AutoModel
    module.AutoProcessor = AutoProcessor
    module.AutoImageProcessor = AutoImageProcessor
    module.AutoTokenizer = AutoTokenizer
    return module


def _fake_open_clip_module(call_log: dict[str, list], *, should_fail: bool) -> types.ModuleType:
    module = types.ModuleType("open_clip")

    def create_model_and_transforms(model_name, pretrained, device):
        call_log["open_clip_create"].append((model_name, pretrained, device))
        if should_fail:
            raise RuntimeError("forced-open-clip-failure")
        return _DummyOpenClipModel(), None, _DummyProcessor()

    def get_tokenizer(model_name):
        call_log["open_clip_tokenizer"].append(model_name)
        return lambda texts: torch.zeros((len(texts), 4), dtype=torch.long)

    module.create_model_and_transforms = create_model_and_transforms
    module.get_tokenizer = get_tokenizer
    return module


class Siglip2BackendSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.call_log: dict[str, list] = {
            "open_clip_create": [],
            "open_clip_tokenizer": [],
            "auto_model": [],
            "auto_processor": [],
            "auto_image_processor": [],
            "auto_tokenizer": [],
        }

    def test_siglip2_default_uses_open_clip(self) -> None:
        fake_open_clip = _fake_open_clip_module(self.call_log, should_fail=False)
        fake_transformers = _fake_transformers_module(self.call_log)
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip, "transformers": fake_transformers}, clear=False):
            with mock.patch.dict(os.environ, {}, clear=False):
                vlm = common.load_model("siglip2_b16", device="cpu")

        self.assertEqual(vlm.spec.backend, "open_clip")
        self.assertEqual(len(self.call_log["open_clip_create"]), 1)
        self.assertEqual(len(self.call_log["auto_model"]), 0)

    def test_siglip2_hf_override_skips_open_clip(self) -> None:
        fake_open_clip = _fake_open_clip_module(self.call_log, should_fail=False)
        fake_transformers = _fake_transformers_module(self.call_log)
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip, "transformers": fake_transformers}, clear=False):
            with mock.patch.dict(os.environ, {"MCDO_SIGLIP2_BACKEND": "hf"}, clear=False):
                vlm = common.load_model("siglip2_b16", device="cpu")

        self.assertEqual(vlm.spec.backend, "siglip2")
        self.assertEqual(len(self.call_log["open_clip_create"]), 0)
        self.assertEqual(len(self.call_log["auto_model"]), 1)

    def test_siglip2_open_clip_failure_falls_back_to_hf(self) -> None:
        fake_open_clip = _fake_open_clip_module(self.call_log, should_fail=True)
        fake_transformers = _fake_transformers_module(self.call_log)
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip, "transformers": fake_transformers}, clear=False):
            with mock.patch.dict(os.environ, {}, clear=False):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    vlm = common.load_model("siglip2_b16", device="cpu")

        self.assertEqual(vlm.spec.backend, "siglip2")
        self.assertEqual(len(self.call_log["open_clip_create"]), 1)
        self.assertEqual(len(self.call_log["auto_model"]), 1)
        messages = [str(item.message) for item in caught]
        self.assertTrue(any("Falling back to HuggingFace backend" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()
