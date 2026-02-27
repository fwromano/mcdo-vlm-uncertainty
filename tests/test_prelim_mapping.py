from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from prelim_investigation import get_classification_signals


class _FakeVLM:
    def encode_texts(self, prompts, normalize=True):  # noqa: ARG002
        # Use one-hot text features so logits mirror image features.
        return torch.eye(len(prompts), dtype=torch.float32)

    def encode_images(self, images, normalize=True):  # noqa: ARG002
        # Two samples, two classes:
        # sample0 -> class1, sample1 -> class0
        return torch.tensor(
            [
                [0.1, 0.9],
                [0.9, 0.1],
            ],
            dtype=torch.float32,
        )

    def similarity_logits(self, image_features, text_features):
        return image_features @ text_features.T


class PrelimMappingTests(unittest.TestCase):
    def test_gt_labels_use_synset_folder_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            # Intentionally create unsorted folder names.
            (data_dir / "n200").mkdir()
            (data_dir / "n100").mkdir()

            loader = [
                (
                    [object(), object()],
                    ["img0.jpg", "img1.jpg"],
                    ["n200", "n100"],
                )
            ]
            class_names = ["class_n100", "class_n200"]  # sorted-folder order

            signals = get_classification_signals(
                vlm=_FakeVLM(),
                loader=loader,
                class_names=class_names,
                data_dir=str(data_dir),
                templates=["a photo of a {}"],
            )

            self.assertEqual(signals["gt"].tolist(), [1, 0])
            self.assertEqual(signals["pred"].tolist(), [1, 0])
            self.assertNotIn(-1, signals["gt"].tolist())


if __name__ == "__main__":
    unittest.main()
