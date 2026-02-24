import os
from glob import glob
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """Lightweight image dataset that preserves file paths."""

    def __init__(
        self,
        paths: Sequence[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.paths: List[str] = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        path = self.paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path

    @classmethod
    def from_dir(
        cls,
        root: str,
        patterns: Sequence[str] = ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"),
        transform: Optional[Callable] = None,
    ) -> "ImageFolderDataset":
        files: List[str] = []
        for pattern in patterns:
            files.extend(glob(os.path.join(root, pattern), recursive=True))
        files = sorted({p for p in files if os.path.isfile(p)})
        return cls(files, transform=transform)
