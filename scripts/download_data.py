#!/usr/bin/env python3
"""Download and prepare all datasets and model checkpoints.

Usage:
    python scripts/download_data.py --all
    python scripts/download_data.py --coco --siglip2 --imagenet-map
    python scripts/download_data.py --reorganize-imagenet /path/to/ILSVRC2012_img_val.tar

Steps that require manual action:
  - Kaggle: set up ~/.kaggle/kaggle.json first (API token from kaggle.com/settings)
  - ImageNet val: download the tar from https://image-net.org (login required),
    then pass it to --reorganize-imagenet.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_ROOT / "raw"

# ---------------------------------------------------------------------------
# 1. COCO 2017 val via kagglehub
# ---------------------------------------------------------------------------
def download_coco():
    """Download COCO 2017 val images via kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    print("\n=== Downloading COCO 2017 val images ===")
    path = kagglehub.dataset_download("xthink/coco-2017-val-images")
    print(f"COCO val images at: {path}")

    # Create a symlink in data/raw for convenience
    link = RAW / "coco_val2017"
    if not link.exists():
        link.symlink_to(path)
        print(f"Symlinked: {link} -> {path}")
    return path


# ---------------------------------------------------------------------------
# 2. SigLIP2 checkpoints (HuggingFace transformers)
# ---------------------------------------------------------------------------
SIGLIP2_MODELS = [
    "google/siglip2-base-patch16-224",
    "google/siglip2-so400m-patch14-384",
]

def download_siglip2():
    """Pre-cache SigLIP2 model weights via transformers."""
    from transformers import AutoModel, AutoProcessor

    for model_id in SIGLIP2_MODELS:
        print(f"\n=== Caching {model_id} ===")
        # Download model + processor (tokenizer + image processor)
        AutoProcessor.from_pretrained(model_id)
        AutoModel.from_pretrained(model_id)
        print(f"  Cached: {model_id}")


# ---------------------------------------------------------------------------
# 3. ImageNet class-name mapping (wnid -> human label)
# ---------------------------------------------------------------------------
IMAGENET_CLASSES_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)
# LOC_synset_mapping.txt style: "n01440764 tench, Tinca tinca"
SYNSET_WORDS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/research/"
    "slim/datasets/imagenet_2012_validation_synset_labels.txt"
)
# Direct wnid -> label mapping (1000 classes, ILSVRC2012 order)
WNID_TO_LABEL_URL = (
    "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/"
    "raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/"
    "imagenet1000_clsidx_to_labels.txt"
)

def download_imagenet_classmap():
    """Download ImageNet-1K class mapping files."""
    import urllib.request

    out_dir = RAW / "imagenet_meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simple human-readable labels (list of 1000 strings, index = class id)
    simple_labels_path = out_dir / "imagenet_simple_labels.json"
    if not simple_labels_path.exists():
        print("\n=== Downloading ImageNet simple labels ===")
        urllib.request.urlretrieve(IMAGENET_CLASSES_URL, simple_labels_path)
        print(f"  Saved: {simple_labels_path}")
    else:
        print(f"  Already exists: {simple_labels_path}")

    # ILSVRC2012 synset -> folder mapping (for reorganizing val set)
    # This is the official mapping from devkit
    synset_map_path = out_dir / "ILSVRC2012_val_synsets.txt"
    if not synset_map_path.exists():
        print("\n=== Downloading ILSVRC2012 val synset list ===")
        url = (
            "https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
        )
        urllib.request.urlretrieve(url, synset_map_path)
        print(f"  Saved: {synset_map_path}")

    # Download the official PyTorch imagenet class index (class_id -> [wnid, label])
    class_index_path = out_dir / "imagenet_class_index.json"
    if not class_index_path.exists():
        print("\n=== Downloading PyTorch ImageNet class index ===")
        url = ("https://storage.googleapis.com/download.tensorflow.org/"
               "data/imagenet_class_index.json")
        urllib.request.urlretrieve(url, class_index_path)
        print(f"  Saved: {class_index_path}")

    # Build wnid -> human-readable label mapping
    wnid_map_path = out_dir / "wnid_to_label.json"
    if not wnid_map_path.exists():
        print("\n=== Building wnid -> label mapping ===")
        class_index = json.loads(class_index_path.read_text())
        # class_index: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
        mapping = {v[0]: v[1] for v in class_index.values()}
        with open(wnid_map_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"  Saved: {wnid_map_path} ({len(mapping)} classes)")

    return out_dir


# ---------------------------------------------------------------------------
# 4. Reorganize ImageNet val into ImageFolder structure
# ---------------------------------------------------------------------------
def reorganize_imagenet_val(tar_path: str):
    """
    Convert flat ILSVRC2012_img_val.tar into ImageFolder layout:
        imagenet_val/
          n01440764/
            ILSVRC2012_val_00000001.JPEG
            ...
          n01443537/
            ...

    Requires the synset label file (auto-downloaded).
    """
    tar_path = Path(tar_path)
    if not tar_path.exists():
        sys.exit(f"File not found: {tar_path}")

    # Make sure we have the synset mapping
    meta_dir = download_imagenet_classmap()
    synset_file = meta_dir / "ILSVRC2012_val_synsets.txt"
    if not synset_file.exists():
        sys.exit(f"Synset mapping not found: {synset_file}")

    # Read synset labels (one per val image, in order)
    synsets = synset_file.read_text().strip().split("\n")
    assert len(synsets) == 50000, f"Expected 50000 synsets, got {len(synsets)}"

    out_dir = RAW / "imagenet_val"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Extracting & reorganizing ImageNet val -> {out_dir} ===")
    with tarfile.open(tar_path, "r:*") as tf:
        members = sorted(tf.getmembers(), key=lambda m: m.name)
        # Filter to actual image files
        members = [m for m in members if m.name.endswith(".JPEG")]
        assert len(members) == 50000, f"Expected 50000 images, got {len(members)}"

        for i, member in enumerate(members):
            synset = synsets[i]
            class_dir = out_dir / synset
            class_dir.mkdir(exist_ok=True)

            # Extract file into class subdirectory
            member.name = Path(member.name).name  # strip any path prefix
            tf.extract(member, class_dir)

            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/50000 images extracted...")

    n_classes = len(list(out_dir.iterdir()))
    print(f"Done. {n_classes} class folders in {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="Download everything (except ImageNet val tar)")
    parser.add_argument("--coco", action="store_true",
                        help="Download COCO 2017 val via kagglehub")
    parser.add_argument("--siglip2", action="store_true",
                        help="Cache SigLIP2 checkpoints from HuggingFace")
    parser.add_argument("--imagenet-map", action="store_true",
                        help="Download ImageNet class-name mapping files")
    parser.add_argument("--reorganize-imagenet", type=str, default=None,
                        metavar="TAR_PATH",
                        help="Path to ILSVRC2012_img_val.tar to reorganize")
    args = parser.parse_args()

    if not any([args.all, args.coco, args.siglip2, args.imagenet_map,
                args.reorganize_imagenet]):
        parser.print_help()
        return

    RAW.mkdir(parents=True, exist_ok=True)

    if args.all or args.coco:
        download_coco()

    if args.all or args.siglip2:
        download_siglip2()

    if args.all or args.imagenet_map:
        download_imagenet_classmap()

    if args.reorganize_imagenet:
        reorganize_imagenet_val(args.reorganize_imagenet)

    print("\n=== Summary ===")
    print(f"Data root: {DATA_ROOT}")
    for d in sorted(RAW.iterdir()):
        if d.is_dir() or d.is_symlink():
            print(f"  {d.name}/")


if __name__ == "__main__":
    main()
