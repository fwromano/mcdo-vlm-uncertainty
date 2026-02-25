# Data & Model Setup

This project needs external datasets and model weights that can't live in git.
Some can be scripted, others require manual downloads with authenticated accounts.

---

## Quick Reference

| Asset | Size | How to get it |
|-------|------|---------------|
| ImageNet val (ILSVRC2012) | ~6.3 GB | Manual download (account required) |
| COCO 2017 val | ~1 GB | Script (Kaggle credentials required) |
| SigLIP2-base-patch16-224 | ~350 MB | Script (auto from HuggingFace) |
| SigLIP2-so400m-patch14-384 | ~1.7 GB | Script (auto from HuggingFace) |
| ImageNet class map | ~50 KB | Script (auto) |
| CLIP ViT-B/32, ViT-L/14 | auto | Downloaded on first use by `open_clip` |

---

## 1. Manual Downloads (can't be scripted)

### ImageNet ILSVRC2012 Validation Set

Required for Phase 1 experiments (Exp 0, 0b, 4, 5) and most of Phase 2.

1. Create an account at https://image-net.org
2. Go to https://image-net.org/download-images
3. Under **ILSVRC 2012**, download one of:
   - **Blurred validation images** (recommended, privacy-aware version)
   - Or the standard `ILSVRC2012_img_val.tar`
4. Run the reorganize script to create ImageFolder layout:
   ```bash
   python scripts/download_data.py --reorganize-imagenet /path/to/ILSVRC2012_img_val.tar
   ```
   This creates `data/raw/imagenet_val/` with 1000 class subdirectories (50K images total).

### Kaggle Credentials (needed for COCO)

1. Go to https://www.kaggle.com/settings
2. Under **API**, click **Create New Token** — downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json`
4. `chmod 600 ~/.kaggle/kaggle.json`

---

## 2. Scripted Downloads

### Everything at once (except ImageNet val tar)

```bash
python scripts/download_data.py --all
```

### Individual components

```bash
# SigLIP2 model checkpoints (cached to ~/.cache/huggingface/hub/)
python scripts/download_data.py --siglip2

# ImageNet class-name mapping (wnid -> label, saved to data/raw/imagenet_meta/)
python scripts/download_data.py --imagenet-map

# COCO 2017 val images (requires Kaggle credentials, symlinked to data/raw/coco_val2017/)
python scripts/download_data.py --coco
```

---

## 3. Expected Directory Layout

After setup, `data/raw/` should look like:

```
data/raw/
  imagenet_val/          # 1000 subdirs (n01440764/, n01443537/, ...)
    n01440764/
      ILSVRC2012_val_00000001.JPEG
      ...
    n01443537/
      ...
  imagenet_meta/
    wnid_to_label.json           # {"n01440764": "tench", ...}
    imagenet_class_index.json    # PyTorch-style {idx: [wnid, label]}
    imagenet_simple_labels.json  # ["tench", "goldfish", ...]
    ILSVRC2012_val_synsets.txt   # 50K lines, one synset per val image
  coco_val2017/          # symlink -> kagglehub cache
    000000000139.jpg
    ...
```

---

## 4. Model Weights

| Model | Library | Cache location |
|-------|---------|----------------|
| CLIP ViT-B/32 | `open_clip` | `~/.cache/clip/` (auto on first use) |
| CLIP ViT-L/14 | `open_clip` | `~/.cache/clip/` (auto on first use) |
| SigLIP2-base-patch16-224 | `transformers` | `~/.cache/huggingface/hub/` |
| SigLIP2-so400m-patch14-384 | `transformers` | `~/.cache/huggingface/hub/` |

CLIP weights download automatically when you first run an experiment. SigLIP2 weights should be pre-cached with the download script to avoid timeouts mid-experiment.

---

## 5. What each phase needs

| Phase | ImageNet val | COCO val | SigLIP2 | CLIP |
|-------|:---:|:---:|:---:|:---:|
| Phase 1 (Exp 0, 0b, 4, 5) | required | - | required (Exp 4) | required |
| Phase 2 (retrieval) | required | optional | optional | required |
| Phase 3 (expansion) | required for Exp 6-8 | optional (only if using external retrieval assets) | optional | required |
