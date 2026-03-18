# Round 3

`round3/` is currently a notebook-first workspace for baseline embedding generation.

## Current contents

- `launch_notebook.sh`: starts Jupyter Lab rooted at `round3/`.
- `part1_baseline_embeddings/compute_embeddings.ipynb`: builds deterministic image embeddings for an ImageNet validation subset.
- `part1_baseline_embeddings/compute_embeddings.html`: exported notebook view.
- `part1_baseline_embeddings/outputs/`: saved `.npz` artifacts from notebook runs.

## Setup

From the repo root, the intended setup is a single command:

```bash
bash round3/launch_notebook.sh
```

On first run, the launcher will:

- create the conda env from `environment.yml` if it does not exist,
- install the repo into that env in editable mode,
- install notebook-only packages if they are missing,
- register the Jupyter kernel,
- open Jupyter Lab rooted at `round3/`, with the browser enabled by default.

By default it uses the env name from `environment.yml`. You can override that with `MCDO_NOTEBOOK_ENV` if needed.
If you do not want it to try opening a browser window, run with `MCDO_OPEN_BROWSER=0`.

### Data required by the notebook

The notebook uses:

- `data/raw/imagenet_val/`: ImageFolder-style ImageNet validation tree.
- `data/imagenet_class_map.json`: JSON mapping from WNID to class label.

Expected structure:

```text
data/
  imagenet_class_map.json
  raw/
    imagenet_val/
      n01440764/
        ILSVRC2012_val_00000001.JPEG
      n01443537/
        ...
```

If `imagenet_val/` is not prepared yet, follow [`../docs/DATA_SETUP.md`](../docs/DATA_SETUP.md) for the ImageNet validation download and reorganization step.

### Model weights

Part 1 defaults to CLIP ViT-B/32 via `open_clip`. Weights are downloaded automatically on first use and cached locally.

## Use

### Launch the default notebook

From the repo root:

```bash
bash round3/launch_notebook.sh
```

You can also point the launcher at another notebook or subdirectory under `round3/`:

```bash
bash round3/launch_notebook.sh round3/part1_baseline_embeddings/compute_embeddings.ipynb
bash round3/launch_notebook.sh round3/part1_baseline_embeddings
```

The launcher restricts targets to files under `round3/` and starts Jupyter Lab with `round3/` as the server root.

### Notebook flow

`part1_baseline_embeddings/compute_embeddings.ipynb` is organized into four steps:

1. Preview data and labels.
2. Load the model.
3. Compute embeddings in batches.
4. Run basic embedding analysis.

The main config lives in the notebook code cell near the top:

```python
MODEL_KEY = "clip_b32"
REQUESTED_DEVICE = "auto"
NUM_IMAGES = 5000
BATCH_SIZE = default_batch_size(DEVICE)
NUM_WORKERS = default_num_workers()
CHECKPOINT_EVERY = 10
SEED = 42
RESUME = True
```

Useful knobs:

- Change `MODEL_KEY` to swap models supported by the notebook registry.
- Lower `NUM_IMAGES` for a quick smoke run.
- Set `REQUESTED_DEVICE` to `cpu`, `mps`, `cuda`, or leave `auto`.
- Keep `RESUME = True` if you want interrupted runs to continue from checkpoint.

## Outputs

Step 3 writes to `round3/part1_baseline_embeddings/outputs/`:

- `embeddings.npz`: final archive.
- `embeddings_checkpoint.npz`: temporary checkpoint file while a run is in progress.

The final archive contains:

- `embeddings`: `float32` array of shape `(N, D)`.
- `wnids`: class folder names.
- `labels`: human-readable labels from `data/imagenet_class_map.json`.
- `image_paths`: source image paths.
- `processed`: number of completed images.

The current checked-in sample output is a `5000 x 512` CLIP embedding archive.
