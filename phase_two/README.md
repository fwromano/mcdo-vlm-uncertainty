# Phase 2 (paper_outline_v3)

This folder implements the Phase 2 experiment set:
1. Exp 1: Rank stability across dropout rates `p`.
2. Exp 3: Dropout type ablation (`A/B/C/D/E`).
3. Exp 5 full: Ambiguity prediction (classification + optional retrieval JSON).
4. Exp 2: Synthetic + natural baselines.
5. Exp 4 full: Cross-model matrix.

## One-command Phase 2 run

```bash
python -m phase_two.run_phase2 /path/to/imagenet_val outputs/phase_two --device cuda
```

## Run each experiment directly

```bash
python -m phase_two.exp1_rank_p /path/to/imagenet_val outputs/phase_two/exp1 --device cuda
python -m phase_two.exp2_synthetic_natural /path/to/imagenet_val outputs/phase_two/exp2 --device cuda
python -m phase_two.exp3_dropout_type /path/to/imagenet_val outputs/phase_two/exp3 --device cuda
python -m phase_two.exp4_full_matrix /path/to/imagenet_val outputs/phase_two/exp4 --device cuda
python -m phase_two.exp5_full_ambiguity /path/to/imagenet_val outputs/phase_two/exp5 --device cuda
```

## Retrieval JSON format (Exp 5 optional)

`--retrieval-json` should point to a JSON list of objects:

```json
[
  {
    "image_path": "relative/or/absolute/path/to/image.jpg",
    "captions": ["caption 1", "caption 2", "caption 3"],
    "correct_index": 0
  }
]
```

Notes:
- `image_path` may be absolute, relative to the retrieval JSON file, or relative to `data_dir`.
- `captions` must contain at least 2 strings.
- `correct_index` defaults to `0` if omitted.

## Outputs

Each experiment writes:
- `*.json` summary files for metrics and settings.
- `*.npz` arrays with per-image uncertainty and intermediate targets.
- Periodic partial checkpoints are enabled by default (`--save-every 1`), including trial-progress JSON files.
