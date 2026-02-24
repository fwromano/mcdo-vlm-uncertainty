# Phase 3 (paper_outline_v3)

This folder implements the Phase 3 experiment set:
1. Exp 6: MCDO vs projection-Laplace comparison.
2. Exp 7: Aleatoric vs epistemic separation diagnostics.
3. Exp 8: Semantic-space directional uncertainty diagnostics.
4. Exp 9: MOT adaptive association evaluation on precomputed costs.

## One-command Phase 3 run

```bash
python -m phase_three.run_phase3 /path/to/imagenet_val outputs/phase_three --device cuda
```

## Run each experiment directly

```bash
python -m phase_three.exp6_laplace_comparison /path/to/imagenet_val outputs/phase_three/exp6 --device cuda
python -m phase_three.exp7_aleatoric_epistemic /path/to/imagenet_val outputs/phase_three/exp7 --device cuda
python -m phase_three.exp8_semantic_space /path/to/imagenet_val outputs/phase_three/exp8 --device cuda
python -m phase_three.exp9_mot_adaptive_demo /path/to/mot_costs.json outputs/phase_three/exp9
```

## Exp 9 input format

`exp9_mot_adaptive_demo.py` expects a JSON list (or object with `frames`) where each frame has:

```json
{
  "frame_id": 1,
  "tracks": [
    {"id": "t1", "gt_id": "person_7"}
  ],
  "detections": [
    {
      "id": "d1",
      "gt_id": "person_7",
      "uncertainty": 0.24,
      "oracle_uncertainty": 0.20,
      "laplace_uncertainty": 0.27
    }
  ],
  "motion_cost": [[0.11]],
  "appearance_cost": [[0.08]]
}
```

Notes:
- `motion_cost` and `appearance_cost` must both be `num_tracks x num_detections`.
- `adaptive` mode reads detection-level `uncertainty` (or `--uncertainty-key`).
- `oracle` and `laplace` modes are automatically skipped when their keys are missing.

## Outputs

Each experiment writes:
- `*.json` summary files for metrics and settings.
- `*.npz` arrays for per-image outputs (Exp 6-8).
- Periodic partial checkpoints are enabled by default (`--save-every 1` for Exp 6-8).
