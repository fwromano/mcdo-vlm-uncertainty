# Phase 4 (paper_outline_v3, optional)

This folder implements the optional Phase 4 experiments:
1. Exp 10: Text encoder uncertainty.
2. Exp 11: Concrete-style layerwise dropout-rate search proxy.

## One-command Phase 4 run

```bash
python -m phase_four.run_phase4 /path/to/imagenet_val outputs/phase_four --device cuda
```

## Run each experiment directly

```bash
python -m phase_four.exp10_text_encoder_uncertainty /path/to/imagenet_val outputs/phase_four/exp10 --device cuda
python -m phase_four.exp11_concrete_dropout_proxy /path/to/imagenet_val outputs/phase_four/exp11 --device cuda
```

## Exp 10 prompt file format (optional)

`--prompt-file` accepts one prompt per line.
You can optionally prefix a style tag using `style<TAB>prompt`.

Example:

```text
generic<TAB>an object
specific<TAB>a detailed close-up photo of a golden retriever in sunlight
something that might be a dog
```

## Exp 11 note

Exp 11 is a practical proxy for concrete dropout:
- It searches per-group dropout rates over vision linear-layer groups.
- It optimizes ambiguity-alignment metrics from classification margins/entropy.
- It does **not** train true continuous Bernoulli/dropout parameters end-to-end.

## Outputs

Each experiment writes:
- `*.json` summary files.
- `*.npz` arrays with per-prompt/per-image uncertainties.
- Periodic partial checkpoints are enabled by default (`--save-every 1`).
