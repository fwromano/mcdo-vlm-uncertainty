# Phase 1 (paper_outline_v3)

This folder implements Phase 1 from `paper_outline_v3.md`:

1. **Exp 0**: Nested MC estimator validation (`K=10`, `T={4,16,64}`, pre/post norm).
2. **Exp 0b**: Pre-norm vs post-norm covariance geometry + angular variance.
3. **Exp 4 subset**: CLIP-B/32 vs SigLIP2-B/16 (recipe test).
4. **Exp 5 subset**: Ambiguity prediction on a 5K subset.

## Supported model keys

- `clip_b32`
- `clip_l14`
- `siglip2_b16`
- `siglip2_so400m`
- `siglip2_g16`

## One-command Phase 1 run

```bash
python -m phase_one.run_phase1 /path/to/imagenet_val outputs/phase_one --device cuda
```

## Run each experiment directly

```bash
python -m phase_one.exp0_nested_mc /path/to/imagenet_val outputs/phase_one/exp0 --device cuda
python -m phase_one.exp0b_norm_geometry /path/to/imagenet_val outputs/phase_one/exp0b --device cuda
python -m phase_one.exp4_subset_recipe /path/to/imagenet_val outputs/phase_one/exp4 --device cuda
python -m phase_one.exp5_subset_ambiguity /path/to/imagenet_val outputs/phase_one/exp5 --device cuda
```

## Notes

- The scripts inject uniform dropout into **vision linear layers** for MC sampling.
- Exp 5 class prompts default to folder names (or mapped labels via `--class-map`).
- Each experiment saves both a compact `*.json` summary and `*.npz` arrays.
