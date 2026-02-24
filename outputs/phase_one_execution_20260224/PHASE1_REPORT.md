# Phase 1 Execution Report (Constrained Pilot)

## Scope
This report documents an executed **Phase 1 pilot** for:
- Exp 0: Nested MC estimator validation
- Exp 0b: Pre-norm vs post-norm covariance geometry
- Exp 4 subset: cross-model comparison
- Exp 5 subset: ambiguity prediction

Run root:
- `outputs/phase_one_execution_20260224`

## Important Constraints
This was **not** a full exact replication of the paper-outline Phase 1, due to environment limits in this shell:
1. `torch.cuda` was unavailable (CPU-only execution).
2. SigLIP2 weights were unavailable offline (`google/siglip2-base-patch16-224`, `google/siglip2-so400m-patch14-384` not cached; network download blocked).
3. ImageNet val was not available locally in this workspace, so a local class-structured CIFAR-100 test imagefolder was used.

## Data and Models Used
Dataset:
- `data/processed/cifar100_test_imagefolder`
- 10,000 images, 100 classes

Models available and used:
- `clip_b32` (OpenCLIP OpenAI weights)
- `clip_l14` (OpenCLIP OpenAI weights) for Exp 4 subset only

Unavailable in this environment:
- `siglip2_b16`
- `siglip2_so400m`

## Executed Configs
### Exp 0 (nested MC)
- Model: `clip_b32`
- Images: 200
- Dropout: `p=0.01`
- Trials: `K=5`
- Passes: `T in {4,16,32}`
- Output: `exp0_nested_mc/clip_b32/exp0_summary.json`

### Exp 0b (geometry)
- Model: `clip_b32`
- Images: 200
- Dropout: `p=0.01`
- Trials: `K=3`
- Passes: `T=32`
- Output: `exp0b_norm_geometry/exp0b_summary.json`

### Exp 4 subset (comparison)
- Models: `clip_b32`, `clip_l14`
- Images: 40
- Dropout: `p=0.01`
- Trials: `K=2`
- Passes: `T=8`
- Output: `exp4_subset_recipe/exp4_subset_summary.json`

### Exp 5 subset (ambiguity)
- Model: `clip_b32`
- Images: 2,000
- Dropout: `p=0.01`
- Trials: `K=1`
- Passes: `T=32`
- Prompts: `a photo of a {}`, `a {}`, `an image of a {}`
- Output: `exp5_subset_ambiguity/exp5_subset_clip_b32_summary.json`

## Results
## Exp 0: Nested MC reliability
Go/no-go thresholds from outline:
- Usable: median pairwise Spearman >= 0.8 AND SNR >= 2 AND ICC >= 0.75

Observed (`clip_b32`):

| T | Space | SNR | ICC | Median pairwise Spearman | Status |
|---|---|---:|---:|---:|---|
| 4 | pre-norm | 0.255 | 0.052 | 0.261 | failed |
| 4 | post-norm | 0.229 | 0.028 | 0.254 | failed |
| 16 | pre-norm | 0.300 | 0.091 | 0.202 | failed |
| 16 | post-norm | 0.253 | 0.051 | 0.171 | failed |
| 32 | pre-norm | 0.445 | 0.197 | 0.234 | failed |
| 32 | post-norm | 0.383 | 0.155 | 0.200 | failed |

Interpretation:
- Reliability improves as `T` increases, but remains far below usable thresholds.
- Estimator noise still dominates between-image signal (`SNR < 1` everywhere).
- No evidence of stable per-image uncertainty ranking under this setup.

## Exp 0b: Geometry diagnostics
From `exp0b_summary.json`:
- Pre-norm trace/d mean: `0.01111`
- Post-norm trace/d mean: `8.04e-05`
- Pre off-diagonal mass ratio mean: `0.9502`
- Post off-diagonal mass ratio mean: `0.9765`
- Angular variance mean: `0.00625`

Additional derived correlations from `exp0b_geometry_trials.npz`:
- corr(trace_pre, angular_var): `0.712`
- corr(trace_post, angular_var): `0.794`
- corr(trace_pre, trace_post): `0.934`

Interpretation:
- Post-norm covariance is more strongly dominated by off-diagonal mass (consistent with hypersphere coupling concerns).
- Pre/post scalar uncertainty remain strongly aligned in ranking for this dataset/model.
- Geometry differences exist, but they did not translate into usable reliability in Exp 0.

## Exp 4 subset: Cross-model comparison (constrained)
Observed (N=40, K=2, T=8):

| Model | Metric | SNR | ICC | Pairwise Spearman | Status |
|---|---|---:|---:|---:|---|
| clip_b32 | trace_pre | 0.362 | -0.160 | 0.060 | failed |
| clip_b32 | angular_var | 0.361 | -0.161 | -0.116 | failed |
| clip_l14 | trace_pre | 0.660 | 0.138 | 0.215 | failed |
| clip_l14 | angular_var | 0.487 | -0.013 | -0.005 | failed |

Means:
- trace mean: `clip_l14 - clip_b32 = +0.00274`
- angular mean: `clip_l14 - clip_b32 = -0.00394`

Interpretation:
- `clip_l14` shows somewhat better trace reliability than `clip_b32` in this tiny subset, but still fails thresholds.
- This comparison is underpowered (small N/K/T due CPU constraints) and should be treated as directional only.

## Exp 5 subset: Ambiguity prediction
From summary:
- rho(uncertainty, -margin): `0.047`
- rho(uncertainty, entropy): `0.040`
- rho(uncertainty, prompt sensitivity): `0.086`
- AUROC low-margin detection: `0.504`
- AUROC high-entropy detection: `0.472`

Bootstrap (n=500 resamples) on saved arrays:
- rho(uncertainty, -margin): 95% CI `[0.005, 0.091]`
- rho(uncertainty, entropy): 95% CI `[-0.005, 0.080]`
- rho(uncertainty, prompt sensitivity): 95% CI `[0.042, 0.128]`
- AUROC low-margin: 95% CI `[0.464, 0.545]`
- AUROC high-entropy: 95% CI `[0.430, 0.518]`

Interpretation:
- Practical predictive utility is weak-to-null in this run.
- Low-margin AUROC is effectively random.
- High-entropy AUROC trends below 0.5 in this sample.
- Small positive signal appears only for prompt-sensitivity correlation, and effect size is modest.

## Phase 1 Gate Assessment
Under this executed pilot:
- Exp 0 gate: **failed** (not enough signal above estimator noise).
- Exp 5 subset gate: **failed for practical usefulness** (near-random ambiguity prediction metrics).

So, for this constrained setup, the phase outcome aligns with a **negative-result direction**.

## What This Means for Your Paper Decisions
1. With current conditions (offline, CPU-only, CLIP-only, CIFAR substitute), evidence does **not** support proceeding with a positive MCDO framing.
2. The cleanest current narrative is a **diagnostic/negative pilot**: instability and weak ambiguity linkage under frozen CLIP.
3. You still need the intended full environment to test your core hypothesis properly:
- ImageNet val
- SigLIP2 models
- GPU run

## Recommended Next Run (to convert this pilot into decisive Phase 1)
1. Provide local paths for:
- ImageNet val root
- SigLIP2 model cache directories (or network access to download once)
2. Re-run with intended near-outline settings:
- Exp0: `clip_b32,siglip2_b16,siglip2_so400m`, `N=500`, `K=10`, `T=4,16,64`
- Exp0b: `clip_b32`, `N=500`, `K=5`, `T=64`
- Exp4 subset: `clip_b32,siglip2_b16`, `N=500`, `K=10`, `T=64`
- Exp5 subset: best two models from Exp4, `N=5000`, `T=64`

## Artifacts
- Exp0 summary: `outputs/phase_one_execution_20260224/exp0_nested_mc/clip_b32/exp0_summary.json`
- Exp0b summary: `outputs/phase_one_execution_20260224/exp0b_norm_geometry/exp0b_summary.json`
- Exp4 subset summary: `outputs/phase_one_execution_20260224/exp4_subset_recipe/exp4_subset_summary.json`
- Exp5 subset summary: `outputs/phase_one_execution_20260224/exp5_subset_ambiguity/exp5_subset_clip_b32_summary.json`
- Raw arrays: corresponding `.npz` files in those directories
