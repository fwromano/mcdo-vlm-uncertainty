# Codebase State Report

Date: 2026-02-24
Repo: `mcdo-vlm-uncertainty`

## Executive Summary

The project now has an end-to-end experiment scaffold implemented for:
1. Phase 1 (foundation)
2. Phase 2 (core experiments)
3. Phase 3 (comparison, diagnostics, MOT evaluator)
4. Phase 4 (optional extensions)

The primary remaining work is operational:
1. Ensure required datasets/artifacts are available.
2. Run experiments at full scale.
3. Aggregate and interpret results for report/paper outputs.

## What Is Implemented

## Phase 1

Folder: `phase_one/`

Implemented scripts:
1. `exp0_nested_mc.py`
2. `exp0b_norm_geometry.py`
3. `exp4_subset_recipe.py`
4. `exp5_subset_ambiguity.py`
5. `run_phase1.py`

Capabilities:
1. Model loading and MC Dropout uncertainty estimation.
2. Reliability metrics (ICC, SNR, pairwise Spearman).
3. Pre/post-norm geometry diagnostics and angular variance.
4. Subset ambiguity correlation metrics.

## Phase 2

Folder: `phase_two/`

Implemented scripts:
1. `dropout_types.py`
2. `exp1_rank_p.py`
3. `exp2_synthetic_natural.py`
4. `exp3_dropout_type.py`
5. `exp4_full_matrix.py`
6. `exp5_full_ambiguity.py`
7. `run_phase2.py`

Capabilities:
1. Rank stability across dropout rates.
2. Synthetic/natural baseline generation and comparison.
3. Dropout type ablation (A/B/C/D/E).
4. Cross-model matrix evaluation.
5. Full ambiguity metrics (classification), plus retrieval support via JSON input.

## Phase 3

Folder: `phase_three/`

Implemented scripts:
1. `exp6_laplace_comparison.py`
2. `exp7_aleatoric_epistemic.py`
3. `exp8_semantic_space.py`
4. `exp9_mot_adaptive_demo.py`
5. `run_phase3.py`

Capabilities:
1. MCDO vs projection-Laplace proxy comparison.
2. Aleatoric vs epistemic perturbation diagnostics.
3. Semantic-subspace directional uncertainty analysis.
4. MOT adaptive-association evaluator on precomputed frame cost payloads.

## Phase 4 (Optional)

Folder: `phase_four/`

Implemented scripts:
1. `text_dropout.py`
2. `layerwise_dropout.py`
3. `exp10_text_encoder_uncertainty.py`
4. `exp11_concrete_dropout_proxy.py`
5. `run_phase4.py`

Capabilities:
1. Text-tower MC Dropout uncertainty.
2. Concrete-style layerwise dropout-rate search proxy for vision tower.

## Top-Level Orchestration

Implemented:
1. `run_all_phases.py` for sequential phase execution.

## Execution Readiness

## Ready now

1. All phase runners are implemented and CLI-accessible.
2. Periodic checkpointing is implemented across long trial loops.
3. Experiment scripts produce structured `*.json` and `*.npz` outputs.

## Required inputs/artifacts

1. Image dataset root for phase image experiments (ImageNet-val style layout expected by scripts).
2. Optional retrieval JSON for Phase 2 Exp 5 retrieval metrics.
3. MOT frame-cost JSON for Phase 3 Exp 9.
4. Model weights must be downloadable/available in runtime environment.

## Periodic Checkpointing Status

Checkpointing is now integrated via `--save-every` (default `1` in runners).

Behavior:
1. Writes partial arrays to `*_partial.npz` during trial loops.
2. Writes progress metadata to `*_progress.json`.
3. Final full outputs are still written at experiment completion.

Current limitation:
1. Scripts checkpoint progress, but do not yet implement automatic resume-from-partial logic.

## Validation Status

Completed:
1. Broad `py_compile` checks across modified scripts passed.
2. CLI help checks passed for phase runners and major experiment entrypoints.
3. Smoke runs were executed on tiny local mock datasets/payloads to verify orchestration paths.

Not completed yet:
1. Full-scale GPU benchmark runs with final datasets.
2. Final consolidated results write-up from completed large runs.

## Git/Workspace State

Current tree includes substantial new phase code and docs that are present locally.

At report generation time:
1. Multiple files are modified or untracked in the working tree.
2. This indicates implementation is present in workspace but not yet fully committed as a clean git snapshot.

## Practical Next Step

Run the full pipeline with checkpointing enabled:

```bash
python run_all_phases.py /path/to/imagenet_val outputs/full_run --phases 1,2,3,4 --device cuda --save-every 1
```

Add Exp 9 input when ready:

```bash
--exp9-cost-json /path/to/mot_costs.json
```
