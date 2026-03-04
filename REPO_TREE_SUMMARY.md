# Repo Tree Summary

This is a high-level map of the repository. Generated/runtime-heavy directories such as `.git/` and `__pycache__/` are omitted, and `outputs/` is summarized rather than expanded exhaustively.

```text
mcdo-vlm-uncertainty/
├── AGENT_RUNBOOK.md                  # Notes for running/operating the repo
├── CODEBASE_STATE.md                 # Snapshot of repo status and major decisions
├── CONTEXT_HANDOFF.md                # Handoff notes from prior work
├── DATA_SETUP.md                     # Dataset setup instructions
├── EXPERIMENT_ALGEBRA.md             # Experiment relationships / design notes
├── PERFORMANCE_GUIDE.md              # Performance and runtime tuning notes
├── PHASE_ONE_REPORT.md               # Phase 1 findings/report
├── PRELIM_FINDINGS.md                # Preliminary analysis summary
├── README.md                         # Main project entry point
├── WHY_CLIP_VS_SIGLIP2.md            # Model comparison / rationale notes
├── paper_outline_v3.md               # Paper-aligned experiment spec
├── REPO_TREE_SUMMARY.md              # This file
├── environment.yml                   # Conda environment definition
├── pyproject.toml                    # Packaging / project metadata
├── requirements.txt                  # Pip requirements
├── setup_env.sh                      # One-shot environment bootstrap
├── run                               # Friendly CLI wrapper: `./run phase one|two|...`
├── run_all_phases.py                 # Sequential multi-phase runner
├── prelim_ablation.py                # Standalone preliminary ablation analysis
├── prelim_investigation.py           # Standalone preliminary investigation
├── phase2_exp1_resume_command.txt    # Saved command for resumed Phase 2 Exp1 runs
│
├── data/
│   ├── imagenet_class_map.json       # WNID/synset -> human label mapping
│   └── raw/
│       ├── imagenet_meta/            # ImageNet metadata assets
│       └── imagenet_val/             # ImageNet validation image tree
│
├── src/
│   ├── mcdo_clip/
│   │   ├── __init__.py
│   │   ├── data.py                   # Dataset/data loading helpers
│   │   ├── metrics.py                # Metric helpers
│   │   ├── models.py                 # Model wrappers/utilities
│   │   └── sampling.py               # MC dropout sampling logic
│   ├── mcdo_clip_uncertainty.egg-info/
│   └── mcdo_vlm_uncertainty.egg-info/
│
├── phase_one/
│   ├── README.md
│   ├── __init__.py
│   ├── common.py                     # Shared model loading, manifests, MC trial logic
│   ├── exp0_nested_mc.py             # Exp 0: nested MC estimator validation
│   ├── exp0b_norm_geometry.py        # Exp 0b: pre/post-norm covariance geometry
│   ├── exp4_subset_recipe.py         # Exp 4 subset: CLIP-B/32 vs SigLIP2-B/16
│   ├── exp5_subset_ambiguity.py      # Exp 5 subset: ambiguity prediction
│   ├── run_phase1.py                 # Standard Phase 1 orchestrator
│   └── run_phase1_fast.py            # Fast Apple-Silicon-oriented Phase 1 runner
│
├── phase_two/
│   ├── README.md
│   ├── __init__.py
│   ├── dropout_types.py              # Phase 2 dropout-ablation injection logic
│   ├── exp1_rank_p.py                # Exp 1: rank stability across dropout rates
│   ├── exp2_synthetic_natural.py     # Exp 2: synthetic vs natural baselines
│   ├── exp3_dropout_type.py          # Exp 3: dropout type ablation
│   ├── exp4_full_matrix.py           # Exp 4 full: cross-model matrix
│   ├── exp5_full_ambiguity.py        # Exp 5 full: ambiguity prediction / retrieval
│   ├── exp6_mean_convergence.py      # Exp 6: mean convergence vs pass count
│   └── run_phase2.py                 # Phase 2 orchestrator
│
├── phase_three/
│   ├── README.md
│   ├── __init__.py
│   ├── exp6_laplace_comparison.py    # Laplace vs MCDO comparison
│   ├── exp7_aleatoric_epistemic.py   # Aleatoric vs epistemic decomposition
│   ├── exp8_semantic_space.py        # Semantic-space uncertainty analysis
│   ├── exp9_mot_adaptive_demo.py     # MOT adaptive demo
│   ├── exp_conformal.py              # Conformal-prediction-related experiment
│   └── run_phase3.py                 # Phase 3 orchestrator
│
├── phase_four/
│   ├── README.md
│   ├── __init__.py
│   ├── exp10_text_encoder_uncertainty.py  # Text encoder uncertainty
│   ├── exp11_concrete_dropout_proxy.py    # Concrete-dropout proxy search
│   ├── layerwise_dropout.py               # Layerwise dropout utilities
│   ├── text_dropout.py                    # Text-side dropout helpers
│   └── run_phase4.py                      # Phase 4 orchestrator
│
├── scripts/
│   ├── download_data.py              # Data acquisition helper
│   ├── monitor.sh                    # Shell helper for monitoring runs
│   ├── run_complexity_correlation.py # Legacy/smaller standalone experiment
│   ├── run_covariance_blur_comparison.py
│   ├── run_error_correlation.py
│   ├── run_rank_stability.py
│   ├── run_resolution_effect.py
│   └── run_trivial_baseline.py
│
├── tests/
│   ├── test_exp1_resume.py           # Resume checkpoint loading tests
│   ├── test_phase_two_resume_cli.py  # Phase 2 CLI flag coverage
│   ├── test_prelim_mapping.py        # GT label mapping regression test
│   ├── test_run_wrapper_smoke.py     # `./run` smoke coverage
│   └── test_siglip2_backend_selection.py # SigLIP2 backend/fallback coverage
│
└── outputs/                          # Generated artifacts, smoke runs, reports, logs
    ├── prelim_ablation.json
    ├── prelim_investigation.json
    ├── prelim_investigation_smoke.json
    ├── phase_two/
    │   ├── manifest_exp1.json
    │   ├── manifest_exp3.json
    │   ├── manifest_exp4.json
    │   ├── manifest_exp5.json
    │   └── exp1_rank_p/             # Active/previous Phase 2 Exp1 checkpoints/results
    ├── smoke_* / fast_* / run_*     # Smoke tests, local runs, timestamped experiment outputs
    ├── rank_stability_cifar100_*    # Standalone rank-stability results
    ├── cov_blur*                    # Covariance blur comparison outputs
    └── trivial*                     # Trivial-baseline outputs
```

## Mental Model

```text
root docs/config
├── explain why the repo exists and how to run it
├── phase_one/ .. phase_four/ contain the main paper-aligned experiments
├── phase_one/common.py is the main shared runtime core
├── run + run_all_phases.py are the operator-facing entry points
├── tests/ covers regressions around wrappers, resume, and backend selection
└── outputs/ is mostly generated data and should be treated as runtime artifacts
```
