# Experimental Plan: MC Dropout as Relative Uncertainty for CLIP

## Hypothesis Summary
- **H1 (Core):** MC Dropout variance is a monotonic function of true uncertainty: $$\sigma^2_{\text{MC}}(x) = p \cdot f(\sigma^2_{\text{true}}(x))$$
- **H2 (Rank Invariance):** Sample rankings by MC variance are stable across dropout rates p ∈ [0.005, 0.1]
- **H3 (Image Complexity):** MC variance correlates positively with image complexity (entropy, edge density)
- **H4 (Trivial Inputs):** Constant-color / low-information images yield near-zero MC variance
- **H5 (Resolution Redundancy):** Upsampled images (pixel space > native content) have lower uncertainty than native resolution
- **H6 (Optimal p):** There exists a principled criterion for selecting dropout rate

---

## Experiment 1: Rank Stability Across Dropout Rates
- **Goal:** Test H2
- **Setup:** 1000+ diverse images (ImageNet val subset or COCO); p ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1}; T = 64; dropout at all 13 encoder sites.
- **Procedure:**
  1. For each p, compute per-sample MC variance: $\sigma^2_i = \tfrac{1}{T}\sum_t \|\mathbf{e}_{i,t} - \bar{\mathbf{e}}_i\|^2$.
  2. Rank samples by $\sigma^2_i$ for each p.
  3. Compute pairwise Spearman ρ between rankings at different p values.
- **Expected:** High ρ (>0.9) for adjacent p in mid-range; degradation at extremes (p < 0.005 too little signal; p > 0.1 noise dominates).
- **Output:** 6×6 rank-correlation matrix.

## Experiment 2: Image Complexity vs MC Variance
- **Goal:** Test H3
- **Setup:** Same 1000+ images; fixed p = 0.01 (or best from Exp 1); T = 64.
- **Complexity metrics:** pixel entropy; edge density (Canny); colorfulness (Hasler–Süsstrunk); JPEG compressibility (compressed/raw size); semantic complexity (#objects, optional detector).
- **Procedure:** compute metrics → compute MC variance → scatter + Pearson/Spearman correlations.
- **Expected:** Positive correlation; likely strongest for edge density or entropy.

## Experiment 3: Trivial Input Baseline
- **Goal:** Test H4
- **Setup:** Synthetic sets (100 solid-color, 100 Gaussian-noise, 100 gradients) + 100 natural images; p = 0.01; T = 64.
- **Procedure:** compute MC variance per category; compare distributions.
- **Expected:** solid ≪ gradient < noise < natural. If solid ≈ natural, H4 fails (variance is architectural noise).

## Experiment 4: Resolution / Upsampling Effect
- **Goal:** Test H5
- **Setup:** 200 natural images; variants: native 224×224, downsample→upsample {112→224, 56→224, 28→224}, high-res 448→224; p = 0.01; T = 64.
- **Procedure:** compute MC variance per image×resolution; plot variance vs effective resolution; paired comparisons.
- **Expected:** Downsample→upsample raises variance; native baseline; high-res downsampled to 224 may lower variance (redundancy).

## Experiment 5: Correlation with Downstream Error
- **Goal:** Validate uncertainty is predictive of error
- **Setup:** Zero-shot classification (ImageNet-1K or CIFAR-100); p = 0.01; T = 64.
- **Procedure:** compute MC variance, mean embedding, correctness; correlate variance with error; calibration plot by variance bins; AUROC for "high variance predicts misclassification."
- **Expected:** Higher variance → lower accuracy; monotonic calibration curve.

## Experiment 6: Comparison with BayesVLM
- **Goal:** Benchmark against a principled Bayesian method
- **Setup:** Same data as Exp 5; methods = MC Dropout (p=0.01, T=64), BayesVLM Laplace on projection, temperature-scaling baseline.
- **Metrics:** Spearman ρ between uncertainty rankings; ECE; AUROC for OOD/misclassification; wall-clock per 1k samples.
- **Expected:** BayesVLM better calibrated; MC Dropout competitive on ranking if H1–H2 hold; MC faster.

## Experiment 7: Optimal Dropout Selection
- **Goal:** Test H6
- **Criteria:**
  - Rank stability: choose p maximizing ρ to adjacent p values.
  - Error correlation: choose p maximizing correlation with downstream error.
  - Variance ratio: maximize between-sample / within-sample variance.
  - Concrete Dropout: learn p (if implemented).
- **Procedure:** use data from Exp 1 + Exp 5; pick p per criterion; compare agreement and use-case fit.

## Experiment 8: Dropout Placement Ablation
- **Goal:** Identify layers carrying useful uncertainty signal
- **Configs:** all 13 sites; final 3 only; attention only; MLP only; projection only; p = 0.01; T = 64.
- **Procedure:** repeat Exp 2 (complexity) and Exp 5 (error correlation) per config; compare signal quality.
- **Expected:** Later layers likely more task-relevant; early-layer dropout may add noise.

## Experiment 9: Text Encoder Uncertainty
- **Goal:** Extend to text modality
- **Setup:** 1000 prompts (specific → general → ambiguous); p = 0.01; T = 64 on text encoder.
- **Procedure:** compute MC variance per prompt; correlate with prompt specificity (token count, WordNet depth, or manual labels).
- **Expected:** General/short prompts show higher variance; specific prompts lower variance.

---

## Summary Table

| Exp | Hypothesis | Key Metric | Dataset Size |
| --- | --- | --- | --- |
| 1 | H2 (rank invariance) | Spearman ρ matrix | 1000 images × 6 p values |
| 2 | H3 (complexity) | Correlation coefficients | 1000 images |
| 3 | H4 (trivial inputs) | Variance by category | 400 synthetic + 100 natural |
| 4 | H5 (resolution) | Variance vs resolution | 200 images × 5 resolutions |
| 5 | Validation | Accuracy calibration | 10K+ (ImageNet val) |
| 6 | Comparison | ECE, AUROC, ρ | 10K+ |
| 7 | H6 (optimal p) | Criterion agreement | From Exp 1 + 5 |
| 8 | Ablation | Signal quality by layer config | 1000 images × 5 configs |
| 9 | Extension | Text variance vs specificity | 1000 prompts |

---

## Priority Order
1. Exp 3 (trivial inputs) — fastest sanity check; if solid colors ≈ natural variance, core hypothesis fails.
2. Exp 1 (rank stability) — tests fundamental claim.
3. Exp 2 (complexity correlation) — tests interpretability.
4. Exp 5 (error correlation) — tests utility.
5. Exp 4, 7, 8, 9 — secondary analyses.
6. Exp 6 (BayesVLM comparison) — needs external code.

## Open Questions
1. Variance metric: covariance trace, Frobenius norm, or top eigenvalue? Each captures different aspects.
2. Embedding normalization: compute variance pre- or post-L2 norm? Post-normalization bounds variance.
3. Aggregation: per-dimension variance summed, or variance of embedding norm?
4. Compute budget: GPU-hours available? Exp 5–6 on full ImageNet are expensive.

---

## Repo Layout
- `src/mcdo_clip/`: library code for CLIP loading, dropout toggling, MC sampling, and complexity metrics.
- `scripts/`: CLIs for core experiments.
  - `run_rank_stability.py`: Exp 1 (Spearman matrix across dropout rates).
  - `run_complexity_correlation.py`: Exp 2 (complexity metrics vs. MC variance).
  - `run_trivial_baseline.py`: Exp 3 (solid/gradient/noise vs. natural).
  - `run_resolution_effect.py`: Exp 4 (downsample/upsample variants).
  - `run_error_correlation.py`: Exp 5 (variance vs. zero-shot accuracy on ImageFolder data).
- `data/`: placeholder for raw/processed assets.
- `requirements.txt` / `pyproject.toml`: dependency pins.

## Quickstart
```bash
cd MCDO_CLIP_Uncertainty
python -m pip install -e .

# Exp 1: Rank stability
python scripts/run_rank_stability.py /path/to/images outputs/rank_stability \
  --device cuda --passes 64 --dropout-rates 0.001,0.005,0.01,0.02,0.05,0.1

# Exp 2: Complexity correlation
python scripts/run_complexity_correlation.py /path/to/images outputs/complexity --device cuda

# Exp 3: Trivial baseline (add natural images with --natural-dir)
python scripts/run_trivial_baseline.py outputs/trivial --device cuda

# Exp 4: Resolution / upsampling
python scripts/run_resolution_effect.py /path/to/images outputs/resolution --device cuda

# Exp 5: Variance vs. error on an ImageFolder zero-shot dataset
python scripts/run_error_correlation.py /path/to/imagefolder outputs/error_corr --device cuda
```

Scripts default to CLIP ViT-B/32 (OpenAI weights). Override with `--model` / `--pretrained`. Add `--no-l2` to disable embedding normalization when you want raw variance.

## Environment Setup (reproducible)
- Conda: `conda env create -f environment.yml && conda activate mcdo-clip-uncertainty`
- Pip/venv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

Notes:
- `networkx==3.2.1` is pinned to avoid known SyntaxError regressions seen with newer builds when importing `torchvision`/`open_clip`.
- If you are CPU-only, drop `pytorch-cuda` from `environment.yml`; if you use a different CUDA stack, adjust the version tag.

## Phase 1 (paper_outline_v3)
- New Phase 1 implementation lives under `phase_one/`.
- Run all Phase 1 experiments:
```bash
python -m phase_one.run_phase1 /path/to/imagenet_val outputs/phase_one --device cuda
```
- See `phase_one/README.md` for direct per-experiment commands.
- Use `--save-every N` to control periodic partial checkpoint frequency (default `1`).

## Phase 2 (paper_outline_v3)
- Phase 2 implementation lives under `phase_two/`.
- Run all Phase 2 experiments:
```bash
python -m phase_two.run_phase2 /path/to/imagenet_val outputs/phase_two --device cuda
```
- See `phase_two/README.md` for direct per-experiment commands and retrieval JSON format.

## Phase 3 (paper_outline_v3)
- Phase 3 implementation lives under `phase_three/`.
- Run core Phase 3 image experiments:
```bash
python -m phase_three.run_phase3 /path/to/imagenet_val outputs/phase_three --device cuda
```
- To include Exp 9 MOT evaluation, pass `--only exp9 --exp9-cost-json /path/to/costs.json`.
- See `phase_three/README.md` for direct commands and the Exp 9 input schema.

## Phase 4 (paper_outline_v3, optional)
- Phase 4 implementation lives under `phase_four/`.
- Run optional Phase 4 experiments:
```bash
python -m phase_four.run_phase4 /path/to/imagenet_val outputs/phase_four --device cuda
```
- See `phase_four/README.md` for prompt-file format and concrete-style search details.

## Full Pipeline Runner
- Run multiple phases in sequence:
```bash
python run_all_phases.py /path/to/imagenet_val outputs/full_run --phases 1,2,3 --device cuda
```
- Include Phase 3 Exp 9 with `--exp9-cost-json /path/to/mot_costs.json`.
- Add `--save-every N` to tune partial checkpoint cadence across phases.
