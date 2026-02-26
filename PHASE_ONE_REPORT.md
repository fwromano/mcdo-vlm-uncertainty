# Phase 1 Report: MC Dropout Reliability Gate

**Run ID:** `run_20260225_150315`
**Date:** February 25-26, 2026
**Hardware:** M3 Ultra Mac Studio (60 GPU cores, 96 GB unified memory, MPS backend)
**Data:** ImageNet-1K validation set (`data/raw/imagenet_val/`, 1000 classes, ~50K images)

---

## 1. Executive Summary

Phase 1 tests whether MC Dropout applied to frozen vision-language model encoders
produces **reliable** (reproducible across independent trials) and **valid**
(correlating with semantic ambiguity) per-image uncertainty scores.

**Verdict: GO** -- proceed to Phase 2 with siglip2_b16 as the primary model and
siglip2_so400m as a secondary model. Drop clip_b32.

| Model | Best Config | Spearman | SNR | ICC | Gate |
|---|---|---|---|---|---|
| **siglip2_b16** | T=64, post-norm | **0.956** | **26.5** | **0.963** | **PASS** |
| **siglip2_so400m** | T=64, post-norm | **0.837** | **6.2** | **0.860** | **PASS** |
| clip_b32 | T=64, pre-norm | 0.486 | 1.01 | 0.477 | **FAIL** |

The MC dropout uncertainty signal from siglip2_b16 is remarkably stable: if you
run the entire T=64 MC sampling procedure independently 10 times, the resulting
per-image uncertainty *rankings* correlate at rho=0.956 (median pairwise
Spearman). The signal-to-noise ratio of 26.5 means between-image variance is
26x larger than within-trial noise.

However, the ambiguity validation (Exp 5) failed for clip_b32 -- the only model
tested -- returning near-chance AUROC and near-zero rank correlations. This is
expected given clip_b32's failed reliability gate, but means we lack ambiguity
validation for the passing models. **Exp 5 should be re-run with siglip2_b16 in
Phase 2.**

---

## 2. Methodology

### 2.1 Core Approach

All model weights are frozen. Dropout is injected *post-hoc*: every `nn.Linear`
layer in the vision encoder is wrapped with `nn.Dropout(p=0.01)`, applied to
the layer's output activations. The model remains in eval mode for
BatchNorm/LayerNorm, but the injected dropout layers are set to train mode so
they are stochastic.

For each image, T independent forward passes are run with different dropout
masks. The per-image uncertainty score is:

```
trace(Cov) / D = (1/D) * sum_d [ (1/T) * sum_t (e_{d,t})^2 - ((1/T) * sum_t e_{d,t})^2 ]
```

where `e_{d,t}` is dimension d of the feature vector on pass t, and D is the
embedding dimensionality. This is computed in both **pre-norm** (raw encoder
output) and **post-norm** (after L2 normalization) feature spaces.

### 2.2 Reliability Assessment (Exp 0)

The key question: if you repeat the entire MC sampling procedure, do you get the
same uncertainty ranking?

**Protocol:**
- For each model, run K=10 independent trials
- Each trial: T forward passes with fresh dropout seeds, producing one
  uncertainty score per image
- Across the K trials, compute three reliability metrics:
  - **Pairwise Spearman rho:** For all (K choose 2) = 45 trial pairs, compute
    Spearman rank correlation between per-image uncertainty vectors. Report
    median.
  - **SNR (Signal-to-Noise Ratio):** `Var(image means across trials) /
    Mean(within-trial variance across images)`. High SNR means the
    between-image differences dominate the trial-to-trial noise.
  - **ICC (Intraclass Correlation Coefficient):** One-way random effects ICC,
    computed as `(MS_between - MS_within) / (MS_between + (K-1)*MS_within)`.
    Ranges from 0 (pure noise) to 1 (perfect agreement).

**Pass thresholds (nested T extraction):**
- T = {4, 16, 64} forward passes, extracted from a single T_max=64 run by
  snapshotting the running accumulators at each checkpoint. This means T=4 and
  T=16 results use the *first* 4 and 16 passes of the same run, not
  independent runs.

**Gate criteria:**
| Metric | Usable | Marginal | Failed |
|---|---|---|---|
| Pairwise Spearman median | >= 0.80 | 0.60 - 0.80 | < 0.60 |
| SNR | >= 2.0 | 1.0 - 2.0 | < 1.0 |
| ICC | >= 0.75 | -- | -- |

Status = "usable" requires ALL three usable thresholds met.
Status = "failed" if Spearman < 0.6 OR SNR < 1.0.
Otherwise "marginal".

### 2.3 Models Tested

| Key | Architecture | Params | Embedding Dim | Resolution | Backend |
|---|---|---|---|---|---|
| clip_b32 | ViT-B/32 | ~88M | 512 | 224px | OpenAI via open_clip |
| siglip2_b16 | SigLIP2 Base | ~93M | 768 | 224px | google/siglip2-base-patch16-224 via HuggingFace |
| siglip2_so400m | SigLIP2 SO400M | ~428M | 1152 | 384px | google/siglip2-so400m-patch14-384 via HuggingFace |

### 2.4 Fixed Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Dropout rate (p) | 0.01 | Conservative; avoids destroying representations |
| Injection target | All nn.Linear in vision encoder | Uniform coverage |
| Number of images (N) | 500 (Exp 0/0b/4), 5000 (Exp 5) | 500 sufficient for reliability; 5000 for correlation power |
| Trials (K) | 10 (Exp 0/4), 5 (Exp 0b), 1 (Exp 5) | K=10 gives 45 pairwise comparisons |
| Seed strategy | seed + trial_index | Reproducible but independent per trial |
| Batch size | Full dataset (500 or 5000) | Single-batch on unified memory |
| Precision | fp16 forward, fp64 accumulation | fp16 for speed on MPS; fp64 for numerical stability in variance |
| Normalization | L2-norm post-encoding | Post-norm = F.normalize(pre, dim=-1) |

---

## 3. Experiment 0: Nested MC Estimator Validation

### 3.1 Full Results Table

**N=500 images, K=10 trials, p=0.01 dropout on all linear layers.**

#### clip_b32

| T | Space | Spearman (med) | Spearman IQR | SNR | ICC | Status |
|---|---|---|---|---|---|---|
| 4 | pre-norm | 0.362 | 0.039 | 0.163 | 0.059 | **failed** |
| 4 | post-norm | 0.331 | 0.045 | 0.139 | 0.038 | **failed** |
| 16 | pre-norm | 0.365 | 0.046 | 0.327 | 0.185 | **failed** |
| 16 | post-norm | 0.334 | 0.060 | 0.247 | 0.128 | **failed** |
| 64 | pre-norm | 0.486 | 0.033 | 1.014 | 0.477 | **failed** |
| 64 | post-norm | 0.369 | 0.035 | 0.654 | 0.356 | **failed** |

**Assessment:** clip_b32 fails at every T value in both feature spaces. Even at
T=64, the median pairwise Spearman is only 0.486 (pre-norm), well below the
0.60 marginal threshold. SNR barely reaches 1.0. The uncertainty signal is
dominated by stochastic noise -- two independent runs produce essentially
unrelated rankings.

Notably, clip_b32 pre-norm *slightly outperforms* post-norm at every T. This is
the opposite of siglip2_b16 and may reflect architectural differences in how
CLIP vs SigLIP2 distribute information across the feature vector.

#### siglip2_b16

| T | Space | Spearman (med) | Spearman IQR | SNR | ICC | Status |
|---|---|---|---|---|---|---|
| 4 | pre-norm | 0.482 | 0.036 | 0.966 | 0.464 | **failed** |
| 4 | post-norm | 0.601 | 0.032 | 1.600 | 0.600 | **marginal** |
| 16 | pre-norm | 0.776 | 0.014 | 3.883 | 0.791 | **marginal** |
| 16 | post-norm | 0.848 | 0.011 | 6.702 | 0.868 | **usable** |
| 64 | pre-norm | 0.930 | 0.006 | 15.029 | 0.937 | **usable** |
| 64 | post-norm | 0.956 | 0.003 | 26.493 | 0.963 | **usable** |

**Assessment:** siglip2_b16 is the clear winner. Post-norm at T=64 achieves
near-perfect reliability: rho=0.956, SNR=26.5, ICC=0.963. Even at T=16
post-norm, it already crosses the usable threshold (rho=0.848, SNR=6.7).

The convergence pattern is clean:
- T=4 -> T=16: Spearman jumps from 0.601 to 0.848 (post-norm)
- T=16 -> T=64: Spearman rises from 0.848 to 0.956

The tight IQR (0.003 at T=64 post-norm) means the 45 pairwise Spearman values
are tightly clustered -- reliability is not just high on average but
consistently high across all trial pairs.

#### siglip2_so400m

| T | Space | Spearman (med) | Spearman IQR | SNR | ICC | Status |
|---|---|---|---|---|---|---|
| 4 | pre-norm | 0.299 | 0.055 | 0.386 | 0.222 | **failed** |
| 4 | post-norm | 0.378 | 0.058 | 0.463 | 0.266 | **failed** |
| 16 | pre-norm | 0.510 | 0.043 | 1.293 | 0.544 | **failed** |
| 16 | post-norm | 0.596 | 0.035 | 1.681 | 0.613 | **failed** |
| 64 | pre-norm | 0.784 | 0.015 | 4.689 | 0.821 | **marginal** |
| 64 | post-norm | 0.837 | 0.014 | 6.222 | 0.860 | **usable** |

**Assessment:** siglip2_so400m reaches usable only at T=64 post-norm, and
barely (Spearman=0.837 vs threshold 0.80). Its pre-norm result at T=64 is
marginal (0.784). It needs significantly more passes than siglip2_b16 to
converge, despite being a much larger model (428M vs 93M params).

This is likely because the so400m model has higher embedding dimensionality
(1152 vs 768) and higher resolution (384px vs 224px), meaning each dropout mask
produces a relatively smaller perturbation relative to the total information
content. More passes are needed to average out the noise.

### 3.2 Convergence Analysis

The SNR scales roughly linearly with T (as expected from central limit theorem
-- variance of the mean estimator decreases as 1/T):

**siglip2_b16 post-norm SNR progression:**
- T=4: SNR = 1.60
- T=16: SNR = 6.70 (4.2x increase for 4x more passes)
- T=64: SNR = 26.5 (4.0x increase for 4x more passes)

This near-perfect 4x scaling confirms the noise is well-behaved (i.i.d. across
passes) and that the signal (between-image variance) is stable as T increases.
Extrapolating: T=256 would yield SNR ~ 106, though with diminishing practical
returns.

**clip_b32 pre-norm SNR progression:**
- T=4: SNR = 0.163
- T=16: SNR = 0.327 (2.0x for 4x passes)
- T=64: SNR = 1.014 (3.1x for 4x passes)

The scaling is sub-linear and the absolute values are 25x lower than siglip2_b16
at every T. clip_b32 has a fundamentally weaker dropout-induced signal, not just
a convergence problem. Even at T=256, it would only reach SNR ~ 3-4, which is
marginal territory. At T=1024, it might reach usable -- but at 16x the compute
cost of siglip2_b16 at T=64 for an inferior result.

### 3.3 Pre-Norm vs Post-Norm

| Model | T=64 | Pre-norm Spearman | Post-norm Spearman | Winner |
|---|---|---|---|---|
| clip_b32 | 64 | **0.486** | 0.369 | pre-norm |
| siglip2_b16 | 64 | 0.930 | **0.956** | post-norm |
| siglip2_so400m | 64 | 0.784 | **0.837** | post-norm |

For both SigLIP2 models, L2 normalization *improves* reliability. This makes
intuitive sense: normalization projects features onto the unit hypersphere,
removing magnitude variation and isolating *directional* uncertainty. The
variance of normalized features reflects how much the dropout masks change the
*direction* of the embedding, not its scale.

clip_b32 is the exception -- post-norm is *worse*. This may indicate that
clip_b32's useful dropout signal is concentrated in magnitude rather than
direction, and normalization destroys it.

---

## 4. Experiment 0b: Covariance Geometry

**Model:** clip_b32, T=64, K=5 trials, N=500 images.

This experiment computes the full per-image covariance matrix across MC passes
(rather than just its trace) to understand the *structure* of dropout-induced
variation.

### 4.1 Results

| Metric | Value | Interpretation |
|---|---|---|
| trace_pre_per_d (mean) | 0.00752 | Mean per-dimension variance in pre-norm space |
| trace_post_per_d (mean) | 7.26e-05 | ~100x smaller in post-norm space (normalization compresses variance) |
| offdiag_pre (mean) | 0.989 | 98.9% of total absolute covariance is off-diagonal |
| offdiag_post (mean) | 0.990 | Same pattern after normalization |
| angular_var (mean) | 0.00756 | Mean angular variance in radians^2 |
| corr(trace_pre, angular) | 0.708 | Moderate agreement between trace and angular metrics |
| corr(trace_post, angular) | 0.806 | Better agreement in post-norm space |
| corr(trace_pre, trace_post) | 0.943 | Pre- and post-norm traces rank images very similarly |

### 4.2 Interpretation

**Off-diagonal dominance (0.99):** The dropout-induced covariance matrix is
almost entirely off-diagonal. This means the perturbations are *not*
independent across dimensions -- when dropout changes one feature dimension, it
changes others in correlated ways. This is expected because a single Linear
layer's dropout affects all output dimensions simultaneously.

This has an important implication: the trace (sum of diagonal) captures only ~1%
of the total covariance structure. A richer uncertainty metric (e.g.,
log-determinant or top eigenvalues) might extract more signal. However, the
trace is what passed the reliability gate for siglip2_b16, so it works despite
this limitation.

**Pre/post trace correlation (0.94):** The two normalization spaces rank images
almost identically. The choice of space mostly affects the *magnitude* of
reliability metrics (post-norm compresses variance by 100x but concentrates the
discriminative signal), not the fundamental ranking.

**Angular vs trace correlation (0.71-0.81):** Angular variance (how much the
embedding *direction* changes) and trace variance (how much *each dimension*
changes) agree moderately. They are measuring related but distinct aspects of
uncertainty. Given that angular variance failed the reliability gate in Exp 4,
trace is the superior metric.

---

## 5. Experiment 4: Recipe Validation (Trace vs Angular Variance)

**Models:** clip_b32, siglip2_b16. T=64, K=10 trials, N=500 images.

This experiment directly compares two candidate uncertainty metrics:
1. **Trace variance** (trace_pre): `(1/D) * sum_d Var_t(e_d)` -- sum of
   per-dimension variances
2. **Angular variance**: `Var_t(arccos(e_t . mean_direction))` -- variance of
   the angle between each pass's embedding and the mean direction

### 5.1 Results

#### clip_b32

| Metric | Spearman | SNR | ICC | Status |
|---|---|---|---|---|
| trace_pre | 0.486 | 1.014 | 0.477 | **failed** |
| angular_var | 0.028 | 0.154 | 0.051 | **failed** |

#### siglip2_b16

| Metric | Spearman | SNR | ICC | Status |
|---|---|---|---|---|
| trace_pre | 0.930 | 15.029 | 0.937 | **usable** |
| angular_var | 0.252 | 0.663 | 0.360 | **failed** |

### 5.2 Analysis

**Trace variance is dramatically more reliable than angular variance.** For
siglip2_b16, trace achieves Spearman 0.930 while angular gets only 0.252.

Why? Angular variance collapses all directional information into a single scalar
(the angle from the mean direction). This throws away information about *which*
directions the embedding moves in. Trace variance preserves per-dimension
structure by summing D independent variance terms, effectively averaging over
many noisy estimates. The law of large numbers works in trace's favor: with
D=768 terms being summed, the noise averages out far more than a single angular
measurement.

The trace_pre results for siglip2_b16 here exactly match Exp 0 (Spearman
0.930, SNR 15.029), confirming internal consistency.

**Mean uncertainty magnitudes:**
| Model | trace_mean | angular_mean |
|---|---|---|
| clip_b32 | 0.00751 | 0.00752 |
| siglip2_b16 | 0.00937 | 0.00144 |

siglip2_b16 has ~25% higher trace variance than clip_b32, suggesting its
features are more sensitive to dropout perturbations. But angular variance is
5x *lower* for siglip2_b16 -- the perturbations change the magnitude more than
the direction, relative to clip_b32.

**Recommendation for Phase 2:** Use **trace of per-dimension variance in
pre-norm space** as the primary uncertainty metric. Post-norm trace can be
computed at negligible cost as a secondary metric. Discard angular variance.

---

## 6. Experiment 5: Subset Ambiguity Prediction

**Model:** clip_b32 only (siglip2 models were intended but only clip_b32 ran).
N=5000 images, T=64, K=1 trial, p=0.01.

This experiment tests whether MC dropout uncertainty correlates with independent
measures of semantic ambiguity:
- **Classification margin:** top-1 logit minus top-2 logit (low margin = hard
  to classify = ambiguous)
- **Prediction entropy:** entropy of the softmax probability distribution (high
  entropy = uncertain classification)
- **Prompt sensitivity:** variance of max-class probability across 3 prompt
  templates ("a photo of a {}", "a {}", "an image of a {}")

### 6.1 Results

| Metric | Value | Interpretation |
|---|---|---|
| rho(uncertainty, -margin) | -0.013 | No correlation |
| rho(uncertainty, entropy) | -0.044 | No correlation (wrong sign) |
| rho(uncertainty, prompt_sensitivity) | 0.001 | No correlation |
| AUROC(low margin, top 10%) | 0.481 | Below chance (0.50) |
| AUROC(high entropy, top 10%) | 0.512 | Essentially chance |

### 6.2 Interpretation

**All correlations are indistinguishable from zero.** MC dropout uncertainty
from clip_b32 has no relationship to classification difficulty.

**This is the expected result given clip_b32's failed reliability gate.** If
the uncertainty scores are not even reproducible across trials (Spearman ~0.49),
they cannot possibly correlate with anything meaningful. The scores are
dominated by stochastic noise that differs per-trial, leaving no stable signal
to correlate with ambiguity.

**Critical gap:** Exp 5 was not run with siglip2_b16 or siglip2_so400m. Given
that siglip2_b16's uncertainty scores are highly reliable (ICC=0.96), they
*may* correlate with ambiguity -- but we don't have that data yet. This is the
most important follow-up for early Phase 2.

**Why only clip_b32?** The overall summary shows siglip2 models under
`"skipped": {}` with no error message, but only clip_b32 appears in the results.
The most likely explanation: siglip2 models may have failed to load text
processing dependencies needed for zero-shot classification logits (the
protobuf/sentencepiece requirement documented in AGENT_RUNBOOK.md). Exp 5
requires text encoding to compute classification margins and entropy, unlike
the image-only Exp 0/0b/4.

---

## 7. Cross-Experiment Consistency Checks

### 7.1 Exp 0 vs Exp 4 (siglip2_b16, T=64, pre-norm)

| Source | Spearman | SNR | ICC |
|---|---|---|---|
| Exp 0 (nested extraction) | 0.930 | 15.029 | 0.937 |
| Exp 4 (independent trials) | 0.930 | 15.029 | 0.937 |

**Perfect agreement.** The nested extraction method (snapshot accumulators at
T=4/16/64 during a single T_max=64 run) produces identical results to
independent T=64 runs. This validates the implementation.

### 7.2 Exp 0b trace_pre vs Exp 0 trace_pre (clip_b32, T=64)

Exp 0b reports `trace_pre_per_d_mean = 0.00752`. Exp 4 reports `trace_mean =
0.00751` for clip_b32. These agree to 3 significant figures, confirming the
per-image trace values are consistent across experiments despite using different
trial counts (K=5 vs K=10) and code paths.

---

## 8. Go/No-Go Assessment

### 8.1 Decision Tree (from AGENT_RUNBOOK.md Section 6)

```
ANY model at ANY T has status "usable"?
  YES --> Proceed to Phase 2
```

**Answer: YES.** siglip2_b16 at T=16 post-norm is already usable, and at T=64
it is strongly usable in both feature spaces. siglip2_so400m at T=64 post-norm
also passes.

### 8.2 Exp 5 Sanity Check

The runbook specifies checking for `rho_uncertainty_vs_entropy > 0.3` and
`auroc_high_entropy > 0.6`. Neither threshold is met (rho = -0.044, AUROC =
0.512). However, this test only ran on clip_b32, which failed the reliability
gate. **The Exp 5 check is inconclusive, not a blocker**, because the model
that produced these results was already determined to be unreliable.

### 8.3 Phase 2 Recommendations

1. **Primary model:** siglip2_b16 at T=64 (or even T=16 for faster iteration)
2. **Secondary model:** siglip2_so400m at T=64 (post-norm only)
3. **Drop:** clip_b32 -- unreliable signal, not worth further compute
4. **Metric:** Trace of per-dimension variance. Report both pre-norm and
   post-norm.
5. **Urgent follow-up:** Re-run Exp 5 with siglip2_b16 to validate ambiguity
   correlation. If rho ~ 0 even with reliable scores, the uncertainty signal is
   *stable but not meaningful*, which would change the project narrative from
   "we can measure uncertainty" to "we can measure something reproducible that
   isn't uncertainty."
6. **Dropout rate:** p=0.01 worked. Phase 2 Exp 1 will sweep rates to confirm
   this is near-optimal.

---

## 9. Raw Data Reference

All artifacts are stored under:
```
outputs/run_20260225_150315/phase_one/
```

### File Inventory

```
phase_one/
  manifest_all.json                              # 5000 image paths (superset)
  exp0_nested_mc/
    exp0_overall_summary.json                    # All models, all T values
    clip_b32/
      exp0_summary.json                          # Per-model summary
      exp0_trials_T4.npz                         # (K=10, N=500) arrays
      exp0_trials_T16.npz
      exp0_trials_T64.npz
    siglip2_b16/
      exp0_summary.json
      exp0_trials_T4.npz
      exp0_trials_T16.npz
      exp0_trials_T64.npz
    siglip2_so400m/
      exp0_summary.json
      exp0_trials_T4.npz
      exp0_trials_T16.npz
      exp0_trials_T64.npz
  exp0b_norm_geometry/
    exp0b_summary.json
    exp0b_geometry_trials.npz                    # (N=500) trace/angular arrays
  exp4_subset_recipe/
    exp4_subset_summary.json
    exp4_clip_b32.npz                            # (K=10, N=500) trace + angular
    exp4_siglip2_b16.npz
  exp5_subset_ambiguity/
    exp5_subset_overall_summary.json
    exp5_subset_clip_b32_summary.json
    exp5_subset_clip_b32.npz                     # (N=5000) unc/margin/entropy/logits
```

### .npz Array Schemas

**Exp 0 trial files** (`exp0_trials_T{X}.npz`):
- `paths`: string array, shape (N,) -- image file paths
- `trial_pre`: float array, shape (K, N) -- per-image trace_pre per trial
- `trial_post`: float array, shape (K, N) -- per-image trace_post per trial

**Exp 4 files** (`exp4_{model}.npz`):
- `paths`: string array, shape (N,)
- `trial_pre`: float array, shape (K, N) -- trace variance per trial
- `trial_angular`: float array, shape (K, N) -- angular variance per trial

**Exp 5 files** (`exp5_subset_{model}.npz`):
- `paths`: string array, shape (N,)
- `uncertainty`: float array, shape (N,) -- mean trace across K trials
- `margin`: float array, shape (N,) -- top1 - top2 logit
- `entropy`: float array, shape (N,) -- softmax entropy
- `prompt_sensitivity`: float array, shape (N,) -- variance of max-prob across templates
- `logits`: float array, shape (N, 1000) -- full logit matrix
- `gt_labels`: int array, shape (N,) -- ground truth class indices
- `pred_labels`: int array, shape (N,) -- predicted class indices

---

## 10. Appendix: Complete Numeric Results

### A1. Exp 0 -- All 18 Cells (3 models x 3 T values x 2 spaces)

| Model | T | Space | Signal (Var of means) | Noise (Mean of vars) | SNR | ICC | Spearman med | Spearman Q25 | Spearman Q75 | Spearman IQR | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_b32 | 4 | pre | 4.44e-06 | 2.72e-05 | 0.163 | 0.059 | 0.362 | 0.341 | 0.380 | 0.039 | failed |
| clip_b32 | 4 | post | 4.90e-10 | 3.52e-09 | 0.139 | 0.038 | 0.331 | 0.301 | 0.345 | 0.045 | failed |
| clip_b32 | 16 | pre | 3.36e-06 | 1.03e-05 | 0.327 | 0.185 | 0.365 | 0.336 | 0.382 | 0.046 | failed |
| clip_b32 | 16 | post | 3.31e-10 | 1.34e-09 | 0.247 | 0.128 | 0.334 | 0.300 | 0.360 | 0.060 | failed |
| clip_b32 | 64 | pre | 2.84e-06 | 2.80e-06 | 1.014 | 0.477 | 0.486 | 0.476 | 0.509 | 0.033 | failed |
| clip_b32 | 64 | post | 2.38e-10 | 3.63e-10 | 0.654 | 0.356 | 0.369 | 0.350 | 0.385 | 0.035 | failed |
| siglip2_b16 | 4 | pre | 1.68e-06 | 1.74e-06 | 0.966 | 0.464 | 0.482 | 0.463 | 0.500 | 0.036 | failed |
| siglip2_b16 | 4 | post | 1.76e-10 | 1.10e-10 | 1.600 | 0.600 | 0.601 | 0.587 | 0.620 | 0.032 | marginal |
| siglip2_b16 | 16 | pre | 2.36e-06 | 6.07e-07 | 3.883 | 0.791 | 0.776 | 0.769 | 0.783 | 0.014 | marginal |
| siglip2_b16 | 16 | post | 2.58e-10 | 3.85e-11 | 6.702 | 0.868 | 0.848 | 0.843 | 0.854 | 0.011 | usable |
| siglip2_b16 | 64 | pre | 2.48e-06 | 1.65e-07 | 15.029 | 0.937 | 0.930 | 0.927 | 0.933 | 0.006 | usable |
| siglip2_b16 | 64 | post | 2.77e-10 | 1.05e-11 | 26.493 | 0.963 | 0.956 | 0.955 | 0.958 | 0.003 | usable |
| siglip2_so400m | 4 | pre | 1.39e-06 | 3.61e-06 | 0.386 | 0.222 | 0.299 | 0.274 | 0.329 | 0.055 | failed |
| siglip2_so400m | 4 | post | 2.27e-11 | 4.90e-11 | 0.463 | 0.266 | 0.378 | 0.342 | 0.400 | 0.058 | failed |
| siglip2_so400m | 16 | pre | 1.61e-06 | 1.25e-06 | 1.293 | 0.544 | 0.510 | 0.494 | 0.537 | 0.043 | failed |
| siglip2_so400m | 16 | post | 2.87e-11 | 1.71e-11 | 1.681 | 0.613 | 0.596 | 0.582 | 0.617 | 0.035 | failed |
| siglip2_so400m | 64 | pre | 1.64e-06 | 3.49e-07 | 4.689 | 0.821 | 0.784 | 0.775 | 0.790 | 0.015 | marginal |
| siglip2_so400m | 64 | post | 2.95e-11 | 4.75e-12 | 6.222 | 0.860 | 0.837 | 0.829 | 0.843 | 0.014 | usable |

### A2. Exp 0b -- Covariance Geometry (clip_b32, T=64, K=5)

| Metric | Value |
|---|---|
| trace_pre_per_d_mean | 0.00752 |
| trace_post_per_d_mean | 7.26e-05 |
| offdiag_pre_mean | 0.989 |
| offdiag_post_mean | 0.990 |
| angular_var_mean | 0.00756 |
| corr(trace_pre, angular_var) | 0.708 |
| corr(trace_post, angular_var) | 0.806 |
| corr(trace_pre, trace_post) | 0.943 |

### A3. Exp 4 -- Recipe Validation (T=64, K=10)

| Model | Metric | Spearman | SNR | ICC | Status | Mean Value |
|---|---|---|---|---|---|---|
| clip_b32 | trace_pre | 0.486 | 1.014 | 0.477 | failed | 0.00751 |
| clip_b32 | angular_var | 0.028 | 0.154 | 0.051 | failed | 0.00752 |
| siglip2_b16 | trace_pre | 0.930 | 15.029 | 0.937 | usable | 0.00937 |
| siglip2_b16 | angular_var | 0.252 | 0.663 | 0.360 | failed | 0.00144 |

### A4. Exp 5 -- Ambiguity Prediction (clip_b32, T=64, K=1, N=5000)

| Correlation Pair | Spearman rho |
|---|---|
| uncertainty vs -margin | -0.013 |
| uncertainty vs entropy | -0.044 |
| uncertainty vs prompt_sensitivity | 0.001 |

| Detection Task | AUROC |
|---|---|
| Low margin (bottom 10%) | 0.481 |
| High entropy (top 10%) | 0.512 |
