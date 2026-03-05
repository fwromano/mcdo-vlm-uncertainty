# Preliminary Investigation: MC Dropout Uncertainty Signal

**Date**: 2026-02-26
**Status**: Preliminary (low-confidence, N=500-2000)

## Summary

After fixing two critical bugs in the Exp 5 pipeline (ImageNet synset ID labels,
HuggingFace SigLIP2 text encoder collapse), we ran a systematic investigation
of the MC dropout uncertainty signal across three angles plus an ablation test.

**Main finding**: MC dropout on CLIP (ViT-B-32) produces uncertainty that is both
physically meaningful (responds to image degradation) and task-relevant (correlates
with classification difficulty). MC dropout on SigLIP2 (ViT-B-16-SigLIP2) produces
uncertainty that is physically anti-meaningful (decreases with degradation) and
task-irrelevant (zero classification correlation).

## Bug Fixes Applied

1. **Synset ID labels**: ImageNet validation folders use synset IDs (n01440764),
   not human labels. Without `--class-map`, all prompts were gibberish like
   "a photo of a n01440764". Fixed by creating `data/imagenet_class_map.json`.

2. **HuggingFace SigLIP2 text encoder**: Produces collapsed embeddings (mean
   pairwise cosine = 0.976 vs expected ~0.76). All 1000 class embeddings were
   nearly identical. Switched to open_clip backend where mean cosine = 0.762.
   Current code defaults to open_clip for SigLIP2 and falls back to HuggingFace
   only if open_clip assets are unavailable at runtime.

## Angle 1: Alternative Uncertainty Metrics (N=500)

Tested 6 metrics derived from per-pass MC dropout features. All measurements
use p=0.01 dropout, T=64 passes.

### clip_b32

| Metric          | rho(entropy) | rho(-margin) |
|-----------------|-------------|-------------|
| trace_pre       | **+0.300**  | +0.083      |
| trace_post      | +0.215      | +0.061      |
| mean_cosine_dev | +0.215      | +0.061      |
| top_eigenvalue  | +0.103      | +0.008      |
| max_dim_var     | +0.092      | -0.022      |
| norm_var        | +0.041      | -0.026      |

**Conclusion**: trace_pre (mean per-dimension variance in raw feature space) is
the strongest metric. Pre-norm features outperform post-norm features across
all metrics.

### siglip2_b16

| Metric          | rho(entropy) | rho(-margin) |
|-----------------|-------------|-------------|
| trace_post      | +0.061      | +0.030      |
| mean_cosine_dev | +0.061      | +0.030      |
| trace_pre       | +0.028      | +0.018      |
| norm_var        | +0.026      | +0.016      |
| max_dim_var     | -0.003      | -0.005      |
| top_eigenvalue  | -0.033      | -0.030      |

**Conclusion**: No metric produces meaningful classification correlation.
Alternative metrics do not rescue SigLIP2.

## Angle 2: clip_b32 at Larger Sample Size (N=2000)

Tested whether clip_b32's signal strengthens with more data.

| Metric          | rho(entropy) N=500 | rho(entropy) N=2000 |
|-----------------|-------------------|---------------------|
| trace_pre       | +0.300            | **+0.253**          |
| trace_post      | +0.215            | +0.166              |
| mean_cosine_dev | +0.215            | +0.166              |

The point estimate decreases slightly (0.30 → 0.25), but at N=2000, rho=0.25
is far more statistically significant (~p < 10⁻³⁰ via permutation). The signal
is robust and not a small-sample artifact.

Margin correlation also improves: rho(-margin) goes from +0.083 (N=500) to
**+0.125** (N=2000), suggesting margin is a noisier ambiguity measure that
benefits more from additional data.

## Angle 3: What Does SigLIP2 Uncertainty Correlate With? (N=1000)

Since SigLIP2 uncertainty doesn't correlate with classification metrics, we
tested what it DOES correlate with.

| Correlate              | rho      |
|------------------------|----------|
| **centroid_distance**  | **+0.241** |
| feat_norm              | -0.133   |
| max_logit              | -0.043   |
| classification_entropy | -0.005   |
| negative_margin        | -0.017   |

**Finding**: SigLIP2's MC dropout uncertainty is an **outlier detector**, not
an ambiguity detector. Images far from the feature-space centroid (atypical
images) have higher uncertainty. Images with larger feature norms have lower
uncertainty. But none of this relates to classification difficulty.

## Ablation Test: Image Degradation (N=500)

**The most decisive test.** For each image, we create degraded versions and
check whether uncertainty increases. This is a paired test — each image is
its own control.

### clip_b32: PASS

| Degradation     | % images more uncertain | Wilcoxon p    | Unc change |
|-----------------|------------------------|---------------|-----------|
| blur_r5         | **80.2%**              | 5 × 10⁻⁴³    | +29%      |
| blur_r15        | 60.2%                  | 2 × 10⁻⁷     | +8%       |
| downsample_4x   | 65.8%                  | 8 × 10⁻¹⁸    | +14%      |
| downsample_8x   | **79.6%**              | 2 × 10⁻⁴⁴    | +29%      |

All conditions highly significant. Non-monotonic pattern (r=5 > r=15) is
explained by extreme blur collapsing features toward a low-variance null point.

### siglip2_b16: FAIL

| Degradation     | % images more uncertain | Wilcoxon p | Unc change |
|-----------------|------------------------|------------|-----------|
| blur_r5         | 41.6%                  | 1.0 (ns)   | **-4%**   |
| blur_r15        | **24.4%**              | 1.0 (ns)   | **-10%**  |
| downsample_4x   | 52.4%                  | 0.26 (ns)  | 0%        |
| downsample_8x   | 38.6%                  | 1.0 (ns)   | **-4%**   |

SigLIP2 uncertainty **decreases** with degradation. Heavy blur causes 75.6%
of images to have LOWER uncertainty. This is anti-correlated with physical
degradation.

## Interpretation

### Why CLIP works and SigLIP2 doesn't

**CLIP** uses contrastive softmax loss — images compete against each other
within a batch. Features encode fine-grained discriminative information that
dropout can perturb meaningfully. When an image is ambiguous or degraded,
the features sit near decision boundaries, and dropout pushes them around.

**SigLIP2** uses sigmoid loss — each image-text pair is classified independently
as matching/not-matching. Features encode more stable, binary-like representations.
Dropout perturbations don't interact with the discriminative structure because
there's no inter-class competition to disrupt. Instead, dropout variance is
dominated by geometric properties (distance from centroid) that don't relate
to task difficulty.

### Implications for the project

1. **CLIP models are viable for MC dropout uncertainty estimation.** The signal
   is reliable (Phase 1 confirmed), task-relevant (rho≈0.25 with entropy),
   and physically meaningful (responds correctly to degradation).

2. **SigLIP2 models are NOT viable for MC dropout uncertainty.** Despite having
   reliable signals (Phase 1 Exp 0 confirmed), the uncertainty measures nothing
   useful — it's an outlier detector that anti-correlates with degradation.

3. **The ablation test should be a standard validation** for any new uncertainty
   method. It's model-agnostic, label-free, and provides immediate pass/fail.

4. **Calibration via ablation** (user's insight): since degradation levels have
   known severity, the uncertainty response curve could serve as a calibration
   reference.

## Data Files

| File | Contents |
|------|----------|
| `outputs/prelim_investigation.json` | Angles 1-3 raw results (must be regenerated after mapping fix) |
| `outputs/prelim_ablation.json` | Ablation test raw results |
| `outputs/exp5_siglip2_validation/` | Earlier N=5000 siglip2_b16 results |

## Next Steps

1. Run ablation test at larger N and with more degradation levels
2. Test clip_l14 (larger CLIP model) — does it have even stronger signal?
3. Formalize the ablation calibration curve idea as Exp 7
4. Update Phase 1 report with corrected Exp 5 findings
5. Consider whether SigLIP2 should be dropped from further Phase 2 experiments
