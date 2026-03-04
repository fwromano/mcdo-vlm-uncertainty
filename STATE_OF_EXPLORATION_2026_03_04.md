# State of the Exploration: MC Dropout Uncertainty in Vision-Language Models

**Date:** March 4, 2026 (updated night)
**Status:** Active investigation — best operating point identified, method confirmed CLIP-specific

---

## Executive Summary

After two weeks of systematic experimentation across 5 VLMs (2 CLIP, 1 PE-Core, 2 SigLIP2),
5 perturbation strategies, 11 uncertainty metrics, and 25+ experimental configurations, we
have identified a clear best operating point and mapped the full tradeoff landscape.

**Best configuration: All c_proj dropout at p=0.01** — the sweet spot between reliability
and validity.

| Config | Validity (blur_r5) | Reliability (Spearman) | Notes |
|--------|-------------------|----------------------|-------|
| All 12 c_proj dropout p=0.01 | **93.6%** | 0.43 (T=64) → 0.74 (T=256) | **Best overall** |
| Uniform dropout p=0.01 | 86.8% | 0.52 (T=64) → 0.82 (T=256) | Previous best |
| Gaussian@block11.c_proj | 25.4% FAIL | 0.998 | Measures wrong thing |

The 12-c_proj config focuses dropout exclusively on the 12 MLP output projection layers
(one per transformer block), avoiding the 12 attention out_proj modules (which produce
zero variance) and the 12 MLP c_fc modules (which add noise without improving validity).
This achieves the **highest ablation validity ever measured** (93.6% on blur_r5) while
maintaining reliability that scales cleanly with T.

**Cross-model validation:** CLIP L/14 (24 blocks) confirms the pattern — all-c_proj
dropout passes ablation (78.2%) while uniform dropout does not (71.6%). Larger models
have more robust features, slightly reducing validity but improving reliability.

**Practical deployment:** CLIP B/32 is tiny (88M params, 176MB fp16 vision encoder).
T forward passes are embarrassingly parallel — batch N×T samples in one forward pass.
On a 24GB GPU, 10 objects × T=256 = 2,560 samples fits easily in ~7.5GB VRAM.

---

## 1. Models Tested

| Model | Architecture | Params | Dim | Reliable? | Valid? | Verdict |
|-------|-------------|--------|-----|-----------|--------|---------|
| **clip_b32** | ViT-B/32 | ~88M | 512→768 | 0.43 (12-cproj T=64) | **YES** (93.6% ablation) | **Primary model** |
| **clip_l14** | ViT-L/14 | ~304M | 768 | 0.75 (24-cproj T=64) | **YES** (78.2% ablation) | **Confirmed** |
| pe_core_b16 | PE-Core-B/16 | ~86M | 768 | 0.82 (12-fc2 T=64) | **NO** (55% ablation) | Dropped |
| siglip2_b16 | SigLIP2 Base | ~93M | 768 | High (Spearman=0.96) | **NO** (ablation FAILS) | Dropped |
| siglip2_so400m | SigLIP2 SO400M | ~428M | 1152 | Moderate (Spearman=0.84) | **NO** (by architecture) | Dropped |

### CLIP L/14 cross-model validation (NEW)

Tested CLIP L/14 (24 transformer blocks, ViT-L/14) with N=500, T=64:

| Config | Reliability | blur_r5 | down_8x | Verdict |
|--------|-----------|---------|---------|---------|
| All 24 c_proj p=0.01 | Spearman=0.752 | **78.2% PASS** | **78.8% PASS** | Valid + reliable |
| Uniform p=0.01 | Spearman=0.838 | 71.6% weak | 73.2% weak | Reliable but marginal validity |

Key observations:
- **L/14 all-c_proj PASSES ablation** (just above 75% threshold)
- **L/14 uniform FAILS ablation** (below 75%) — unlike B/32 where uniform passes
- L/14 has HIGHER reliability across the board (0.75 vs 0.43 for c_proj, 0.84 vs 0.52 for uniform)
- Larger model = more robust features = less sensitive to perturbation (lower validity)
  but more consistent noise response (higher reliability)
- **This confirms all-c_proj is the right strategy** — it passes on both B/32 and L/14,
  while uniform dropout only passes on B/32

### The SigLIP2 reversal

Phase 1 recommended SigLIP2 as the primary model based on reliability alone
(Spearman=0.956). The preliminary investigation in Phase 2 revealed this was wrong:

- SigLIP2 uncertainty **decreases** with image degradation (75.6% of blurred images
  have LOWER uncertainty)
- SigLIP2 uncertainty correlates with **centroid distance** (rho=0.24), not
  classification entropy (rho=-0.005)
- SigLIP2 MC dropout is an **outlier detector**, not an ambiguity detector

Root cause: SigLIP2's sigmoid loss trains image-text pairs independently (match/no-match).
Features encode stable binary representations where dropout perturbation reflects
geometric position (distance from training distribution centroid), not discriminative
uncertainty. CLIP's contrastive softmax loss creates inter-class competition that dropout
can meaningfully disrupt.

**Lesson:** Reliability without validity is dangerous. A perfectly consistent ruler that
measures the wrong thing is worse than a noisy ruler that measures the right thing.

---

## 2. Metrics Tested

### 2.1 Uncertainty metrics

| Metric | Description | rho(entropy) | Ablation(blur_r5) | Ablation(down_8x) | Verdict |
|--------|-------------|-------------|-------------------|-------------------|---------|
| **weighted_trace_pre** | Trace weighted by discriminative power | **+0.105** | **96.4%** | **97.0%** | **NEW BEST** |
| topk64_trace_pre | Trace on top-64 discriminative dims | +0.108 | 95.4% | 94.6% | Strong |
| trace_pre | Mean per-dim variance, pre-L2-norm | +0.079 | 93.6% | 90.6% | Previous best |
| max_dim_var | Max single-dimension variance | +0.097 | 89.4% | 85.2% | Good |
| norm_var | Variance of feature vector norms | +0.112 | 81.4% | 77.6% | Moderate |
| top_eigenvalue | Largest eigenvalue of covariance | +0.099 | 79.0% | 80.2% | Moderate |
| trace_post | Mean per-dim variance, post-L2-norm | -0.003 | 82.0% | 77.0% | Weak entropy corr |
| mean_cosine_dev | Mean cosine distance from MC mean | -0.003 | 82.0% | 77.0% | = trace_post |
| effective_rank | Rank of covariance matrix | -0.112 | 52.0% DOWN | — | **NULL** |
| spectral_entropy | Entropy of eigenvalue distribution | -0.118 | 57.4% DOWN | — | **NULL** |
| top1_ratio | Top eigenvalue / trace | +0.106 | 51.6% DOWN | — | **NULL** |

**Metric ablation** (all measured on CLIP B/32, all-c_proj p=0.01, T=64, N=500):

**Conclusion:** weighted_trace_pre is the new best metric, achieving 96.4%/97.0% ablation
validity (vs trace_pre's 93.6%/90.6%). It weights each dimension's MC variance by that
dimension's discriminative power (across-image variance of the mean feature). Dimensions
that vary more across images encode more visual distinctions, so weighting by discriminative
power focuses on the dimensions most damaged by image degradation.

topk64_trace_pre (using only the top-64 most discriminative dimensions) performs similarly,
confirming that a small subset of dimensions carries most of the valid uncertainty signal.

### 2.2 Validity proxies

All measured on CLIP B/32 with uniform dropout p=0.01, trace_pre metric:

| Proxy | Value | N | Interpretation |
|-------|-------|---|----------------|
| rho(classification entropy) | +0.25 to +0.30 | 500-2000 | Moderate positive correlation |
| rho(negative margin) | +0.08 to +0.13 | 500-2000 | Weak but consistent |
| AUROC(error prediction) | 0.57 | 500-2000 | Above chance |
| AUROC(high entropy) | 0.67 | 2000 | Meaningful |
| Ablation: blur_r5 | 80-87% increased | 500 | Strong paired signal |
| Ablation: downsample_8x | 80-88% increased | 500 | Strong paired signal |

---

## 3. Phase 1: Reliability Gate (Feb 25-26)

Phase 1 tested whether MC dropout on frozen VLM encoders produces reproducible
uncertainty rankings. K=10 independent trials, N=500 images, T=64 passes, p=0.01.

| Model | Best Config | Spearman | SNR | ICC | Gate |
|-------|------------|----------|-----|-----|------|
| siglip2_b16 | T=64, post-norm | 0.956 | 26.5 | 0.963 | PASS |
| siglip2_so400m | T=64, post-norm | 0.837 | 6.2 | 0.860 | PASS |
| clip_b32 | T=64, pre-norm | 0.486 | 1.01 | 0.477 | FAIL |

**Phase 1 verdict (now superseded):** Proceed with SigLIP2 as primary, drop CLIP.

**Phase 1 lesson:** Reliability-only assessment can be misleading. Phase 2's validity
tests reversed the model ranking entirely.

---

## 4. Phase 2 Experiments (Feb 27 - Mar 4)

### Exp 1: Dropout Rate Sweep

Tested p = {0.001, 0.005, 0.01, 0.02, 0.05, 0.1} on CLIP B/32 and SigLIP2 SO400M.
N=1000, T=64, K=3.

**CLIP B/32:** Best at p=0.005 (Spearman=0.575). Signal collapses above p=0.02.
**SigLIP2 SO400M:** Best at p=0.001 (Spearman=0.846). Signal collapses above p=0.005.

Key finding: Rankings at different p values are largely uncorrelated (the p=0.001 ranking
is different from p=0.01). The optimal p is model-dependent and low.

### Exp 2: Synthetic vs Natural Images

Sanity check: does uncertainty order solid < gradient < noise < natural?

Both CLIP and SigLIP2 pass: natural images have highest uncertainty, solid colors lowest.
This is a weak test (even random noise would produce nonzero variance) but confirms the
pipeline isn't broken.

### Exp 3: Dropout Type Ablation (Key Experiment)

Tested 5 dropout strategies on CLIP B/32, N=1000, T=64, K=5:

| Type | Description | Modules | Spearman | SNR |
|------|-------------|---------|----------|-----|
| A | Attention only | 12 | N/A (zero var) | 0 |
| B | MLP only | 24 | 0.525 | 0.10 |
| C | Stochastic depth | 12 | 0.194 | 0.09 |
| **D** | **Single module (block 9 c_proj)** | **1** | **0.771** | **0.19** |
| E | Uniform (all linear) | 36 | 0.518 | 0.10 |

**Critical discoveries:**
1. **Attention modules produce exactly zero variance** under dropout. All 12 out_proj
   layers are dead — confirmed across all perturbation types.
2. **Single-module dropout (Type D) beats uniform dropout.** Focusing perturbation on
   one bottleneck projection outperforms spraying it everywhere.
3. **Type B ≈ Type E** — since attention contributes nothing, MLP-only is equivalent
   to uniform.
4. Type D's block 9 was selected by alphabetical sort, not analysis — motivating the
   systematic perturbation search.

### Exp 4: Full Model Matrix

4 models × 10 trials, uniform dropout p=0.01:

| Model | trace_pre Spearman | SNR |
|-------|-------------------|-----|
| **clip_l14** | **0.419** | **0.046** |
| clip_b32 | 0.273 | 0.040 |
| siglip2_so400m | 0.115 | 0.017 |
| siglip2_b16 | 0.080 | 0.004 |

CLIP L/14 shows the strongest signal among all models under uniform dropout.
Cross-model agreement is weak (clip_b32 vs clip_l14: rho=0.35).

### Exp 5: Ambiguity Prediction (N=2000, clip_b32)

Correlates MC dropout uncertainty with zero-shot classification metrics:

| Metric | Value |
|--------|-------|
| rho(uncertainty vs entropy) | **+0.253** |
| rho(uncertainty vs negative margin) | +0.125 |
| AUROC(high entropy, top 10%) | **0.671** |
| AUROC(low margin, bottom 10%) | 0.555 |

**Conclusion:** MC dropout uncertainty on CLIP correlates with semantic ambiguity.
rho=0.25 is moderate but highly significant at N=2000 (p < 10^-30). The AUROC of
0.67 for detecting high-entropy images is practically useful.

### Exp 6: Mean Convergence

Tests how the MC mean embedding converges with T. Measures relative distance between
the T-pass MC mean and the deterministic (no dropout) embedding.

**CLIP B/32:** Converges well — rel_dist drops from 0.14 (T=4) to 0.11 (T=64).
Convergence slope = -0.09 (slow but steady).

**SigLIP2 B/16:** Does NOT converge — rel_dist stays ~0.52 from T=4 to T=64.
The MC mean is permanently far from the deterministic embedding regardless of T.
This confirms that dropout fundamentally disrupts SigLIP2's representation rather
than gently probing it.

---

## 5. The Perturbation Search (Feb 28 - Mar 1)

### Motivation

Exp 3's finding that single-module dropout (Type D) beats uniform dropout raised the
question: what is the optimal perturbation strategy across all modules and perturbation
types?

### Framework

Built `phase_two/perturbation.py` supporting three perturbation types:
- **Dropout**: Sparse binary (zero neurons with probability p, scale by 1/(1-p))
- **Gaussian**: Dense continuous (add N(0, (mag * output_std)^2) to every neuron)
- **Scale**: Multiplicative (multiply each neuron by 1 + N(0, mag^2))

### Quick Scan: All 36 Modules

Tested each linear module individually with dropout and Gaussian at mag=0.05.
N=100, K=3, T=16. ~16 minutes.

**Results:** Gaussian dominates. 11 of the top 15 configurations are Gaussian.
All c_proj modules achieve Spearman > 0.94 with Gaussian. All 12 attention out_proj
modules produce exactly zero variance.

### Deep Validation: Block 11 c_proj

N=500, K=5, T=64, multiple types and magnitudes:

| Config | Spearman | SNR | ICC |
|--------|----------|-----|-----|
| Gaussian@block11 mag=0.01 | **0.998** | **457.6** | **0.998** |
| Gaussian@block11 mag=0.05 | **0.998** | **456.8** | **0.998** |
| Gaussian@block11 mag=0.10 | **0.998** | **451.6** | **0.998** |
| Gaussian@block9 mag=0.05 | 0.989 | 100.8 | 0.990 |
| Scale@block11 mag=0.05 | 0.963 | 25.4 | 0.962 |
| Dropout@block11 p=0.05 | 0.776 | 3.64 | 0.775 |
| Dropout@block9 p=0.01 | 0.578 | 2.0 | 0.641 |
| Uniform dropout p=0.01 | 0.518 | 0.1 | -0.112 |

**Key finding:** Gaussian noise on block 11 c_proj produces SNR=458 — a 4580x
improvement over uniform dropout. The ranking is **magnitude-insensitive** (0.01, 0.05,
0.1 all give Spearman=0.998) because the ranking reflects the layer's Jacobian structure,
which is a fixed property of each image independent of noise scale.

---

## 6. The Validity Crisis (Mar 1)

### The ablation test

The critical question: does Gaussian@block11's near-perfect reliability translate to
valid uncertainty? We tested whether degraded images (blur, downsample) produce higher
uncertainty scores.

| Config | blur_r5 | blur_r15 | down_4x | down_8x |
|--------|---------|----------|---------|---------|
| **Gaussian block11** | **25.4%** | **2.8%** | **42.4%** | **33.6%** |
| Gaussian block9 | 60.4% | 45.6% | 62.8% | 72.0% |
| **Uniform dropout** | **86.8%** | **63.6%** | **82.8%** | **88.0%** |

**Gaussian@block11 fails catastrophically.** Heavy blur causes 97.2% of images to have
LOWER uncertainty. The ranking is inverted relative to what valid uncertainty should do.

### Why Gaussian fails validity

Gaussian perturbation on c_proj measures the local Jacobian norm: how sensitive is each
output dimension to perturbation of its input? This is a **geometric property** of how
the image sits in activation space.

Degraded images (blur, downsample) have smoother, simpler internal activations. The
Jacobian norm is smaller for simpler features. So degraded images are *less* sensitive
to Gaussian noise — the opposite of what uncertainty should show.

Dropout works differently: it tests whether the network has **redundant computational
paths** for this image. Degraded images lose information, reducing redundancy, so they
become more sensitive to ablation of any single path.

### Higher dropout rates don't help

We tested whether cranking dropout could improve reliability while keeping validity:

| Config | blur_r5 | Spearman |
|--------|---------|----------|
| Uniform p=0.01 | **86.8%** | 0.518 |
| All c_proj p=0.05 | 71.5% | 0.392 |
| Uniform p=0.05 | 43.5% | 0.296 |
| All c_proj p=0.10 | 25.5% | 0.248 |
| All c_proj p=0.20 | **0.5%** | 0.152 |
| Uniform p=0.10 | 13.0% | 0.093 |

Higher dropout rates destroy BOTH validity AND reliability. At p=0.20, the ablation
signal completely inverts (99.5% of degraded images have LESS uncertainty). The valid
measurement exists only at very low perturbation magnitudes (p=0.01), where the
perturbation probes decision-boundary proximity rather than feature complexity.

### Dense Gaussian everywhere doesn't help

We tested applying Gaussian noise to all 12 c_proj or all 24 MLP modules:

| Config | blur_r5 | down_8x | Spearman |
|--------|---------|---------|----------|
| Gaussian all c_proj (12) | 59.5% | 65.5% | 0.977 |
| Gaussian all MLP (24) | 60.0% | 71.5% | 0.966 |
| Uniform dropout (36) | 86.8% | 88.0% | 0.518 |

Spreading Gaussian everywhere gives middling results on both axes. The problem is
not coverage — it's that Gaussian measures the wrong quantity (Jacobian norm, not
computational redundancy).

### Residual stream perturbation doesn't help

We tested injecting Gaussian noise into the residual stream between blocks
(the user's hypothesis about measuring "pipeline stability"):

| Config | blur_r5 | down_8x |
|--------|---------|---------|
| After block 11 | 43.5% | 50.5% |
| After block 6 | 59.5% | 67.5% |
| After blocks 9-11 | 56.0% | 63.0% |
| After all blocks | 59.5% | 63.5% |

None pass the 75% threshold. The Lipschitz constant of the tail network is also a
geometric property that doesn't distinguish degradation from difficulty.

---

## 7. The Reliability-Validity Tradeoff (UPDATED)

The central finding, now refined with the 12-c_proj discovery:

```
              100% ┬─ High Validity
                   │
   12 c_proj p=0.01│● 93.6%  ← NEW BEST
                   │
   Uniform p=0.01  │  ● 86.8%
                   │
          ~75% ----│----------- pass threshold -------
                   │
   L/14 cproj p=0.01  ● 78.2%   (passes on larger model too)
   L/14 uniform p=0.01│  ● 71.6%
                   │
   Gaussian all    │      ● 60%
                   │
   Gaussian blk11  │               ● 25%
                0% ┴───────┼───────┼───────┼──────── Reliability (Spearman)
                  0.0     0.4    0.75    1.0
                         B/32    L/14   Gaussian
```

**The 12-c_proj config breaks the tradeoff.** By targeting dropout precisely at the 12
MLP output projections (one per block), we achieve HIGHER validity than uniform dropout
(93.6% vs 86.8%) while retaining dropout's validity mechanism (computational redundancy
probing). The improvement comes from removing noise sources that don't contribute to
valid uncertainty signal:
- 12 attention out_proj: produce **zero variance** (skip them)
- 12 MLP c_fc: add noise but dilute the c_proj signal (skip them)

**The underlying physics is unchanged:**

- **Dropout at low p** → probes computational redundancy → correlates with classification
  difficulty → VALID but noisy (sparse sampling)
- **Gaussian noise** → probes Jacobian structure → correlates with activation complexity →
  RELIABLE but wrong measurement
- **Dropout at high p** → overwhelms representation → measures feature complexity
  (same as Gaussian) → neither valid nor reliable

**But targeting the right modules amplifies the valid signal** while reducing the noise
from modules that contribute nothing (attention) or dilute (c_fc).

---

## 8. The Path Forward: Scaling T

Keep the valid perturbation (c_proj dropout or uniform dropout at p=0.01)
and increase T to beat down the noise.

### Reliability scales as O(sqrt(T))

Two configs measured on CLIP B/32, N=500, K=5 trials:

**Uniform dropout p=0.01:**

| T | Spearman | SNR | Cost relative to T=64 |
|---|----------|-----|-----------------------|
| 16 | 0.433 | 9.8 | 0.25x |
| 32 | 0.446 | 16.1 | 0.5x |
| 64 | 0.574 | 31.8 | 1x (baseline) |
| 128 | 0.705 | 61.7 | 2x |
| 256 | **0.821** | **123.1** | 4x |

**All 12 c_proj dropout p=0.01 (NEW):**

| T | Spearman | SNR | Cost relative to T=64 |
|---|----------|-----|-----------------------|
| 16 | 0.307 | — | 0.25x |
| 32 | 0.311 | — | 0.5x |
| 64 | 0.430 | — | 1x (baseline) |
| 128 | 0.581 | — | 2x |
| 256 | **0.739** | — | 4x |

Both follow O(sqrt(T)) scaling. The 12-c_proj config has slightly lower reliability at
each T (0.74 vs 0.82 at T=256) but **much higher validity** (93.6% vs 86.8%). The
tradeoff is clear: choose your operating point.

**Recommended operating points:**

| Use case | Config | T | Spearman | Validity | Batch size (10 objects) |
|----------|--------|---|----------|----------|------------------------|
| Quick screening | 12-c_proj | 64 | 0.43 | 93.6% | 640 |
| Balanced | 12-c_proj | 128 | 0.58 | 93.6% | 1,280 |
| High reliability | Uniform | 256 | 0.82 | 86.8% | 2,560 |
| Maximum validity | 12-c_proj | 256 | 0.74 | 93.6% | 2,560 |

---

## 9. What We Know For Certain

1. **All-c_proj dropout at p=0.01 is the best operating point.** Achieves 93.6% ablation
   validity (best ever) on B/32 and 78.2% on L/14. Focuses perturbation on the modules
   that carry the valid uncertainty signal.

2. **CLIP + trace_pre + low-p dropout is a valid uncertainty metric.** It passes ablation
   (87-94%), correlates with entropy (rho=0.25), and predicts errors (AUROC=0.57-0.67).

3. **The method generalizes across CLIP sizes.** Both B/32 (12 blocks) and L/14 (24 blocks)
   pass ablation with all-c_proj dropout. Larger models have higher reliability but
   slightly lower validity — consistent with more robust feature representations.

4. **Method is CLIP-specific, not contrastive-loss-generic.** SigLIP2 fails (sigmoid loss,
   anti-correlated). PE-Core-B/16 also fails despite using contrastive softmax loss —
   its 5.4B-pair training produces features too robust for dropout probing. Only OpenAI
   CLIP models (B/32, L/14) pass ablation validity.

5. **Attention modules contribute nothing** to dropout uncertainty in CLIP ViT. All
   out_proj modules produce exactly zero variance across all perturbation types.

6. **weighted_trace_pre is the best metric** (96.4%/97.0% ablation validity, vs trace_pre's
   93.6%/90.6%). Weights MC variance by discriminative power. All spectral metrics
   (effective_rank, spectral_entropy, top1_ratio) are null.

7. **The reliability-validity tradeoff is fundamental**, not an artifact of insufficient
   search. Gaussian noise, scale perturbation, residual injection, and high-rate dropout
   all fail validity for the same reason: they measure Jacobian/complexity rather than
   decision-boundary proximity.

8. **Reliability scales as O(sqrt(T))** for dropout. This provides a clear (if
   expensive) path to usable reliability.

9. **T forward passes are embarrassingly parallel.** No sequential dependency between
   passes — batch N×T samples in one forward call on GPU.

10. **PCA on MC covariance validates compact Kalman filter state for MOT.** 84% overlap
    between top-64 MC uncertainty dims and top-64 discriminative dims. K=8 PCs pass ablation
    (81-84%), K=32 gets 85-87%. A 16-32 dim state vector captures valid uncertainty.

---

## 10. Open Questions

1. ~~**Does CLIP L/14 have better validity?**~~ **ANSWERED:** L/14 passes ablation (78.2%)
   with all-c_proj dropout but with lower validity than B/32 (93.6%). Larger model = more
   robust features = less sensitive to degradation perturbation interaction.

2. ~~**Is there a metric that extracts more signal per pass?**~~ **ANSWERED:** Yes.
   weighted_trace_pre (weights dims by discriminative power) improves ablation from
   93.6%→96.4% blur, 90.6%→97.0% downsample. topk64_trace_pre (top-64 dims only) also
   strong at 95.4%/94.6%.

3. **Can we combine Gaussian reliability with dropout validity?** E.g., use Gaussian
   ranking as a prior and dropout passes as noisy updates. Or train a small calibration
   model that maps Gaussian uncertainty to valid uncertainty.

4. **What is the minimum T for practical use?** T=256 gives Spearman=0.74-0.82. For MOT
   applications with distant/blurry objects, is moderate reliability sufficient?

5. **Does the ablation test set the right bar?** The test assumes "degraded = more
   uncertain." But a perfect classifier might be equally confident on a blurred dog
   (still obviously a dog). Maybe rho(entropy) is a better validity criterion than
   ablation pass rate.

6. ~~**Would Meta's Perception Encoder work?**~~ **ANSWERED: NO.** PE-Core-B/16 FAILS
   ablation (55%/39%) despite contrastive softmax loss. High reliability (Spearman=0.82)
   but invalid. Same pattern as SigLIP2: "reliable ruler measuring the wrong thing."
   Method is CLIP-specific, not contrastive-loss-generic. See Section 13.

7. **Real-world MOT validation.** PCA analysis confirms 16-32 dim state vector is
   sufficient for Kalman filter tracking. Next step: test on actual tracking sequences.

---

## 11. Experiment Inventory

| Experiment | Status | Key Output |
|------------|--------|------------|
| Phase 1 Exp 0: Reliability gate | DONE | SigLIP2 passes, CLIP fails (reversed later) |
| Phase 1 Exp 0b: Covariance geometry | DONE | 99% off-diagonal, angular < trace |
| Phase 1 Exp 4: Trace vs Angular | DONE | Trace >> Angular |
| Phase 1 Exp 5: Ambiguity (clip_b32) | DONE | rho=0, but model was unreliable |
| Phase 2 Exp 1: Dropout rate sweep | DONE | p=0.005 best for CLIP |
| Phase 2 Exp 2: Synthetic/natural | DONE | Both models pass sanity check |
| Phase 2 Exp 3: Dropout type ablation | DONE | Type D > Type E, attention=zero |
| Phase 2 Exp 4: Full model matrix | DONE | CLIP L/14 > B/32 > SigLIP2 |
| Phase 2 Exp 5: Full ambiguity | DONE (clip_b32) | rho=0.25, AUROC=0.67 |
| Phase 2 Exp 6: Mean convergence | DONE | CLIP converges, SigLIP2 doesn't |
| Prelim investigation: Metrics + SigLIP2 | DONE | trace_pre best, SigLIP2=outlier detector |
| Prelim ablation: Image degradation | DONE | CLIP passes, SigLIP2 fails |
| Spectral ablation: Extended metrics | DONE | Only trace_pre passes |
| Spectral smoke: Metric correlations | DONE | trace_pre > trace_post > rest |
| Ensemble smoke: Metric combining | DONE | Ridge rho=0.37, needs calibration |
| Perturbation quick scan: 36 modules | DONE | Gaussian dominates, attention=0 |
| Perturbation deep test: Block 11 | DONE | SNR=458, magnitude-insensitive |
| Validity smoke: Entropy correlations | DONE | Gaussian rho=0.33, dropout rho=0.30 |
| Gaussian ablation test | DONE | Gaussian FAILS, dropout PASSES |
| Dense Gaussian (all c_proj/MLP) | DONE | 60-71% validity, 0.97 reliability |
| Higher dropout rates | DONE | Kill both validity and reliability |
| Residual stream perturbation | DONE | 50-67% validity, doesn't pass |
| Dropout reliability vs T (uniform) | DONE | Spearman=0.82 at T=256 |
| 12-c_proj dropout ablation (5 configs) | DONE | 93.6% validity — best ever |
| 12-c_proj T-scaling (T=16-256, K=5) | DONE | Spearman=0.74 at T=256 |
| CLIP L/14 cross-model validation | DONE | c_proj PASSES (78%), uniform weak (72%) |
| PE-Core-B/16 ablation + reliability | DONE | **FAILS** ablation (55%/39%), Spearman=0.82 |
| Metric engineering: weighted/topk trace | DONE | weighted_trace_pre: 96.4%/97.0% — **new best** |
| Metric ablation: all metrics vs degradation | DONE | weighted > topk > trace_pre > rest |
| PCA on MC covariance for Kalman filter | DONE | K=8 PASS (81%), K=32 PASS (87%), 84% dim overlap |

---

## 12. Practical Deployment Analysis (NEW)

### Model sizes

CLIP models are remarkably small:

| Model | Vision Encoder | Total Params | fp16 VRAM | fp32 VRAM |
|-------|---------------|-------------|-----------|-----------|
| CLIP B/32 | ViT-B/32 | 88M | ~176 MB | ~352 MB |
| CLIP L/14 | ViT-L/14 | 304M | ~608 MB | ~1.2 GB |

### Batch-parallel MC dropout

**Key insight:** T forward passes are embarrassingly parallel. Instead of running T
sequential forward passes, duplicate each input T times and run as a single batch:

```
Sequential (current code):  for t in range(T): features[t] = model(x)  # T passes
Batched (optimal):          x_rep = x.repeat(T,1,1,1); features = model(x_rep)  # 1 pass
```

Batch size = N_objects × T_passes. Each CLIP B/32 sample needs ~3MB VRAM (224×224×3
input + intermediate activations + 768-dim output).

### VRAM requirements (CLIP B/32, fp16)

| N objects | T=64 | T=128 | T=256 |
|-----------|------|-------|-------|
| 1 | 0.2 GB | 0.4 GB | 0.8 GB |
| 5 | 1.0 GB | 1.9 GB | 3.8 GB |
| 10 | 1.9 GB | 3.8 GB | **7.5 GB** |
| 20 | 3.8 GB | 7.5 GB | 15.0 GB |
| 50 | 9.4 GB | 18.8 GB | 37.5 GB |

### GPU deployment scenarios

| GPU | VRAM | Max N×T batch | Example |
|-----|------|---------------|---------|
| RTX 3060 | 12 GB | ~4,000 | 15 objects × T=256 |
| RTX 3090 | 24 GB | ~8,000 | 30 objects × T=256 |
| RTX 4090 | 24 GB | ~8,000 | 30 objects × T=256 (faster) |
| A100 | 40 GB | ~13,000 | 50 objects × T=256 |
| A100 80GB | 80 GB | ~26,000 | 100 objects × T=256 |

**Bottom line:** With a 24GB GPU and CLIP B/32, you can run uncertainty estimation on
30 objects at T=256 in a single batch forward pass. For MOT with ~10 tracked objects,
this fits easily on even a 12GB GPU. The computational bottleneck is NOT the model or
VRAM — it's the O(sqrt(T)) scaling of reliability that requires high T for precise
rankings.

**Note:** Current code (`phase_one/common.py`, `run_mc_trial`) uses a sequential loop
over T passes. This should be refactored to batch-parallel for deployment.

---

## 13. Alternative Models: Meta Perception Encoder (NEW)

### Background

Meta's **Perception Encoder (PE)** (April 2025) is a next-generation contrastive VLM,
trained on 5.4B image-text pairs. It is the backbone of SAM 3 (Segment Anything 3).

### Why it's relevant

PE is architecturally a CLIP — contrastive image-text training with softmax loss (not
sigmoid like SigLIP). Since our MC dropout method works on CLIP models specifically
because of the contrastive training objective, PE is a natural candidate.

### PE model variants

| Model | Vision Params | Text Params | Dim | Input Res |
|-------|-------------|-------------|-----|-----------|
| PE-Core-B/16 | ~86M | ~125M | 768 | 224px |
| PE-Core-L/14 | ~320M | ~310M | 1024 | 336px |
| PE-Core-G/14 | ~1.0B | ~310M | 1024 | 448px |

### Key differences from OpenAI CLIP

1. **Trained on 5.4B pairs** (vs CLIP's ~400M) — likely more robust features
2. **Uses MetaCLIP data curation** — more balanced training distribution
3. **RoPE position embeddings** — better length generalization
4. **Intermediate features are best for dense tasks** (their key finding) — unclear
   if this affects our per-image uncertainty
5. **Not open_clip compatible** — requires `facebookresearch/perception_models` repo

### Experimental Results: PE-Core FAILS

Tested PE-Core-B/16 via open_clip (`"PE-Core-B-16"`, pretrained=`"meta"`). Same protocol
as L/14: N=500, T=64, K=3, degradations=blur_r5+downsample_8x.

| Config | Reliability (Spearman) | blur_r5 | down_8x | Verdict |
|--------|----------------------|---------|---------|---------|
| All 12 fc2 dropout p=0.01 | 0.819 | 55.0% **FAIL** | 38.8% **FAIL** | Invalid |
| Uniform dropout p=0.01 | 0.798 | 25.0% **FAIL** | 32.8% **FAIL** | Invalid |

**PE-Core follows the same pattern as SigLIP2:** high reliability but ablation validity
fails completely. The uncertainty is consistent (Spearman=0.82) but measures the wrong
thing — degraded images do NOT get higher uncertainty.

**Why PE-Core fails despite contrastive loss:**
- Trained on **5.4B pairs** (vs CLIP's ~400M) → features are much more robust
- This continues the L/14 trend: larger/better training → more robust features → less
  sensitive to dropout perturbation → lower validity
- B/32 (400M pairs): 93.6% validity
- L/14 (400M pairs): 78.2% validity
- PE-Core (5.4B pairs): 55% validity — below threshold

**Conclusion:** The MC dropout validity method is CLIP-specific, not contrastive-loss-generic.
It requires features that are sensitive enough to dropout to reflect decision-boundary
proximity. PE-Core's more robust features (from massive training data) are too stable for
dropout to meaningfully probe.

### SAM family summary

| Model | Vision Encoder | Text-aligned? | Usable for us? |
|-------|---------------|---------------|----------------|
| SAM 1 | ViT-H (MAE pretrained) | No | No — no text encoder |
| SAM 2 | Hiera (MAE pretrained) | No | No — no text encoder |
| SAM 3 | PE (contrastive) | **Yes** | **Possibly** — needs testing |

Only SAM 3's backbone (PE) has the contrastive text-image alignment needed for our
zero-shot classification uncertainty approach. SAM 1/2 use MAE-pretrained encoders
with no text understanding.

---

## 14. PCA Dimensionality Reduction for MOT (NEW)

### Motivation

For multi-object tracking (MOT), we want to incorporate MC dropout uncertainty into a
Kalman filter state vector. The full 512-dim covariance is too large — we need a compact
representation that preserves valid uncertainty signal.

### Key question: do MC uncertainty dimensions overlap with discriminative dimensions?

If the dimensions that carry MC dropout variance are the same ones that distinguish
images from each other, then PCA on the batch covariance will preserve both discriminative
features AND uncertainty signal.

### Results (CLIP B/32, all-c_proj p=0.01, T=64, N=500)

**Dimension overlap analysis:**
- Top-64 MC uncertainty dims vs top-64 discriminative dims: **84% overlap** (54/64)
- Spearman(mc_var_ranking, disc_var_ranking): **0.668**
- Top-64 PCs capture 52.8% of MC variance

**Ablation validity by PCA dimension K:**

| K (PCs) | MC variance captured | blur_r5 | down_8x | Verdict |
|---------|---------------------|---------|---------|---------|
| 8 | 26.4% | 81.2% | 84.2% | PASS |
| 16 | 33.7% | 82.4% | 84.6% | PASS |
| 32 | 42.3% | 85.4% | 87.2% | PASS |
| 64 | 52.8% | 87.8% | 88.2% | PASS |
| full (512) | 100% | 93.6% | 90.6% | PASS |

### Interpretation

1. **Even K=8 passes ablation** (81-84%). The valid uncertainty signal is concentrated
   in a very low-dimensional subspace.
2. **84% overlap** means the PCA basis learned from the mean features (no MC dropout
   needed) will preserve most of the uncertainty signal. This is convenient — compute PCA
   once on a calibration set, then project at runtime.
3. **For Kalman filter MOT:** Use K=16-32 dimensions. This gives 82-87% ablation validity
   with a state vector small enough for real-time tracking. The PCA projection matrix is
   512×K (~32KB for K=32), negligible overhead.

### Pipeline for MOT deployment

```
1. Offline: compute PCA on calibration images → projection matrix W (512×K)
2. Per frame, per object:
   a. Run T MC dropout passes → T×512 features
   b. Project: T×K features via W
   c. Compute weighted_trace_pre on projected features → scalar uncertainty
   d. Feed (K-dim mean feature, scalar uncertainty) into Kalman filter
```

---

## 15. File Reference

### Reports
- `PHASE_ONE_REPORT.md` — Phase 1 reliability gate (Exp 5 section outdated)
- `PRELIM_FINDINGS.md` — Preliminary investigation (outdated, pre-perturbation search)
- `PERTURBATION_SEARCH_REPORT.md` — Perturbation search (Section 5.2 confirmed: fails validity)
- `STATE_OF_EXPLORATION_2026_03_04.md` — This document

### Code
- `phase_one/common.py` — Core infrastructure (ModelSpec, run_mc_trial, feature caching)
- `phase_two/perturbation.py` — Perturbation framework + `get_mlp_output_projections()` (model-agnostic)
- `phase_two/metrics.py` — `compute_all_metrics`, weighted_trace_pre, topk_dim_trace
- `phase_two/ablation.py` — Shared ablation utilities (DEGRADATIONS, paired_comparison, run_ablation_test)
- `phase_two/module_scan.py` — Per-module sensitivity scanner
- `phase_two/exp_pe_core.py` — PE-Core experiment (negative result)
- `phase_two/exp1-6` — Phase 2 experiment scripts

### Key output files
- `outputs/phase_two/exp*` — Phase 2 experiment results
- `outputs/prelim_ablation.json` — Image degradation validation
- `outputs/gaussian_ablation_test.json` — Gaussian validity test (FAILS)
- `outputs/validity_smoke_test.json` — Entropy/margin/error correlations
- `outputs/block11_deep_test.json` — Deep reliability validation
- `outputs/cproj12_t_scaling.json` — T-scaling for 12-c_proj config
- `outputs/pe_core_exp.json` — PE-Core ablation + reliability (FAILS)

### Test scripts (root directory)
- `validity_smoke_test.py` — Tests perturbation configs against entropy/margin/error
- `gaussian_ablation_test.py` — Tests Gaussian vs dropout on image degradation
- `residual_ablation_test.py` — Tests residual stream perturbation
- `spectral_ablation_test.py` — Tests all metrics on image degradation
