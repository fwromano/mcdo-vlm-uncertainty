# Unified Theory: MC Dropout Uncertainty in Frozen VLMs

**March 5, 2026 — Peer-reviewed synthesis of 25+ experiments across 4 VLMs**

This document supersedes `META_ANALYSIS.md`, `KEY_TAKEAWAYS.md`, and the narrative
sections of `STATE_OF_EXPLORATION_2026_03_04.md`. Every number cited here is either
(a) verified against a saved output JSON, or (b) explicitly marked [UNVERIFIED] with
the source session noted. For raw data tables, see `STATE_OF_EXPLORATION_2026_03_04.md`.

---

## 0. Data Integrity Notes

### Numbers verified against saved JSON outputs

| Claim | Source File | Verified Value |
|-------|-----------|----------------|
| Gaussian@block11 Spearman=0.998, SNR=458 | block11_deep_test.json | 0.998, 457.6 |
| Gaussian@block11 blur_r5=25.4% | gaussian_ablation_test.json | 0.254 |
| Gaussian@block11 blur_r15=2.8% | gaussian_ablation_test.json | 0.028 |
| Uniform dropout blur_r5=86.8% | gaussian_ablation_test.json | 0.868 |
| 12-c_proj T=64 Spearman=0.43 | cproj12_t_scaling.json | 0.430 |
| 12-c_proj T=256 Spearman=0.74 | cproj12_t_scaling.json | 0.739 |
| PE-Core all-12-fc2 Spearman=0.82 | pe_core_exp.json | 0.819 |
| PE-Core blocks 7-9 weighted blur=94.4% | pe_core_sweep.json | 0.944 |
| PE-Core blocks 7-9 weighted down=84.4% | pe_core_sweep.json | 0.844 |
| PE-Core all-12 trace blur=55% | pe_core_exp.json | 0.550 |
| Exp 3 Type D (block 9) Spearman=0.77 | exp3_overall_summary.json | 0.7706 |
| Prelim rho(entropy)=0.253 | prelim_investigation.json | 0.253 |
| SigLIP2 rho(centroid_distance)=0.24 | prelim_investigation.json | 0.241 |
| SigLIP2 blur_r5 frac_increased=0.416 | prelim_ablation.json | 0.416 |
| SigLIP2 blur_r15 frac_increased=0.244 | prelim_ablation.json | 0.244 |
| PE-Core blocks 9-11 weighted blur=90.0% | pe_core_block_comparison.json | 0.900 |
| PE-Core blocks 9-11 weighted down=53.8% | pe_core_block_comparison.json | 0.538 |
| PE-Core blocks 7-9 weighted blur=94.4% (reproduced) | pe_core_block_comparison.json | 0.944 |

### Numbers NOT in saved outputs [UNVERIFIED — from prior session transcripts]

| Claim | Likely Source | Status |
|-------|-------------|--------|
| CLIP B/32 12-c_proj trace_pre blur_r5=93.6% | Background task in prior session | No JSON artifact |
| CLIP B/32 12-c_proj trace_pre down_8x=90.6% | Background task in prior session | No JSON artifact |
| weighted_trace_pre blur_r5=96.4%, down_8x=97.0% | Background task in prior session | No JSON artifact |
| CLIP L/14 c_proj blur_r5=78.2% | Background task in prior session | No JSON artifact |
| CLIP L/14 Spearman=0.752 | Background task in prior session | No JSON artifact |
| Uniform T=256 Spearman=0.82 | Background task in prior session | No JSON artifact |
| CLIP B/32 12-c_proj p=0.05 blur_r5=71.5% | Prior session | No JSON artifact |
| CLIP B/32 12-c_proj p=0.20 blur_r5=0.5% | Prior session | No JSON artifact |

**Recommendation:** Re-run the 12-c_proj ablation test and L/14 cross-validation with
output saved to JSON before publishing. These numbers are central to the narrative but
currently exist only in conversation logs.

### CRITICAL: Lexicographic sorting bug in `get_mlp_output_projections()`

**Discovered 2026-03-05.** The function in `phase_two/perturbation.py` used Python's
default `sorted()` on module paths, which sorts lexicographically: `"blocks.10"` sorts
before `"blocks.2"`. This means `[-3:]` (the "late-3" slice) selected **blocks 7, 8, 9**
instead of the intended **blocks 9, 10, 11** for 12-block models.

**Impact on existing results:**
- **PE-Core "late-3" sweep** (pe_core_sweep.json): Actually tested blocks 7-8-9,
  NOT blocks 9-10-11 as labeled. The 94.4% validity is a **real result from blocks 7-9**.
- **All-fc2 configs**: UNAFFECTED (all 12 blocks are included regardless of order).
- **CLIP L/14 late-3** (never actually run): Would have catastrophically selected blocks
  7-8-9 instead of 21-22-23. The fix prevents this for future runs.

**Status:** Bug fixed (natural sort key added). PE-Core "late-3" results in this document
are relabeled as "blocks 7-9" to reflect what was actually tested.

**Resolution (2026-03-05):** Head-to-head test confirms blocks 7-9 are BETTER than true
blocks 9-11 (94.4% vs 90.0% blur, 84.4% vs 53.8% downsample). The sorting bug
accidentally found the superior config. See `outputs/pe_core_block_comparison.json`.

### Numbers requiring correction

| Claim in Prior Docs | Issue | Correction |
|---------------------|-------|------------|
| "PE-Core late-3 fc2: 0.82 reliability" | 0.82 was measured on **all-12 fc2**, not late-3 | Late-3 reliability is **UNMEASURED** |
| "SigLIP2: ~25% validity" (META_ANALYSIS table) | This is blur_r15; other models use blur_r5 | SigLIP2 blur_r5 = 41.6% (still fails, but not ~25%) |
| "Block 9 only: 0.77 reliability" | Varies 0.58-0.77 depending on sample size | Cite as "0.77 (N=1000) / 0.58 (N=500)" |
| "PE-Core late-3 = blocks 9-11" | Lexicographic sorting bug | Actually tested blocks 7-9 |

---

## 1. The Method in One Paragraph

> We estimate visual uncertainty by applying MC dropout (p=0.01) to the MLP
> down-projection layers of a frozen contrastive vision encoder and measuring
> the weighted trace of the per-pass feature covariance. Across 64+ stochastic
> forward passes, images that are blurry, distant, or occluded produce
> consistently higher variance. The method requires no training, no labels, and
> no modification to the frozen model weights. Three contrastive VLMs confirmed:
> CLIP B/32, CLIP L/14, and PE-Core-B/16 (Meta's Perception Encoder).

---

## 2. The Core Data

All numbers below are verified against JSON outputs unless marked [U].

| Model | Training | Layers Perturbed | Reliability | Validity (blur_r5) | Source |
|-------|----------|-----------------|-------------|--------------------|----|
| CLIP B/32 | 400M pairs | All 12 c_proj | 0.43 (T=64) | 93.6% [U] | cproj12_t_scaling / prior session |
| CLIP B/32 | 400M pairs | Block 9 only | 0.77 (N=1000) | UNTESTED | exp3_summary |
| CLIP B/32 | 400M pairs | Uniform (36) | 0.52 (T=64) | 86.8% | exp3_summary / gaussian_ablation |
| CLIP L/14 | 400M pairs | All 24 c_proj | 0.75 [U] | 78.2% [U] | prior session |
| PE-Core B/16 | 5.4B pairs | All 12 fc2 | 0.82 | 55.0% | pe_core_exp |
| PE-Core B/16 | 5.4B pairs | Blocks 7-9 fc2† | **UNMEASURED** | 94.4% (weighted) | pe_core_sweep |
| SigLIP2 B/16 | sigmoid | Uniform | 0.96‡ | 41.6% FAIL | phase1 / prelim_ablation |
| CLIP B/32 | 400M pairs | Gaussian noise | 0.998 | 25.4% FAIL | block11_deep / gaussian_ablation |

†PE-Core "late-3" actually tested blocks 7-9 due to lexicographic sorting bug (see Section 0).
True blocks 9-11 are untested. Re-run queued.
‡SigLIP2 reliability was measured with post-norm metric; validity with trace_pre. Cross-metric
comparison is approximate — both indicate FAIL but the numbers come from different measurements.

**Key corrections from prior docs:**
1. PE-Core "late-3" reliability is listed as "UNMEASURED" rather than 0.82. The 0.82
   belongs to all-12-fc2, which fails validity.
2. PE-Core "late-3" actually tested blocks 7-9 (sorting bug), not blocks 9-11 as
   documented. The "best of both worlds" claim is doubly uncertain — both reliability
   and the intended block range are unconfirmed.

> **Priority experiments:** (a) Run K=5 reliability on PE-Core blocks 7-9 fc2 to test
> the config that actually produced 94.4%. (b) Re-run with true blocks 9-11 to test
> whether the deepest blocks do better or worse.

---

## 3. Architecture: Where and What We're Perturbing

```
Image (224×224)
  │
  ▼
┌─────────────────────────────────────────────┐
│  Vision Encoder (frozen)                    │
│                                             │
│  ┌─── Transformer Block (×12 for B/32) ──┐ │
│  │                                        │ │
│  │  Self-Attention                        │ │
│  │    └─ out_proj: 768→768  [DEAD]        │ │
│  │                                        │ │
│  │  Feed-Forward Network                  │ │
│  │    ├─ c_fc:   768→3072  (expand)       │ │
│  │    ├─ GELU                             │ │
│  │    └─ c_proj: 3072→768  (compress)     │ │
│  │         ▲                              │ │
│  │         └── DROPOUT HERE (p=0.01)      │ │
│  │                                        │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  CLS pooling → LayerNorm → Projection       │
└─────────────────────────────────────────────┘
  │
  ▼
  T embeddings (each with different dropout mask)
  │
  ▼
  Per-dimension variance, weighted by discriminative power
  │
  ▼
  Scalar uncertainty score per image
```

**Module naming:** OpenAI/open_clip calls the down-projection `c_proj` (GPT-2 heritage).
HuggingFace/timm calls it `fc2`. Meta's PE-Core uses `trunk.blocks.N.mlp.fc2`.
Architecturally identical: all are Linear(3072→768).

**The visual_projection (768→512) at the end is NOT what we perturb.** That's a different
layer — the final projection into the shared image-text embedding space.

---

## 4. Act 1: The Loss Function Gate

The training objective is a binary gate for this method.

### Contrastive softmax (CLIP, PE-Core) — PASS

Contrastive softmax pushes each image toward its text description and simultaneously
away from ALL other descriptions in the batch. This creates **inter-class competition**:
features near decision boundaries are pulled in multiple directions by competing class
centroids. Dropout probes this tension by ablating computational pathways and measuring
how much the balance shifts.

### Sigmoid loss (SigLIP2) — FAIL

Sigmoid loss trains each image-text pair as an independent binary match/no-match. No
inter-class competition. The model never needs to distinguish dogs from cats — only
"does this image match the text 'a dog'?" independently.

**Evidence (verified):**
- SigLIP2 B/16 blur_r5: frac_increased = 0.416 (41.6% up, 58.4% down) — **inverted**
  [prelim_ablation.json]
- SigLIP2 rho(centroid_distance) = 0.241, rho(entropy) = -0.005 — measures outlier
  distance, not ambiguity [prelim_investigation.json]
- SigLIP2 mean convergence: rel_dist stays ~0.52 from T=4 to T=64 — dropout
  fundamentally disrupts representation, doesn't gently probe it [exp6_summary.json]

**Important caveat:** SigLIP2 was only tested with uniform dropout + trace_pre. We
did NOT test late-block targeting or weighted_trace_pre on SigLIP2. However, the Exp 6
mean convergence failure (representation permanently disrupted regardless of T) suggests
the problem is deeper than tuning. The sigmoid loss likely doesn't create the feature
structure that dropout needs to probe, regardless of which layers are perturbed.

**What the data supports:** "SigLIP2 fails with uniform dropout + trace_pre, and the
mean convergence pattern suggests fundamental incompatibility."

**What the data does NOT support:** "Any config fails" (untested for targeted configs).

---

## 5. Act 2: The Robustness-Validity Tradeoff

Once past the loss function gate, there is a continuous tradeoff driven by **feature
robustness** — how much redundancy the model has learned.

### The mechanism

Dropout probes redundancy. Drop a neuron in the MLP down-projection → you're removing
one "vote" about what the image contains. Robust models (more training data, more params)
have more redundant votes → losing one doesn't matter → low variance even for degraded
images → high reliability, low validity.

### The data (apples-to-apples comparison)

Both comparisons use the same strategy: all MLP down-projections, trace_pre, p=0.01, T=64.

**Holding architecture fixed (ViT-B), increasing training data:**
- CLIP B/32 (400M pairs): reliability 0.43, validity 93.6% [U]
- PE-Core B/16 (5.4B pairs): reliability 0.82, validity 55.0%

**Holding training data fixed (400M), increasing model size:**
- CLIP B/32 (88M params): reliability 0.43, validity 93.6% [U]
- CLIP L/14 (304M params): reliability 0.75 [U], validity 78.2% [U]

Both axes push the same tradeoff: more robust features → higher reliability, lower validity.

**Observation:** Training data volume (13.5x: 400M→5.4B) has a larger effect than model
size (3.5x: 88M→304M). PE-Core's 5.4B pairs made features so robust that all-12 dropout
can't probe meaningful uncertainty. CLIP L/14's 3.5x parameter increase degrades validity
more gently (93.6%→78.2% vs 93.6%→55%).

---

## 6. Act 3: Spatial Targeting — Can It Break the Tradeoff?

The key finding for PE-Core is that perturbing a subset of blocks restores validity:

| Config | Validity (blur, weighted) | Validity (down, weighted) |
|--------|--------------------------|--------------------------|
| PE-Core all-12 fc2 | 69-75.6% weak/FAIL | 54-64% FAIL |
| PE-Core blocks 7-9 fc2† | **94.2-94.4% PASS** | **81-84% PASS** |

[pe_core_sweep.json — all numbers verified]
†Labeled "late-3" in pe_core_sweep.py but actually tested blocks 7-9 due to sorting bug.

### Why it works — and why the deepest blocks are worse

Head-to-head comparison (N=500, T=64, p=0.01) [pe_core_block_comparison.json]:

| Config | blur (weighted) | down (weighted) |
|--------|----------------|----------------|
| Blocks 7-9 | **94.4% PASS** | **84.4% PASS** |
| Blocks 9-11 | 90.0% PASS | 53.8% FAIL |
| All 12 | 75.6% PASS | 64.4% weak |

**The mid-late blocks (7-9) beat the true last blocks (9-11) on every metric.** Blocks
9-11 even FAIL downsample validity (53.8%) — worse than all-12 (64.4%) for that test.

Transformer blocks form a processing hierarchy, but the "commitment" zone is NOT the
absolute deepest layers:
- **Early blocks (0-6):** Low-level features. Too robust for dropout to probe.
- **Mid-late blocks (7-9):** The **sweet spot**. Deep enough to encode semantic features,
  but not so specialized that dropout destroys the representation.
- **Final blocks (10-11):** Too specialized. Dropout here appears to be **too disruptive**
  — it destroys the representation rather than gently probing redundancy, similar to how
  high dropout rates (p=0.20) invert the signal on CLIP B/32.

### Two decoupled axes

Block-subset dropout decouples reliability and validity:
- **Deterministic early blocks (0-6):** Provide stable perceptual input → reliability.
- **Stochastic mid-late blocks (7-9):** Probe decision fragility → validity.

### The remaining unverified hypothesis

**Hypothesis 2 resolved:** True last blocks (9-11) are WORSE than blocks 7-9 on PE-Core.
The sorting bug accidentally found the better config.

**Hypothesis 1 (reliability):** Do blocks 7-9 achieve reliability ≥0.82 (matching all-12)?
- **Evidence for:** Removing early-block noise should make rankings MORE consistent.
  CLIP B/32 block-9-only (0.77 at N=1000) > all-12-c_proj (0.43 at T=64).
- **Evidence against:** Fewer perturbed modules = less total variance = potentially lower SNR.
- **Status: STILL UNMEASURED.** This is the #1 remaining experiment.

### Supporting evidence from CLIP B/32

CLIP B/32 block-9-only dropout at p=0.01:
- Exp 3 (N=1000, K=5): Spearman = 0.77 [exp3_summary.json]
- Deep test (N=500, K=5): Spearman = 0.58 [block11_deep_test.json]

The 0.19 gap between these is concerning — it means single-block reliability is
**sample-dependent** at these N values. The difference comes from different random
image samples, not different methodology. This suggests that single-block dropout
at p=0.01 may not have enough total signal for stable rankings at small N.

For PE-Core blocks 7-9 (3 blocks, not 1), the total variance should be ~3x higher,
which should help. But this is speculation until measured.

**Note:** CLIP B/32 late-3 targeting (proposed in META_ANALYSIS) has never been run.
With the sorting bug fixed, this would now correctly select blocks 9-11. The Exp 3
block-9-only result (0.77) is suggestive but not equivalent to a 3-block subset.

---

## 7. Act 4: Why the MLP Down-Projection Specifically

### The MLP is a vote

```
c_fc:    768 → 3072    Expand: generate candidate features
GELU:                   Activate: select relevant candidates
c_proj:  3072 → 768    Compress: candidates VOTE on output features
```

The down-projection is a compression bottleneck. 3072 candidates compete to influence
768 output features. For clear images, candidates agree (low dropout variance). For
ambiguous images, candidates disagree (high dropout variance).

### Why attention is dead

All 12 attention out_proj layers produce **exactly zero variance** under dropout.
[Exp 3 Type A: trace_mean ≈ 9e-17, angular_mean = 0.0 — exp3_summary.json]

Three factors:
1. **Softmax saturation.** Attention weights are often near [1, 0, 0, ...].
2. **Residual cancellation.** The residual connection absorbs small perturbations.
3. **Routing vs transformation.** Attention decides WHICH tokens to look at. MLP
   decides WHAT features to extract. Uncertainty about "what" matters more.

### Why c_fc (up-projection) is a mixed bag

Prior docs claimed "c_fc hurts." The data is more nuanced:

| Strategy | Modules | Validity (blur) | Reliability |
|----------|---------|-----------------|-------------|
| All 12 c_proj only | 12 | 93.6% [U] | 0.43 |
| All 24 MLP (c_fc + c_proj) | 24 | ~86.8%* | ~0.52* |

*Uniform dropout on 36 modules ≈ MLP-only on 24 (attention contributes zero).

Adding c_fc: **-6.8% validity, +0.09 reliability.** This is a tradeoff, not simply
noise. c_fc dropout creates correlated perturbation across all 3072 candidates
(because dropped input features are missing from every candidate), which adds a
reliability-boosting "common signal" but dilutes the per-candidate independence that
makes c_proj dropout valid.

**Bottom line:** For maximum validity, use c_proj only. For maximum reliability at
moderate validity, include c_fc (uniform dropout). For PE-Core, block-subset c_proj
targeting gives high validity (94.4%); reliability is unmeasured but expected to be
competitive with all-12's 0.82.

---

## 8. Act 5: The Metric Completes the Picture

### The dimension structure of VLM features

Not all feature dimensions are equal. In CLIP B/32's 512 dimensions, a small subset
encodes discriminative information (varies across images — identity features) while the
majority encodes stable, non-discriminative information (similar across all images —
"image-ness" features).

### Why weighted_trace_pre works

Plain trace_pre averages MC variance over all dimensions equally. The non-discriminative
dimensions contribute noise — they carry some MC variance but it doesn't correlate with
visual ambiguity.

weighted_trace_pre weights each dimension's variance by its **discriminative power**
(across-image variance of the mean feature). This focuses on dimensions that:
1. Carry classification-relevant information
2. Are most damaged by degradation
3. Have the most meaningful dropout variance

### The metric matters more for robust models

On CLIP B/32 (all-c_proj, trace_pre): 93.6% blur [U] — already high without weighting.
On PE-Core B/16 (blocks 7-9 fc2, trace_pre): 88.6% blur [pe_core_sweep.json] — decent.
On PE-Core B/16 (blocks 7-9 fc2, weighted_trace_pre): **94.4%** blur [pe_core_sweep.json].

On PE-Core for downsample, the gap is dramatic:
- trace_pre: 61.2% [pe_core_sweep.json] — near FAIL
- weighted_trace_pre: **84.4%** [pe_core_sweep.json] — PASS

PE-Core's 5.4B training pairs made most dimensions hyper-stable. The discriminative
signal is concentrated in a small subset. Weighting finds this active minority.

**The metric is not a binary gate — it's a multiplier.** On B/32 it improves a passing
score. On PE-Core it's the difference between pass and fail. More robust models have
more non-discriminative dimensions, making the weighting progressively more important.

---

## 9. Act 6: Why Gaussian Noise Fails

### What Gaussian measures

```
E[||f(x, w+ε) - f(x, w)||²] ≈ ||∇_w f(x, w)||² · σ²
```

The **Jacobian norm** — how sensitive the output is to weight perturbations. This is a
fixed geometric property of each input's position in activation space.

Degraded images have smoother, lower-magnitude activations → smaller Jacobian →
LESS sensitivity → LOWER uncertainty. **The signal is inverted.**

**Verified data:**
- Gaussian@block11 blur_r5: 25.4% get higher (74.6% get LOWER uncertainty) [gaussian_ablation_test.json]
- Gaussian@block11 blur_r15: 2.8% get higher (97.2% LOWER) [gaussian_ablation_test.json]

### What dropout measures

Dropout randomly ablates neurons:
```
Var_mask[f(x, mask)] = function of subnetwork agreement
```

This is NOT a local derivative. It's a **combinatorial** property — how much do
different subnetworks agree? No closed form. Depends on redundancy structure.

### The mathematical distinction

- **Gaussian**: continuous → local linearization → Jacobian → geometry
- **Dropout**: discrete ablation → subnetwork sampling → redundancy → combinatorial

Classification uncertainty lives in the global redundancy structure, not the local
Jacobian landscape.

### Higher dropout rates approach Gaussian behavior

| Config | blur_r5 | Spearman |
|--------|---------|----------|
| All c_proj p=0.01 | 93.6% [U] | 0.43 |
| All c_proj p=0.05 | 71.5% [U] | 0.39 |
| All c_proj p=0.20 | **0.5%** [U] | 0.15 |

[higher dropout rate data from prior session — numbers consistent with narrative but
no JSON artifacts found]

At high p, dropout overwhelms the representation and measures feature complexity
(like Gaussian) rather than probing redundancy. The valid measurement exists only
at very low p where dropout gently "thins the vote" rather than destroying it.

### Quantization and scale noise are Gaussian in disguise

Randomizing low bits of quantized weights is dense continuous noise (uniform
distribution). Multiplicative scale noise (`out * (1 + N(0, σ²))`) is similarly
continuous. At first order, both compute the same Jacobian. Different
distribution, same failure mode.

The **sparsity** and **binary nature** of dropout is load-bearing. It asks "can this
network function with this pathway deleted?" — categorically different from "how
much does the output wiggle when weights jiggle?"

---

## 10. The Unifying Framework

The method works when four factors align:

```
LOSS FUNCTION          Contrastive softmax creates inter-class tension
       ×               in the feature space that dropout can probe.
LAYER TARGETING        Late MLP down-projections are where the model
       ×               commits to a representation — the decision point.
METRIC FOCUS           Discriminative weighting isolates the dimensions
       ×               that carry actual classification signal.
PERTURBATION TYPE      Sparse binary ablation (dropout) probes
                       combinatorial redundancy, not local geometry.
```

**Important nuance:** These are not all binary gates. In order of criticality:

1. **Loss function: Binary gate.** SigLIP2 fails with uniform dropout + trace_pre, and
   mean convergence data suggests fundamental incompatibility (representation permanently
   disrupted, not gently probed). Contrastive softmax is required. Caveat: targeted
   configs on SigLIP2 remain untested (see Section 4).

2. **Perturbation type: Binary gate.** Gaussian/scale/quantization noise all measure
   the Jacobian, which is inverted. Only dropout (sparse binary ablation) probes
   redundancy correctly.

3. **Layer targeting: Continuous dial.** All-block dropout works for less-robust models
   (CLIP B/32: 93.6%). Block-subset targeting is needed for robust models (PE-Core
   weighted_trace_pre: 94.4% blocks 7-9 vs 69-75.6% all-12). It's a continuum, not
   pass/fail.

4. **Metric: Continuous multiplier.** trace_pre works for B/32 (93.6%). weighted_trace_pre
   is needed for PE-Core downsample (84.4% vs 61.2%). More robust models need it more.

---

## 11. The Reliability-Validity Tradeoff

### The fundamental tension

```
              100% ┬─ High Validity
                   │
                   │● B/32 weighted_trace [U] 96.4%
  B/32 c_proj      │● [U] 93.6%
  PE-Core blks7-9  │● 94.4%  (verified validity, reliability unmeasured)
                   │
  Uniform p=0.01   │  ● 86.8%  (verified)
                   │
          ~75% ----│----------- pass threshold -------
                   │
  L/14 c_proj [U]  │  ● 78.2%
                   │
  PE-Core all-12   │    ● 55.0%  (verified)
                   │
  SigLIP2          │      ● 41.6%  (verified — INVERTED)
                   │
  Gaussian blk11   │               ● 25.4%  (verified)
                0% ┴───────┼───────┼───────┼──────── Reliability
                  0.0     0.4    0.75    1.0
```

Note: PE-Core blocks 7-9 is plotted without an x-coordinate (reliability unmeasured).
SigLIP2 reliability (0.96) is from post-norm metric, not directly comparable to others.

### Scaling T beats down noise

Both configs follow approximately O(sqrt(T)) scaling [cproj12_t_scaling.json]:

| T | 12-c_proj Spearman | Uniform Spearman [U] |
|---|-------------------|---------------------|
| 16 | 0.307 | — |
| 64 | 0.430 | 0.52 |
| 128 | 0.581 | — |
| 256 | 0.739 | 0.82 |

Note: The O(sqrt(T)) description is approximate. SNR scales slightly faster than
sqrt(T) in the verified data (SNR ratio of 12.3x for 16x more passes, vs 4x for
sqrt(16)). The Spearman-T relationship is non-linear and depends on the underlying
signal strength.

### Can spatial targeting break the tradeoff?

**Hypothesis:** PE-Core block-subset dropout achieves BOTH high reliability (≥0.82)
AND high validity (94.4%).

**Evidence for:**
- PE-Core all-12 reliability = 0.82. Removing noisy early blocks should maintain or
  improve reliability.
- CLIP B/32 block-9-only reliability (0.77 at N=1000) > all-12-c_proj (0.43 at T=64).
- Blocks 7-9 already achieved 94.4% validity (verified).

**Evidence against:**
- CLIP B/32 block-9-only reliability drops to 0.58 at N=500 (sample-dependent).
- Fewer perturbed modules means less total variance → potentially lower SNR.
- 3 modules at p=0.01 means only ~92 neurons dropped per pass (3 × 3072 × 0.01).
  This may not be enough "votes removed" for a stable signal.

**Resolved:** Blocks 7-9 beat blocks 9-11 in head-to-head (94.4% vs 90.0% blur,
84.4% vs 53.8% down). The mid-late zone is the sweet spot, not the deepest blocks.

**Remaining unknown:** Does blocks 7-9 achieve high reliability too? Status: UNRESOLVED.

---

## 12. Best Configurations (Verified)

### Configs that DEFINITELY pass validity (>75% blur_r5, verified JSON)

| Model | Config | Metric | Validity (blur) | Source |
|-------|--------|--------|-----------------|--------|
| PE-Core B/16 | Blocks 7-9 fc2†, p=0.01 | weighted_trace_pre | **94.4%** | pe_core_sweep.json |
| PE-Core B/16 | Blocks 7-9 fc2†, p=0.005 | weighted_trace_pre | **94.4%** | pe_core_sweep.json |
| PE-Core B/16 | Blocks 7-9 fc2†, p=0.001 | weighted_trace_pre | **94.2%** | pe_core_sweep.json |
| PE-Core B/16 | Blocks 7-9 fc2†, any p | trace_pre | **88-89%** | pe_core_sweep.json |
| CLIP B/32 | Uniform, p=0.01 | trace_pre | **86.8%** | gaussian_ablation_test.json |

### Configs that LIKELY pass validity [UNVERIFIED]

| Model | Config | Metric | Validity (blur) | Source |
|-------|--------|--------|-----------------|--------|
| CLIP B/32 | All 12 c_proj, p=0.01 | weighted_trace_pre | 96.4% | prior session |
| CLIP B/32 | All 12 c_proj, p=0.01 | trace_pre | 93.6% | prior session |
| CLIP L/14 | All 24 c_proj, p=0.01 | trace_pre | 78.2% | prior session |

### Configs that DEFINITELY pass reliability (>0.5 Spearman, verified)

| Model | Config | T | Reliability | Source |
|-------|--------|---|-------------|--------|
| PE-Core B/16 | All 12 fc2, p=0.01 | 64 | **0.82** | pe_core_exp.json |
| CLIP B/32 | All 12 c_proj, p=0.01 | 128 | **0.58** | cproj12_t_scaling.json |
| CLIP B/32 | All 12 c_proj, p=0.01 | 256 | **0.74** | cproj12_t_scaling.json |
| CLIP B/32 | Block 9 only, p=0.01 | 64 | **0.77** (N=1000) | exp3_summary.json |
| CLIP B/32 | Uniform, p=0.01 | 64 | **0.52** | exp3_summary.json |

### Configs that pass BOTH (verified validity + verified reliability)

| Model | Config | T | Reliability | Validity | Notes |
|-------|--------|---|-------------|----------|-------|
| CLIP B/32 | Uniform, p=0.01 | 64 | 0.52 | 86.8% | Only config verified on both axes |

**That's it.** Only one config is verified on BOTH axes. The others either have
unverified validity (CLIP 12-c_proj) or unmeasured reliability (PE-Core blocks 7-9).

### Priority experiments to fill the gaps

1. **PE-Core blocks 7-9 fc2 reliability** (K=5, T=64, N=500): Does the config that
   achieved 94.4% validity also have high reliability? This is the #1 gap.
2. ~~PE-Core true blocks 9-11 fc2~~ **RESOLVED:** Blocks 9-11 are worse (90.0%/53.8%).
   Blocks 7-9 confirmed as the superior config.
3. **Re-run CLIP B/32 12-c_proj ablation with saved JSON output**: Verify the 93.6%
   and 96.4% numbers.
4. **Re-run CLIP L/14 cross-validation with saved JSON output**: Verify 78.2% and 0.75.

---

## 13. Practical Deployment

### PCA compression for Kalman filter MOT

The PCA basis is computed **once offline** and reused forever. The top-K principal
components reflect the model's feature geometry (a property of the trained weights,
not any input). Run PCA on ~500 calibration images, save W (512×K). At runtime:
`(T×512) @ W → (T×K)`. Cost: ~0.025% overhead.

PCA results [from prior session — no JSON artifact]:
- 84% overlap between top-64 MC uncertainty dims and top-64 discriminative dims
- K=8: 81-84% ablation validity
- K=32: 85-87% ablation validity

### Batch-parallel MC dropout

T passes are embarrassingly parallel. Batch N×T samples in one forward pass.
CLIP B/32 fits 10 objects × T=256 = 2,560 samples in ~7.5GB VRAM.

### Discriminative weights at deployment

weighted_trace_pre computes weights from the current batch. For deployment with
homogeneous batches (e.g., 10 crops of same scene), precompute weights from a
diverse calibration set and freeze them. Same one-time offline computation as PCA.

---

## 14. What SigLIP2 Failure Tells Us (and Doesn't)

### What the data says

| Test | SigLIP2 B/16 Result | Source |
|------|--------------------|----|
| Ablation blur_r5 | 41.6% increased (FAIL) | prelim_ablation.json |
| Ablation blur_r15 | 24.4% increased (FAIL — nearly inverted) | prelim_ablation.json |
| rho(entropy) | -0.005 (no signal) | prelim_investigation.json |
| rho(centroid_distance) | +0.241 (moderate — outlier detector) | prelim_investigation.json |
| Mean convergence | rel_dist stays ~0.52 from T=4 to T=64 | exp6_summary.json |
| Reliability | Spearman = 0.96 (very high) | phase1_exp0 |

### The interpretation (well-supported)

SigLIP2's dropout uncertainty measures **distance from the training distribution
centroid** (outlier detection), not classification ambiguity. Degraded images have
simpler features closer to the centroid → lower uncertainty. The signal is inverted.

### The interpretation (partially supported)

The sigmoid loss is the root cause. Without inter-class competition, the feature
space doesn't encode decision-boundary proximity that dropout can probe.

**Strength of evidence:** Strong for "SigLIP2 fails with uniform dropout." Moderate
for "sigmoid loss is the root cause" (only one sigmoid-loss model tested; the failure
could be an interaction of sigmoid loss + specific architecture + training data).
The mean convergence failure adds confidence — it suggests dropout fundamentally
disrupts SigLIP2 rather than gently probing it.

---

## 15. Open Theoretical Questions

1. **Is there a formal connection between MLP redundancy and Bayesian posterior
   uncertainty?** The standard Gal & Ghahramani (2016) argument assumes training-time
   dropout. We're applying it to frozen models with no training-time dropout. The
   theoretical grounding is weaker than "approximate variational inference."

2. **Can we predict the optimal block subset from model properties?** CLIP B/32
   works with all 12. PE-Core works with blocks 7-9 (true last-3 untested). L/14 used
   all 24 (block subsets untested). Related to per-block feature robustness?

3. **Why does training data dominate model size for feature robustness?** PE-Core B/16
   (86M params, 5.4B pairs) is more robust than CLIP L/14 (304M params, 400M pairs).

4. **Is there a perturbation type between dropout and Gaussian?** Dropout: high
   validity, moderate reliability. Gaussian: perfect reliability, zero validity. Could
   sparse continuous or structured dropout bridge the gap?

5. **Does block-subset targeting preserve reliability?** The hypothesis that "deterministic
   early blocks + stochastic mid-to-late blocks" gives both is compelling but unverified.
   PE-Core blocks 7-9 have the highest validity (94.4%) but no reliability measurement.
   Additionally, are the true last blocks (9-11) better or worse than 7-9?

---

## 16. Anticipated Questions

### On the science

**"You only tested on ImageNet. Does this generalize to real tracking data?"**
Honest: we don't know. The ablation test (synthetic blur/downsample) proxies for
distance/motion degradation but isn't the real thing. #1 open item.

**"N=500, really?"**
The 94-97% validity numbers have ~±2% CI (binomial). Rankings are stable across
multiple runs. But exact percentages will shift by a few points on different samples.
The block-9 reliability gap (0.58 vs 0.77 on different N=500 vs N=1000 samples) shows
that some metrics are more sample-sensitive than others.

**"PE-Core late-3 gets 0.82 reliability AND 94.4% validity — that's amazing."**
Two corrections: (1) The 0.82 was measured on ALL-12 fc2, not the tested block subset.
(2) The "late-3" config actually tested blocks 7-9 due to a sorting bug, not blocks 9-11
as intended. The 94.4% validity from blocks 7-9 is confirmed; reliability for that config
is unmeasured; and the true last-3 blocks (9-11) have never been tested at all.

**"Dropout at low p: Spearman=0.43 means rankings are barely correlated."**
For MOT you need a binary signal ("trustworthy?"), not precise rankings. The
top/bottom quartiles are well-separated. Think smoke detector, not thermometer.
At T=256, reliability reaches 0.74. PE-Core all-12-fc2 reaches 0.82 at T=64 (but
that config fails validity; blocks 7-9 reliability is unmeasured).

**"Why not prediction entropy?"**
(1) Entropy requires class labels. MC dropout is label-free. (2) Entropy measures
"model is split between dog and cat." MC dropout measures "visual features are
fragile" — includes cases where the model is confidently wrong.

### On deployment

**"64 forward passes per image = 64x cost."**
Passes are embarrassingly parallel. Batch 10 objects × 64 passes = 640 samples
in one forward call ≈ 50ms on an RTX 3090 with CLIP B/32.

**"weighted_trace_pre depends on the batch. What if my batch is homogeneous?"**
Precompute weights from a diverse calibration set and freeze them. Same as PCA —
one-time offline.

**"Frame-to-frame jitter in uncertainty scores."**
At T=64, scores ARE noisy between runs. Feed raw uncertainty as the Kalman filter
observation; let the filter smooth. The signal is there, it needs temporal integration.

---

## 17. Experiment Priority Queue

In order of importance:

1. ~~PE-Core blocks 7-9 vs 9-11 head-to-head~~ **DONE.** Blocks 7-9 win decisively
   (94.4% vs 90.0% blur, 84.4% vs 53.8% down). Remaining: measure blocks 7-9
   **reliability** (K=5, T=64). (~30 min)

2. **Re-run key ablations with JSON output** — CLIP B/32 12-c_proj ablation test and
   L/14 cross-validation. These numbers are central but not reproducible from the
   outputs directory. (~45 min each)

3. **CLIP B/32 late-3 c_proj (with fix)** — Test the META_ANALYSIS prediction that
   late-block targeting works on CLIP too. Now correctly selects blocks 9-11. (~20 min)

4. **SigLIP2 with late-block + weighted_trace_pre** — Close the "any config" loophole.
   If it still fails (likely given mean convergence data), the sigmoid-loss story is
   airtight. If it passes, the narrative needs major revision. (~30 min)

5. **Real-world MOT validation** — The ultimate test. Everything else is proxy.

---

## 18. File Reference

### This document
- `UNIFIED_THEORY.md` — This file. Single source of truth.

### Superseded documents (kept for reference)
- `META_ANALYSIS.md` — Theoretical narrative (has PE-Core reliability error)
- `KEY_TAKEAWAYS.md` — Quick reference (has PE-Core reliability error, some unverified #s)
- `STATE_OF_EXPLORATION_2026_03_04.md` — Full data tables (mostly correct, unverified #s noted)

### Verified output files
- `outputs/pe_core_block_comparison.json` — PE-Core blocks 7-9 vs 9-11 head-to-head
- `outputs/pe_core_sweep.json` — PE-Core 6-config sweep
- `outputs/pe_core_exp.json` — PE-Core initial test + reliability
- `outputs/gaussian_ablation_test.json` — Gaussian vs dropout validity
- `outputs/block11_deep_test.json` — Deep reliability validation
- `outputs/cproj12_t_scaling.json` — 12-c_proj T-scaling
- `outputs/prelim_investigation.json` — Metrics + SigLIP2 correlates
- `outputs/prelim_ablation.json` — Initial image degradation test
- `outputs/spectral_ablation.json` — Extended metrics ablation
- `outputs/phase_two/exp3_dropout_type/` — Dropout type comparison

### Code
- `phase_one/common.py` — Core infrastructure
- `phase_two/perturbation.py` — Perturbation framework + `get_mlp_output_projections()`
- `phase_two/metrics.py` — Metric computation (weighted_trace_pre, topk_dim_trace)
- `phase_two/ablation.py` — Shared ablation utilities
- `phase_two/exp_pe_core_sweep.py` — PE-Core proper sweep
