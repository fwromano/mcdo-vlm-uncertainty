# Meta-Analysis: Why MC Dropout Uncertainty Works (and When It Doesn't)

**March 5, 2026 — Synthesized from 25+ experiments across 5 VLMs**

This document explains the *why* behind the results, not the results themselves.
For raw numbers, see `STATE_OF_EXPLORATION_2026_03_04.md`. For actionable configs,
see `KEY_TAKEAWAYS.md`.

---

## The Core Data

| Model | Training Data | Layers Perturbed | Reliability | Validity (blur) |
|-------|--------------|-----------------|-------------|-----------------|
| CLIP B/32 | 400M pairs | All 12 c_proj | 0.43 | 93.6% |
| CLIP B/32 | 400M pairs | Block 9 only | 0.77 | untested (Gaussian@block9 was 60% FAIL) |
| CLIP L/14 | 400M pairs | All 24 c_proj | 0.75 | 78.2% |
| PE-Core B/16 | 5.4B pairs | All 12 fc2 | 0.82 | 55% |
| PE-Core B/16 | 5.4B pairs | Late 3 fc2 | 0.82 | 94.4% |
| SigLIP2 B/16 | sigmoid loss | Any config | 0.96 | ~25% FAIL |
| Any model | — | Gaussian noise | 0.998 | 25% FAIL |

---

## Act 1: The Loss Function Gate

The training objective is a binary gate. Either the loss function creates the right
kind of feature space, or it doesn't. No amount of tuning overcomes the wrong loss.

### Contrastive softmax (CLIP, PE-Core) — PASS

Contrastive softmax training pushes each image toward its text description and
simultaneously away from ALL other descriptions in the batch. This creates
**inter-class competition**: features near decision boundaries are being pulled in
multiple directions by competing class centroids.

An image of a dog that looks slightly like a cat has features that are torn between
the "dog" attractor and the "cat" attractor. This tension is encoded in the feature
geometry. Dropout can probe it because ablating a computational pathway shifts the
balance between competing attractors.

### Sigmoid loss (SigLIP2) — FAIL

Sigmoid loss trains each image-text pair as an independent binary classification:
match or no-match. There is no inter-class competition. The model never needs to
distinguish dogs from cats — it only needs to decide "does this image match the
text 'a dog'?" independently for each text.

This creates features that encode **distance from the training distribution centroid**
rather than **position relative to decision boundaries**. Dropout perturbation
measures how far the image is from "typical" — an outlier detector, not an ambiguity
detector. Degraded images have simpler features that are actually CLOSER to the
centroid (more "typical"), so they get LOWER uncertainty. The signal is inverted.

This is architectural and unfixable. SigLIP2 could have infinite training data and
perfect features — the sigmoid loss still doesn't create the inter-class tension
that dropout needs to probe.

---

## Act 2: The Robustness-Validity Tradeoff

Once past the loss function gate, there is a continuous tradeoff driven by **feature
robustness** — how much redundancy the model has learned.

### The mechanism

A well-trained vision encoder learns multiple ways to represent the same concept.
A dog might be recognizable by its ears, its nose, its body shape, its fur texture,
etc. Each MLP down-projection neuron encodes one of these "features." The more
training data and parameters the model has, the more redundant features it learns.

**Dropout probes redundancy.** When you drop a neuron in the MLP down-projection,
you're removing one "vote" about what the image contains. If the model has many
redundant features (well-trained, robust), losing one vote doesn't change the
outcome — low variance, low uncertainty. If the model has few features for this
particular image (ambiguous, degraded), losing one vote matters — high variance,
high uncertainty.

### The tradeoff

More robust features (from more training data or larger models) means:

- **Higher reliability**: The redundancy structure is stable. Each dropout mask
  samples a slightly different subset of redundant features, but the variance
  pattern across images is consistent. The signal-to-noise ratio is high because
  the "signal" (which images are ambiguous) is a stable property of the feature
  space.

- **Lower validity**: The redundancy is SO high that even degraded images have
  enough features to survive dropout. A blurry dog still activates enough
  "dog-like" features that dropping one doesn't change the output much. The
  degraded image isn't flagged as uncertain because the model is robust enough
  to handle the degradation.

### The data confirms this

Holding architecture fixed, increasing training data:
- B/32 (400M) → PE-Core (5.4B): reliability 0.43 → 0.82, validity 93.6% → 55%

Holding training data fixed, increasing model size:
- B/32 (88M params) → L/14 (304M params): reliability 0.43 → 0.75, validity 93.6% → 78.2%

Both axes — more data, more parameters — push the SAME tradeoff.
**Robustness is a single dimension that simultaneously improves reliability and
degrades validity.**

---

## Act 3: Spatial Targeting Breaks the Tradeoff

PE-Core late-3-fc2 achieves 0.82 reliability AND 94.4% validity. This shouldn't be
possible if the tradeoff were fundamental. What's happening?

### Transformer blocks are a pipeline, not a bag

The 12 blocks are not interchangeable. They form a processing hierarchy:

```
Blocks 0-3  │  PERCEPTION
(early)     │  Low-level features: edges, textures, colors, spatial frequency.
            │  Very robust — these features are the same whether the image
            │  is a dog or a cat. Dropout here adds noise to a signal that
            │  carries no uncertainty information.
            │
Blocks 4-8  │  COMPOSITION
(middle)    │  Mid-level features: parts, shapes, spatial relationships.
            │  Moderate information. Some dropout signal, but mixed with
            │  noise from non-discriminative features.
            │
Blocks 9-11 │  COMMITMENT
(late)      │  High-level features: "this is a dog" vs "this is a cat."
            │  The MLP down-projection HERE is where 3072 candidate features
            │  get compressed to 768 output features. The model is CHOOSING
            │  which interpretation to commit to. This choice is fragile for
            │  ambiguous images and robust for clear ones.
```

### The key insight

**You want robustness in the early layers (for reliability) and fragility in the
late layers (for validity).**

- **Deterministic early blocks** provide stable input to the late blocks. The
  model's robust perceptual features create a consistent "canvas" regardless of
  dropout masks. This is where reliability comes from.

- **Stochastic late blocks** probe the model's commitment. For ambiguous or
  degraded images, the model's decision is fragile — different dropout masks
  lead to different compression choices. This is where validity comes from.

Late-block-only dropout decouples the two: the early blocks are a **noise-free
telescope**, the late blocks are the **measurement instrument**.

### This explains all the data

- **PE-Core all-12**: Perturbing early blocks adds noise to a robust feature
  extractor → drowns the signal from late blocks → 55% validity
- **PE-Core late-3**: Clean early blocks, stochastic late blocks → 94.4% validity,
  0.82 reliability
- **CLIP B/32 all-12**: CLIP's features are fragile enough that even early-block
  dropout carries SOME signal → still passes validity (93.6%) but reliability
  suffers (0.43) because early-block noise makes rankings inconsistent
- **CLIP B/32 block-9-only**: 0.77 reliability (vs 0.43 for all-12) — removing
  early-block noise dramatically improves reliability

### The prediction we haven't tested

CLIP B/32 with late-3 c_proj should give ~0.65-0.75 reliability + ~90-93% validity.
The same principle that rescued PE-Core should improve CLIP B/32. We have partial
evidence: single-block-9 gives 0.77 reliability, suggesting late-block targeting
works on CLIP too.

**Caveat:** Single-block Gaussian@block9 FAILED validity (60%). Gaussian and dropout
measure different quantities (Jacobian vs redundancy), so this doesn't prove single-
block dropout would also fail — but it's a warning. The PE-Core rescue used 3 blocks,
not 1. There may be a minimum number of perturbed blocks needed for the "vote thinning"
mechanism to produce valid signal. One block might not be enough dropout to probe
redundancy; it might just measure that single layer's sensitivity (more like Gaussian).
Testing CLIP B/32 late-3 would resolve this.

---

## Act 4: Why the MLP Down-Projection Specifically

### The MLP is a vote

Each transformer block's MLP is:
```
c_fc:    768 → 3072    Expand: generate candidate features
GELU:                   Activate: select relevant candidates
c_proj:  3072 → 768    Compress: candidates VOTE on output features
```

The down-projection is a compression bottleneck. 3072 candidates compete to
influence 768 output features. This is the "vote."

For **clear images**: Most candidates agree on the output. Dropping a few voters
doesn't change the election. Low variance.

For **ambiguous images**: Candidates disagree. Some say "dog features," others say
"cat features." Dropping a voter can swing the election. High variance.

### Why attention is dead

All 12 attention out_proj layers produce **exactly zero variance** under dropout.
Three reasons:

1. **Softmax saturation.** Attention weights are computed via softmax over
   key-query similarities. When one token dominates (common), the softmax output
   is ~[1, 0, 0, ...] regardless of small perturbations. Dropout on the output
   projection shifts all token representations similarly.

2. **Residual cancellation.** The residual connection adds the attention output
   to the input. Dropout shifts the attention output → the shifted output +
   original input ≈ original input + small perturbation → the residual absorbs
   the perturbation.

3. **Routing vs transformation.** Attention decides WHICH tokens to look at
   (routing). MLP decides WHAT features to extract (transformation). Uncertainty
   about "what" matters; uncertainty about "which token to attend to" doesn't
   change the output meaningfully for single-image classification.

### Why c_fc (the up-projection) hurts

Dropout on c_fc (768 → 3072) drops input features BEFORE the nonlinearity. This
means:
- Dropped features are missing from ALL 3072 candidates
- The GELU can't compensate because it never sees the dropped inputs
- The resulting noise is correlated across all 3072 candidates
- This creates systematic bias rather than probing redundancy

Dropout on c_proj (3072 → 768) drops candidates AFTER the nonlinearity:
- Each dropped candidate is independent
- Remaining candidates still have access to all input features
- The vote is properly "thinned" — some voices removed, but remaining voices
  are fully informed
- This cleanly probes redundancy

---

## Act 5: The Metric Completes the Picture

### The dimension structure of VLM features

Not all feature dimensions are equal. In CLIP B/32's 512 dimensions:
- ~64 are highly discriminative (vary a lot across images — encode identity)
- ~200 are moderately discriminative
- ~250 are near-zero discriminative (stable across all images — encode "image-ness")

### Why weighted_trace_pre works

Plain trace_pre averages MC dropout variance over all 512 dimensions equally.
The ~250 non-discriminative dimensions contribute noise — they carry MC variance
(dropout changes them) but this variance doesn't correlate with visual ambiguity
(it's just random noise in stable features).

weighted_trace_pre asks: "for each dimension, how much does it vary across different
images in the batch?" Dimensions with high across-image variance encode **visual
distinctions** — the features the model uses to tell images apart. These are exactly
the dimensions that:

1. Carry classification-relevant information
2. Are most damaged by degradation (blur destroys the fine features they encode)
3. Have the most meaningful dropout variance (the model USES these dims for decisions)

### For PE-Core this is critical

PE-Core's 5.4B training pairs made most dimensions hyper-stable. The discriminative
signal is concentrated in a small subset of dimensions. Plain trace_pre is dominated
by the stable majority → fails (61% on downsample). weighted_trace_pre finds the
active minority → passes (84%).

This is why the metric gap is larger on PE-Core (trace: 61% → weighted: 84%) than
on CLIP B/32 (trace: 91% → weighted: 97%). More robust models have more non-
discriminative dimensions, making the weighting more important.

---

## Act 6: Why Gaussian/Quantization Noise Fails

### What Gaussian measures

Adding Gaussian noise to weights and measuring output variance computes:

```
E[||f(x, w+ε) - f(x, w)||²] ≈ ||∇_w f(x, w)||² · σ²
```

This is the **Jacobian norm** — how sensitive the output is to weight perturbations.
It's a fixed geometric property of each input image's position in activation space.

Degraded images have smoother, lower-magnitude activations → smaller Jacobian →
LESS sensitivity to noise → LOWER uncertainty. The signal is inverted relative to
what valid uncertainty should show.

### What dropout measures

Dropout randomly ablates neurons, producing:

```
Var_mask[f(x, mask)] = function of subnetwork agreement
```

This is NOT a local derivative. It's a **combinatorial** property — how much do
different subnetworks (exponentially many) agree on the output? There's no simple
closed-form. It depends on the redundancy structure of the full network, not the
local curvature.

### The mathematical distinction

- **Gaussian**: continuous perturbation → local linearization → Jacobian → geometry
- **Dropout**: discrete ablation → subnetwork sampling → redundancy → combinatorial

These are fundamentally different quantities. The Jacobian tells you about the
LOCAL landscape around the current weights. Dropout tells you about the GLOBAL
structure of subnetwork agreement. Classification uncertainty lives in the global
structure, not the local landscape.

### Quantization noise is just Gaussian in disguise

Randomizing the low bits of quantized weights is dense continuous noise with uniform
distribution. It measures the same Jacobian as Gaussian noise. Different distribution,
same first-order approximation, same failure mode.

The **sparsity** and **binary nature** of dropout is load-bearing. It's not "adding
noise" — it's asking "can this network still function with this pathway deleted?"
That's a categorically different question from "how much does the output wiggle when
I jiggle the weights?"

---

## The Unifying Principle

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

Remove any one factor:
- Wrong loss (SigLIP2) → no inter-class tension to probe → fails
- Wrong layers (early blocks, attention) → probing perception, not decisions → noise
- Wrong metric (plain trace, spectral) → signal drowned by non-discriminative dims → weak
- Wrong perturbation (Gaussian, quantization) → measures geometry, not redundancy → inverted

All four aligned → valid, reliable uncertainty from a frozen model with no training.

---

## Open Theoretical Questions

1. **Is there a formal connection between MLP redundancy and Bayesian posterior
   uncertainty?** Our method feels like approximate Bayesian inference but the
   standard MC dropout → variational inference argument (Gal & Ghahramani 2016)
   assumes dropout was used during training. We're applying it to frozen models
   with no training-time dropout.

2. **Can we predict the optimal late-block count from model properties?** CLIP B/32
   works with all 12. PE-Core needs only 3. L/14 uses all 24 (but we didn't test
   late-only). Is there a rule? Perhaps related to the effective depth of the
   network or some measure of per-block feature robustness.

3. **Why does the reliability-validity tradeoff track training data MORE than model
   size?** PE-Core B/16 (86M params, 5.4B pairs) is more robust than CLIP L/14
   (304M params, 400M pairs). Training data volume seems to dominate parameter
   count for determining feature robustness. This is consistent with scaling laws
   but worth formalizing.

4. **Is there a perturbation type between dropout and Gaussian that gets both?**
   Dropout is sparse binary (high validity, low reliability). Gaussian is dense
   continuous (high reliability, zero validity). Is there a sweet spot — e.g.,
   sparse continuous, or structured dropout — that probes redundancy more
   efficiently?
