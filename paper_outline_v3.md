# MC Dropout as Measurement Uncertainty for Vision Encoders in Sensor Fusion

## v3 Master Reference Document

**Author:** Franky
**Last updated:** February 2026
**Status:** Pre-experiment comprehensive plan

---

# PART I — THE PAPER

---

## 1. ABSTRACT (Draft)

Sensor fusion frameworks require per-measurement noise characterization for each sensor. In multi-object tracking (MOT), data association balances motion predictions against appearance similarity — but the appearance encoder's reliability varies per detection. We investigate whether Monte Carlo Dropout (MCDO), applied post-hoc to frozen vision encoders, can estimate per-detection appearance uncertainty for adaptive association weighting.

We propose that MCDO on frozen encoders samples over implicit sub-networks, and its effectiveness depends on whether training produced *independently coherent sub-networks* — representations that degrade gracefully under partial ablation. We test this across model families: CLIP ViT-B/32 (87M, contrastive-only training) and SigLIP 2 (86M–1B, trained with masked prediction and self-distillation). We introduce a nested Monte Carlo methodology with intraclass correlation analysis to rigorously separate estimator noise from genuine inter-detection uncertainty signal.

Our evaluation targets *ambiguity prediction* — does MCDO variance predict when the encoder is internally conflicted about a detection? — and culminates in an adaptive MOT association experiment where appearance distance is downweighted for uncertain detections. We compare MCDO against post-hoc Laplace approximation (BayesVLM / projection-layer Laplace) and provide practical guidance for practitioners integrating vision encoder uncertainty into tracking and fusion pipelines.

**Keywords:** Monte Carlo Dropout, CLIP, SigLIP, uncertainty estimation, sensor fusion, Kalman filter, multi-object tracking, data association, appearance embedding, vision-language models

---

## 2. THESIS & THEORETICAL FRAMEWORK

### 2.1 The Engineering Problem

In multi-object tracking, a Kalman filter (or variant) predicts each track's state (position, velocity). New detections arrive each frame from a detector like YOLO. The tracker must solve *data association*: which detection belongs to which track?

Standard approach: compute a cost matrix combining motion distance and appearance distance, then solve assignment (e.g., Hungarian algorithm). The cost for assigning detection j to track i is typically:

```
cost(i,j) = α · d_motion(i,j) + β · d_appearance(i,j)
```

Where d_motion is Mahalanobis distance from the KF prediction, and d_appearance is cosine distance between the track's appearance embedding and the detection's appearance embedding (from a vision encoder like CLIP).

**The problem:** α and β are currently *fixed*. But the vision encoder's reliability varies per detection:
- Clear, well-lit, unoccluded detections → encoder is reliable → trust appearance more
- Blurry, occluded, unusual-angle detections → encoder is unreliable → trust motion more

**What we want:** A per-detection uncertainty scalar s(x) such that:

```
cost(i,j) = α · d_motion(i,j) + β · d_appearance(i,j) / s(x_j)
```

When s(x_j) is high (uncertain detection), appearance distance is downweighted. When s(x_j) is low (confident detection), appearance contributes fully.

**This is the adaptive R concept from Kalman filter theory, applied to appearance-based association.**

### 2.2 The Core Hypothesis

**Empirical claim (what we test):**

MCDO on a frozen vision encoder estimates *expected representation change under structured unit ablation*. This quantity — Tr(Σ_MC(x)) — is useful as a measurement uncertainty scalar **if and only if**:

1. **(a) Input-dependent with SNR > 1:** The between-image variance of Tr(Σ_MC) exceeds the within-image estimator noise. That is, different images produce detectably different uncertainty levels.
2. **(b) Predictive of downstream ambiguity:** Images where Tr(Σ_MC) is high should correspond to images where the encoder is genuinely conflicted (low classification margin, unstable retrieval ranking).

**Explanatory hypothesis (why we think it works or fails):**

The sub-network coherence hypothesis provides the mechanistic explanation. A frozen encoder contains implicit sub-networks — subsets of neurons that collectively process input. Each dropout mask activates a different sub-network. If these sub-networks are *independently coherent* (each produces a meaningful, if noisy, embedding on its own), then variance across sub-networks reflects genuine disagreement about the input. This is epistemic uncertainty in the Bayesian sense.

Whether a model has independently coherent sub-networks depends on training:

- **Contrastive-only training (CLIP):** Optimizes the *full* network to produce one correct embedding. Neurons may develop entangled, non-redundant roles. Dropping a neuron shifts the embedding in an arbitrary direction. MCDO variance reflects *how the network is wired*, not *what the input means*.

- **Masked prediction training (SigLIP 2):** The model must produce coherent representations from partial inputs (50% of patches masked during training). This forces redundancy — multiple pathways converge on the same representation. Sub-networks are more likely to be independently meaningful. MCDO samples genuine partial interpretations.

**What this framework predicts:**

| Model | Training Recipe | MCDO Prediction |
|-------|----------------|----------------|
| CLIP ViT-B/32 (87M) | Contrastive only | Low SNR, poor ambiguity prediction |
| SigLIP 2 ViT-B/16 (86M) | Contrastive + masked pred + self-distill | Higher SNR (training recipe effect) |
| SigLIP 2 So400m (400M) | Same | Higher still (scale effect) |
| CLIP ViT-L/14 (428M) | Contrastive only | Tests scale vs recipe: if worse than SigLIP2-B, recipe dominates |

### 2.3 What We Are NOT Claiming

- **Not "image uncertainty":** Uncertainty is a property of the *model-image pair*, not the image alone. We are characterizing a specific measurement device, like characterizing an IMU's noise profile.
- **Not absolute calibration:** We only need *rank preservation* — more uncertain detections consistently get higher scores.
- **Not claiming MCDO is Bayesian inference on frozen models:** We acknowledge the training-inference mismatch (Gal & Ghahramani 2016 assumes dropout during training). Our claim is empirical: under certain conditions, the variance is useful *regardless* of whether it satisfies the variational inference interpretation.

### 2.4 Formal Definitions

**Measurement function (appearance):** For detection x with bounding box crop, the vision encoder produces embedding e = f(x) ∈ ℝ^d. In tracking, the "measurement" is this embedding used to compute appearance similarity.

**MC Dropout uncertainty:** T stochastic forward passes with dropout mask z_t ~ Bernoulli(p):

```
e_t = f(x; θ ⊙ z_t),  t = 1..T
ē = (1/T) Σ_t e_t
Σ_MC = (1/(T-1)) Σ_t (e_t - ē)(e_t - ē)ᵀ
```

**Scalar uncertainty for association weighting:**

```
s(x) = u(Σ_MC)
```

Where u is one of: Tr(Σ)/d (trace per dimension), angular variance, or log-det per dimension (see Section 5.6).

**Relative uncertainty hypothesis:** For detections x_i, x_j, if the encoder is truly more uncertain about x_i than x_j, then s(x_i) > s(x_j). We require monotonicity of s with respect to true measurement quality, not absolute calibration.

---

## 3. INTRODUCTION

### 3.1 Vision Encoders as Sensors in Multi-Object Tracking

Modern multi-object tracking systems (DeepSORT, ByteTrack, BoT-SORT, StrongSORT) combine motion models (Kalman filters) with appearance models (vision encoder embeddings). YOLO-family detectors provide bounding boxes; a vision encoder (often a ReID model, but increasingly a foundation model like CLIP) extracts appearance features for data association.

The Kalman filter side is well-characterized: the state prediction covariance P and measurement noise covariance R together determine optimal fusion via the Kalman gain K = PH^T(HPH^T + R)^{-1}. For motion measurements (position, velocity), R is straightforward to estimate.

For appearance measurements, R is unknown and input-dependent. Currently, appearance distance is weighted by a fixed hyperparameter, which is equivalent to assuming constant measurement noise — an assumption that is clearly wrong (occluded detections should be trusted less than clear ones).

### 3.2 Why MC Dropout Is Appealing

- **Zero retraining:** Applies to any frozen encoder. No architectural changes needed.
- **Computational simplicity:** T stochastic forward passes, trivially parallelizable on GPU.
- **Directly produces covariance:** The sample covariance of MC embeddings is a candidate for R.
- **Theoretically grounded** (Gal & Ghahramani 2016) — but under assumptions we examine critically.
- **Model-agnostic:** Same approach works across CLIP, SigLIP 2, ReID models, etc.

### 3.3 The Gap

- MCDO was validated on models *trained* with dropout, for classification/regression.
- No study of MCDO on frozen VLM encoders for appearance uncertainty in tracking/fusion.
- Existing VLM uncertainty methods (ProLIP, BayesVLM, ProbVLM) target calibration and OOD detection, not data association weighting.
- No investigation of how *training recipe* affects MCDO validity.
- No rigorous methodology for separating MCDO estimator noise from genuine uncertainty signal.

### 3.4 Contributions

1. **Problem framing:** We formulate vision encoder uncertainty as an *appearance measurement noise* problem for adaptive data association in MOT — distinct from classification calibration or OOD detection.
2. **Nested MC methodology:** We introduce a nested Monte Carlo design with ICC analysis that rigorously separates estimator noise from true inter-image signal, with explicit go/no-go thresholds.
3. **Training recipe hypothesis:** We propose and empirically test that MCDO effectiveness depends on whether training produced independently coherent sub-networks, with a clean matched-parameter comparison (CLIP-B/32 vs SigLIP2-B/16).
4. **Comprehensive ablations:** Layer-type-specific dropout (attention probabilities vs MLP hidden activations vs stochastic depth), pre-norm vs post-norm embedding space, scale-free uncertainty metrics.
5. **Ambiguity-based evaluation:** We evaluate against *model ambiguity* (classification margin, retrieval gap) rather than correctness, properly testing whether MCDO captures the encoder's internal conflict.
6. **MOT integration demo:** Adaptive appearance weighting in data association, evaluated with HOTA/MOTA/ID-switches.

---

## 4. LITERATURE REVIEW

### 4.1 Multi-Object Tracking & Data Association

Kalman (1960) established optimal state estimation with known measurement noise. Modern MOT extends this: DeepSORT (Wojke et al., 2017) introduced appearance embeddings for association; ByteTrack (Zhang et al., 2022) and BoT-SORT (Aharon et al., 2022) improved detection-level handling. All use fixed appearance weighting.

Adaptive measurement noise in tracking has been studied for traditional sensors (Sage-Husa filters, innovation-based adaptation), but not for neural network appearance features. GNN and JPDA (Bar-Shalom & Li, 1995) handle measurement-to-track assignment under uncertainty — but assume known measurement noise characteristics.

### 4.2 MC Dropout as Approximate Bayesian Inference

Gal & Ghahramani (2016) proved dropout training approximates variational inference over network weights, with the dropout distribution as the approximate posterior q(ω). **Critical assumption:** the model must be *trained* with dropout so that learned weights lie in the support of q(ω).

Concrete Dropout (Gal et al., 2017) learns per-layer dropout rates via continuous relaxation of the Bernoulli distribution. This is relevant because optimal p may vary by layer — and our layer-ablation experiments (Exp 3) explore this dimension manually.

Known limitations: MCDO can be overconfident far from training data (Ovadia et al., 2019), produces poorer uncertainty estimates than deep ensembles (Lakshminarayanan et al., 2017), and its quality depends heavily on dropout placement (Verdoja & Kyrki, 2021).

### 4.3 Uncertainty in Vision-Language Models

**ProLIP (Chun et al., 2025):** Probabilistic VLM trained from scratch with an uncertainty token that outputs per-embedding variance. Captures aleatoric uncertainty. Requires full retraining — not applicable post-hoc.

**BayesVLM (Baumann et al., 2024):** Post-hoc Laplace approximation over the final projection layers. Estimates epistemic uncertainty analytically from the Hessian of the loss at the learned weights. No retraining required. Closest alternative to our approach.

**ProbVLM (Upadhyay et al., 2023):** Trains a probabilistic adapter on frozen embeddings. Requires adapter training but not encoder retraining.

### 4.4 Sub-Network Structure and Robustness to Ablation

The Lottery Ticket Hypothesis (Frankle & Carbin, 2018) demonstrated that dense networks contain sparse sub-networks capable of matching full-network performance. This supports the notion that dropout samples over functional sub-networks.

Masked Autoencoders (He et al., 2022) showed that ViTs trained with 75% patch masking learn robust, distributed representations. SigLIP 2 (Tschannen et al., 2025) incorporates 50% masked prediction during training, directly building in the partial-information robustness we hypothesize is necessary for meaningful MCDO.

DINO/DINOv2 (Caron et al., 2021; Oquab et al., 2023) use self-distillation with local-to-global consistency, training the student on partial crops to match teacher on full images — another form of partial-information training.

### 4.5 Uncertainty Taxonomy for This Paper

**Aleatoric uncertainty:** Irreducible noise from the data itself (blur, occlusion, sensor noise). Present even with a perfect model. Probe: fix the model, vary the input.

**Epistemic uncertainty:** The model's lack of knowledge — it hasn't seen enough similar examples to be confident. Reducible with more data. Probe (ideally): posterior over weights. Proxy: MCDO if sub-networks are coherent.

**Representational sensitivity:** What MCDO *definitely* measures on frozen models — how much does the output change when you randomly ablate internal units? This is Var_z[f(x; θ ⊙ z)]. Our empirical question: does this correlate with epistemic uncertainty?

**Distributional uncertainty:** How far the input is from training data. OOD inputs should get high uncertainty. Probe: compare in-distribution vs. shifted inputs.

**The key point for this paper:** We don't need to resolve the philosophical question of whether representational sensitivity "is" epistemic uncertainty. We need to determine empirically whether it's *useful* as a measurement noise proxy — specifically, whether it predicts when the encoder is unreliable.

---

## 5. METHOD

### 5.1 System Architecture

```
[Camera Frame]
    │
    ▼
[YOLO Detector] → bounding box detections {x_1, ..., x_M}
    │
    ▼ (crop each detection)
[Vision Encoder f(·)] → appearance embeddings {e_1, ..., e_M}
    │
    │  ← THIS IS WHERE WE ADD MCDO
    │     T stochastic passes per detection → Σ_MC per detection
    │     s(x_j) = uncertainty scalar from Σ_MC
    │
    ▼
[Data Association]
    cost(i,j) = α · d_motion(i,j) + β · d_appearance(i,j) / s(x_j)
    │
    ▼
[Kalman Filter Update] → updated tracks
```

### 5.2 MC Dropout Protocol

For each detection crop x:

```
For t = 1..T:
    Sample dropout mask z_t ~ Bernoulli(p) at specified layers
    e_t = f(x; θ ⊙ z_t)         # stochastic forward pass

# Pre-normalization covariance (recommended primary metric):
Σ_pre = SampleCovariance({e_1, ..., e_T})     # before L2 norm

# Post-normalization angular variance (secondary metric):
ê_t = e_t / ||e_t||                            # L2 normalize
ê_0 = mean(ê_t)  / ||mean(ê_t)||              # normalized mean
angular_var = Var_t(arccos(ê_t · ê_0))         # variance of angles

# Scalar uncertainty:
s(x) = Tr(Σ_pre) / d                           # trace per dimension
```

### 5.3 Pre-Normalization vs Post-Normalization Embedding Space

**Why this matters (pedagogical explanation):**

CLIP and SigLIP 2 both L2-normalize their final embeddings so that all embeddings live on a unit hypersphere (||e|| = 1). This normalization is important for cosine similarity to work, but it creates problems for covariance estimation.

**The problem with post-norm covariance:** When all vectors have unit length, they're constrained to a sphere. If you perturb one dimension, the others *must* change to maintain ||e|| = 1. This creates artificial correlations between dimensions — the "high off-diagonal mass" we observed in early experiments (~3.4k off-diagonal covariance) is largely an artifact of this constraint, not meaningful structure.

**Pre-norm features** live in unconstrained ℝ^d. Their covariance reflects actual feature variation without spherical artifacts. The principal components of pre-norm covariance are more interpretable — they represent actual directions of variation in the encoder's internal representation.

**Angular variance** is the natural metric on the sphere. Instead of asking "how far apart are the endpoints?" (Euclidean), it asks "how spread out are the directions?" (angular). This is immune to the spherical constraint issue and is directly meaningful for cosine-similarity-based association.

**Protocol: compute both and compare.** Report Tr(Σ_pre)/d as the primary metric, angular variance as secondary, and Tr(Σ_post)/d for reference. If pre-norm and angular variance agree but post-norm disagrees, the post-norm results are artifactual.

### 5.4 Nested Monte Carlo Design

A single MCDO run of T passes produces one estimate û = Tr(Σ_MC)/d. But this estimate is itself a *random variable* — run it again with different dropout masks and you'll get a different number. With small T, the estimate is very noisy.

**The nested design separates two sources of variation:**

```
For each image x_i (i = 1..N):
    For trial k = 1..K:                    # K independent MCDO runs
        For pass t = 1..T:                  # T MC passes per run
            e_ikt = f(x_i; θ ⊙ z_ikt)      # stochastic forward pass
        Σ_ik = SampleCovariance({e_ik1, ..., e_ikT})
        u_ik = Tr(Σ_ik) / d               # uncertainty from trial k

    # Per-image aggregation:
    û_i = mean(u_i1, ..., u_iK)            # best estimate of this image's uncertainty
    within_var_i = Var(u_i1, ..., u_iK)     # how noisy is our estimate?
```

**Three critical outputs:**

**1. Signal-to-Noise Ratio (SNR):**
```
Signal = Var_i(û_i)                    # how much does uncertainty differ BETWEEN images?
Noise  = Mean_i(within_var_i)          # how noisy is each estimate?
SNR    = Signal / Noise
```
If SNR >> 1: rankings are meaningful (real between-image differences dominate estimator noise).
If SNR ≈ 1: rankings are unreliable (you can't tell images apart through the noise).

**2. Intraclass Correlation Coefficient (ICC):**
ICC measures the proportion of total variance that comes from between-image differences vs. within-image estimator noise. Ranges from 0 (pure noise) to 1 (perfectly reliable).
- ICC > 0.75: "good" reliability (Koo & Li, 2016)
- ICC > 0.9: "excellent" reliability
This is the standard measurement reliability metric from psychometrics — exactly the right tool for asking "is this instrument producing consistent readings?"

**3. Pairwise trial rank stability:**
For each pair of trials (k_a, k_b), compute Spearman ρ and Kendall τ between the image rankings. Report the distribution (median, IQR) across all (K choose 2) pairs.

**Go/no-go thresholds:**
- **Usable:** median pairwise Spearman ρ ≥ 0.8 AND SNR ≥ 2 AND ICC ≥ 0.75
- **Marginal:** ρ ∈ [0.6, 0.8) or SNR ∈ [1, 2)
- **Failed:** ρ < 0.6 or SNR < 1

**Recommended parameters:** K = 10 trials, T = 64 passes, N ≥ 500 images.
Also run with T ∈ {4, 16, 64} at K = 10 to quantify where estimator noise stops dominating (i.e., find the minimum T where ICC plateaus).

### 5.5 Dropout Application Strategies

In a Vision Transformer, there are distinct perturbation types. "Dropping attention layers" is ambiguous — here are the precise options:

**Perturbation Type A — Attention probability dropout:**
Apply dropout to the attention weight matrix *after* softmax, before multiplying with V.
Location: inside `MultiheadAttention`, on `attn_weights`.
Effect: randomly zeroes out attention connections between patches → some patches can't "see" some other patches.

**Perturbation Type B — MLP hidden activation dropout:**
Apply dropout to the hidden layer of the feed-forward block (between the two linear layers).
Location: inside each transformer block's MLP, after GELU activation.
Effect: randomly zeroes intermediate feature computations.

**Perturbation Type C — Stochastic depth (drop entire residual blocks):**
With probability p, skip an entire transformer block (replace its output with the identity/residual).
Effect: the coarsest ablation — removes an entire processing "stage." May produce the most coherent sub-networks because you're dropping functional units, not random neurons.

**Perturbation Type D — Projection dropout:**
Apply dropout only to the final linear projection from ViT hidden dim to embedding dim.
Effect: the narrowest intervention — only perturbs the final mapping.

**Perturbation Type E — Uniform (current approach):**
Apply the same dropout to all linear layers throughout the network.
Effect: maximum perturbation but confounds all the above types.

Each type probes a different aspect of the architecture and may produce different uncertainty signals. Experiment 3 tests these systematically.

### 5.6 Scale-Free Uncertainty Metrics

When comparing models with different embedding dimensions (CLIP: 512, SigLIP 2-B: 768, So400m: 1152, ViT-g: 1536), raw Tr(Σ) is meaningless — bigger embeddings have bigger traces mechanically. We need scale-free metrics:

**Trace per dimension:** u_trace = Tr(Σ) / d
Interpretation: average per-dimension variance. Simple, interpretable.

**Log-determinant per dimension:** u_logdet = log det(Σ + εI) / d
Interpretation: entropy of a Gaussian with this covariance, normalized by dimension. Captures the *volume* of the uncertainty ellipsoid, accounting for correlations. (ε is a small regularizer, e.g., 1e-10, to handle singular matrices.)

**Anisotropy ratio:** u_aniso = λ_max / Tr(Σ)
Interpretation: what fraction of total variance is concentrated in the top principal component? High anisotropy → uncertainty is structured (one dominant direction). Low anisotropy → uncertainty is isotropic (spread equally). If MCDO is measuring meaningful uncertainty, we'd expect *some* anisotropy — uncertainty concentrated in specific feature dimensions rather than uniform noise.

**Angular variance:** u_angular = Var_t(arccos(ê_t · ê_0))
Interpretation: how spread out are the normalized embeddings on the hypersphere? Naturally scale-free, directly meaningful for cosine-similarity-based metrics.

**For all cross-model comparisons, report at minimum: u_trace AND u_angular.**

### 5.7 Separating Aleatoric and Epistemic Components

Two complementary perturbation experiments:

**Aleatoric probe (fix model, vary input):**
Apply realistic sensor degradations to the input while keeping the model deterministic:
- JPEG quality sweep: quality ∈ {100, 80, 60, 40, 20}
- Gaussian blur: σ_blur ∈ {0, 1, 2, 4, 8}
- Random occlusion: mask 0%, 10%, 25%, 50% of detection crop
- Additive Gaussian noise: σ ∈ {0, 0.01, 0.05, 0.1} (keep as standard baseline)

For each degraded input, compute the deterministic embedding. The variance of embeddings across degradation levels = sensitivity to input noise = aleatoric proxy.

**Epistemic probe (fix input, vary model):**
Apply MCDO to the model while keeping the input fixed.
Variance across dropout masks = sensitivity to model perturbation = epistemic proxy (if sub-networks are coherent).

**Key analysis:** Compute both probes for the same set of images. If they correlate, MCDO may be conflating aleatoric and epistemic. If they capture different signals (different images rank high on each), they provide complementary information — potentially both useful for R.

### 5.8 Connection to Semantic Decomposition

Beyond tracking, vision encoders are used for semantic decomposition — breaking an embedding into interpretable semantic components. Two methods:

**Method 1: Cosine similarity distribution.** Compute cos(e, text_embedding_i) for semantic concepts i → probability distribution over semantics. MCDO variance in this distribution = uncertainty about *which* semantic concepts apply.

**Method 2: PCA-based factorization.** Find the principal subspace of embeddings over representative semantic samples. Project new embeddings onto this subspace. MCDO variance in projection = uncertainty about semantic decomposition.

**Diagnostic for signal vs noise in semantic space:**
- Variance isotropic across all directions → architectural noise → uninformative
- Variance concentrated along semantic axes → meaningful uncertainty about concepts
- Variance concentrated orthogonal to semantic subspace → off-manifold perturbation → noise

---

## 6. EXPERIMENTS

### Experiment 0: Nested MC Estimator Validation [PHASE 1 — RUN FIRST]

**Goal:** Determine whether MCDO produces detectable signal above estimator noise. Establish minimum T. Go/no-go for entire project.

**Setup:**
- Models: CLIP ViT-B/32, SigLIP 2 ViT-B/16, SigLIP 2 So400m
- Dataset: ImageNet val (N = 500)
- Dropout: p = 0.01, uniform all layers
- Design: K = 10 trials, T ∈ {4, 16, 64}
- Compute on BOTH pre-norm and post-norm embeddings

**Analysis:**
1. Per-T: ICC, SNR, distribution of pairwise Spearman ρ across 45 trial pairs
2. T-sweep: at which T does ICC plateau? This is the minimum T for all subsequent experiments
3. Pre-norm vs post-norm: which space gives higher SNR? (Expect pre-norm to be better)
4. Cross-model comparison: which model has highest SNR / ICC?

**Go/no-go:**
- If ICC < 0.5 for all models at T=64 on pre-norm features: MCDO is fundamentally noisy for frozen encoders. Paper pivots to pure negative result + BayesVLM comparison.
- If ICC > 0.75 for SigLIP 2 but < 0.5 for CLIP: Training recipe hypothesis has support. Proceed.
- If ICC > 0.75 for all models: Prior rank instability was due to T=4 and/or post-norm artifacts. Proceed with strong positive framing.

---

### Experiment 0b: Pre-Norm vs Post-Norm + Angular Variance [PHASE 1]

**Goal:** Determine which embedding space yields the most meaningful covariance structure. Resolve whether early "global noise injection" finding was a normalization artifact.

**Setup:**
- Model: CLIP ViT-B/32
- Dataset: Same 500 images from Exp 0
- p = 0.01, T = 64, K = 5

**For each image, compute:**
- Σ_pre (covariance on pre-normalization features)
- Σ_post (covariance on L2-normalized embeddings)
- Angular variance: Var(arccos(ê_t · ê_0))
- Eigenspectrum of Σ_pre and Σ_post (top-10 eigenvalues)
- Off-diagonal mass ratio: ||Σ - diag(Σ)||_F / ||Σ||_F

**Expected findings:**
- Σ_post will show high off-diagonal correlation (spherical constraint artifact)
- Σ_pre will show more structured, interpretable covariance
- Angular variance will agree more with Tr(Σ_pre)/d than Tr(Σ_post)/d
- This resolves the car sim "off-diagonal mass ~3.4k" finding

---

### Experiment 1: Rank Stability Across Dropout Rates (Hyperparameter Robustness) [PHASE 2]

**Goal:** Determine how sensitive uncertainty rankings are to the choice of p. This is a *hyperparameter robustness* test — not a validity test.

**Reframing from v2:** Different dropout rates probe different *ablation regimes*. At p=0.01, you make tiny perturbations — the network is 99% intact and you're probing fine-grained sensitivity. At p=0.1, the network is substantially damaged and may operate in a qualitatively different mode (residual connections dominate, main pathways too disrupted). Comparing rankings at p=0.01 vs p=0.1 is like comparing two different instruments, not two readings from the same instrument.

**Why we still care:** Practitioners need to know if their choice of p matters a lot or a little. If rankings are stable across p ∈ [0.005, 0.02], the method isn't brittle and p-selection is forgiving. If rankings diverge wildly even between p=0.005 and p=0.01, the method requires careful tuning, which limits practical adoption.

**Setup:**
- Models: CLIP ViT-B/32, SigLIP 2 So400m
- Dataset: ImageNet val (N = 1000)
- p ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1}, uniform all layers
- T = 64, K = 3 trials per (image, p)
- Pre-norm features (based on Exp 0b findings)

**Analysis:**
- Spearman ρ matrix across all p-pairs, per model
- Identify "stable range" of p where ρ > 0.8 (if it exists)
- Compare stability range: wider for SigLIP 2 than CLIP? (predicted by sub-network hypothesis)

---

### Experiment 2: Synthetic & Natural Image Baselines [PHASE 2]

**Goal:** Test whether MCDO variance reflects model familiarity / training distribution proximity.

**Setup:**
- Categories: solid colors (10), gradients (10), uniform noise (10), **natural images from ImageNet (10)**
- Models: CLIP ViT-B/32, SigLIP 2 So400m
- p = 0.01, T = 64, K = 5
- Pre-norm features

**Prior results (CLIP, T=64, post-norm):**
Solid (0.112) > Noise (0.090) ≈ Gradient (0.090)

**Predictions:**
- If MCDO captures training-distribution distance: Natural < Gradient < Noise < Solid
- If architectural noise: All ≈ equal
- SigLIP 2 should show a clearer pattern (if sub-network hypothesis holds)

**Additional:** Compute image entropy, edge density. Correlate with uncertainty.

---

### Experiment 3: Dropout Type Ablation [PHASE 2 — KEY]

**Goal:** Identify which *type* of dropout produces the most stable and meaningful uncertainty signal.

**Setup:**
- Model: CLIP ViT-B/32 (and SigLIP 2 So400m if promising)
- Dropout types (precisely defined):
  - **A: Attention probability dropout** — on post-softmax attention weights
  - **B: MLP hidden dropout** — on intermediate activations in feed-forward blocks
  - **C: Stochastic depth** — skip entire residual blocks with probability p
  - **D: Projection-only dropout** — on final projection layer only
  - **E: Uniform** — all linear layers (current baseline)
- p = 0.01, T = 64, K = 5
- Dataset: ImageNet val (N = 1000), pre-norm features

**Analysis:**
1. ICC and SNR per dropout type
2. Pairwise trial rank stability per type
3. Anisotropy: which type produces the most structured (non-isotropic) variance?
4. Cross-type rank correlation: do different dropout types produce correlated or orthogonal rankings?

---

### Experiment 4: Cross-Model Comparison (Training Recipe × Scale) [PHASE 1 — CENTRAL]

**Goal:** Test whether training recipe or model scale is the primary determinant of MCDO effectiveness.

**Models:**

| Model | Params | Emb Dim | Training Recipe | Role |
|-------|--------|---------|----------------|------|
| CLIP ViT-B/32 | 87M | 512 | Contrastive only | Baseline |
| SigLIP 2 ViT-B/16 | 86M | 768 | Contrastive + masked pred + self-distill | **Matched-params comparison** |
| SigLIP 2 So400m/14 | 400M | 1152 | Same | Scale comparison |
| CLIP ViT-L/14 | 428M | 768 | Contrastive only | **Scale vs recipe control** |
| SigLIP 2 ViT-g/16 | 1B | 1536 | Same | Maximum scale |

**The critical comparisons:**
- CLIP-B/32 (87M) vs SigLIP2-B/16 (86M): **same parameter count, different training → isolates recipe effect**
- SigLIP2-B vs So400m vs ViT-g: **same recipe, increasing scale → isolates scale**
- CLIP-L/14 (428M) vs SigLIP2-So400m (400M): **similar scale, different recipe → recipe at scale**
- If SigLIP2-B (86M, masked) outperforms CLIP-L (428M, contrastive): **training recipe dominates scale**

**Setup:** p = 0.01, T = 64, K = 10, N = 500, pre-norm features.
**Metrics:** ICC, SNR, pairwise ρ, using scale-free metrics (Tr/d, angular variance).

---

### Experiment 5: Ambiguity Prediction (Validation) [PHASE 2 — GROUND TRUTH]

**Goal:** The ultimate test — does MCDO uncertainty predict when the encoder is genuinely conflicted?

**Why ambiguity, not correctness (pedagogical explanation):**

Consider two misclassified images:
- Image A: Top scores are [husky: 0.42, wolf: 0.40]. The model is *genuinely uncertain* — MCDO should show high variance.
- Image B: Top scores are [golden retriever: 0.95, labrador: 0.005]. The model is *confident but wrong* — MCDO should show low variance.

If we evaluate on correct/incorrect, both are "wrong" and we're asking MCDO to predict both kinds. But MCDO should only predict the first kind (genuine ambiguity). The *margin* (top1 score − top2 score) captures this distinction.

**Setup:**
- Dataset: ImageNet val (5K-10K images)
- Models: Best from Exp 4
- p = 0.01, T = 64 (single run sufficient if ICC is high from Exp 0)

**Ambiguity metrics for zero-shot classification:**
- **Margin:** top-1 logit − top-2 logit. Low margin = model is conflicted.
- **Entropy:** H = -Σ p_i log p_i over class logits. High entropy = model is conflicted.
- **Prompt sensitivity:** Variance of classification score across different prompt templates (e.g., "a photo of a {class}" vs "a {class}" vs "an image of a {class}"). High variance = unstable classification.

**Ambiguity metrics for retrieval (COCO val, 5K):**
- **Retrieval gap:** sim(best caption) − sim(2nd best caption). Low gap = ambiguous match.
- **Rank of correct caption:** Higher rank = harder retrieval.

**Evaluation:**
- Spearman ρ between s(x) and each ambiguity metric
- AUROC for detecting low-margin cases (e.g., bottom 10% margin)
- AUROC for detecting high-entropy cases
- Correlation between s(x) and negative margin (should be positive)

---

### Experiment 6: BayesVLM / Laplace Comparison [PHASE 3]

**Goal:** Compare MCDO against post-hoc Laplace approximation as an alternative zero-retraining uncertainty method.

**Primary approach:** BayesVLM (if codebase is available and tractable).

**Fallback (if BayesVLM is impractical):**
- Projection-layer Laplace: fit a Laplace approximation to just the final linear projection of the vision encoder (using frozen features as input). This is a standard ridge-regression Laplace — implementable in an afternoon.
- Alternatively: deep ensemble over linear heads (train 5 linear classifiers on frozen features, use disagreement as uncertainty). Cheap, no encoder modification.

**Comparison dimensions:**

| Criterion | MCDO | Laplace (BayesVLM) |
|-----------|------|---------------------|
| Per-image inference cost | T × forward pass | 1 forward pass + matrix multiply |
| Stochastic? | Yes (needs K trials for CI) | No (deterministic) |
| Setup cost | Zero | Hessian computation (one-time) |
| What it captures | Representational sensitivity | Weight-space posterior curvature |

**Evaluate both on Exp 5 ambiguity metrics.** The better method for ambiguity prediction is the better choice for adaptive association.

---

### Experiment 7: Aleatoric vs Epistemic Separation [PHASE 3]

**Goal:** Determine whether input perturbation variance and dropout variance capture different signals.

**Setup:**
- N = 500 images
- Aleatoric probe: JPEG quality ∈ {100, 80, 60, 40, 20}, blur σ ∈ {0, 1, 2, 4}, occlusion ∈ {0%, 10%, 25%} — compute embedding for each degradation, measure variance across degradations
- Epistemic probe: MCDO with p = 0.01, T = 64 on clean input
- Models: CLIP ViT-B/32, SigLIP 2 So400m

**Analysis:**
1. Correlation between aleatoric and epistemic Tr/d across images
2. Which correlates better with Exp 5 ambiguity metrics?
3. Does combining both (e.g., sum or product) improve ambiguity prediction?
4. From prior data: dropout-on variance dominated input-perturbation variance in car sim. Does this hold at T=64 with SigLIP 2?

---

### Experiment 8: Semantic Space Uncertainty [PHASE 3 — EXTENSION]

**Goal:** Test whether MCDO variance has meaningful *direction*, not just magnitude, in semantic space.

**Setup:**
- 20 text prompts spanning diverse categories
- Cosine similarity distribution variance across MC passes
- PCA projection: measure variance in-subspace vs orthogonal-to-subspace
- Test with semantically ambiguous images (should show high variance in competing semantic dimensions) vs unambiguous images

**Key diagnostic:**
- Isotropic variance across all directions → architectural noise
- Variance concentrated along semantic axes → meaningful semantic uncertainty
- Variance concentrated orthogonal to semantic subspace → off-manifold noise

---

### Experiment 9: MOT Association Demo [PHASE 3 — CAPSTONE]

**Goal:** End-to-end demonstration that adaptive appearance weighting improves tracking.

**Setup:**
- Tracker: DeepSORT-style or BoT-SORT-style with KF motion model
- Detection: YOLO (pre-computed detections on standard benchmark)
- Appearance: Vision encoder (best model from Exp 4)
- Dataset: MOT17 or MOT20 benchmark sequences

**Association cost formulations:**
```
# Baseline (fixed weighting):
cost(i,j) = α · d_mahalanobis(i,j) + β · d_cosine(i,j)

# Adaptive (our method):
cost(i,j) = α · d_mahalanobis(i,j) + β · d_cosine(i,j) / s(x_j)

# Where s(x_j) = MCDO uncertainty of detection j's appearance embedding
# Higher uncertainty → appearance distance is divided by larger number → downweighted
```

**Metrics:**
- **HOTA** (Higher Order Tracking Accuracy) — balances detection and association
- **MOTA** (Multiple Object Tracking Accuracy) — standard aggregate metric
- **ID switches** — how often tracks swap identities (directly measures association quality)
- **Fragmentation** — how often tracks are broken and restarted

**Controls:**
- Fixed α/β (current standard practice)
- Oracle s(x) from Exp 5 ambiguity (upper bound on adaptive weighting)
- MCDO-based s(x)
- Laplace-based s(x) (from Exp 6)

**This closes the loop:** even if MCDO itself is imperfect, demonstrating that *any* per-detection uncertainty improves association quality validates the overall framework.

---

### Experiment 10: Text Encoder Uncertainty [OPTIONAL EXTENSION]

**Goal:** Apply same methodology to text encoder. Test whether MCDO on text embeddings captures prompt specificity.

**Setup:** 1000 text prompts of varying specificity. Apply MCDO to text encoder.
**Prediction:** General prompts → higher variance (per ProLIP findings).

---

## 7. EXISTING RESULTS SUMMARY

### Data We Have

| Experiment | Dataset | Model | T | Norm | Finding |
|-----------|---------|-------|---|------|---------|
| Rank stability | CIFAR-100 (N=100) | CLIP-B/32 | 4 | Post | ρ ∈ [-0.13, +0.17] |
| Synthetic baselines | Synthetic (N=30) | CLIP-B/32 | 64 | Post | Solid (0.112) > Noise ≈ Gradient (0.090) |
| Accuracy vs dropout | MNIST, car sim | CLIP-B/32 | 10-16 | Post | Non-monotonic; p=0.01 mild benefit |
| Perturbation robustness | Car sim | CLIP-B/32 | varies | Post | Dropout variance dominates input variance |
| Covariance geometry | Car sim | CLIP-B/32 | varies | Post | Off-diagonal mass ~3.4k |
| Entropy/MI | MNIST | CLIP-B/32 | varies | Post | Entropy locked at ~2.29; MI negligible |

### Key observation about existing data

**Everything was run on CLIP ViT-B/32, mostly with post-norm embeddings and often with T=4.** The "negative results" may be attributable to three confounds:
1. **T=4 is far too few** (estimated ~50% relative error on variance estimates)
2. **Post-norm covariance** is dominated by spherical constraint artifacts
3. **CLIP's contrastive-only training** may lack sub-network coherence

Phase 1 experiments (Exp 0, 0b, 4) directly address all three confounds.

---

## 8. EXPERIMENT EXECUTION PLAN

### Phase 1 — Foundation (1 day GPU)

**Run these first. Everything else depends on the outcomes.**

| # | Experiment | Purpose | Go/No-Go Decision |
|---|-----------|---------|-------------------|
| 1 | Exp 0: Nested MC validation | Is there signal above noise? | ICC < 0.5 everywhere → negative result paper |
| 2 | Exp 0b: Pre-norm vs post-norm | Which space is meaningful? | Determines metric for all subsequent experiments |
| 3 | Exp 4 (subset): CLIP-B vs SigLIP2-B | Training recipe test | SigLIP2-B >> CLIP-B → recipe headline; SigLIP2-B ≈ CLIP-B → scale/other headline |
| 4 | Exp 5 (subset): Ambiguity validation on 5K | Does uncertainty predict margin? | No correlation → MCDO is not useful, regardless of ICC |

**If Phase 1 shows clear SigLIP 2 advantage:** Paper framing = Option A (training recipe determines MCDO validity). Proceed to Phase 2 with full experiments.

**If CLIP-L closes the gap with SigLIP2-B:** Paper framing = Option D (scale dominates). Add CLIP-L to all experiments.

**If no model shows useful signal:** Paper framing = Option B (diagnostic negative result + BayesVLM comparison).

### Phase 2 — Core Experiments (1-2 days GPU)

| # | Experiment | Purpose |
|---|-----------|---------|
| 5 | Exp 1: Rank stability across p | Hyperparameter robustness characterization |
| 6 | Exp 3: Dropout type ablation | Which perturbation type is best? |
| 7 | Exp 5 (full): Ambiguity on 10K + COCO | Full validation |
| 8 | Exp 2: Synthetic + natural baselines | Training distribution sensitivity |
| 9 | Exp 4 (full): All 5 models | Complete recipe × scale matrix |

### Phase 3 — Comparison, Diagnostics, Demo (1-2 days)

| # | Experiment | Purpose |
|---|-----------|---------|
| 10 | Exp 6: BayesVLM / Laplace comparison | Alternative method benchmark |
| 11 | Exp 7: Aleatoric vs epistemic | Diagnostic: different uncertainty types |
| 12 | Exp 8: Semantic space | Extension for semantic decomposition |
| 13 | Exp 9: MOT association demo | Capstone: adaptive weighting improves tracking |

### Phase 4 — Optional

| # | Experiment | Purpose |
|---|-----------|---------|
| 14 | Exp 10: Text encoder | Extension to text modality |
| 15 | Concrete Dropout | Learned per-layer rates (if Exp 3 is promising) |

---

## 9. PAPER FRAMING OPTIONS

### Option A: Training Recipe Determines MCDO Validity ★ (preferred)

**Title:** "When Does MC Dropout Work? Training Recipe as the Determinant of Measurement Uncertainty Quality in Vision Encoders for Tracking"

**Requires:** CLIP-B vs SigLIP2-B shows meaningfully higher ICC/SNR/ambiguity-prediction for SigLIP 2 at matched parameter count.

### Option B: Diagnostic Negative Result

**Title:** "Does MC Dropout Provide Useful Measurement Uncertainty for Vision Encoders? A Diagnostic Study"

**When to use:** No model shows useful signal. Paper becomes a cautionary tale + comparison with BayesVLM.

### Option C: Comprehensive Comparison

**Title:** "Post-Hoc Uncertainty for Visual Tracking: Comparing MC Dropout, Laplace, and Ensemble Methods for Adaptive Data Association"

**When to use:** Multiple methods show partial success; no single clear winner.

### Option D: Scale Dominates

**Title:** "Bigger Models, Better Uncertainty: How Encoder Scale Enables Dropout-Based Measurement Noise Estimation for Tracking"

**When to use:** CLIP-L closes the gap with SigLIP2-B, suggesting scale matters more than training recipe.

---

## 10. MODELS & DATASETS

### Models

| Model | HuggingFace ID | Params | Emb Dim | Layers | Patch | Training |
|-------|---------------|--------|---------|--------|-------|----------|
| CLIP ViT-B/32 | openai/clip-vit-base-patch32 | 87M | 512 | 12 | 32 | Contrastive |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 | 428M | 768 | 24 | 14 | Contrastive |
| SigLIP 2 ViT-B/16 | google/siglip2-base-patch16-224 | 86M | 768 | 12 | 16 | Contrastive + masked pred + self-distill |
| SigLIP 2 So400m/14 | google/siglip2-so400m-patch14-* | 400M | 1152 | ~27 | 14 | Same |
| SigLIP 2 ViT-g/16 | google/siglip2-giant-patch16-* | 1B | 1536 | ~48 | 16 | Same |

### Datasets

| Dataset | Size | Role | Priority |
|---------|------|------|----------|
| ImageNet val | 50K images | Primary benchmark | Critical |
| COCO val | 5K images, 25K captions | Retrieval evaluation | High |
| MOT17 / MOT20 | Tracking sequences | MOT demo (Exp 9) | Medium |
| Car sim | Controlled | Supplement only | Low |
| Synthetic (solid/gradient/noise) | 30+ images | Baselines | Low |

---

## 11. COMPUTE ESTIMATES

| Experiment | Models | N | Passes/Image | Total Passes | Est. GPU Time |
|-----------|--------|---|-------------|-------------|---------------|
| Exp 0 (nested MC) | 3 | 500 | K=10 × T=64 = 640 | 960K | ~40 min |
| Exp 0b (norm comparison) | 1 | 500 | K=5 × T=64 = 320 | 160K | ~7 min |
| Exp 1 (rank vs p) | 2 | 1000 | 6p × T=64 × K=3 = 1152 | 2.3M | ~1.5 hr |
| Exp 2 (synthetic) | 2 | 40 | K=5 × T=64 = 320 | 26K | ~2 min |
| Exp 3 (dropout type) | 2 | 1000 | 5 types × T=64 × K=5 = 1600 | 3.2M | ~2 hr |
| Exp 4 (cross-model) | 5 | 500 | K=10 × T=64 = 640 | 1.6M | ~1.5 hr |
| Exp 5 (ambiguity) | 2 | 10K | T=64 = 64 | 1.3M | ~50 min |
| Exp 6 (Laplace) | 2 | 10K | 1 + Hessian | ~20K + Hessian | ~1 hr |
| Exp 7 (aleatoric) | 2 | 500 | ~15 degradations + T=64 | ~140K | ~10 min |
| **Phase 1 total** | | | | ~2.7M | ~2.5 hr |
| **Phase 1-3 total** | | | | ~10M | ~8 hr GPU |

---

# PART II — APPENDIX & REFERENCE MATERIAL

---

## A. Conceptual Glossary

### A.1 Kalman Filter Essentials

The Kalman filter estimates the state x (e.g., position and velocity of a tracked object) by combining a *prediction* from a motion model with a *measurement* from a sensor.

**State prediction:** x̂⁻ = F·x̂ + noise, with prediction covariance P⁻
**Measurement:** z = H·x + noise, with measurement noise covariance R
**Update:** Kalman gain K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹

The key insight: **R controls how much you trust the measurement.**
- Small R → high K → the update trusts the sensor heavily
- Large R → low K → the update mostly trusts the motion prediction

**Adaptive R:** In our context, R varies per detection based on the vision encoder's uncertainty about that detection's appearance. This is the core engineering contribution — replacing fixed R with per-detection adaptive R.

### A.2 Data Association in MOT

Given N existing tracks and M new detections per frame, data association solves: "which detection belongs to which track?"

**Cost matrix:** An N × M matrix where entry (i,j) is the cost of assigning detection j to track i. This cost typically combines:
- **Motion distance:** Mahalanobis distance between KF prediction and detection position (accounts for prediction uncertainty)
- **Appearance distance:** Cosine distance between track's appearance embedding and detection's embedding

**Assignment:** Hungarian algorithm finds minimum-cost matching.

**Our contribution:** Replace fixed appearance weighting with adaptive weighting based on per-detection MCDO uncertainty.

### A.3 Ablation Regimes

When we drop neurons from a network, the behavior depends on *how many* we drop:

**Low p (e.g., 0.01):** 99% of neurons survive. The network is almost intact. Tiny perturbations → we're probing *local* sensitivity around the full network. Like gently tapping a structure to test for resonance.

**High p (e.g., 0.1):** 90% survive. Substantial damage. The network may operate in a qualitatively different mode — residual connections may do all the work because main pathways are too disrupted. Like removing a load-bearing wall — the structure doesn't just flex, it reconfigures.

These are different *regimes* — the physics is different at each level. Rankings at p=0.01 reflect "which images are sensitive to fine-grained perturbation?" while rankings at p=0.1 reflect "which images still work when the network is substantially damaged?" These could be different questions with different answers, which is why cross-p rank stability is hyperparameter robustness (does my choice matter?) rather than validity (is the answer right?).

### A.4 Sub-Network Coherence

A neural network can be thought of as containing many overlapping "sub-networks" — subsets of neurons that together perform some function. Dropout randomly deactivates neurons, effectively selecting a random sub-network.

**Independently coherent sub-networks:** Each sub-network produces a *meaningful* (if noisy) output on its own. Like an orchestra where any random subset of musicians could still play a recognizable version of the song.

**Entangled sub-networks:** Individual sub-networks produce *meaningless* outputs. Each neuron's role only makes sense in the context of the full network. Like a JPEG file — remove any random 10% of bytes and you get garbage, not a slightly blurry image.

The Lottery Ticket Hypothesis suggests large networks contain winning sub-networks. The question is whether *random* sub-networks (from dropout) happen to be coherent. Models trained with masked prediction (like SigLIP 2) are explicitly forced to produce coherent outputs from partial networks — they're trained to be like the orchestra, not the JPEG.

### A.5 Pre-Normalization vs Post-Normalization

**Pre-norm features:** The raw output of the Vision Transformer's final layer, before the L2 normalization step. These live in unconstrained ℝ^d — a vector could point anywhere and have any magnitude.

**Post-norm features (what CLIP typically outputs):** After dividing by the L2 norm, ||e||=1. All embeddings live on a unit hypersphere. Cosine similarity = dot product (because vectors are already unit length).

**Why it matters for covariance:** On the sphere, dimensions are mechanically coupled. If one dimension increases, others must decrease (to maintain ||e||=1). This creates artificial correlations. Computing covariance on the sphere is like measuring the spread of points on the surface of a ball — the geometry constrains what you can see. Pre-norm covariance is unconstrained and more interpretable.

### A.6 Scale-Free Metrics

If CLIP has 512 dimensions and SigLIP 2 So400m has 1152 dimensions, the raw trace Tr(Σ) will be ~2x larger for SigLIP just because there are more dimensions to sum over, even if the per-dimension variance is identical. This makes raw traces incomparable across models.

**Tr(Σ)/d:** Dividing by dimension count gives the average per-dimension variance. Now a CLIP trace of 51.2 and a SigLIP trace of 115.2 both correspond to 0.1 per dimension — they're equal.

**Angular variance:** Measures angular spread on the hypersphere in radians. It's inherently dimension-free — an angle is an angle regardless of the ambient dimension.

---

## B. Theoretical Notes

### B.1 Why Gal & Ghahramani Doesn't Directly Apply

Gal & Ghahramani (2016) proved that a network trained with dropout minimizes the KL divergence between an approximate posterior q(ω) (the dropout distribution over weights) and the true posterior p(ω|data). This means dropout samples are *approximate posterior samples*, and their variance is *approximate posterior variance* — i.e., epistemic uncertainty.

**The catch:** This proof requires that the learned weights θ lie in the *support* of q(ω). The dropout distribution q places mass on weight matrices of the form θ ⊙ z where z ~ Bernoulli(p). If θ was learned WITH dropout, the optimization naturally finds weights that are in this support. If θ was learned WITHOUT dropout (as in CLIP), the weights may lie outside the support of q, and the variational inference interpretation breaks.

**Analogy:** It's like fitting a Gaussian posterior around a point estimate that was found by a completely different optimization procedure. The Gaussian might not approximate the actual posterior at all — it's just centered at an arbitrary point that happened to be found by contrastive loss.

**Our position:** We don't rely on the variational inference interpretation. We instead ask the empirical question: *regardless* of theoretical justification, does MCDO variance on frozen encoders happen to predict downstream ambiguity?

### B.2 Why Masked Prediction Training Might Help

SigLIP 2's training includes 50% masked patch prediction: half the input patches are replaced with mask tokens, and the model must still produce a coherent representation. This is analogous to — but not identical to — dropout:

- **Masked prediction:** Input-level masking. The model learns that partial inputs should produce consistent representations. This builds *input-side* redundancy.
- **Dropout:** Weight-level masking. The model (if trained with it) learns that partial networks should produce consistent outputs. This builds *weight-side* redundancy.

The hypothesis is that input-side redundancy from masked prediction *transfers* to weight-side robustness. If the model has learned to extract the same features from many different subsets of input patches, its internal representations are likely distributed across many neurons — making it naturally robust to individual neuron ablation.

This is plausible but unproven. Our experiments test it empirically.

### B.3 Separating Aleatoric and Epistemic Uncertainty

**Aleatoric uncertainty** comes from the input: noise, blur, occlusion, inherent ambiguity. It's irreducible by the model. Even a perfect model with infinite training data would still be uncertain about a heavily occluded detection.

**Epistemic uncertainty** comes from the model: insufficient training data, limited capacity, or unfamiliar inputs. It's reducible in principle — more data or a better model would reduce it.

**How to probe each:**
- Fix the model, perturb the input → measures sensitivity to input noise → aleatoric proxy
- Fix the input, perturb the model (MCDO) → measures sensitivity to model perturbation → epistemic proxy

**For sensor fusion, both matter.** An occluded detection should get high R (aleatoric). An unusual detection that CLIP has never seen should also get high R (epistemic). Ideally, our uncertainty estimate captures both — or we combine separate estimates.

---

## C. Citation List

### Core Theory
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML 2016*.
- Gal, Y., Hron, J., & Kendall, A. (2017). Concrete Dropout. *NeurIPS 2017*.
- Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*.

### VLM Uncertainty
- Baumann, P., et al. (2024). BayesVLM: Bayesian Uncertainty Estimation in Vision-Language Models.
- Chun, S., et al. (2025). ProLIP: Probabilistic Contrastive Language-Image Pretraining.
- Upadhyay, U., et al. (2023). ProbVLM: Probabilistic Adapter-Based Approach for VLM Uncertainty.

### Vision-Language Models
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML 2021*.
- Tschannen, M., et al. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. *arXiv:2502.14786*.

### Multi-Object Tracking
- Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT). *ICIP 2017*.
- Zhang, Y., et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV 2022*.
- Aharon, N., et al. (2022). BoT-SORT: Robust Associations Multi-Pedestrian Tracking.
- Bar-Shalom, Y., & Li, X. R. (1995). *Multitarget-Multisensor Tracking: Principles and Techniques*. YBS Publishing.

### Sub-Network Structure
- Frankle, J., & Carbin, M. (2018). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. *ICLR 2019*.
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*.
- Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). *ICCV 2021*.
- Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision.

### Uncertainty Estimation
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS 2017*.
- Ovadia, Y., et al. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *NeurIPS 2019*.
- Verdoja, F., & Kyrki, V. (2021). Notes on the Behavior of MC Dropout. *arXiv:2108.07276*.

### Statistical Methods
- Koo, T. K., & Li, M. Y. (2016). A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research. *Journal of Chiropractic Medicine*.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass Correlations: Uses in Assessing Rater Reliability. *Psychological Bulletin*.

### Sensor Fusion
- Sage, A. P., & Husa, G. W. (1969). Algorithms for Sequential Adaptive Estimation of Prior Statistics. *IEEE Symposium on Adaptive Processes*.

---

## D. Open Questions & Risks

1. **SigLIP 2 dropout injection:** Does HuggingFace SigLIP 2 expose forward hooks at the same granularity as CLIP? Need to verify module names for each dropout type (A-E). If using `transformers` library, may need to monkey-patch or wrap modules.

2. **Embedding normalization implementation:** Need to extract features *before* the final L2 norm. For CLIP this is straightforward (grab `visual.ln_post` output before projection+norm). For SigLIP 2, need to check the model architecture.

3. **BayesVLM availability:** Is the codebase public and usable? If not, fallback to projection-layer Laplace (Section 5, Exp 6).

4. **Compute for ViT-g:** 1B param model × T=64 × K=10 × N=500 = 320K forward passes. May need batching or reduced K.

5. **MOT benchmark logistics (Exp 9):** Need pre-computed YOLO detections and evaluation toolkit (TrackEval). Consider MOT17 as it's most standard and well-supported.

6. **What if SigLIP 2 ALSO fails?** If no model shows useful signal (ICC < 0.5, no ambiguity correlation), the paper is a diagnostic negative result. This is still publishable — it's a cautionary tale for practitioners and clarifies the scope of MCDO — but less exciting.

7. **Confident misclassifications:** The ambiguity evaluation (Exp 5) handles this better than correct/incorrect, but there's still a subtle issue: MCDO might correlate with uncertainty types that *don't* affect downstream tracking performance. Need to verify that the ambiguity metrics (margin, entropy) actually predict association failure.

8. **Concrete Dropout as follow-up:** If Exp 3 shows that dropout type matters, learning per-layer rates (Concrete Dropout) would be a natural next step but adds significant implementation complexity. Consider as future work rather than paper scope.
