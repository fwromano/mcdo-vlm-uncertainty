# Why MC Dropout Works on CLIP but Not SigLIP2

Implementation note: SigLIP2 now defaults to the `open_clip` backend in code.
If `open_clip` assets are unavailable, loader falls back to HuggingFace
(`MCDO_SIGLIP2_BACKEND=hf` can force this fallback path).

## What MC dropout actually computes

A ViT vision encoder is a stack of transformer blocks, each containing two
linear layers (in the MLP) and three linear layers (QKV projections in
attention). For a ViT-B, that's ~12 blocks × 5 linears = ~60 linear layers.

`inject_uniform_linear_dropout(model, p=0.01)` wraps each of those 60 linears
with `Dropout(0.01)` applied to the output. On every forward pass, each linear
layer independently zeroes out 1% of its output activations at random.

One image goes through T=64 passes. Each pass uses a different random dropout
mask — a different subset of activations is zeroed. This produces 64 feature
vectors **f₁, f₂, ..., f₆₄** ∈ ℝ⁵¹². The uncertainty score is:

```
trace_pre = (1/512) × Σ_d Var({f₁[d], f₂[d], ..., f₆₄[d]})
```

The average per-dimension variance across the 64 vectors.

## What determines whether dropping a specific neuron matters

When a particular activation in layer L is zeroed, the effect on the final
feature vector depends on the gradient of the output with respect to that
activation. High gradient = big effect. Low gradient = negligible effect.

For a linear layer computing `y = Wx + b`, zeroing output unit j removes
`y_j`'s contribution to all downstream computation. The first-order
approximation of the resulting feature change is:

```
Δf ≈ (∂f/∂y_j) × y_j
```

The contribution of dropping unit j depends on:

1. **y_j** — how large the activation is (what the unit is computing for this
   input)
2. **∂f/∂y_j** — how sensitive the final feature is to that unit (the
   downstream pathway's gain)

The total variance from dropout is approximately the sum of these contributions
across all droppable units, weighted by the dropout probability p.

## Why CLIP's variance tracks classification difficulty

CLIP is trained with contrastive softmax loss:

```
L = -log( exp(sim(image, text_correct)) / Σ_k exp(sim(image, text_k)) )
```

This loss pushes the image feature **away from** wrong-class text features and
**toward** the correct-class text feature. The gradient is largest for images
where multiple classes have similar similarity scores — images near class
decision boundaries in feature space.

During training, the network allocates representational capacity to resolve
these ambiguous cases. The linear layers in blocks that handle fine-grained
discrimination develop large gradients ∂f/∂y_j for inputs near boundaries.
These same layers also have large activations y_j because they're actively
computing the discriminative signal.

When dropout zeroes one of these high-gradient, high-activation units:

- The feature shifts significantly (large Δf)
- The shift direction depends on which specific unit was dropped
- Different masks → different shifts → high variance

For easy, unambiguous images, the features land far from any decision boundary.
The discriminative layers have small gradients (the output is already
saturated — the correct class dominates the softmax). Dropping a unit doesn't
change the feature much. Low variance.

**Result**: Var(f) correlates with proximity to decision boundaries, which
correlates with classification entropy. Measured: rho(entropy) = 0.25.
Ablation test: 80% of blurred images have higher uncertainty (p = 5 × 10⁻⁴³).

## Why CLIP's variance is noisy across trials

Each trial runs 64 passes with independent random masks. The variance estimate
from 64 samples has sampling error.

For each dimension d, we estimate Var({f₁[d]...f₆₄[d]}) from 64 samples.
For normally distributed data, the standard error of a variance estimate is:

```
SE(Var) = Var × √(2/(n-1)) = Var × √(2/63) ≈ 0.178 × Var
```

About 18% relative error per dimension. The trace averages 512 dimensions, so
if they were independent, the trace error would be ~0.178 / √512 ≈ 0.8%.

But the actual noise is much higher (SNR = 1.0) because the per-dimension
variances are correlated. When unit j in some layer is dropped, it shifts
multiple output dimensions simultaneously — the Δf vector from dropping one
unit has nonzero entries across many of the 512 dimensions. This means the
variance estimates across dimensions are not independent, and the √512
averaging benefit is largely lost.

The effective number of independent variance components is much smaller than
512. Two trials with different random seeds rank images differently because the
mask-specific correlation structure differs between trials.

The per-image signal IS present (some images genuinely have more variance than
others) but the estimation noise is comparable to the signal. Measured:
SNR = 1.0, pairwise Spearman between trials = 0.49.

## Why SigLIP2's variance doesn't track classification difficulty

SigLIP2 is trained with sigmoid loss:

```
L = Σ_k [ -log(σ(±sim(image, text_k))) ]
```

Each image-text pair is independently classified as matching (+) or not
matching (−). There is no softmax denominator coupling the classes together.

The gradient for a given image does NOT depend on how similar it is to other
classes. Each class contributes independently to the loss. There is no concept
of "near a decision boundary between class A and class B" because classes A
and B are never compared against each other in the loss function.

The network therefore doesn't develop the same structure of high-gradient
pathways at inter-class boundaries. The gradients ∂f/∂y_j are distributed more
uniformly across images. They depend on image-level properties like feature
magnitude and activation pattern, not class-boundary proximity.

When dropout zeroes a unit, the resulting Δf is proportional to that unit's
activation. Images far from the feature-space centroid (atypical images) tend
to have more extreme activations in certain layers → larger Δf → higher
variance.

**Result**: Var(f) correlates with centroid distance (rho = 0.24) but not
classification entropy (rho ≈ 0). Ablation test: blurred images are actually
LESS uncertain (75.6% of heavily blurred images have lower uncertainty).

Blurring reduces uncertainty because blurred images lose their distinctive
features and move TOWARD the centroid in feature space. The network sees them
as more "generic" → lower activation magnitudes in discriminative layers →
less variance under dropout.

## Why SigLIP2's variance has low noise

SigLIP2's variance is dominated by a few gross geometric factors (centroid
distance, feature norm). These are deterministic properties of the image that
don't change between trials.

The variance Var(f) for a given image is a function of the network's Jacobian
at that input and the dropout distribution. For SigLIP2, the Jacobian's
dominant singular vectors point in stable directions determined by the image's
position in feature space — not by the specific dropout mask.

When you estimate the trace from 64 passes, you're sampling a variance
distribution whose dominant modes are low-rank (a few eigenvectors of the
per-image covariance matrix capture most of the total variance). Estimating a
low-rank variance is statistically easy — 64 samples is more than enough for
the trace to converge.

For CLIP, the per-image covariance has higher effective rank (many independent
directions contribute, each driven by different decision-boundary
interactions). More modes to estimate from finite samples → slower convergence
→ more noise between trials.

Measured: SigLIP2 SNR = 15.0, pairwise Spearman = 0.93.

## Summary

| Property | CLIP (ViT-B-32) | SigLIP2 (ViT-B-16-SigLIP2) |
|---|---|---|
| Training loss | Contrastive softmax (classes compete) | Sigmoid (classes independent) |
| What dropout variance measures | Decision boundary sensitivity | Centroid distance / feature scale |
| Variance rank structure | High-rank (many competing modes) | Low-rank (few geometric modes) |
| Estimation from 64 passes | Noisy (SNR = 1.0) | Clean (SNR = 15.0) |
| Trial-to-trial ranking consistency | Spearman = 0.49 | Spearman = 0.93 |
| Correlation with classification difficulty | rho(entropy) = 0.25 | rho(entropy) ≈ 0 |
| Response to image degradation | +29% uncertainty (p < 10⁻⁴³) | −10% uncertainty (anti-correlated) |

The contrastive loss creates inter-class competition that dropout can
meaningfully disrupt, producing valid but noisy uncertainty. The sigmoid loss
creates independent per-class pathways that dropout disrupts along geometric
axes unrelated to task difficulty, producing stable but meaningless uncertainty.

---

## Appendix: Definitions

### Vision Transformer (ViT)

Architecture that processes images by:
1. Splitting the image into fixed-size patches (e.g., 16×16 pixels)
2. Linearly projecting each patch into a vector (the "patch embedding")
3. Passing the sequence of patch embeddings through a stack of transformer
   blocks
4. Taking the output of a special [CLS] token (or mean-pooling all tokens) as
   the final feature vector

Each transformer block contains:
- **Multi-head self-attention**: computes Query, Key, Value matrices via three
  separate linear layers (the QKV projections), then computes attention weights
  and a weighted sum
- **MLP (feedforward network)**: two linear layers with a nonlinearity (GELU)
  between them
- **Layer normalization** and **residual connections** around each sub-block

### Linear layer

A function `y = Wx + b` where W is a learned weight matrix, x is the input
vector, b is a bias vector. This is the basic building block of neural
networks. In a ViT-B, each linear layer operates on vectors of dimension 768.

### Dropout

A regularization technique. During a forward pass, each activation (output of
a neuron) is independently set to zero with probability p, and the remaining
activations are scaled by 1/(1-p) to preserve the expected value. Different
random masks are drawn each time.

Normally used only during training. MC dropout repurposes it at inference time
by running multiple passes with dropout enabled to sample from an approximate
posterior distribution over network outputs.

### Gradient (∂f/∂y_j)

The partial derivative of the output feature f with respect to an intermediate
activation y_j. Computed via the chain rule through all layers between y_j and
f. Represents how much the final output would change per unit change in y_j.
Not explicitly computed in our code — we observe its effect through the
variance of f across dropout masks.

### Jacobian

The matrix of all partial derivatives of the output with respect to some set
of intermediate activations. For a function f: ℝⁿ → ℝᵐ, the Jacobian J is an
m × n matrix where J_ij = ∂f_i/∂x_j. The "rank" of the Jacobian determines
how many independent directions of variation exist in the output.

### Contrastive softmax loss (CLIP)

Given a batch of N image-text pairs, compute cosine similarity between every
image and every text (an N × N matrix). The loss maximizes the diagonal
entries (correct pairs) relative to off-diagonal entries (incorrect pairs)
using a softmax over each row and column. This forces different classes to
actively compete for representational space.

### Sigmoid loss (SigLIP2)

Each image-text pair is independently classified as matching (sigmoid output
→ 1) or not matching (sigmoid output → 0). No softmax denominator means
classes don't compete with each other. The network can assign high similarity
to multiple classes without penalty, as long as the correct pair also gets
high similarity.

### Spearman rank correlation

A measure of monotonic association between two ranked variables. Computed by
ranking both variables and taking the Pearson correlation of the ranks.
Ranges from −1 (perfect inverse) to +1 (perfect agreement). In our context,
it measures whether two independent MC dropout trials rank images in the same
order. Our gate requires Spearman ≥ 0.80.

### ICC (Intraclass Correlation Coefficient)

Measures the proportion of total variance that is "between subjects" (between
images) versus "within subjects" (between trials for the same image). ICC = 1
means all variance is between images (perfect reliability). ICC = 0 means
within-image variance equals between-image variance (pure noise). Computed
from a one-way random-effects ANOVA decomposition. Our gate requires ICC ≥
0.75.

### SNR (Signal-to-Noise Ratio)

```
SNR = Var(image means across images) / Mean(within-image variance across images)
```

Signal is the variance of the per-image mean uncertainty scores. Noise is the
average within-image variance (how much the uncertainty score fluctuates across
trials for the same image). SNR > 1 means the signal exceeds the noise. Our
gate requires SNR ≥ 2.0.

### AUROC (Area Under the Receiver Operating Characteristic)

Measures how well a continuous score (uncertainty) discriminates between two
classes (e.g., correct vs incorrect predictions). AUROC = 0.5 is chance (the
score contains no discriminative information). AUROC = 1.0 is perfect
separation. Computed by sweeping all possible thresholds on the score and
plotting the true positive rate vs false positive rate.

### Wilcoxon signed-rank test

A non-parametric paired hypothesis test. Given N paired observations
(uncertainty_clean, uncertainty_degraded), it tests whether the distribution
of differences is symmetric around zero. Unlike a paired t-test, it doesn't
assume normality — it works on ranks. The p-value indicates the probability
of observing the data if degradation had no effect on uncertainty.

### Centroid distance

The mean feature vector across all images is the centroid. Centroid distance
for a given image is `1 - cosine_similarity(image_feature, centroid)`. Images
with unusual content (rare classes, atypical viewpoints) have high centroid
distance.

### Eigenvalue / eigenvector

For a square matrix A, an eigenvector v satisfies Av = λv — multiplication
by A just scales v by the scalar λ (the eigenvalue). For a covariance matrix,
eigenvalues represent the variance along each principal direction. The "top
eigenvalue" is the largest one — the direction of maximum variance.

### Effective rank

The number of eigenvalues of a covariance matrix that are "significantly
nonzero." A matrix with one dominant eigenvalue has effective rank ~1 (the
variance is concentrated in one direction). A matrix where all eigenvalues
are equal has effective rank equal to the matrix dimension (variance is
spread uniformly). Low effective rank → easy to estimate from few samples.
High effective rank → need many samples for accurate estimation.
