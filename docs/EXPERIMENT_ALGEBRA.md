# Experiment Algebra

The uncertainty estimation pipeline is a composition of four independent axes.
Each experiment is a specific cross-product of choices from these axes.

## Axis 1: Models

```
Model := (architecture, backend, pretrained_weights)

MODELS = {
    clip_b32:       (ViT-B-32,            open_clip, openai)
    clip_l14:       (ViT-L-14,            open_clip, openai)
    siglip2_b16:    (ViT-B-16-SigLIP2,    open_clip, webli)
    siglip2_so400m: (ViT-SO400M-14-SigLIP2, open_clip, webli)
    siglip2_g16:    (ViT-gopt-16-SigLIP2-384, open_clip, webli)
}
```

Runtime note for SigLIP2:
- Default backend is `open_clip`.
- If `open_clip` assets are unavailable, code falls back to HuggingFace (`siglip2`) with a runtime warning.
- Force backend with `MCDO_SIGLIP2_BACKEND=open_clip` or `MCDO_SIGLIP2_BACKEND=hf`.

**Input**:  PIL Image → pixel tensor
**Output**: feature vector ∈ R^D (D varies by model: 512, 768, 1152)

## Axis 2: Dropout Strategy

```
DropoutStrategy := (injection_method, p, target_modules)

inject_uniform_linear_dropout(model, p) → wraps every nn.Linear in model.visual
    with LinearDropoutWrapper(linear, Dropout(p))

set_dropout_mode(model, enabled, p) → toggle train/eval on all dropout layers
```

**Parameters**:
- `p` ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1} — dropout probability
- `T` ∈ {4, 8, 16, 32, 64} — number of MC forward passes
- `K` ∈ {1, 3, 5} — independent trials (for reliability estimation)

**Process**: For each of T passes, forward the same pixel tensor through the model
with dropout enabled, collecting T feature vectors per image.

```
run_mc_trial(model, images, T) → {
    features_pre:  Tensor[T, N, D]    # raw features (before L2 norm)
    features_post: Tensor[T, N, D]    # L2-normalized features
}
```

## Axis 3: Uncertainty Metrics

```
UncertaintyMetric := features[T, N, D] → scores[N]
```

All metrics reduce T passes × D dimensions to a single scalar per image.

| Metric | Formula | What it measures |
|---|---|---|
| `trace_pre` | `(1/D) Σ_d Var_t(f_d)` on raw features | Total variance in pre-norm space |
| `trace_post` | `(1/D) Σ_d Var_t(f̂_d)` on L2-normed features | Directional spread |
| `max_dim_var` | `max_d Var_t(f_d)` | Worst-case dimension instability |
| `top_eigenvalue` | `λ_max(Cov_t(f))` via Gram trick | Dominant mode of variation |
| `norm_var` | `Var_t(‖f_t‖)` | Magnitude instability |
| `mean_cosine_dev` | `1 - mean_t(cos(f̂_t, mean_t(f̂)))` | Angular deviation from mean direction |
| `angular_var` | `Var_t(arccos(cos(f̂_t, mean_t(f̂))))` | Variance of angle from mean direction |

**Current best**: `trace_pre` (strongest correlation with task difficulty for CLIP).

## Axis 4: Validation Criteria

How we assess whether an uncertainty metric is useful.

### 4a. Reliability (Is it consistent?)

```
ReliabilityTest := scores[K, N] → {ICC, SNR, pairwise_spearman}
```

Run K independent MC trials. Check that per-image uncertainty rankings
are stable across trials.

| Metric | Gate | Meaning |
|---|---|---|
| ICC | >= 0.75 | Intraclass correlation |
| SNR | >= 2.0 | signal variance / noise variance |
| Pairwise Spearman | >= 0.80 | Rank consistency across trials |

### 4b. Validity — Classification Correlation

```
ClassificationCorrelation := (scores[N], classification_metrics[N]) → rho
```

Check that uncertainty correlates with classification difficulty.

| Target metric | How computed | Interpretation |
|---|---|---|
| entropy | `-Σ p_c log p_c` from softmax logits | Higher = model confused about class |
| negative_margin | `-(logit_1st - logit_2nd)` | Higher = top two classes are close |
| error | `1 - correct` | Binary: did model get it wrong? |
| AUROC(error) | area under ROC for uncertainty → error | Discrimination power |

### 4c. Validity — Ablation Sensitivity

```
AblationTest := (scores_clean[N], scores_degraded[N]) → {
    frac_increased,    # proportion where unc(degraded) > unc(clean)
    wilcoxon_p,        # paired non-parametric significance
    effect_size        # mean(unc_degraded - unc_clean)
}
```

Degrade each image (blur, downsample) and check that uncertainty increases.
Paired design — each image is its own control.

| Degradation | Parameter | Expected effect |
|---|---|---|
| Gaussian blur | radius ∈ {5, 15} | moderate → large unc increase |
| Downsample+upsample | factor ∈ {4, 8} | moderate → large unc increase |

### 4d. Validity — Mean Convergence (Exp 6)

```
MeanConvergence := (mc_mean[T, N, D], det_embedding[N, D]) → {
    relative_distance(T),   # ‖mean_mc(T) - det‖ / ‖det‖
    log_log_slope            # should be ~ -0.5 for unbiased estimator
}
```

Check that the MC dropout mean converges to the deterministic embedding
as T increases. Validates that dropout doesn't introduce systematic bias.

## Experiment Map

Each experiment is a specific slice through the axes:

| Experiment | Models | Dropout | Metric | Validation |
|---|---|---|---|---|
| Exp 0 | all | p=0.01, T=64, K=3 | trace_pre, trace_post | Reliability |
| Exp 0b | all | p=0.01, T=64, K=3 | angular_var | Reliability |
| Exp 1 | clip_b32, siglip2_so400m | p ∈ {0.001..0.1}, T=64, K=3 | trace_pre | Reliability × p |
| Exp 4 | all | p=0.01, T∈{4..64}, K=3 | trace_pre | Reliability × T |
| Exp 5 | clip_b32, siglip2_b16 | p=0.01, T=64, K=1 | trace_pre | Classification correlation |
| Exp 6 | siglip2_b16, siglip2_so400m | p=0.01, T∈{4..64}, K=3 | trace_pre | Mean convergence |
| Ablation | clip_b32, siglip2_b16 | p=0.01, T=64 | trace_pre | Ablation sensitivity |
