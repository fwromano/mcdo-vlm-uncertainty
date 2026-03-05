# Agent Runbook: MCDO-VLM Uncertainty Experiments

> Self-contained execution guide for running the full experiment pipeline on an
> M3 Ultra Mac Studio. Written for an autonomous agent with no prior context.

---

## 1. Project Summary

**What this tests:** Whether MC Dropout applied to frozen CLIP/SigLIP2 vision
encoders produces reliable per-image uncertainty estimates.

**Hypothesis:** Injecting stochastic dropout into the linear layers of a frozen
vision encoder and running T independent forward passes creates a feature
distribution whose variance (trace of covariance) is a meaningful, per-image
uncertainty score.

**Structure:**
- **Phase 1** (4 experiments) — go/no-go gate. Tests whether the uncertainty
  signal is reliable (reproducible across trials) and valid (correlates with
  ambiguity). If this fails, stop.
- **Phase 2** (5 experiments) — core results. Dropout rate sweeps, model
  comparisons, full-scale ambiguity prediction.
- **Phase 3** (4 experiments) — comparisons with Laplace, aleatoric baselines,
  semantic analysis.
- **Phase 4** (2 experiments) — extensions: text encoder dropout, concrete
  dropout proxy.

**Current state:** All experiment code is written. Nothing has been run
successfully at full scale. A previous pilot on CPU/CIFAR-100 failed all gates.

---

## 2. Hardware Profile (M3 Ultra Mac Studio)

| Spec | Value | Implication |
|------|-------|-------------|
| GPU cores | 60 (Metal 3) | Saturated at batch=500 for ViT-B/32 |
| Unified memory | 96 GB | All models + data fit simultaneously (~10 GB used) |
| Memory bandwidth | ~800 GB/s | **This is the bottleneck**, not compute |
| CPU cores | 20P + 8E | Available for parallel post-processing |
| PyTorch backend | MPS | No torch.compile(), no multi-process GPU |

### Benchmark Results (ViT-B/32, 500 images)

| Batch size | Throughput | Per-image |
|-----------|-----------|-----------|
| 100 | 254 img/s | 3.93 ms |
| 250 | 309 img/s | 3.24 ms |
| **500** | **327 img/s** | **3.06 ms** |
| 1000 | 308 img/s | 3.25 ms |
| 2000 | 9 img/s | OOM/swap thrash |

**batch=500 is the sweet spot.** fp16 gave only 1.06x speedup — confirms
memory-bandwidth-bound.

### Per-Pass Cost Floors (500 images)

| Model | Resolution | Time/pass | Per-image |
|-------|-----------|-----------|-----------|
| clip_b32 | 224px | ~1.5s | 3.0 ms |
| siglip2_b16 | 224px | ~1.8s | 3.6 ms |
| siglip2_so400m | 384px | ~5-8s | 10-16 ms |

These are hard floors on this hardware.

### Memory Budget

| What | Size |
|------|------|
| All 4 model weights | 4.8 GB |
| All pixel tensors (all models, all image counts) | ~5 GB |
| **Total working set** | **~10 GB** |
| **Remaining** | **86 GB free** |

---

## 3. Environment Setup

### 3a. Create Conda Environment

```bash
cd /Users/agc/Documents/GitHub/mcdo-vlm-uncertainty
bash setup_env.sh
```

This auto-detects the Mac, creates conda env `mcdo` (Python 3.11), installs
all dependencies, and caches model weights.

### 3b. Install SigLIP2 Text Encoding Dependencies

```bash
conda run -n mcdo pip install protobuf sentencepiece
```

Required for Exp5+ (text encoding with SigLIP2 models).

### 3c. Verify

```bash
conda run -n mcdo python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
import open_clip
print('open_clip OK')
from transformers import AutoModel
print('transformers OK')
"
```

### 3d. Data

ImageNet validation set at `data/raw/imagenet_val/` with 1000 class
subdirectories (~50K images total). See `DATA_SETUP.md` for download
instructions if not present.

Verify:
```bash
ls data/raw/imagenet_val/ | wc -l   # Should be 1000
```

---

## 4. Code Changes Before Running

Three modifications to `phase_one/run_phase1_fast.py` to maximize throughput.

### 4a. On-Device MPS Accumulation

**Why:** Each `.cpu()` call in the inner loop forces an MPS sync barrier
(~0.1-0.2s). Over T=256 passes, that's 25-50s of pure sync waste per trial.
Unified memory means `.cpu()` copies zero bytes but still blocks the GPU
command queue pipeline.

**File:** `phase_one/run_phase1_fast.py`

**Change 1 — `_mc_pass()` (line 122-124):**

Current:
```python
@torch.inference_mode()
def _mc_pass(vlm: VisionLanguageModel, pixels: torch.Tensor) -> torch.Tensor:
    """One MC dropout forward pass -> pre-norm features as CPU float64."""
    return vlm.encode_pixel_values(pixels, normalize=False).cpu().to(torch.float64)
```

Change to:
```python
@torch.inference_mode()
def _mc_pass(vlm: VisionLanguageModel, pixels: torch.Tensor) -> torch.Tensor:
    """One MC dropout forward pass -> pre-norm features on-device float32."""
    return vlm.encode_pixel_values(pixels, normalize=False)
```

**Change 2 — `run_nested_trial()` (lines 127-175):**

Change accumulator initialization (lines 152-157) from float64 CPU to float32
on-device:
```python
# Before:
sum_pre = torch.zeros(N, D, dtype=torch.float64)

# After:
sum_pre = torch.zeros(N, D, dtype=torch.float32, device=pre.device)
```

Apply the same change to `sq_pre`, `sum_post`, `sq_post`.

At the snapshot point (lines 166-173), convert to float64 only for the final
variance computation:
```python
var_p = (sq_pre / T_done - (sum_pre / T_done) ** 2).to(torch.float64)
var_q = (sq_post / T_done - (sum_post / T_done) ** 2).to(torch.float64)
snapshots[T_done] = {
    "trace_pre": (var_p.sum(1) / D).cpu().float(),
    "trace_post": (var_q.sum(1) / D).cpu().float(),
}
```

**Change 3 — `run_simple_trial()` (lines 238-263):**

Same pattern: accumulate on-device in float32, convert to float64 only at the
end when computing variance.

**Change 4 — `run_trial_with_features()` (lines 178-235):**

The `all_pre.append(pre.float())` calls (lines 195-196) already keep tensors
on-device. No change needed here.

**Numerical safety:** fp32 has ~7 decimal digits. For sums of T<=512
unit-scale features, values stay well within range. Verify after implementation:

```bash
# Quick validation: compare fp32 vs fp64 on small run
conda run -n mcdo python -m phase_one.run_phase1_fast \
    data/raw/imagenet_val outputs/fp_test \
    --device mps --only exp0 \
    --exp0-models clip_b32 --exp0-num-images 50 --exp0-trials 2 \
    --exp0-passes 4,64
```

Compare output traces against a reference run with the old fp64 code. They
should agree to 4+ decimal places.

### 4b. Extended T Snapshot List

**File:** `phase_one/run_phase1_fast.py`, line 67

Current:
```python
p.add_argument("--exp0-passes", type=str, default="4,16,64")
```

Change to:
```python
p.add_argument("--exp0-passes", type=str, default="4,8,16,32,64,128,256")
```

This costs zero extra compute — the nested extraction runs T_max=256 once and
snapshots at each value. It maps the full convergence curve.

### 4c. (Low Priority) Parallel CPU Eigendecompositions

For Phase 2/3 experiments that compute per-image eigendecompositions (Exp0b,
Exp3, Exp8), wrap the per-image loop with
`concurrent.futures.ProcessPoolExecutor(max_workers=20)`. This is pure CPU work
and doesn't contend with MPS. Only needed when running those experiments at
scale — skip for initial Phase 1 run.

---

## 5. Execution Commands

### Phase 1: The Gate (~2 hours)

```bash
conda run -n mcdo python -m phase_one.run_phase1_fast \
    data/raw/imagenet_val outputs/phase1 \
    --device mps \
    --dropout 0.01 \
    --exp0-passes 4,8,16,32,64,128,256 \
    --exp0-num-images 500 --exp0-trials 10 \
    --exp0-models clip_b32,siglip2_b16,siglip2_so400m \
    --exp0b-num-images 500 --exp0b-passes 64 --exp0b-trials 5 \
    --exp4-num-images 500 --exp4-passes 64 --exp4-trials 10 \
    --exp4-models clip_b32,siglip2_b16 \
    --exp5-num-images 5000 --exp5-passes 64 --exp5-trials 1 \
    --exp5-models clip_b32,siglip2_b16
```

Run in a tmux/screen session. Monitor progress with:
```bash
bash scripts/monitor.sh /path/to/logfile
```

Or watch stdout directly — the fast runner prints progress bars via tqdm.

### What Phase 1 Runs

| Exp | What it Tests | Config |
|-----|--------------|--------|
| Exp0 | Reliability of uncertainty scores across T values | 3 models, K=10 trials, T=4..256 (nested) |
| Exp0b | Covariance geometry (diagonality, angular var) | clip_b32, K=5, T=64 |
| Exp4 | Recipe validation (trace vs angular var agreement) | 2 models, K=10, T=64 |
| Exp5 | Ambiguity prediction (uncertainty vs margin/entropy) | 2 models, K=1, T=64, N=5000 |

---

## 6. Go/No-Go Decision Criteria

After Phase 1 completes, read the summary:
```bash
cat outputs/phase1/exp0_nested_mc/exp0_overall_summary.json | python -m json.tool
```

### Thresholds (from `run_phase1_fast.py:269-277`)

| Metric | Pass | Marginal | Fail |
|--------|------|----------|------|
| Pairwise Spearman median | >= 0.8 | 0.6-0.8 | < 0.6 |
| SNR (signal-to-noise) | >= 2.0 | 1.0-2.0 | < 1.0 |
| ICC (intraclass correlation) | >= 0.75 | — | — |

**Status logic:**
- `"usable"`: Spearman >= 0.8 AND SNR >= 2.0 AND ICC >= 0.75
- `"marginal"`: Not usable, but Spearman >= 0.6 AND SNR >= 1.0
- `"failed"`: Spearman < 0.6 OR SNR < 1.0
- `"insufficient_trials"`: Any metric is NaN (need K >= 2)

### Decision Tree

```
Read exp0_overall_summary.json
  |
  +-- ANY model at ANY T has status "usable"?
  |     YES --> Proceed to Phase 2
  |
  +-- Best result is "marginal"?
  |     YES --> Run convergence sweep (Section 7) at higher T
  |             If "usable" at T=512 --> Proceed
  |             If still "marginal" at T=512 --> Judgment call
  |
  +-- ALL models at T=256 are "failed"?
        YES --> Negative result. Stop here. Write it up.
```

Also check Exp5 results for sanity:
```bash
cat outputs/phase1/exp5_subset_ambiguity/exp5_subset_overall_summary.json | python -m json.tool
```
Look for `rho_uncertainty_vs_entropy > 0.3` and `auroc_high_entropy > 0.6`.

---

## 7. Convergence Sweep (Optional, 1-2 hours)

If Phase 1 shows promising but unconverged results, run a focused sweep to find
where reliability saturates:

```bash
conda run -n mcdo python -m phase_one.run_phase1_fast \
    data/raw/imagenet_val outputs/convergence_sweep \
    --device mps --only exp0 \
    --exp0-models clip_b32 \
    --exp0-num-images 200 --exp0-trials 5 \
    --exp0-passes 4,8,16,32,64,128,256,512
```

Small N (200) x high T (512) x single model = maps the reliability-vs-T curve
with minimal compute. Plot Spearman/ICC/SNR vs T to find the knee.

---

## 8. Phase 2 Execution (Only if Phase 1 Passes)

**Priority order** (by information value):

### 8a. Exp3 — Dropout Types (~1 hour)
```bash
conda run -n mcdo python -m phase_two.run_phase2 \
    data/raw/imagenet_val outputs/phase2 \
    --only exp3 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp3-models clip_b32 \
    --exp3-num-images 1000 --exp3-passes 64 --exp3-trials 5
```

### 8b. Exp1 — Rank Stability Across Dropout Rates (~1 hour)
```bash
conda run -n mcdo python -m phase_two.run_phase2 \
    data/raw/imagenet_val outputs/phase2 \
    --only exp1 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp1-models clip_b32,siglip2_so400m \
    --exp1-num-images 1000 --exp1-passes 64 --exp1-trials 3
```

### 8c. Exp4 — Full Model Matrix (~2 hours)
```bash
conda run -n mcdo python -m phase_two.run_phase2 \
    data/raw/imagenet_val outputs/phase2 \
    --only exp4 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp4-models clip_b32,siglip2_b16,siglip2_so400m,clip_l14 \
    --exp4-num-images 500 --exp4-passes 64 --exp4-trials 10
```

Note: `siglip2_g16` excluded by default — it may OOM at batch=500 due to
384px resolution + giant model size. Test with batch=250 first if needed.

### 8d. Exp5 — Full Ambiguity (~1 hour)
```bash
conda run -n mcdo python -m phase_two.run_phase2 \
    data/raw/imagenet_val outputs/phase2 \
    --only exp5 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp5-models clip_b32,siglip2_b16 \
    --exp5-num-images 10000 --exp5-passes 64 --exp5-trials 1
```

### 8e. Exp2 — Synthetic Baselines (~30 min, lowest priority)
```bash
conda run -n mcdo python -m phase_two.run_phase2 \
    data/raw/imagenet_val outputs/phase2 \
    --only exp2 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42
```

---

## 9. Phase 3 Execution (Selective)

### 9a. Exp6 — Laplace Comparison (~1 hour, most paper-worthy)
```bash
conda run -n mcdo python -m phase_three.run_phase3 \
    data/raw/imagenet_val outputs/phase3 \
    --only exp6 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp6-models clip_b32,siglip2_b16 \
    --exp6-num-images 10000 --exp6-passes 64
```

### 9b. Exp7 — Aleatoric vs Epistemic (~1.5 hours)
```bash
conda run -n mcdo python -m phase_three.run_phase3 \
    data/raw/imagenet_val outputs/phase3 \
    --only exp7 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp7-num-images 500 --exp7-passes 64 --exp7-trials 3
```

### 9c. Exp8 — Semantic Space (~1 hour)
```bash
conda run -n mcdo python -m phase_three.run_phase3 \
    data/raw/imagenet_val outputs/phase3 \
    --only exp8 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp8-num-images 500 --exp8-passes 64 --exp8-trials 3
```

### 9d. Exp9 — MOT Adaptive Demo (~5 min, CPU only)
Requires an external MOT cost JSON file. Skip if not available.
```bash
conda run -n mcdo python -m phase_three.run_phase3 \
    data/raw/imagenet_val outputs/phase3 \
    --only exp9 \
    --exp9-cost-json /path/to/mot_costs.json
```

---

## 10. Phase 4 Execution (Optional Extensions)

### 10a. Exp10 — Text Encoder Dropout (~15 min)
```bash
conda run -n mcdo python -m phase_four.run_phase4 \
    data/raw/imagenet_val outputs/phase4 \
    --only exp10 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp10-models clip_b32,siglip2_b16 \
    --exp10-num-prompts 1000 --exp10-passes 64 --exp10-trials 3
```

### 10b. Exp11 — Concrete Dropout Proxy (~4 hours, most expensive)
```bash
conda run -n mcdo python -m phase_four.run_phase4 \
    data/raw/imagenet_val outputs/phase4 \
    --only exp11 \
    --device mps --batch-size 500 --num-workers 0 \
    --dropout 0.01 --seed 42 \
    --exp11-models clip_b32 \
    --exp11-num-images 2000 --exp11-passes 32 --exp11-trials 2
```

---

## 11. Output Structure

After a full run, expect:

```
outputs/
  phase1/
    manifest_all.json                          # Image paths used
    exp0_nested_mc/
      exp0_overall_summary.json                # <-- PRIMARY GO/NO-GO FILE
      clip_b32/
        exp0_summary.json                      # Per-model reliability metrics
        exp0_trials_T4.npz                     # Trial arrays per T value
        exp0_trials_T16.npz
        exp0_trials_T64.npz
        ...
      siglip2_b16/
        ...
      siglip2_so400m/
        ...
    exp0b_norm_geometry/
      exp0b_summary.json                       # Geometry diagnostics
      exp0b_geometry_trials.npz
    exp4_subset_recipe/
      exp4_subset_summary.json                 # Recipe validation
      exp4_clip_b32.npz
      exp4_siglip2_b16.npz
    exp5_subset_ambiguity/
      exp5_subset_overall_summary.json         # Ambiguity prediction
      exp5_subset_clip_b32.npz
      exp5_subset_clip_b32_summary.json
      ...
  phase2/
    exp1_rank_stability/                       # Dropout rate sweep
    exp2_synthetic_natural/                    # Synthetic baselines
    exp3_dropout_types/                        # Dropout variant comparison
    exp4_full_model_matrix/                    # Multi-model comparison
    exp5_full_ambiguity/                       # Large-scale ambiguity
  phase3/
    exp6_laplace_comparison/                   # MC vs Laplace
    exp7_aleatoric_epistemic/                  # Degradation analysis
    exp8_semantic_space/                       # Semantic structure
    exp9_mot_adaptive/                         # MOT demo
  phase4/
    exp10_text_encoder/                        # Text dropout
    exp11_concrete_proxy/                      # Concrete dropout
```

---

## 12. Known Issues & Workarounds

### SigLIP2 Returns Dataclass Instead of Tensor
**Symptom:** `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'float'`
**Cause:** HuggingFace transformers 5.x changed return types.
**Fix:** Already handled by `_as_feature_tensor()` in `phase_one/common.py:203-210`.
No action needed.

### SigLIP2 Text Encoding Fails Without protobuf/sentencepiece
**Symptom:** `RuntimeError: Text processor is unavailable for siglip2_b16`
**Fix:** `conda run -n mcdo pip install protobuf sentencepiece` (Section 3b).
Exp5 in the fast runner wraps this in try/except and skips failing models
(`run_phase1_fast.py:549`).

### MPS Batch Size Limits
**Symptom:** Catastrophic slowdown or hang at batch > 1500.
**Fix:** Use batch=500 (the fast runner defaults to full-dataset single-batch,
which is 500 for most experiments). For Exp5 (N=5000), the fast runner still
uses a single batch — monitor memory pressure. If it thrashes, add chunked
encoding.

### torch.compile() Not Available on MPS
Don't attempt it. The fast runner doesn't use it.

### pyproject.toml Version Mismatch
`pyproject.toml` pins `torch==2.1.2` but conda env has a newer torch. The
editable install works fine. Don't try to "fix" this — just leave it.

### Exp9 Requires External MOT Cost File
Exp9 (MOT adaptive demo) needs a pre-computed cost JSON from an object
tracking pipeline. If you don't have one, skip Exp9 entirely.

---

## 13. What NOT to Optimize

These have been benchmarked and proven unhelpful on this hardware:

| Idea | Why It's a Dead End |
|------|-------------------|
| fp16 model weights | Only 1.06x — we're bandwidth-bound, not compute-bound |
| `torch.compile()` | Not supported on MPS backend |
| Multi-process GPU | MPS serializes all GPU work to one queue |
| Batch size > 500 | Flat scaling past 500, OOM at 2000 |
| More DataLoader workers | Pixels are precomputed in memory, no I/O |
| Moving models to CPU between experiments | Unified memory, no benefit |

---

## 14. Model Registry

Available models (defined in `phase_one/common.py:36-65`):

| Key | Backend | Architecture | Pretrained | Resolution |
|-----|---------|-------------|------------|------------|
| `clip_b32` | open_clip | ViT-B-32 | OpenAI | 224px |
| `clip_l14` | open_clip | ViT-L-14 | OpenAI | 224px |
| `siglip2_b16` | HuggingFace | SigLIP2 Base | google/siglip2-base-patch16-224 | 224px |
| `siglip2_so400m` | HuggingFace | SigLIP2 SO400M | google/siglip2-so400m-patch14-384 | 384px |
| `siglip2_g16` | HuggingFace | SigLIP2 Giant | google/siglip2-giant-patch16-384 | 384px |

---

## 15. Key Code Architecture

### Fast Runner (`phase_one/run_phase1_fast.py`)

The fast runner pre-loads everything into unified memory at startup:

1. **Discover needed models** from CLI args (lines 582-593)
2. **Load all PIL images** into RAM (line 619)
3. **Load all model weights** to MPS (lines 624-636)
4. **Precompute pixel tensors** per model per image count (lines 646-657)
5. **Run experiments sequentially**, reusing cached tensors (lines 662-710)

### MC Trial Pattern

Every MC dropout trial follows this pattern:

```
inject_dropout(model, p=0.01)  # One-time: wrap all Linear layers with Dropout
set_seeds(seed + trial_k)      # Deterministic per trial
for t in range(T):
    features = model.encode(precomputed_pixels)  # Dropout is stochastic
    accumulate(sum, sum_of_squares, features)
variance = sum_sq/T - (sum/T)^2
trace = variance.sum(dim=-1) / D    # Per-image uncertainty score
```

### Nested T Extraction

Instead of running T=4, T=16, T=64 as separate experiments (84 passes),
run T_max=64 once and snapshot accumulators at T=4, 16, 64 (64 passes total).
This is implemented in `run_nested_trial()` (lines 127-175).

### Reliability Metrics

`reliability_from_trials()` in `phase_one/common.py` computes:
- **ICC** (intraclass correlation): signal / (signal + noise)
- **SNR**: between-image variance / within-trial variance
- **Pairwise Spearman**: median rank correlation across all trial pairs

---

## 16. Recommended Execution Order (Total: ~15-17 hours)

| Stage | What | Time | Gate |
|-------|------|------|------|
| 1 | Phase 1 (Section 5) | ~2h | Read Section 6. Stop if all "failed". |
| 2 | Convergence sweep (Section 7) | ~1-2h | Only if Phase 1 "marginal" |
| 3 | Phase 2: Exp3, Exp1, Exp4 (Section 8a-c) | ~4h | Only if Phase 1 passes |
| 4 | Phase 2: Exp5, Exp2 (Section 8d-e) | ~1.5h | Continue if time permits |
| 5 | Phase 3: Exp6, Exp7, Exp8 (Section 9) | ~3.5h | Run overnight |
| 6 | Phase 4: Exp10, Exp11 (Section 10) | ~4h | Lowest priority |

**Critical path:** Stage 1 is the gate. Everything else is conditional on its
results. Do not proceed past Stage 1 if all experiments show `"failed"` status.
