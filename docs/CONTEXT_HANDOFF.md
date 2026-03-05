# Context Handoff: MC Dropout VLM Uncertainty Project

This document distills all project knowledge for use in a separate Claude conversation.
Copy-paste the relevant sections to give the web UI full context.

---

## 1. What This Project Is

We're testing whether **MC (Monte Carlo) Dropout** can produce useful per-image
uncertainty estimates from frozen vision-language models (CLIP, SigLIP2). The
idea: run the same image through the vision encoder multiple times with random
dropout, measure how much the output features vary, and use that variance as an
uncertainty score.

**Application**: Multi-object tracking (MOT). A blurry/distant object should
have higher uncertainty than a crisp close-up. We want the model to know what
it doesn't know.

**Repo**: `mcdo-vlm-uncertainty` on GitHub. Python, PyTorch, runs on M3 Ultra
Mac Studio with MPS backend. Conda env `mcdo`.

---

## 2. The Core Finding (as of 2026-02-26)

**MC Dropout works on CLIP but not on SigLIP2**, and neither model alone gives
you everything you need.

| Property | CLIP ViT-B-32 | SigLIP2 ViT-B-16 |
|---|---|---|
| Training loss | Contrastive softmax | Sigmoid (independent) |
| Reliability (rank consistency across trials) | FAIL (Spearman=0.49, SNR=1.0) | PASS (Spearman=0.93, SNR=15.0) |
| Validity (correlates with classification difficulty) | PASS (rho(entropy)=0.25) | FAIL (rho≈0) |
| Ablation test (degraded images → more uncertain?) | PASS (80% images, p=5×10⁻⁴³) | FAIL (uncertainty DECREASES, anti-correlated) |
| What uncertainty actually measures | Decision boundary sensitivity | Centroid distance (outlier detection) |

**The paradox**: CLIP's uncertainty is valid (measures the right thing) but
unreliable (noisy between trials). SigLIP2's uncertainty is reliable (rock
solid between trials) but invalid (measures the wrong thing — blurred images
are LESS uncertain).

---

## 3. Why the Paradox Exists (First Principles)

**CLIP** (contrastive softmax loss): Classes compete against each other in the
loss function. Features encode inter-class boundaries. Dropout disrupts these
boundaries — images near boundaries get pushed around more. This produces
task-relevant variance, but it's high-rank (many competing modes) so 64 passes
aren't enough to estimate it reliably.

**SigLIP2** (sigmoid loss): Each class is independently classified as
matching/not-matching. No inter-class competition. Dropout variance is
dominated by a few geometric factors (distance from feature centroid, feature
magnitude) that are stable properties of each image — hence very reliable, but
unrelated to classification difficulty. Blurring moves images TOWARD the
centroid (more "generic"), reducing uncertainty.

Full derivation: `WHY_CLIP_VS_SIGLIP2.md` in the repo.

---

## 4. Experiment Structure

The pipeline has four composable axes (documented in `EXPERIMENT_ALGEBRA.md`):

**Axis 1 — Models**: clip_b32, clip_l14, siglip2_b16, siglip2_so400m, siglip2_g16
**Axis 2 — Dropout Strategy**: p ∈ {0.001–0.1}, T ∈ {4–64} passes, K ∈ {1–10} trials
**Axis 3 — Uncertainty Metrics**: trace_pre (best), trace_post, max_dim_var, top_eigenvalue, norm_var, mean_cosine_dev
**Axis 4 — Validation**: reliability (ICC/SNR/Spearman), classification correlation, ablation sensitivity, mean convergence

Each experiment is a specific cross-product slice.

---

## 5. Phase 1 Results (Complete)

Phase 1 tested **reliability** — do you get the same uncertainty ranking if you
run the procedure twice?

Gate criteria: Spearman ≥ 0.80, SNR ≥ 2.0, ICC ≥ 0.75.

| Model | T=64 pre-norm | T=64 post-norm | Best status |
|---|---|---|---|
| clip_b32 | Spearman=0.49, SNR=1.0 | Spearman=0.37, SNR=0.65 | FAILED |
| siglip2_b16 | Spearman=0.93, SNR=15.0 | Spearman=0.96, SNR=26.5 | USABLE |
| siglip2_so400m | Spearman=0.78, SNR=4.7 | Spearman=0.84, SNR=6.2 | USABLE (post-norm) |

Trace_pre outperforms angular_var. Pre-norm works better for CLIP; post-norm
works better for SigLIP2. SNR scales ~linearly with T.

Full report: `PHASE_ONE_REPORT.md`

---

## 6. Preliminary Investigation Results

### Alternative Metrics (N=500)

trace_pre is the clear winner for CLIP (rho(entropy)=0.30). No metric rescues
SigLIP2 — all near zero.

### Larger Sample (N=2000)

CLIP trace_pre rho(entropy)=0.25 at N=2000. Signal is robust, not a
small-sample artifact. AUROC(error)=0.564. Accuracy 57.3%.

### What SigLIP2 Uncertainty Correlates With (N=1000)

| Correlate | rho |
|---|---|
| centroid_distance | +0.24 |
| feat_norm | −0.13 |
| classification_entropy | ≈0 |

It's an outlier detector, not an ambiguity detector.

### Ablation Test (N=500) — Most Important Result

**CLIP**: 80% of images more uncertain when degraded. Wilcoxon p = 5×10⁻⁴³. Pass.
**SigLIP2**: 75.6% of heavily blurred images LESS uncertain. Anti-correlated. Fail.

Full results: `PRELIM_FINDINGS.md`, `outputs/prelim_ablation.json`

---

## 7. Bugs Found and Fixed

1. **ImageNet synset labels**: Folders are named `n01440764` not `tench`.
   Without class map, prompts were gibberish. Fixed: `data/imagenet_class_map.json`.

2. **HuggingFace SigLIP2 text encoder**: Produces collapsed embeddings (all
   1000 class vectors have cosine 0.976 with each other). Switched to open_clip
   backend (cosine 0.762). Code now defaults to open_clip with HF fallback.
   Env var: `MCDO_SIGLIP2_BACKEND=hf` forces HuggingFace.

3. **Phase 2 launched from wrong conda env**: Base Python lacks open_clip →
   clip_b32 exp1 failed. Must use `conda run -n mcdo`.

---

## 8. Current State (as of 2026-02-26 evening)

**Running**:
- Phase 2 exp1_rank_p: siglip2_so400m on p=0.005 (2 of 6 p-values done, from
  base Python — still works for HF models)
- prelim_investigation.py full re-run with fixed label mapping (background)

**Needs re-run**:
- clip_b32 exp1 (killed mid-run, got 2/3 trials of p=0.001)

**Not yet run**:
- Exp 6 (mean convergence — does MC mean → deterministic embedding as T↑?)
- clip_l14 (larger CLIP model — might have better SNR while keeping validity)
- Ablation calibration curve (Exp 7 — user's idea)

---

## 9. Key Open Questions

1. **Can we improve CLIP's reliability?** More passes (T=256?), higher dropout
   (p=0.05?), or ensembling multiple trials might push SNR above 2.0. Phase 2
   exp1 tests this across p-values.

2. **Does clip_l14 solve both problems?** It's a larger CLIP model — might have
   higher SNR (more stable features) while keeping the contrastive loss
   structure that makes uncertainty valid.

3. **Can we combine CLIP + SigLIP2?** Use SigLIP2 for feature extraction
   (better representations) and CLIP for uncertainty estimation (valid signal).
   Architecturally odd but separates "what is this?" from "how sure am I?"

4. **Ablation calibration**: User's key insight — if blur radius R reliably
   increases uncertainty by X, you can express uncertainty in physically
   meaningful units ("equivalent blur level"). This could be Exp 7.

5. **Does mean convergence work?** (Exp 6) If the MC dropout mean converges to
   the deterministic embedding as T increases, it validates that dropout
   doesn't introduce systematic bias.

---

## 10. Code Architecture

**Core**: `phase_one/common.py`
- `ModelSpec` / `MODEL_REGISTRY`: 5 models with backend routing (open_clip or HF)
- `VisionLanguageModel`: wraps model + processors, handles dropout injection
- `inject_uniform_linear_dropout(model, p)`: wraps every nn.Linear with Dropout
- `run_mc_trial(vlm, loader, T)`: runs T forward passes, returns trace_pre/trace_post
- `reliability_from_trials(values)`: computes ICC, SNR, pairwise Spearman

**Experiments**: `phase_one/exp*.py`, `phase_two/exp*.py`
**Runners**: `phase_one/run_phase1_fast.py`, `phase_two/run_phase2.py`
**Prelim scripts**: `prelim_investigation.py`, `prelim_ablation.py`

---

## 11. Environment

- **Machine**: M3 Ultra Mac Studio, 96GB unified memory, MPS backend
- **Conda env**: `mcdo` — MUST use `conda run -n mcdo` (base Python lacks open_clip)
- **Key packages**: torch, open_clip (3.2.0), transformers, scipy, numpy
- **Data**: `data/raw/imagenet_val/` (ImageNet-1K validation, ~50K images)

---

## 12. User Preferences

- Fix broken things immediately, don't flag for later
- First-principles explanations, not analogies
- Direct experimental results over theory
- Concise communication
- MOT (multi-object tracking) is the target application
- Ablation-based calibration is a key interest
