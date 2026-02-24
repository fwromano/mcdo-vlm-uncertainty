# Performance Optimization Guide

## Current Bottleneck Analysis

Profiling the Exp 0 run on RTX 5000 Ada revealed **GPU utilization near 0%** despite the model being loaded in VRAM. The bottleneck is overwhelmingly **data loading and preprocessing**, not compute.

### Why it's slow

The hot loop in `run_mc_trial()` does this for every MC pass:

```
for pass_idx in range(T):          # T=64 passes
    for images, paths, _ in loader:  # PIL images from disk
        pre = vlm.encode_images(images)  # preprocess + forward
```

For `siglip2_so400m` at T=64, K=10 trials: **640 times** loading and preprocessing the same 500 images from disk via PIL. Each image is:
1. Opened from PNG on disk (`PIL.Image.open`)
2. Converted to RGB
3. Resized/normalized by the model's image processor (CPU)
4. Stacked into a tensor
5. Transferred to GPU
6. Forward pass (fast)
7. Result moved back to CPU for accumulation

Steps 1-5 dominate. The GPU forward pass takes microseconds per batch relative to the I/O + preprocessing overhead.

### Scale of waste

| Config | Images loaded from disk | Redundant loads |
|--------|------------------------|-----------------|
| clip_b32, T=64, K=10 | 320,000 | 319,500 (99.8%) |
| siglip2_so400m, T=64, K=10 | 320,000 | 319,500 (99.8%) |
| Full Exp 0 (3 models × 3 T-values × K=10) | ~1.26M | ~1.26M |

---

## Optimization 1: Precompute pixel tensors (biggest win)

**Expected speedup: 10-50x**

Load and preprocess all images once, store as a tensor. Reuse across all MC passes.

```python
def precompute_pixel_values(vlm, loader):
    """Preprocess all images once, return a single GPU tensor."""
    all_pixels = []
    all_paths = []
    for images, paths, _ in loader:
        pv = vlm._pixel_values_from_pil(images)
        all_pixels.append(pv)
        all_paths.extend(paths)
    pixel_tensor = torch.cat(all_pixels, dim=0)  # (N, C, H, W)
    return pixel_tensor, all_paths
```

Then `run_mc_trial` becomes a pure GPU loop:

```python
pixel_tensor = pixel_tensor.to(device)  # move once
for pass_idx in range(T):
    for i in range(0, N, batch_size):
        batch = pixel_tensor[i:i+batch_size]
        features = model.encode_image(batch)  # GPU only, no I/O
```

### Memory requirements

| Dataset | Model | Tensor size (fp32) | Fits 16GB VRAM? | Fits 128GB unified? |
|---------|-------|--------------------|-----------------|---------------------|
| 500 imgs | CLIP (224px) | 500 × 3 × 224 × 224 × 4 = **300 MB** | Yes | Yes |
| 500 imgs | SigLIP2-so400m (384px) | 500 × 3 × 384 × 384 × 4 = **885 MB** | Yes | Yes |
| 5000 imgs | CLIP (224px) | **3.0 GB** | Yes | Yes |
| 5000 imgs | SigLIP2-so400m (384px) | **8.8 GB** | Tight (with model) | Yes |
| 10000 imgs | SigLIP2-so400m (384px) | **17.6 GB** | No (offload to CPU) | Yes |

---

## Optimization 2: Mixed precision (`torch.float16` / `torch.bfloat16`)

**Expected speedup: 1.5-2x on forward pass**

The RTX 5000 Ada has excellent FP16/BF16 tensor core performance. The forward pass can run in half precision while accumulation stays in fp64.

```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    features = model.encode_image(batch)
features = features.float()  # upcast for accumulation
```

On the M3 Ultra, use `torch.float16` with MPS (bfloat16 support varies by PyTorch version).

---

## Optimization 3: `torch.inference_mode()` instead of `torch.no_grad()`

**Expected speedup: ~5-10% (less bookkeeping overhead)**

```python
@torch.inference_mode()
def encode_images(self, images, normalize=False):
    ...
```

`inference_mode` is stricter than `no_grad` — it disables autograd entirely and avoids version counting on tensors.

---

## Optimization 4: Larger batch sizes

With precomputed tensors already on device, memory is freed from PIL/preprocessing overhead.

| Platform | Model | Recommended batch_size |
|----------|-------|----------------------|
| RTX 5000 Ada (16GB) | CLIP-B/32 | 256-500 (entire dataset) |
| RTX 5000 Ada (16GB) | SigLIP2-B/16 | 128-256 |
| RTX 5000 Ada (16GB) | SigLIP2-so400m | 64-128 |
| M3 Ultra (128GB) | Any model | 500-5000 (entire dataset) |

With the full 500-image dataset fitting in one batch, there's zero batch loop overhead.

---

## Optimization 5: `torch.compile()` (NVIDIA only)

**Expected speedup: 1.2-2x on forward pass**

```python
model = torch.compile(model, mode="reduce-overhead")
```

Fuses operations, reduces kernel launch overhead. First call has compilation cost (~30s), then subsequent calls are faster. Worth it for T=64 × K=10 = 640 forward passes.

**Note:** `torch.compile()` does not yet work reliably with MPS (Apple Silicon). Skip on M3 Ultra.

---

## Optimization 6: Progress output

The `run_mc_trial` already accepts `progress=True` but the experiment scripts don't pass it. Fix:

```python
# In exp0_nested_mc.py, line ~94:
trial = run_mc_trial(
    vlm=vlm, loader=loader, passes=passes,
    collect_pass_features=False,
    progress=True,  # ADD THIS
    progress_desc=f"{model_key} T={passes} K={trial_idx+1}/{args.trials}",
)
```

---

## Platform-Specific: NVIDIA RTX 5000 Ada (16GB VRAM)

### Recommended configuration

```bash
python -m phase_one.run_phase1 \
  /path/to/data \
  /path/to/output \
  --device cuda \
  --batch-size 128 \
  --num-workers 0       # no workers needed with precomputed tensors
```

### Additional CUDA optimizations

```python
# At startup
torch.backends.cudnn.benchmark = True       # auto-tune conv algorithms
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 on Ampere+
torch.backends.cudnn.allow_tf32 = True

# For the so400m model at 5000 images, if VRAM is tight:
# Keep pixel tensor on CPU, transfer batches
for i in range(0, N, batch_size):
    batch = pixel_tensor[i:i+batch_size].to('cuda', non_blocking=True)
    features = model.encode_image(batch)
```

### Expected total Exp 0 time with optimizations

| Optimization | Estimated Exp 0 time |
|-------------|---------------------|
| Current (PIL reload every pass) | ~6-8 hours |
| + Precompute pixels | ~20-40 min |
| + AMP fp16 | ~12-25 min |
| + torch.compile | ~8-18 min |
| + Batch size 256 | ~6-15 min |

---

## Platform-Specific: Apple M3 Ultra (128GB unified memory)

### Key advantages

1. **128GB unified memory** — no CPU↔GPU transfer penalty. All models and all preprocessed tensors fit in memory simultaneously.
2. **No PCIe bottleneck** — CPU and GPU share the same memory pool. `tensor.to('mps')` is essentially a no-op pointer change.
3. **Can load all models at once** — clip_b32 (~350MB) + clip_l14 (~890MB) + siglip2_b16 (~400MB) + siglip2_so400m (~1.6GB) = ~3.2GB total. Trivial.

### Recommended configuration

```bash
python -m phase_one.run_phase1 \
  /path/to/data \
  /path/to/output \
  --device mps \
  --batch-size 500      # entire dataset in one batch
  --num-workers 0       # MPS + multiprocessing can deadlock
```

### MPS-specific code changes

```python
# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# MPS gotchas:
# 1. Some ops fall back to CPU silently — watch for slow individual ops
# 2. Use torch.mps.synchronize() before timing
# 3. torch.compile() does NOT work on MPS — skip it
# 4. float64 is not supported on MPS — accumulate on CPU
#    (current code already does .cpu().to(torch.float64), so this is fine)

# Precompute and keep in unified memory:
pixel_tensor = precompute_pixel_values(vlm, loader)
# No .to('mps') needed — unified memory means it's already accessible
```

### M3 Ultra parallelism strategy

With 128GB, you can exploit the M3 Ultra's multi-model advantage:

```python
# Load ALL models upfront (total ~3.2GB, trivial for 128GB)
models = {
    key: load_model(key, device='mps')
    for key in ['clip_b32', 'siglip2_b16', 'siglip2_so400m']
}

# Precompute pixel values per model (different preprocessors)
pixel_cache = {}
for key, vlm in models.items():
    pixel_cache[key] = precompute_pixel_values(vlm, loader)

# Now trials are pure compute — no I/O at all
```

### Expected M3 Ultra performance

The M3 Ultra GPU has ~22 TFLOPS FP32 (vs RTX 5000 Ada's ~50 TFLOPS). Raw compute is ~2.3x slower than the RTX 5000 Ada.

**However**, with the current I/O-bound code, the M3 Ultra might actually be *faster* because:
- No PCIe transfer overhead
- Unified memory eliminates copies
- High memory bandwidth (~800 GB/s)

With optimizations applied (precomputed tensors, no I/O):

| Config | RTX 5000 Ada (optimized) | M3 Ultra (optimized) |
|--------|--------------------------|----------------------|
| Exp 0 full | ~8-15 min | ~15-30 min |
| Exp 5 (N=5000) | ~10-20 min | ~20-40 min |
| All Phase 1 | ~30-60 min | ~60-120 min |

---

## Implementation Priority

1. **Precompute pixel tensors** — Biggest win. Single change eliminates 99.8% of I/O.
2. **Enable progress bars** — Zero cost, huge UX improvement.
3. **AMP fp16** — Easy to add, meaningful speedup on both platforms.
4. **Larger batch sizes** — Free with precomputed tensors.
5. **torch.compile()** — NVIDIA only, good for long runs.
6. **torch.inference_mode()** — Small but free.

## Summary

The current implementation is **I/O bound, not compute bound**. The GPU sits idle 95%+ of the time waiting for PIL image loads. Precomputing pixel tensors once and reusing them across MC passes would transform a 6-8 hour Exp 0 run into a 10-15 minute one on either platform.
