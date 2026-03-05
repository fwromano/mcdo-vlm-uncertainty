#!/usr/bin/env bash
# Run all remaining Phase 2 experiments sequentially.
# Waits for the current Exp 1 process to finish first.
#
# Usage:
#   nohup bash run_phase2_all.sh >> outputs/phase2_runner.log 2>&1 &

set -euo pipefail

DATA_DIR="data/raw/imagenet_val"
OUT_ROOT="outputs/phase_two"
CLASS_MAP="data/imagenet_class_map.json"
DEVICE="mps"
CONDA="conda run --no-capture-output -n mcdo"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Wait for any existing phase_two experiments to finish ─────────────
wait_for_clear() {
    while pgrep -f "phase_two\.exp" >/dev/null 2>&1; do
        log "Waiting for running experiments to finish..."
        sleep 60
    done
    log "No phase_two experiments running. Proceeding."
}
wait_for_clear

# ── Exp 2: Synthetic + Natural Baselines (fast, ~5 min) ───────────────
log "=== STARTING EXP 2: Synthetic + Natural Baselines ==="
$CONDA python -u -m phase_two.exp2_synthetic_natural \
    "$DATA_DIR" "$OUT_ROOT/exp2_synthetic_natural" \
    --models clip_b32,siglip2_b16 \
    --num-natural 10 --num-each-synth 10 \
    --passes 64 --trials 5 --batch-size 32 --num-workers 0 \
    --device "$DEVICE" --seed 42 --save-every 1 \
    || log "WARNING: Exp 2 failed with exit code $?"
log "=== EXP 2 DONE ==="

# ── Exp 3: Dropout Type Ablation (the important one, ~2-3 hrs) ────────
log "=== STARTING EXP 3: Dropout Type Ablation ==="
$CONDA python -u -m phase_two.exp3_dropout_type \
    "$DATA_DIR" "$OUT_ROOT/exp3_dropout_type" \
    --models clip_b32 \
    --dropout-types A,B,C,D,E \
    --num-images 1000 --dropout 0.01 \
    --passes 64 --trials 5 --batch-size 32 --num-workers 0 \
    --device "$DEVICE" --seed 42 --save-every 1 \
    || log "WARNING: Exp 3 failed with exit code $?"
log "=== EXP 3 DONE ==="

# ── Exp 4: Full Model Matrix (5 models × 10 trials, ~4-6 hrs) ─────────
log "=== STARTING EXP 4: Full Model Matrix ==="
$CONDA python -u -m phase_two.exp4_full_matrix \
    "$DATA_DIR" "$OUT_ROOT/exp4_full_matrix" \
    --models clip_b32,siglip2_b16,siglip2_so400m,clip_l14 \
    --num-images 500 --dropout 0.01 \
    --passes 64 --trials 10 --batch-size 32 --num-workers 0 \
    --device "$DEVICE" --seed 42 --save-every 1 \
    || log "WARNING: Exp 4 failed with exit code $?"
log "=== EXP 4 DONE ==="

# ── Exp 5: Full Ambiguity Prediction N=10K (2 models, ~1-2 hrs) ───────
log "=== STARTING EXP 5: Full Ambiguity Prediction ==="
$CONDA python -u -m phase_two.exp5_full_ambiguity \
    "$DATA_DIR" "$OUT_ROOT/exp5_full_ambiguity" \
    --models clip_b32,siglip2_b16 \
    --num-images 10000 --class-map "$CLASS_MAP" \
    --templates "a photo of a {}|a {}|an image of a {}" \
    --dropout 0.01 --passes 64 --trials 1 \
    --batch-size 32 --num-workers 0 \
    --device "$DEVICE" --seed 42 --save-every 1 \
    || log "WARNING: Exp 5 failed with exit code $?"
log "=== EXP 5 DONE ==="

# ── Exp 6: Mean Convergence (2 models, ~1 hr) ─────────────────────────
log "=== STARTING EXP 6: Mean Convergence ==="
$CONDA python -u -m phase_two.exp6_mean_convergence \
    "$DATA_DIR" "$OUT_ROOT/exp6_mean_convergence" \
    --models clip_b32,siglip2_b16 \
    --num-images 500 --dropout 0.01 \
    --passes "4,8,16,32,64" --trials 3 \
    --batch-size 32 --num-workers 0 \
    --device "$DEVICE" --seed 42 --save-every 1 \
    || log "WARNING: Exp 6 failed with exit code $?"
log "=== EXP 6 DONE ==="

log "=========================================="
log "ALL PHASE 2 EXPERIMENTS COMPLETE"
log "=========================================="
