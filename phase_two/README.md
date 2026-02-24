# Phase 2 Prep (Pending Phase 1 Gate)

This folder is initial scaffolding for Phase 2 from `paper_outline_v3.md`.

Phase 2 experiments in scope:
1. Exp 1: Rank stability across `p`.
2. Exp 3: Dropout type ablation.
3. Exp 5 full: Ambiguity on large set + COCO.
4. Exp 2: Synthetic + natural baselines.
5. Exp 4 full: Cross-model matrix.

## Status
- Prep started.
- Execution blocked on Phase 1 gate metrics (`ICC`, `SNR`, `pairwise rho`) and selected model/embedding settings.

## Next implementation tasks
- Add `run_phase2.py` orchestrator.
- Add per-experiment configs and CLIs (`exp1_rank_p.py`, `exp3_dropout_type.py`, ...).
- Bind to Phase 1 selected defaults (`T*`, embedding space, model shortlist).

