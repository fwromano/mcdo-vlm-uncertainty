#!/usr/bin/env python
"""Per-module sensitivity scan.

Tests each linear module in the vision encoder individually to map which
modules produce the most reliable uncertainty ranking when perturbed.

This replaces ad-hoc dropout type selection (A/B/C/D/E) with a systematic
sweep over ALL modules × perturbation types × magnitudes.

Usage:
    python -m phase_two.module_scan data/raw/imagenet_val outputs/phase_two/module_scan \
        --model clip_b32 \
        --ptypes dropout,gaussian,scale \
        --magnitudes 0.01,0.05 \
        --trials 3 --passes 32 --num-images 200 \
        --device mps --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from phase_one.common import (
    build_loader,
    detect_best_device,
    list_images,
    load_model,
    reliability_from_trials,
    run_mc_trial,
    sample_paths,
    save_json,
    set_all_seeds,
)
from phase_two.perturbation import (
    disable_all_perturbation,
    named_linears,
    perturb_modules,
)


def evaluate_config(
    vlm,
    loader,
    configs: List[Tuple[str, str, float]],
    trials: int,
    passes: int,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a perturbation configuration.

    Args:
        vlm: Vision-language model
        loader: DataLoader for images
        configs: List of (module_path, ptype, magnitude) to apply
        trials: Number of independent trials
        passes: MC passes per trial
        seed: Base seed

    Returns:
        Dict with reliability metrics + trace_mean
    """
    root = vlm.vision_root
    disable_all_perturbation(root)

    trace_arrays: List[np.ndarray] = []
    with perturb_modules(root, configs):
        for trial_idx in range(trials):
            set_all_seeds(seed + trial_idx)
            trial = run_mc_trial(vlm=vlm, loader=loader, passes=passes)
            trace_arrays.append(trial["trace_pre"].numpy())

    values = np.stack(trace_arrays)
    rel = reliability_from_trials(values)
    rel["trace_mean"] = float(values.mean())
    return rel


def scan_all_modules(
    vlm,
    loader,
    perturbation_types: List[str],
    magnitudes: List[float],
    trials: int,
    passes: int,
    seed: int,
    out_dir: Path,
    save_every: int = 5,
) -> List[Dict[str, Any]]:
    """Scan each linear module individually.

    Returns list of results sorted by pairwise_spearman_median descending.
    """
    root = vlm.vision_root
    all_linears = named_linears(root)
    total = len(all_linears) * len(perturbation_types) * len(magnitudes)

    print(f"Scanning {len(all_linears)} modules x {len(perturbation_types)} types x {len(magnitudes)} magnitudes = {total} configs")
    print(f"Trials={trials}, passes={passes}, N={len(loader.dataset)}")

    results: List[Dict[str, Any]] = []
    done = 0

    for mod_idx, (path, linear) in enumerate(all_linears):
        for ptype in perturbation_types:
            for mag in magnitudes:
                t0 = time.time()
                try:
                    rel = evaluate_config(
                        vlm, loader,
                        configs=[(path, ptype, mag)],
                        trials=trials, passes=passes, seed=seed,
                    )
                except Exception as exc:
                    rel = {"error": str(exc)}

                elapsed = time.time() - t0
                entry = {
                    "module": path,
                    "module_idx": mod_idx,
                    "ptype": ptype,
                    "magnitude": mag,
                    "in_features": linear.in_features,
                    "out_features": linear.out_features,
                    "elapsed_sec": round(elapsed, 1),
                    **rel,
                }
                results.append(entry)
                done += 1

                spearman = rel.get("pairwise_spearman_median", 0)
                trace_m = rel.get("trace_mean", 0)
                print(
                    f"  [{done}/{total}] {path}  {ptype}@{mag}  "
                    f"spearman={spearman:.3f}  trace={trace_m:.6f}  "
                    f"({elapsed:.1f}s)"
                )

                if save_every > 0 and done % save_every == 0:
                    _save_partial(results, out_dir)

    results.sort(key=lambda r: r.get("pairwise_spearman_median", -1), reverse=True)
    return results


def greedy_combination_search(
    vlm,
    loader,
    candidates: List[Dict[str, Any]],
    max_modules: int,
    trials: int,
    passes: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Greedy forward selection: add modules one at a time, keeping only additions that improve reliability.

    Args:
        candidates: Ranked scan results (best first)
        max_modules: Maximum modules in the combination
        trials/passes/seed: Evaluation parameters (should be higher than scan)

    Returns:
        List of results for each step of the greedy search
    """
    print(f"\n{'='*70}")
    print(f"Greedy combination search (top {len(candidates)} candidates, max {max_modules} modules)")
    print(f"{'='*70}")

    selected: List[Tuple[str, str, float]] = []
    best_spearman = -1.0
    history: List[Dict[str, Any]] = []

    for step in range(min(max_modules, len(candidates))):
        best_addition = None
        best_addition_score = best_spearman

        for cand in candidates:
            key = (cand["module"], cand["ptype"], cand["magnitude"])
            if key in selected:
                continue

            trial_config = selected + [key]
            rel = evaluate_config(
                vlm, loader,
                configs=trial_config,
                trials=trials, passes=passes, seed=seed,
            )
            score = rel.get("pairwise_spearman_median", -1)

            if score > best_addition_score:
                best_addition = cand
                best_addition_score = score
                best_addition_rel = rel

        if best_addition is None or best_addition_score <= best_spearman:
            print(f"  Step {step + 1}: No improvement found. Stopping.")
            break

        key = (best_addition["module"], best_addition["ptype"], best_addition["magnitude"])
        selected.append(key)
        best_spearman = best_addition_score

        step_result = {
            "step": step + 1,
            "added_module": best_addition["module"],
            "added_ptype": best_addition["ptype"],
            "added_magnitude": best_addition["magnitude"],
            "total_modules": len(selected),
            "config": [{"module": m, "ptype": t, "magnitude": g} for m, t, g in selected],
            **best_addition_rel,
        }
        history.append(step_result)

        print(
            f"  Step {step + 1}: +{best_addition['module']}  "
            f"({best_addition['ptype']}@{best_addition['magnitude']})  "
            f"spearman={best_spearman:.3f}"
        )

    return history


def _save_partial(results: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scan_partial.json", "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-module sensitivity scan")
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--model", default="clip_b32")
    parser.add_argument("--ptypes", default="dropout,gaussian,scale",
                        help="Comma-separated perturbation types")
    parser.add_argument("--magnitudes", default="0.01,0.05",
                        help="Comma-separated magnitudes to test")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    # Combination search options
    parser.add_argument("--combo-top-k", type=int, default=10,
                        help="Take top K from scan for combination search")
    parser.add_argument("--combo-max-modules", type=int, default=5,
                        help="Max modules in combination")
    parser.add_argument("--combo-trials", type=int, default=5,
                        help="Trials for combination evaluation (higher = more reliable)")
    parser.add_argument("--combo-passes", type=int, default=64)
    parser.add_argument("--combo-num-images", type=int, default=500)
    parser.add_argument("--skip-combo", action="store_true",
                        help="Skip combination search (scan only)")
    args = parser.parse_args()

    if args.device is None:
        args.device = detect_best_device()

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    all_paths = list_images(args.data_dir)
    scan_paths = sample_paths(all_paths, args.num_images, args.seed)
    scan_loader = build_loader(scan_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    ptypes = [t.strip() for t in args.ptypes.split(",") if t.strip()]
    magnitudes = [float(m.strip()) for m in args.magnitudes.split(",") if m.strip()]

    # ── Phase 1: Module scan ──
    print(f"\n{'='*70}")
    print(f"PHASE 1: Module Sensitivity Scan — {args.model}")
    print(f"{'='*70}")
    t0 = time.time()

    vlm = load_model(args.model, device=args.device)

    scan_results = scan_all_modules(
        vlm, scan_loader,
        perturbation_types=ptypes,
        magnitudes=magnitudes,
        trials=args.trials, passes=args.passes, seed=args.seed,
        out_dir=out_dir,
        save_every=args.save_every,
    )

    scan_elapsed = time.time() - t0
    print(f"\nScan complete in {scan_elapsed / 60:.1f} min")

    # Top 10
    print(f"\nTop 10 modules by pairwise Spearman:")
    print(f"  {'Module':>55s}  {'Type':>8s}  {'Mag':>6s}  {'Spearman':>8s}  {'Trace':>10s}")
    print(f"  {'-'*95}")
    for r in scan_results[:10]:
        print(
            f"  {r['module']:>55s}  {r['ptype']:>8s}  {r['magnitude']:>6.3f}  "
            f"{r.get('pairwise_spearman_median', 0):>8.3f}  {r.get('trace_mean', 0):>10.6f}"
        )

    save_json(
        {"scan_results": scan_results, "elapsed_sec": scan_elapsed,
         "config": {"model": args.model, "ptypes": ptypes, "magnitudes": magnitudes,
                    "num_images": args.num_images, "trials": args.trials, "passes": args.passes}},
        str(out_dir / "scan_results.json"),
    )

    # ── Phase 2: Combination search ──
    if args.skip_combo:
        print("\nSkipping combination search (--skip-combo)")
        return

    print(f"\n{'='*70}")
    print(f"PHASE 2: Greedy Combination Search")
    print(f"{'='*70}")

    # Use more images for the combination evaluation
    combo_paths = sample_paths(all_paths, args.combo_num_images, args.seed + 999)
    combo_loader = build_loader(combo_paths, batch_size=args.batch_size, num_workers=args.num_workers)

    # Deduplicate candidates by module (keep best ptype/mag per module)
    seen_modules = set()
    deduped: List[Dict[str, Any]] = []
    for r in scan_results:
        if r["module"] not in seen_modules:
            seen_modules.add(r["module"])
            deduped.append(r)
        if len(deduped) >= args.combo_top_k:
            break

    t1 = time.time()
    combo_results = greedy_combination_search(
        vlm, combo_loader,
        candidates=deduped,
        max_modules=args.combo_max_modules,
        trials=args.combo_trials,
        passes=args.combo_passes,
        seed=args.seed,
    )
    combo_elapsed = time.time() - t1

    save_json(
        {"combination_results": combo_results, "elapsed_sec": combo_elapsed,
         "top_k_candidates": deduped[:args.combo_top_k]},
        str(out_dir / "combination_results.json"),
    )

    print(f"\nCombination search complete in {combo_elapsed / 60:.1f} min")
    print(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
