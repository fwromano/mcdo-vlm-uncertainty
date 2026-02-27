#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from phase_one.common import (
    build_loader,
    list_images,
    load_manifest,
    load_model,
    reliability_from_trials,
    run_mc_trial,
    sample_paths,
    save_json,
    save_manifest,
    set_all_seeds,
    should_save_checkpoint,
)
from phase_two.dropout_types import configure_dropout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Exp 1: rank stability across dropout rates p")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--p-values", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from per-p partial checkpoints when available",
    )
    return parser.parse_args()


def spearman_matrix(rows: np.ndarray) -> np.ndarray:
    k = rows.shape[0]
    out = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            rho, _ = spearmanr(rows[i], rows[j])
            out[i, j] = out[j, i] = 0.0 if np.isnan(rho) else float(rho)
    return out


def _scalar_from_npz(payload: np.lib.npyio.NpzFile, key: str, default: float | int) -> float | int:
    if key not in payload:
        return default
    arr = np.asarray(payload[key]).reshape(-1)
    if arr.size == 0:
        return default
    item = arr[0]
    if isinstance(default, int):
        return int(item)
    return float(item)


def load_partial_uncertainty_trials(
    partial_path: Path,
    *,
    expected_trials: int,
    expected_num_images: int,
    expected_passes: int,
    expected_p: float,
) -> List[np.ndarray]:
    if not partial_path.exists():
        return []
    try:
        with np.load(partial_path, allow_pickle=False) as payload:
            if "uncertainty_trials" not in payload:
                return []
            trials = np.asarray(payload["uncertainty_trials"])
            if trials.ndim != 2:
                print(f"[Exp1] Ignoring invalid checkpoint {partial_path} (expected 2-D uncertainty array).")
                return []
            if trials.shape[1] != expected_num_images:
                print(
                    f"[Exp1] Ignoring checkpoint {partial_path} "
                    f"(num_images mismatch: {trials.shape[1]} != {expected_num_images})."
                )
                return []

            saved_total = int(_scalar_from_npz(payload, "total_trials", expected_trials))
            saved_passes = int(_scalar_from_npz(payload, "passes", expected_passes))
            saved_p = float(_scalar_from_npz(payload, "p_value", expected_p))

            if saved_total != expected_trials:
                print(
                    f"[Exp1] Ignoring checkpoint {partial_path} "
                    f"(trials mismatch: {saved_total} != {expected_trials})."
                )
                return []
            if saved_passes != expected_passes:
                print(
                    f"[Exp1] Ignoring checkpoint {partial_path} "
                    f"(passes mismatch: {saved_passes} != {expected_passes})."
                )
                return []
            if not np.isclose(saved_p, expected_p, rtol=0.0, atol=1e-12):
                print(
                    f"[Exp1] Ignoring checkpoint {partial_path} "
                    f"(p mismatch: {saved_p} != {expected_p})."
                )
                return []

            completed = int(_scalar_from_npz(payload, "completed_trials", trials.shape[0]))
            completed = max(0, min(completed, trials.shape[0], expected_trials))
            return [trials[idx] for idx in range(completed)]
    except Exception as exc:  # noqa: BLE001
        print(f"[Exp1] Failed to read checkpoint {partial_path}: {exc}")
        return []


def _acquire_lock(lock_path: Path) -> None:
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = int(payload.get("pid", -1))
            except Exception:  # noqa: BLE001
                pid = -1
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    raise RuntimeError(
                        f"Another exp1_rank_p process is already using {lock_path.parent} (pid={pid}). "
                        "Stop it or use a different output directory."
                    )
                except ProcessLookupError:
                    pass
                except PermissionError:
                    raise RuntimeError(
                        f"Another exp1_rank_p process may already be using {lock_path.parent} (pid={pid})."
                    ) from None
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        except OSError as exc:
            raise RuntimeError(f"Unable to create lock file {lock_path}: {exc}") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid()}, f)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".exp1_rank_p.lock"
    _acquire_lock(lock_path)

    try:
        if args.manifest:
            sampled_paths = load_manifest(args.manifest)
        else:
            sampled_paths = sample_paths(list_images(args.data_dir), args.num_images, args.seed)
            save_manifest(sampled_paths, str(out_dir / "phase2_exp1_manifest.json"))

        p_values = [float(x.strip()) for x in args.p_values.split(",") if x.strip()]
        model_keys = [x.strip() for x in args.models.split(",") if x.strip()]

        loader = build_loader(sampled_paths, batch_size=args.batch_size, num_workers=args.num_workers)

        overall: Dict[str, Dict[str, object]] = {}

        for model_key in model_keys:
            print(f"[Exp1] model={model_key}")
            model_out = out_dir / model_key
            model_out.mkdir(parents=True, exist_ok=True)

            try:
                vlm = load_model(model_key, device=args.device)
            except Exception as exc:  # noqa: BLE001
                overall[model_key] = {"error": str(exc)}
                save_json(overall[model_key], str(model_out / "exp1_error.json"))
                continue

            p_mean_unc: List[np.ndarray] = []
            p_trial_arrays: Dict[str, np.ndarray] = {}
            p_reliability: Dict[str, Dict[str, float]] = {}
            p_wrapped: Dict[str, int] = {}

            for p in p_values:
                p_key = str(p)
                p_token = str(p).replace(".", "_")
                partial_path = model_out / f"exp1_p_{p_token}_partial.npz"
                progress_path = model_out / f"exp1_p_{p_token}_progress.json"

                cfg = configure_dropout(vlm, dropout_type="E", p=p)
                wrapped_count = cfg.wrapped_modules

                trial_unc: List[np.ndarray] = []
                if args.resume:
                    trial_unc = load_partial_uncertainty_trials(
                        partial_path,
                        expected_trials=args.trials,
                        expected_num_images=len(sampled_paths),
                        expected_passes=args.passes,
                        expected_p=p,
                    )
                    if trial_unc:
                        print(f"[Exp1] {model_key} p={p}: resuming from {len(trial_unc)}/{args.trials} trial(s)")

                start_trial = len(trial_unc)
                for trial_idx in range(start_trial, args.trials):
                    set_all_seeds(args.seed + int(p * 1e6) + trial_idx)
                    cfg = configure_dropout(vlm, dropout_type="E", p=p)
                    wrapped_count = max(wrapped_count, cfg.wrapped_modules)
                    trial = run_mc_trial(vlm=vlm, loader=loader, passes=args.passes, collect_pass_features=False)
                    trial_unc.append(trial["trace_pre"].numpy())

                    completed = len(trial_unc)
                    if should_save_checkpoint(completed=completed, total=args.trials, every=args.save_every):
                        partial = np.stack(trial_unc, axis=0)
                        np.savez_compressed(
                            partial_path,
                            paths=np.asarray(sampled_paths),
                            p_value=np.asarray([p], dtype=np.float64),
                            passes=np.asarray([args.passes], dtype=np.int64),
                            uncertainty_trials=partial,
                            completed_trials=np.asarray([completed], dtype=np.int64),
                            total_trials=np.asarray([args.trials], dtype=np.int64),
                        )
                        save_json(
                            {
                                "experiment": "exp1_rank_p",
                                "model": model_key,
                                "p_value": p,
                                "completed_trials": completed,
                                "total_trials": args.trials,
                            },
                            str(progress_path),
                        )

                arr = np.stack(trial_unc, axis=0)
                p_mean_unc.append(arr.mean(axis=0))
                p_trial_arrays[p_key] = arr
                p_reliability[p_key] = reliability_from_trials(arr)
                p_wrapped[p_key] = wrapped_count

            mean_matrix = np.stack(p_mean_unc, axis=0)
            rho_matrix = spearman_matrix(mean_matrix)

            np.savez_compressed(
                model_out / "exp1_rank_p_trials.npz",
                paths=np.asarray(sampled_paths),
                p_values=np.asarray(p_values, dtype=np.float64),
                mean_uncertainty=mean_matrix,
                spearman_matrix=rho_matrix,
                **{f"trials_p_{str(p).replace('.', '_')}": arr for p, arr in p_trial_arrays.items()},
            )

            summary = {
                "experiment": "exp1_rank_p",
                "model": model_key,
                "num_images": len(sampled_paths),
                "passes": args.passes,
                "trials": args.trials,
                "p_values": p_values,
                "reliability_by_p": p_reliability,
                "wrapped_modules_by_p": p_wrapped,
                "spearman_matrix": rho_matrix.tolist(),
            }
            save_json(summary, str(model_out / "exp1_summary.json"))
            overall[model_key] = summary

        overall_summary = {
            "experiment": "exp1_rank_p",
            "models": model_keys,
            "num_images": len(sampled_paths),
            "p_values": p_values,
            "results": overall,
        }
        save_json(overall_summary, str(out_dir / "exp1_overall_summary.json"))

        print(f"[Exp1] Complete: {out_dir}")
    finally:
        _release_lock(lock_path)


if __name__ == "__main__":
    main()
