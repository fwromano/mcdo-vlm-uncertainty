#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Exp 9: MOT adaptive association evaluator")
    parser.add_argument("cost_json", type=str, help="Path to frame-wise MOT association payload")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--uncertainty-key", type=str, default="uncertainty")
    parser.add_argument("--oracle-key", type=str, default="oracle_uncertainty")
    parser.add_argument("--laplace-key", type=str, default="laplace_uncertainty")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--modes", type=str, default="baseline,adaptive,oracle,laplace")
    return parser.parse_args()


def _load_frames(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        frames = payload.get("frames", [])
    elif isinstance(payload, list):
        frames = payload
    else:
        raise ValueError("cost_json must be a list or an object with `frames`")

    if not isinstance(frames, list) or not frames:
        raise ValueError("No frames found in cost_json")

    normalized: List[Dict[str, object]] = []
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        tracks = frame.get("tracks", [])
        detections = frame.get("detections", [])
        motion = frame.get("motion_cost")
        appearance = frame.get("appearance_cost")

        if not isinstance(tracks, list) or not isinstance(detections, list):
            continue
        if motion is None or appearance is None:
            continue

        motion_arr = np.asarray(motion, dtype=np.float64)
        appearance_arr = np.asarray(appearance, dtype=np.float64)
        if motion_arr.ndim != 2 or appearance_arr.ndim != 2:
            continue
        if motion_arr.shape != appearance_arr.shape:
            continue
        if motion_arr.shape[0] != len(tracks) or motion_arr.shape[1] != len(detections):
            continue

        normalized.append(
            {
                "frame_id": frame.get("frame_id", idx),
                "tracks": tracks,
                "detections": detections,
                "motion_cost": motion_arr,
                "appearance_cost": appearance_arr,
            }
        )

    if not normalized:
        raise ValueError("No valid frames found after schema validation")

    return normalized


def _has_detection_key(frames: Sequence[Dict[str, object]], key: str) -> bool:
    for frame in frames:
        detections = frame["detections"]
        if any(isinstance(det, dict) and key in det for det in detections):
            return True
    return False


def _detection_scales(detections: Sequence[Dict[str, object]], key: str, epsilon: float) -> np.ndarray:
    out = np.ones(len(detections), dtype=np.float64)
    for i, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        value = det.get(key, 1.0)
        try:
            value_f = float(value)
        except Exception:  # noqa: BLE001
            value_f = 1.0
        if not np.isfinite(value_f):
            value_f = 1.0
        out[i] = max(value_f, epsilon)
    return out


def _evaluate_mode(
    frames: Sequence[Dict[str, object]],
    mode: str,
    alpha: float,
    beta: float,
    epsilon: float,
    uncertainty_key: str,
    oracle_key: str,
    laplace_key: str,
) -> Dict[str, object]:
    if mode == "baseline":
        key = ""
    elif mode == "adaptive":
        key = uncertainty_key
    elif mode == "oracle":
        key = oracle_key
    elif mode == "laplace":
        key = laplace_key
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_frames = len(frames)
    total_tracks = 0
    total_dets = 0
    total_assignments = 0
    total_correct = 0
    total_unmatched_tracks = 0
    total_unmatched_dets = 0
    id_switches = 0
    fragmentation = 0
    cost_sum = 0.0

    has_gt = False
    last_gt_by_track: Dict[str, str] = {}
    last_correct_frame_by_gt: Dict[str, int] = {}

    for frame_idx, frame in enumerate(frames):
        tracks = frame["tracks"]
        detections = frame["detections"]
        motion = frame["motion_cost"]
        appearance = frame["appearance_cost"]

        n_tracks = len(tracks)
        n_dets = len(detections)
        total_tracks += n_tracks
        total_dets += n_dets

        if n_tracks == 0:
            total_unmatched_dets += n_dets
            continue
        if n_dets == 0:
            total_unmatched_tracks += n_tracks
            continue

        if mode == "baseline":
            scales = np.ones(n_dets, dtype=np.float64)
        else:
            scales = _detection_scales(detections=detections, key=key, epsilon=epsilon)

        combined = alpha * motion + beta * (appearance / scales[None, :])
        row_idx, col_idx = linear_sum_assignment(combined)

        total_assignments += int(len(row_idx))
        total_unmatched_tracks += int(n_tracks - len(row_idx))
        total_unmatched_dets += int(n_dets - len(col_idx))

        for r, c in zip(row_idx.tolist(), col_idx.tolist()):
            cost_sum += float(combined[r, c])
            track = tracks[r] if isinstance(tracks[r], dict) else {}
            det = detections[c] if isinstance(detections[c], dict) else {}

            track_id = str(track.get("id", f"track_{r}"))
            track_gt = track.get("gt_id")
            det_gt = det.get("gt_id")

            if track_gt is not None and det_gt is not None:
                has_gt = True
                track_gt_s = str(track_gt)
                det_gt_s = str(det_gt)

                if track_gt_s == det_gt_s:
                    total_correct += 1
                    prev_correct = last_correct_frame_by_gt.get(det_gt_s)
                    if prev_correct is not None and prev_correct < frame_idx - 1:
                        fragmentation += 1
                    last_correct_frame_by_gt[det_gt_s] = frame_idx

                prev = last_gt_by_track.get(track_id)
                if prev is not None and prev != det_gt_s:
                    id_switches += 1
                last_gt_by_track[track_id] = det_gt_s

    accuracy = float(total_correct / total_assignments) if (has_gt and total_assignments > 0) else float("nan")
    avg_cost = float(cost_sum / total_assignments) if total_assignments > 0 else float("nan")

    return {
        "mode": mode,
        "num_frames": total_frames,
        "num_tracks": total_tracks,
        "num_detections": total_dets,
        "num_assignments": total_assignments,
        "num_unmatched_tracks": total_unmatched_tracks,
        "num_unmatched_detections": total_unmatched_dets,
        "avg_assignment_cost": avg_cost,
        "has_gt_labels": has_gt,
        "assignment_accuracy": accuracy,
        "id_switches": int(id_switches),
        "fragmentation": int(fragmentation),
    }


def _relative_improvement(curr: float, base: float, higher_is_better: bool) -> Optional[float]:
    if not np.isfinite(curr) or not np.isfinite(base):
        return None
    denom = abs(base) if abs(base) > 1e-12 else 1.0
    raw = (curr - base) / denom
    return float(raw if higher_is_better else -raw)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_frames(args.cost_json)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"baseline", "adaptive", "oracle", "laplace"}
    bad_modes = [m for m in requested_modes if m not in valid_modes]
    if bad_modes:
        raise ValueError(f"Unsupported mode(s): {bad_modes}. Valid: {sorted(valid_modes)}")

    selected_modes: List[str] = []
    for mode in requested_modes:
        if mode == "oracle" and not _has_detection_key(frames, args.oracle_key):
            print(f"[Exp9] Skipping oracle mode: no `{args.oracle_key}` present")
            continue
        if mode == "laplace" and not _has_detection_key(frames, args.laplace_key):
            print(f"[Exp9] Skipping laplace mode: no `{args.laplace_key}` present")
            continue
        if mode == "adaptive" and not _has_detection_key(frames, args.uncertainty_key):
            print(f"[Exp9] Skipping adaptive mode: no `{args.uncertainty_key}` present")
            continue
        selected_modes.append(mode)

    if not selected_modes:
        raise ValueError("No runnable modes selected after checking available uncertainty keys")

    results: Dict[str, Dict[str, object]] = {}
    for mode in selected_modes:
        print(f"[Exp9] mode={mode}")
        results[mode] = _evaluate_mode(
            frames=frames,
            mode=mode,
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.epsilon,
            uncertainty_key=args.uncertainty_key,
            oracle_key=args.oracle_key,
            laplace_key=args.laplace_key,
        )
        with open(out_dir / "exp9_progress.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": "exp9_mot_adaptive_demo",
                    "completed_modes": len(results),
                    "total_modes": len(selected_modes),
                    "results_so_far": results,
                },
                f,
                indent=2,
            )

    improvements: Dict[str, Dict[str, Optional[float]]] = {}
    baseline = results.get("baseline")
    if baseline is not None:
        for mode, payload in results.items():
            if mode == "baseline":
                continue
            improvements[mode] = {
                "assignment_accuracy": _relative_improvement(
                    float(payload["assignment_accuracy"]),
                    float(baseline["assignment_accuracy"]),
                    higher_is_better=True,
                ),
                "id_switches": _relative_improvement(
                    float(payload["id_switches"]),
                    float(baseline["id_switches"]),
                    higher_is_better=False,
                ),
                "fragmentation": _relative_improvement(
                    float(payload["fragmentation"]),
                    float(baseline["fragmentation"]),
                    higher_is_better=False,
                ),
                "avg_assignment_cost": _relative_improvement(
                    float(payload["avg_assignment_cost"]),
                    float(baseline["avg_assignment_cost"]),
                    higher_is_better=False,
                ),
            }

    summary = {
        "experiment": "exp9_mot_adaptive_demo",
        "cost_json": str(Path(args.cost_json).resolve()),
        "num_frames": len(frames),
        "alpha": args.alpha,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "uncertainty_key": args.uncertainty_key,
        "oracle_key": args.oracle_key,
        "laplace_key": args.laplace_key,
        "results": results,
        "relative_improvement_vs_baseline": improvements,
    }

    with open(out_dir / "exp9_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Exp9] Complete: {out_dir}")


if __name__ == "__main__":
    main()
