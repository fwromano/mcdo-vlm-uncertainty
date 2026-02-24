#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from phase_one.common import list_images, sample_paths, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 experiments from paper_outline_v3")
    parser.add_argument("data_dir", type=str, help="Image root used for class names / manifests")
    parser.add_argument("out_dir", type=str, help="Output root for phase four")
    parser.add_argument(
        "--only",
        type=str,
        default="exp10,exp11",
        help="Comma-separated subset/order: exp10,exp11",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")

    parser.add_argument("--exp10-models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--exp10-prompt-file", type=str, default="")
    parser.add_argument("--exp10-num-prompts", type=int, default=1000)
    parser.add_argument("--exp10-passes", type=int, default=64)
    parser.add_argument("--exp10-trials", type=int, default=3)
    parser.add_argument("--exp10-specificity-quantile", type=float, default=0.20)
    parser.add_argument("--exp10-text-batch-size", type=int, default=128)
    parser.add_argument("--exp10-class-map", type=str, default="")

    parser.add_argument("--exp11-models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--exp11-num-images", type=int, default=2000)
    parser.add_argument("--exp11-passes", type=int, default=32)
    parser.add_argument("--exp11-trials", type=int, default=2)
    parser.add_argument("--exp11-quantile", type=float, default=0.10)
    parser.add_argument("--exp11-class-map", type=str, default="")
    parser.add_argument("--exp11-templates", type=str, default="a photo of a {}|a {}|an image of a {}")
    parser.add_argument("--exp11-num-groups", type=int, default=4)
    parser.add_argument("--exp11-base-p", type=float, default=0.01)
    parser.add_argument("--exp11-min-p", type=float, default=0.001)
    parser.add_argument("--exp11-max-p", type=float, default=0.1)
    parser.add_argument("--exp11-num-candidates", type=int, default=24)

    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = [x.strip() for x in args.only.split(",") if x.strip()]
    known = {"exp10", "exp11"}
    invalid = [x for x in selected if x not in known]
    if invalid:
        raise ValueError(f"Unknown experiment(s) in --only: {invalid}. Known: {sorted(known)}")

    all_paths = list_images(args.data_dir)

    def make_manifest(count: int, seed: int, filename: str) -> Path:
        path = out_root / filename
        save_manifest(sample_paths(all_paths, count, seed), str(path))
        return path

    manifest_exp11 = make_manifest(args.exp11_num_images, args.seed, "manifest_exp11.json")

    for exp_name in selected:
        if exp_name == "exp10":
            cmd = [
                sys.executable,
                "-m",
                "phase_four.exp10_text_encoder_uncertainty",
                args.data_dir,
                str(out_root / "exp10_text_encoder_uncertainty"),
                "--models",
                args.exp10_models,
                "--num-prompts",
                str(args.exp10_num_prompts),
                "--dropout",
                str(args.dropout),
                "--passes",
                str(args.exp10_passes),
                "--trials",
                str(args.exp10_trials),
                "--specificity-quantile",
                str(args.exp10_specificity_quantile),
                "--text-batch-size",
                str(args.exp10_text_batch_size),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--save-every",
                str(args.save_every),
            ]
            if args.exp10_prompt_file:
                cmd.extend(["--prompt-file", args.exp10_prompt_file])
            if args.exp10_class_map:
                cmd.extend(["--class-map", args.exp10_class_map])
            run_cmd(cmd)
            continue

        if exp_name == "exp11":
            cmd = [
                sys.executable,
                "-m",
                "phase_four.exp11_concrete_dropout_proxy",
                args.data_dir,
                str(out_root / "exp11_concrete_dropout_proxy"),
                "--manifest",
                str(manifest_exp11),
                "--models",
                args.exp11_models,
                "--templates",
                args.exp11_templates,
                "--num-groups",
                str(args.exp11_num_groups),
                "--base-p",
                str(args.exp11_base_p),
                "--min-p",
                str(args.exp11_min_p),
                "--max-p",
                str(args.exp11_max_p),
                "--num-candidates",
                str(args.exp11_num_candidates),
                "--passes",
                str(args.exp11_passes),
                "--trials",
                str(args.exp11_trials),
                "--quantile",
                str(args.exp11_quantile),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--save-every",
                str(args.save_every),
            ]
            if args.exp11_class_map:
                cmd.extend(["--class-map", args.exp11_class_map])
            run_cmd(cmd)

    print(f"Phase 4 run completed. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
