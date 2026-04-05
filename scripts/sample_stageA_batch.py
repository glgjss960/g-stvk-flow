from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of StageA samples by calling eval_stageA repeatedly")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps-per-band", type=int, default=None)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--save-scale", type=float, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_script = Path(__file__).resolve().parent / "eval_stageA.py"
    for i in range(int(args.num_samples)):
        seed = int(args.seed_start + i)
        out_path = out_dir / f"sample_{i:05d}_seed_{seed:08d}.mp4"
        cmd = [
            sys.executable,
            str(eval_script),
            "--config",
            str(args.config.resolve()),
            "--out",
            str(out_path),
            "--seed",
            str(seed),
        ]
        if args.checkpoint is not None:
            cmd.extend(["--checkpoint", str(args.checkpoint.resolve())])
        if args.steps_per_band is not None:
            cmd.extend(["--steps-per-band", str(int(args.steps_per_band))])
        if args.class_label is not None:
            cmd.extend(["--class-label", str(int(args.class_label))])
        if args.fps is not None:
            cmd.extend(["--fps", str(int(args.fps))])
        if args.save_scale is not None:
            cmd.extend(["--save-scale", str(float(args.save_scale))])

        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
