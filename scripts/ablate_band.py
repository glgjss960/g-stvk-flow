from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convenience wrapper for phase-A band ablation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--band", type=str, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps-per-band", type=int, default=None)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_stageA.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--out",
        str(args.out),
        "--ablate-band",
        args.band,
        "--ablate-scale",
        str(args.scale),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.steps_per_band is not None:
        cmd.extend(["--steps-per-band", str(args.steps_per_band)])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
