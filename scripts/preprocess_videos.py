from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.data import preprocess_video_folder



def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw videos into G-STVK-Flow cached clips")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8, help="clip stride in frames")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    preprocess_video_folder(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        frames=args.frames,
        image_size=args.size,
        train_ratio=args.train_ratio,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()


