from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.data import preprocess_video_folder


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def _resolve(root: Path, maybe_path: str | Path | None) -> Path | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else (root / p)


def _pick(args_value, cfg_value, default):
    if args_value is not None:
        return args_value
    if cfg_value is not None:
        return cfg_value
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw videos into G-STVK-Flow cached clips")
    parser.add_argument("--config", type=Path, default=None, help="yaml config (uses `preprocess:` section if present)")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--size", type=int, default=None, help="fallback square size when --height/--width are not set")
    parser.add_argument("--height", type=int, default=None, help="target output frame height")
    parser.add_argument("--width", type=int, default=None, help="target output frame width")
    parser.add_argument("--fps", type=float, default=None, help="target fps before clip slicing")
    parser.add_argument("--stride", type=int, default=None, help="clip stride in frames")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument(
        "--tail-pad-last-window",
        dest="tail_pad_last_window",
        action="store_true",
        default=None,
        help="enable tail window and pad to full clip by repeating last frame",
    )
    parser.add_argument(
        "--no-tail-pad-last-window",
        dest="tail_pad_last_window",
        action="store_false",
        help="disable tail padding; keep only full windows",
    )

    parser.add_argument(
        "--cache-dtype",
        type=str,
        choices=["float32", "float16", "uint8"],
        default=None,
        help="cache tensor dtype on disk; uint8 reduces disk usage the most",
    )

    parser.add_argument(
        "--save-new-zipfile-serialization",
        dest="save_new_zipfile_serialization",
        action="store_true",
        default=None,
        help="use torch new zipfile serialization (default: false for better network-FS compatibility)",
    )
    parser.add_argument(
        "--save-legacy-serialization",
        dest="save_new_zipfile_serialization",
        action="store_false",
        help="use legacy torch serialization (recommended on unstable/network filesystems)",
    )

    parser.add_argument("--save-retries", type=int, default=None, help="retry count when clip write fails")
    parser.add_argument(
        "--save-retry-backoff-seconds",
        type=float,
        default=None,
        help="backoff multiplier between save retries",
    )

    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    cfg_root = Path.cwd()
    if args.config is not None:
        cfg_path = args.config.resolve()
        cfg_root = cfg_path.parent
        cfg = _load_yaml(cfg_path)

    pre_cfg = cfg.get("preprocess", cfg)

    raw_dir = _pick(args.raw_dir, _resolve(cfg_root, pre_cfg.get("raw_dir")), None)
    out_dir = _pick(args.out_dir, _resolve(cfg_root, pre_cfg.get("out_dir")), None)
    frames = int(_pick(args.frames, pre_cfg.get("frames"), 16))
    size = int(_pick(args.size, pre_cfg.get("image_size"), 128))
    height = _pick(args.height, pre_cfg.get("image_height"), None)
    width = _pick(args.width, pre_cfg.get("image_width"), None)
    fps = _pick(args.fps, pre_cfg.get("fps"), None)
    stride = int(_pick(args.stride, pre_cfg.get("stride"), frames))
    train_ratio = float(_pick(args.train_ratio, pre_cfg.get("train_ratio"), 0.9))
    tail_pad_last_window = bool(_pick(args.tail_pad_last_window, pre_cfg.get("tail_pad_last_window"), True))

    cache_dtype = str(_pick(args.cache_dtype, pre_cfg.get("cache_dtype"), "float16"))
    save_new_zipfile_serialization = bool(
        _pick(args.save_new_zipfile_serialization, pre_cfg.get("save_new_zipfile_serialization"), False)
    )
    save_retries = int(_pick(args.save_retries, pre_cfg.get("save_retries"), 5))
    save_retry_backoff_seconds = float(
        _pick(args.save_retry_backoff_seconds, pre_cfg.get("save_retry_backoff_seconds"), 0.5)
    )

    if raw_dir is None or out_dir is None:
        raise ValueError("raw_dir and out_dir must be provided either by CLI or yaml config")

    preprocess_video_folder(
        raw_dir=raw_dir,
        out_dir=out_dir,
        frames=frames,
        image_size=size,
        image_height=(int(height) if height is not None else None),
        image_width=(int(width) if width is not None else None),
        target_fps=(float(fps) if fps is not None else None),
        train_ratio=train_ratio,
        stride=stride,
        tail_pad_last_window=tail_pad_last_window,
        cache_dtype=cache_dtype,
        save_new_zipfile_serialization=save_new_zipfile_serialization,
        save_retries=save_retries,
        save_retry_backoff_seconds=save_retry_backoff_seconds,
    )


if __name__ == "__main__":
    main()

