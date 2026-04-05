from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, maybe_path: str | None) -> Path | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else (root / p)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rec = json.loads(ln)
        name = rec.get("checkpoint_file")
        if isinstance(name, str) and name:
            out.add(name)
    return out


def _epoch_from_checkpoint_name(name: str) -> int | None:
    if not name.startswith("epoch_") or not name.endswith(".pt"):
        return None
    body = name[len("epoch_") : -len(".pt")]
    if body.isdigit():
        return int(body)
    return None


def _iter_epoch_checkpoints(ckpt_dir: Path) -> list[Path]:
    paths = sorted(list(ckpt_dir.glob("epoch_*.pt")))
    return [p for p in paths if _epoch_from_checkpoint_name(p.name) is not None]


def _run_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def _count_mp4(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.mp4")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously monitor StageA quality metrics per checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--real-manifest", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)

    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--steps-per-band", type=int, default=None)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--save-scale", type=float, default=None)

    parser.add_argument("--feature-frames", type=int, default=None)
    parser.add_argument("--feature-size", type=int, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=None)

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--feature-pretrained", dest="feature_pretrained", action="store_true")
    fp_group.add_argument("--no-feature-pretrained", dest="feature_pretrained", action="store_false")
    parser.set_defaults(feature_pretrained=None)

    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval-seconds", type=int, default=600)
    parser.add_argument("--force-resample", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg_root = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    run_dir = args.run_dir.resolve() if args.run_dir is not None else _resolve(cfg_root, cfg["run"]["output_dir"])
    if run_dir is None:
        raise ValueError("Unable to resolve run_dir")
    run_dir.mkdir(parents=True, exist_ok=True)

    qm_cfg = cfg.get("quality_monitor", {})

    real_manifest = (
        args.real_manifest.resolve()
        if args.real_manifest is not None
        else _resolve(
            cfg_root,
            qm_cfg.get("real_manifest", cfg.get("data", {}).get("manifest_val", None)),
        )
    )
    if real_manifest is None or not real_manifest.exists():
        raise FileNotFoundError(f"real manifest not found: {real_manifest}")

    out_jsonl = (
        args.out_jsonl.resolve()
        if args.out_jsonl is not None
        else _resolve(cfg_root, qm_cfg.get("output_jsonl", None))
    )
    if out_jsonl is None:
        out_jsonl = run_dir / "quality_metrics.jsonl"

    num_samples = int(args.num_samples if args.num_samples is not None else qm_cfg.get("num_samples", 64))
    seed_start = int(args.seed_start if args.seed_start is not None else qm_cfg.get("seed_start", 1000))
    max_videos = int(args.max_videos if args.max_videos is not None else qm_cfg.get("max_videos", num_samples))

    sampling_cfg = cfg.get("sampling", {})
    steps_per_band = (
        int(args.steps_per_band) if args.steps_per_band is not None else int(sampling_cfg.get("steps_per_band", 24))
    )
    class_label = args.class_label if args.class_label is not None else sampling_cfg.get("class_label", None)
    fps = int(args.fps) if args.fps is not None else int(sampling_cfg.get("fps", cfg.get("data", {}).get("fps", 8)))
    save_scale = (
        float(args.save_scale) if args.save_scale is not None else float(sampling_cfg.get("save_scale", 1.0))
    )

    feature_frames = int(args.feature_frames if args.feature_frames is not None else qm_cfg.get("feature_frames", 16))
    feature_size = int(args.feature_size if args.feature_size is not None else qm_cfg.get("feature_size", 112))
    feature_batch_size = int(
        args.feature_batch_size if args.feature_batch_size is not None else qm_cfg.get("feature_batch_size", 8)
    )
    if args.feature_pretrained is None:
        feature_pretrained = bool(qm_cfg.get("feature_pretrained", True))
    else:
        feature_pretrained = bool(args.feature_pretrained)

    run_label = args.run_label if args.run_label is not None else run_dir.name

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    script_dir = Path(__file__).resolve().parent
    sample_script = script_dir / "sample_stageA_batch.py"
    quality_script = script_dir / "eval_stageA_quality.py"

    while True:
        processed = _load_processed(out_jsonl)
        checkpoints = _iter_epoch_checkpoints(ckpt_dir)
        pending = [p for p in checkpoints if p.name not in processed]

        if pending:
            print(f"Found {len(pending)} new checkpoints to evaluate.")
        else:
            print("No new checkpoints to evaluate.")

        for ckpt in pending:
            epoch = _epoch_from_checkpoint_name(ckpt.name)
            if epoch is None:
                continue

            sample_dir = run_dir / "quality_samples" / ckpt.stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            if args.force_resample or _count_mp4(sample_dir) < num_samples:
                cmd = [
                    sys.executable,
                    str(sample_script),
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(ckpt),
                    "--out-dir",
                    str(sample_dir),
                    "--num-samples",
                    str(num_samples),
                    "--seed-start",
                    str(seed_start),
                    "--steps-per-band",
                    str(steps_per_band),
                    "--fps",
                    str(fps),
                    "--save-scale",
                    str(save_scale),
                ]
                if class_label is not None:
                    cmd.extend(["--class-label", str(int(class_label))])
                _run_cmd(cmd)

            quality_tmp = sample_dir / "_quality_tmp.json"
            cmd = [
                sys.executable,
                str(quality_script),
                "--generated-dir",
                str(sample_dir),
                "--real-manifest",
                str(real_manifest),
                "--max-videos",
                str(max_videos),
                "--feature-frames",
                str(feature_frames),
                "--feature-size",
                str(feature_size),
                "--feature-batch-size",
                str(feature_batch_size),
                "--out-json",
                str(quality_tmp),
            ]
            if feature_pretrained:
                cmd.append("--feature-pretrained")
            _run_cmd(cmd)

            metrics = json.loads(quality_tmp.read_text(encoding="utf-8"))
            try:
                quality_tmp.unlink()
            except FileNotFoundError:
                pass

            record: dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "run_label": str(run_label),
                "checkpoint_file": ckpt.name,
                "checkpoint_path": str(ckpt.as_posix()),
                "epoch": int(epoch),
                "num_requested_samples": int(num_samples),
                "sample_dir": str(sample_dir.as_posix()),
            }
            record.update(metrics)
            _append_jsonl(out_jsonl, record)
            print(f"Appended quality metrics for {ckpt.name} -> {out_jsonl}")

        if not args.watch:
            break
        time.sleep(max(1, int(args.watch_interval_seconds)))


if __name__ == "__main__":
    main()
