from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision.io import read_video

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _discover_videos(raw_dir: Path) -> List[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _infer_class_name(raw_dir: Path, video_path: Path) -> str:
    rel = video_path.relative_to(raw_dir)
    if len(rel.parts) <= 1:
        return "default"
    return rel.parts[0]


def _window_starts(num_frames_src: int, frames: int, stride: int, tail_pad_last_window: bool = True) -> List[int]:
    if num_frames_src <= 0:
        raise ValueError("Video has no frames.")
    if frames <= 0:
        raise ValueError("frames must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    if tail_pad_last_window:
        # Fixed-stride slicing with tail-padding support:
        # e.g. num_frames=250, frames=120, stride=120 -> starts=[0,120,240]
        return list(range(0, num_frames_src, stride))

    # Backward-compatible behavior: only full windows, but ensure end coverage.
    if num_frames_src <= frames:
        return [0]

    starts = list(range(0, num_frames_src - frames + 1, stride))
    last = num_frames_src - frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def _resample_video_fps(video: torch.Tensor, src_fps: float | None, target_fps: float | None) -> torch.Tensor:
    # video: [T,H,W,C]
    if target_fps is None:
        return video
    tfps = float(target_fps)
    if tfps <= 0.0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    if src_fps is None:
        return video

    sfps = float(src_fps)
    if sfps <= 0.0 or video.shape[0] <= 1:
        return video
    if abs(sfps - tfps) < 1e-6:
        return video

    duration = (float(video.shape[0]) - 1.0) / sfps
    new_t = max(1, int(round(duration * tfps)) + 1)
    indices = torch.linspace(0, video.shape[0] - 1, steps=new_t)
    indices = indices.round().to(torch.long).clamp_(0, video.shape[0] - 1)
    return video.index_select(0, indices)


def _to_clip_tensor(
    video: torch.Tensor,
    start: int,
    frames: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    # video: [T,H,W,C], uint8
    end = min(video.shape[0], start + frames)
    clip = video[start:end]

    if clip.shape[0] < frames:
        pad_count = frames - clip.shape[0]
        tail = clip[-1:].repeat(pad_count, 1, 1, 1)
        clip = torch.cat([clip, tail], dim=0)

    clip = clip.permute(0, 3, 1, 2).float() / 255.0  # [T,C,H,W]
    clip = F.interpolate(clip, size=(image_height, image_width), mode="bilinear", align_corners=False)
    clip = clip.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
    clip = clip * 2.0 - 1.0
    return clip


def preprocess_video_folder(
    raw_dir: str | Path,
    out_dir: str | Path,
    frames: int,
    image_size: int,
    train_ratio: float = 0.9,
    stride: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    target_fps: float | None = None,
    tail_pad_last_window: bool = True,
) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    image_height = int(image_height if image_height is not None else image_size)
    image_width = int(image_width if image_width is not None else image_size)
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"image_height/image_width must be > 0, got {(image_height, image_width)}")

    stride = int(stride or frames)
    if stride <= 0:
        raise ValueError("stride must be > 0")

    videos = _discover_videos(raw_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {raw_dir}")

    class_names = sorted({_infer_class_name(raw_dir, p) for p in videos})
    class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(class_names)}

    entries: List[Dict[str, object]] = []
    source_ids: List[str] = []

    clip_index = 0
    observed_src_fps: List[float] = []

    for _, video_path in enumerate(videos):
        class_name = _infer_class_name(raw_dir, video_path)
        label = class_to_idx[class_name]

        video, _, info = read_video(str(video_path), pts_unit="sec")  # [T,H,W,C], uint8
        src_fps_raw = info.get("video_fps", None)
        src_fps = float(src_fps_raw) if src_fps_raw is not None else None
        if src_fps is not None:
            observed_src_fps.append(src_fps)

        video = _resample_video_fps(video=video, src_fps=src_fps, target_fps=target_fps)
        starts = _window_starts(
            video.shape[0],
            frames=frames,
            stride=stride,
            tail_pad_last_window=bool(tail_pad_last_window),
        )

        source_key = str(video_path.relative_to(raw_dir).as_posix())
        source_ids.append(source_key)

        for start in starts:
            clip = _to_clip_tensor(
                video=video,
                start=start,
                frames=frames,
                image_height=image_height,
                image_width=image_width,
            )
            tensor_path = clips_dir / f"clip_{clip_index:08d}.pt"
            clip_index += 1

            torch.save(
                {
                    "video": clip,
                    "label": label,
                    "class_name": class_name,
                    "source": source_key,
                    "start": int(start),
                    "frames": int(frames),
                    "fps": float(target_fps) if target_fps is not None else (float(src_fps) if src_fps is not None else None),
                },
                tensor_path,
            )

            entries.append(
                {
                    "tensor_path": str(tensor_path.as_posix()),
                    "label": label,
                    "class_name": class_name,
                    "source": source_key,
                    "start": int(start),
                }
            )

    unique_sources = sorted(set(source_ids))
    split_src_idx = max(1, int(len(unique_sources) * train_ratio))
    train_sources = set(unique_sources[:split_src_idx])
    val_sources = set(unique_sources[split_src_idx:])
    if not val_sources:
        val_sources = {unique_sources[-1]}

    train_entries = [e for e in entries if str(e["source"]) in train_sources]
    val_entries = [e for e in entries if str(e["source"]) in val_sources]
    if not train_entries:
        train_entries = entries[:1]
    if not val_entries:
        val_entries = entries[-1:]

    (out_dir / "train_manifest.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=True) for e in train_entries) + "\n",
        encoding="utf-8",
    )
    (out_dir / "val_manifest.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=True) for e in val_entries) + "\n",
        encoding="utf-8",
    )
    (out_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    stats = {
        "num_sources": len(unique_sources),
        "num_clips": len(entries),
        "num_train_clips": len(train_entries),
        "num_val_clips": len(val_entries),
        "frames": int(frames),
        "image_size": int(image_size),
        "image_height": int(image_height),
        "image_width": int(image_width),
        "stride": int(stride),
        "target_fps": float(target_fps) if target_fps is not None else None,
        "tail_pad_last_window": bool(tail_pad_last_window),
        "observed_source_fps": sorted({round(v, 6) for v in observed_src_fps}),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Preprocessed {len(entries)} clips from {len(unique_sources)} videos into {out_dir}")
    print(f"Classes: {class_to_idx}")
    print(f"Split by source videos: train={len(train_sources)}, val={len(val_sources)}")
