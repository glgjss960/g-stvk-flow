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



def _window_starts(num_frames_src: int, frames: int, stride: int) -> List[int]:
    if num_frames_src <= 0:
        raise ValueError("Video has no frames.")
    if num_frames_src <= frames:
        return [0]

    starts = list(range(0, num_frames_src - frames + 1, stride))
    last = num_frames_src - frames
    if starts[-1] != last:
        starts.append(last)
    return starts



def _to_clip_tensor(video: torch.Tensor, start: int, frames: int, image_size: int) -> torch.Tensor:
    # video: [T,H,W,C], uint8
    end = min(video.shape[0], start + frames)
    clip = video[start:end]

    if clip.shape[0] < frames:
        pad_count = frames - clip.shape[0]
        tail = clip[-1:].repeat(pad_count, 1, 1, 1)
        clip = torch.cat([clip, tail], dim=0)

    clip = clip.permute(0, 3, 1, 2).float() / 255.0  # [T,C,H,W]
    clip = F.interpolate(clip, size=(image_size, image_size), mode="bilinear", align_corners=False)
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
) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

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
    for src_id, video_path in enumerate(videos):
        class_name = _infer_class_name(raw_dir, video_path)
        label = class_to_idx[class_name]

        video, _, _ = read_video(str(video_path), pts_unit="sec")  # [T,H,W,C], uint8
        starts = _window_starts(video.shape[0], frames=frames, stride=stride)

        source_key = str(video_path.relative_to(raw_dir).as_posix())
        source_ids.append(source_key)

        for start in starts:
            clip = _to_clip_tensor(video=video, start=start, frames=frames, image_size=image_size)
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
        "stride": int(stride),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"Preprocessed {len(entries)} clips from {len(unique_sources)} videos into {out_dir}")
    print(f"Classes: {class_to_idx}")
    print(f"Split by source videos: train={len(train_sources)}, val={len(val_sources)}")
