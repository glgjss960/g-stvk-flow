from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


def _dtype_from_string(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    n = str(name).strip().lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _decode_cached_video(video: torch.Tensor, video_range: object | None) -> torch.Tensor:
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Expected tensor for video payload, got {type(video)}")

    if video.dtype == torch.uint8:
        # Stored as uint8 [0,255], decode back to float [-1,1].
        return video.to(torch.float32) / 127.5 - 1.0

    out = video.to(torch.float32)
    if isinstance(video_range, str) and video_range.strip().lower() in {"float_0_1", "0_1"}:
        out = out * 2.0 - 1.0
    return out


def _torch_load_compat(path: str | Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CachedVideoDataset(Dataset):
    def __init__(self, manifest_path: str | Path, output_dtype: str | torch.dtype = "float32") -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.output_dtype = _dtype_from_string(output_dtype)

        lines = [ln.strip() for ln in self.manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.items: List[Dict[str, object]] = [json.loads(ln) for ln in lines]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.items[idx]
        payload = _torch_load_compat(rec["tensor_path"])

        if isinstance(payload, dict):
            video = _decode_cached_video(payload["video"], payload.get("video_range", None))
            label = int(payload.get("label", rec.get("label", 0)))
        else:
            video = _decode_cached_video(payload, None)
            label = int(rec.get("label", 0))

        if video.dtype != self.output_dtype:
            video = video.to(dtype=self.output_dtype)

        return {
            "video": video,  # [C,T,H,W], in [-1,1], dtype controlled by output_dtype
            "label": torch.tensor(label, dtype=torch.long),
        }
