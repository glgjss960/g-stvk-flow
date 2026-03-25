from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torchvision.io import write_video
from torchvision.utils import save_image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _torch_load_compat(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    weights_only: bool | None = None,
) -> Any:
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(Path(path), **kwargs)
    except TypeError:
        # Older PyTorch versions do not accept the weights_only argument.
        kwargs.pop("weights_only", None)
        return torch.load(Path(path), **kwargs)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    # Training checkpoints may include custom Python objects (for example config dataclasses),
    # so we must opt out of weights_only mode on PyTorch>=2.6 when loading trusted files.
    return _torch_load_compat(path=path, map_location=map_location, weights_only=False)


def _resize_video_spatial(video: torch.Tensor, scale: float) -> torch.Tensor:
    if float(scale) <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if abs(float(scale) - 1.0) < 1e-8:
        return video

    c, t, h, w = video.shape
    new_h = max(1, int(round(h * float(scale))))
    new_w = max(1, int(round(w * float(scale))))

    frames = video.permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]
    if new_h > 1 and new_w > 1:
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    else:
        frames = F.interpolate(frames, size=(new_h, new_w), mode="nearest")
    return frames.permute(1, 0, 2, 3).contiguous()


def save_video_tensor(video: torch.Tensor, path: str | Path, fps: int = 8, scale: float = 1.0) -> None:
    """
    video: [C,T,H,W] in [-1,1]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    video = video.detach().cpu().to(torch.float32)
    video = _resize_video_spatial(video, scale=float(scale))
    video = video.clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).contiguous()  # [T,H,W,C]

    if path.suffix.lower() == ".mp4":
        try:
            write_video(str(path), video, fps=fps)
            return
        except Exception:
            pass

    # Fallback: write frames
    frame_dir = path.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(video.shape[0]):
        frame = video[i].permute(2, 0, 1).float() / 255.0
        save_image(frame, frame_dir / f"frame_{i:04d}.png")
