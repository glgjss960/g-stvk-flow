from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from g_stvk_flow.gstvk import BandMeta, Haar3DTransform
from g_stvk_flow.utils.io import save_video_tensor


@dataclass
class TraceSavedItem:
    tau: float
    tag: str
    video_path: str
    mid_frame_path: str


def _tau_of(pt: object) -> float:
    return float(getattr(pt, "tau"))


def _tag_of(pt: object) -> str:
    return str(getattr(pt, "tag"))


def _video_of(pt: object) -> torch.Tensor:
    return getattr(pt, "video")


def build_trace_taus(step_percent: float, anchor: float | None = None, dense_window: float = 0.05) -> list[float]:
    p = float(step_percent)
    if p <= 0.0:
        p = 10.0

    step = p / 100.0
    n = max(1, int(round(1.0 / step)))
    taus = {round(i / n, 6) for i in range(n + 1)}

    if anchor is not None:
        a = float(anchor)
        w = max(0.0, float(dense_window))
        extra = [a - w, a, a + w]
        for t in extra:
            taus.add(round(min(1.0, max(0.0, t)), 6))

    return sorted(float(t) for t in taus)


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


def _to_uint8_frames(video: torch.Tensor, scale: float = 1.0) -> np.ndarray:
    # video: [C,T,H,W] in [-1,1]
    v = video.detach().cpu().to(torch.float32)
    v = _resize_video_spatial(v, scale=float(scale))
    v = v.clamp(-1.0, 1.0)
    v = ((v + 1.0) * 127.5).round().to(torch.uint8)
    v = v.permute(1, 2, 3, 0).contiguous()  # [T,H,W,C]
    return v.numpy()


def _sanitize_tag(tag: str) -> str:
    keep = []
    for ch in tag:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def save_trace_videos(
    trace_points: Sequence[object],
    out_dir: Path,
    fps: int,
    scale: float = 1.0,
) -> list[TraceSavedItem]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vids_dir = out_dir / "videos"
    mids_dir = out_dir / "mid_frames"
    vids_dir.mkdir(parents=True, exist_ok=True)
    mids_dir.mkdir(parents=True, exist_ok=True)

    saved: list[TraceSavedItem] = []
    for i, pt in enumerate(trace_points):
        video_t = _video_of(pt)
        video = video_t[0] if video_t.ndim == 5 else video_t
        tau_val = _tau_of(pt)
        tau_str = f"{tau_val:.3f}".replace(".", "p")
        tag = _sanitize_tag(_tag_of(pt))
        stem = f"{i:03d}_tau_{tau_str}_{tag}"

        video_path = vids_dir / f"{stem}.mp4"
        save_video_tensor(video, video_path, fps=fps, scale=scale)

        frames = _to_uint8_frames(video, scale=scale)
        mid_idx = int(frames.shape[0] // 2)
        mid_img = Image.fromarray(frames[mid_idx])
        mid_path = mids_dir / f"{stem}_mid.png"
        mid_img.save(mid_path)

        saved.append(
            TraceSavedItem(
                tau=float(tau_val),
                tag=str(_tag_of(pt)),
                video_path=str(video_path),
                mid_frame_path=str(mid_path),
            )
        )

    return saved


def make_low_high_masks(meta: BandMeta, kt_threshold: float, ks_min_replace: float) -> tuple[torch.Tensor, torch.Tensor]:
    low = (meta.kt < float(kt_threshold)) | (meta.ks < float(ks_min_replace))
    high = ~low
    return low, high


def band_vector(transform: Haar3DTransform, video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # video: [C,T,H,W]
    coeffs, _ = transform.forward(video.unsqueeze(0))
    flat = transform.flatten(coeffs)
    parts: List[torch.Tensor] = []
    for i, band in enumerate(flat):
        if bool(mask[i].item()):
            parts.append(band.reshape(-1).detach().cpu())
    if not parts:
        return torch.zeros(1, dtype=video.dtype)
    return torch.cat(parts, dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    if a.shape != b.shape:
        n = min(a.numel(), b.numel())
        a = a[:n]
        b = b[:n]
    den = a.norm() * b.norm()
    if float(den.item()) < 1e-10:
        return 0.0
    return float((a @ b / den).item())


def save_anchor_compare_panel(pre_video: torch.Tensor, post_video: torch.Tensor, out_png: Path, scale: float = 1.0) -> None:
    # pre/post: [C,T,H,W] in [-1,1]
    pre = _to_uint8_frames(pre_video, scale=scale)
    post = _to_uint8_frames(post_video, scale=scale)
    t = pre.shape[0]
    idxs = sorted(set([0, t // 2, max(0, t - 1)]))

    rows: List[np.ndarray] = []
    for idx in idxs:
        a = pre[idx]
        b = post[idx]
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
        if diff.max() > 0:
            diff = (diff / diff.max()) * 255.0
        diff_rgb = np.stack([diff, np.zeros_like(diff), 255.0 - diff], axis=2).astype(np.uint8)
        row = np.concatenate([a, b, diff_rgb], axis=1)
        rows.append(row)

    canvas = np.concatenate(rows, axis=0)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    w = pre.shape[2]
    h = pre.shape[1]

    draw.text((8, 8), "pre_edit", fill=(255, 255, 255))
    draw.text((w + 8, 8), "post_edit", fill=(255, 255, 255))
    draw.text((2 * w + 8, 8), "abs_diff", fill=(255, 255, 255))

    for i, idx in enumerate(idxs):
        y = i * h + 8
        draw.text((8, y), f"t={idx}", fill=(255, 255, 255))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def save_cosine_curve_png(
    points: Sequence[dict],
    series: Dict[str, str],
    out_png: Path,
    title: str,
    y_min: float = -1.0,
    y_max: float = 1.0,
) -> None:
    width = 1100
    height = 640
    margin_l = 80
    margin_r = 40
    margin_t = 60
    margin_b = 80

    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    x0 = margin_l
    y0 = margin_t
    x1 = width - margin_r
    y1 = height - margin_b

    draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=2)

    for i in range(1, 10):
        x = int(x0 + (x1 - x0) * i / 10.0)
        draw.line([x, y0, x, y1], fill=(50, 50, 50), width=1)
    for i in range(1, 8):
        y = int(y0 + (y1 - y0) * i / 8.0)
        draw.line([x0, y, x1, y], fill=(50, 50, 50), width=1)

    def map_xy(tau: float, v: float) -> tuple[int, int]:
        xx = x0 + (x1 - x0) * float(tau)
        vv = (float(v) - y_min) / max(1e-8, (y_max - y_min))
        vv = min(1.0, max(0.0, vv))
        yy = y1 - (y1 - y0) * vv
        return int(xx), int(yy)

    palette = [
        (255, 99, 71),
        (135, 206, 250),
        (152, 251, 152),
        (255, 215, 0),
        (216, 191, 216),
        (255, 160, 122),
    ]

    for idx, (label, key) in enumerate(series.items()):
        color = palette[idx % len(palette)]
        pts = []
        for item in points:
            tau = float(item["tau"])
            val = float(item.get(key, 0.0))
            pts.append(map_xy(tau, val))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
        elif len(pts) == 1:
            x, y = pts[0]
            r = 3
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

        lx = x0 + 12
        ly = y0 + 12 + idx * 22
        draw.rectangle([lx, ly + 4, lx + 16, ly + 14], fill=color)
        draw.text((lx + 24, ly), label, fill=(230, 230, 230))

    draw.text((x0, 18), title, fill=(240, 240, 240))
    draw.text((x0, height - 40), "tau", fill=(210, 210, 210))
    draw.text((8, y0), f"{y_max:.1f}", fill=(210, 210, 210))
    draw.text((8, y1 - 12), f"{y_min:.1f}", fill=(210, 210, 210))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def sort_trace_points(points: Iterable[object]) -> list[object]:
    return sorted(list(points), key=lambda x: (_tau_of(x), _tag_of(x)))


