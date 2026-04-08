from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from torchvision.models.video import r3d_18


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def _tensor_to_float_video_chw(video: torch.Tensor, *, video_range: str | None = None) -> torch.Tensor:
    # input could be:
    # - [T,H,W,C] uint8
    # - [C,T,H,W] float[-1,1] / float[0,1] / uint8[0,255]
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got {tuple(video.shape)}")

    vr = str(video_range).strip().lower() if video_range is not None else ""

    # [T,H,W,C]
    if video.shape[-1] == 3:
        v = video.permute(3, 0, 1, 2).contiguous().to(torch.float32)
        if video.dtype == torch.uint8 or vr in {"uint8_0_255", "0_255"}:
            v = v / 255.0
            v = v * 2.0 - 1.0
            return v
        if vr in {"float_0_1", "0_1"}:
            v = v * 2.0 - 1.0
        return v

    # [C,T,H,W]
    v = video.to(torch.float32)
    if video.dtype == torch.uint8 or vr in {"uint8_0_255", "0_255"}:
        v = v / 127.5 - 1.0
        return v
    if vr in {"float_0_1", "0_1"}:
        v = v * 2.0 - 1.0
    return v


def _load_generated_videos(gen_dir: Path, max_videos: int) -> list[torch.Tensor]:
    paths = sorted(list(gen_dir.glob("*.mp4")))
    if not paths:
        raise FileNotFoundError(f"No .mp4 files found in {gen_dir}")
    videos: list[torch.Tensor] = []
    for p in paths[:max_videos]:
        frames, _, _ = read_video(str(p), pts_unit="sec")  # [T,H,W,C] uint8
        videos.append(_tensor_to_float_video_chw(frames))
    return videos


def _load_real_videos(manifest_path: Path, max_videos: int) -> list[torch.Tensor]:
    rows = _load_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    videos: list[torch.Tensor] = []
    for r in rows[:max_videos]:
        payload = torch.load(r["tensor_path"], map_location="cpu")
        if isinstance(payload, dict):
            video = payload["video"]
            video_range = payload.get("video_range", None)
        else:
            video = payload
            video_range = None
        videos.append(_tensor_to_float_video_chw(video, video_range=video_range))
    return videos


def _resize_video(video: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    # video: [C,T,H,W], float [-1,1]
    c, tv, hv, wv = video.shape
    if tv != t:
        idx = torch.linspace(0, tv - 1, steps=t).round().to(torch.long).clamp(0, tv - 1)
        video = video.index_select(1, idx)
    if hv != h or wv != w:
        frames = video.permute(1, 0, 2, 3).contiguous()
        frames = F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)
        video = frames.permute(1, 0, 2, 3).contiguous()
    return video


def _extract_r3d_features(
    videos: list[torch.Tensor],
    device: torch.device,
    frames: int,
    size: int,
    batch_size: int,
    pretrained: bool,
) -> np.ndarray:
    weights = None
    if pretrained:
        try:
            from torchvision.models.video import R3D_18_Weights

            weights = R3D_18_Weights.KINETICS400_V1
        except Exception:
            weights = None

    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)

    feats: list[np.ndarray] = []
    with torch.no_grad():
        i = 0
        while i < len(videos):
            batch = videos[i : i + batch_size]
            xs = []
            for v in batch:
                vr = _resize_video(v, t=frames, h=size, w=size)
                x = (vr * 0.5 + 0.5).clamp(0.0, 1.0)  # to [0,1]
                xs.append(x)
            x = torch.stack(xs, dim=0).to(device)
            x = (x - mean) / std
            f = model(x)
            feats.append(f.detach().cpu().numpy())
            i += batch_size

    return np.concatenate(feats, axis=0)


def _frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    mu1 = np.mean(feats_a, axis=0)
    mu2 = np.mean(feats_b, axis=0)
    sigma1 = np.cov(feats_a, rowvar=False)
    sigma2 = np.cov(feats_b, rowvar=False)

    diff = mu1 - mu2

    covmean = None
    try:
        from scipy import linalg

        covmean = linalg.sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        vals = np.linalg.eigvals(sigma1 @ sigma2)
        vals = np.real(vals)
        vals = np.clip(vals, a_min=0.0, a_max=None)
        tr_covmean = float(np.sum(np.sqrt(vals)))
        return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def _compute_gen_tlpips(videos: list[torch.Tensor], device: torch.device) -> float | None:
    try:
        import lpips
    except Exception:
        return None

    model = lpips.LPIPS(net="alex").to(device).eval()
    vals: list[float] = []
    with torch.no_grad():
        for v in videos:
            c, t, h, w = v.shape
            for i in range(t - 1):
                a = v[:, i].unsqueeze(0).to(device)
                b = v[:, i + 1].unsqueeze(0).to(device)
                vals.append(float(model(a, b).item()))
    if not vals:
        return None
    return float(np.mean(vals))


def _compute_gen_warping_error(videos: list[torch.Tensor]) -> float | None:
    try:
        import cv2
    except Exception:
        return None

    errs: list[float] = []
    for v in videos:
        # [C,T,H,W] -> [T,H,W,C] uint8
        x = ((v * 0.5 + 0.5).clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        for i in range(x.shape[0] - 1):
            prev = x[i]
            curr = x[i + 1]
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            h, w = prev_gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(prev.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)
            err = np.mean(np.abs(warped - curr.astype(np.float32))) / 255.0
            errs.append(float(err))

    if not errs:
        return None
    return float(np.mean(errs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality metrics for StageA generated videos")
    parser.add_argument("--generated-dir", type=Path, required=True, help="directory containing generated .mp4 videos")
    parser.add_argument("--real-manifest", type=Path, required=True, help="val_manifest.jsonl from preprocessed UCF cache")
    parser.add_argument("--max-videos", type=int, default=64)
    parser.add_argument("--feature-frames", type=int, default=16)
    parser.add_argument("--feature-size", type=int, default=112)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-pretrained", action="store_true", help="use pretrained R3D-18 weights if available")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_videos = _load_generated_videos(args.generated_dir.resolve(), max_videos=int(args.max_videos))
    real_videos = _load_real_videos(args.real_manifest.resolve(), max_videos=int(args.max_videos))

    n = min(len(gen_videos), len(real_videos), int(args.max_videos))
    gen_videos = gen_videos[:n]
    real_videos = real_videos[:n]

    gen_feats = _extract_r3d_features(
        videos=gen_videos,
        device=device,
        frames=int(args.feature_frames),
        size=int(args.feature_size),
        batch_size=int(args.feature_batch_size),
        pretrained=bool(args.feature_pretrained),
    )
    real_feats = _extract_r3d_features(
        videos=real_videos,
        device=device,
        frames=int(args.feature_frames),
        size=int(args.feature_size),
        batch_size=int(args.feature_batch_size),
        pretrained=bool(args.feature_pretrained),
    )

    metrics = {
        "num_videos": int(n),
        "fvd_proxy_r3d18": _frechet_distance(real_feats, gen_feats),
        "gen_tlpips": _compute_gen_tlpips(gen_videos, device=device),
        "gen_warping_error": _compute_gen_warping_error(gen_videos),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
