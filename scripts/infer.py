from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.config import load_config
from g_stvk_flow.engine import TracePoint, sample_video, sample_video_with_trace
from g_stvk_flow.models import STVKFlowModel
from g_stvk_flow.transforms import Haar3DTransform, SAASchedule
from g_stvk_flow.utils import (
    band_vector,
    build_trace_taus,
    cosine,
    load_checkpoint,
    make_low_high_masks,
    save_cosine_curve_png,
    save_trace_videos,
    save_video_tensor,
    sort_trace_points,
)


def _build_schedule(cfg: object, device: torch.device) -> SAASchedule:
    return SAASchedule(
        num_knots=cfg.flow.num_knots,
        delta_min=cfg.flow.delta_min,
        delta_max=cfg.flow.delta_max,
        radius_min=cfg.flow.radius_min,
        radius_max=cfg.flow.radius_max,
        anisotropy_min=cfg.flow.anisotropy_min,
        derivative_eps=cfg.flow.derivative_eps,
        delta_hidden_dim=cfg.flow.delta_hidden_dim,
        spread_temperature=cfg.flow.spread_temperature,
        reg_grid_size=cfg.flow.reg_grid_size,
        integration_grid_size=getattr(cfg.flow, 'integration_grid_size', 129),
        rate_floor=getattr(cfg.flow, 'rate_floor', 1e-4),
        lambda_replace_thr=getattr(cfg.flow, 'lambda_replace_thr', 0.55),
        tail_start=getattr(cfg.flow, 'tail_start', 0.85),
    ).to(device)


def _load_model(checkpoint: Path, cfg_path: Path, device: torch.device) -> tuple[STVKFlowModel, object, SAASchedule]:
    cfg = load_config(cfg_path)
    model = STVKFlowModel(
        in_channels=cfg.data.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        cond_dim=cfg.model.cond_dim,
        phase_dim=cfg.model.phase_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        backbone=getattr(cfg.model, 'backbone', 'opensora_dit'),
        hidden_size=getattr(cfg.model, 'hidden_size', 512),
        depth=getattr(cfg.model, 'depth', 8),
        num_heads=getattr(cfg.model, 'num_heads', 8),
        mlp_ratio=getattr(cfg.model, 'mlp_ratio', 4.0),
        patch_size_t=getattr(cfg.model, 'patch_size_t', 1),
        patch_size_h=getattr(cfg.model, 'patch_size_h', 2),
        patch_size_w=getattr(cfg.model, 'patch_size_w', 2),
    ).to(device)

    schedule = _build_schedule(cfg, device)

    ckpt = load_checkpoint(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    if not (isinstance(ckpt, dict) and "schedule" in ckpt):
        raise KeyError(
            "Checkpoint does not contain 'schedule'. "
            "This usually indicates a mismatch between training and current G-STVK-Flow inference code."
        )
    schedule.load_state_dict(ckpt["schedule"])

    model.eval()
    schedule.eval()
    return model, cfg, schedule


def _dedup_trace(points: list[TracePoint]) -> list[TracePoint]:
    by_key: dict[tuple[int, str], TracePoint] = {}
    for pt in points:
        key = (int(round(float(pt.tau) * 10000)), str(pt.tag))
        by_key[key] = pt
    return sort_trace_points(by_key.values())


def _save_trace_artifacts(
    trace_points: list[TracePoint],
    out_dir: Path,
    fps: int,
    transform: Haar3DTransform,
    device: torch.device,
    kt_threshold: float,
    ks_min_replace: float,
    save_scale: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = save_trace_videos(trace_points=trace_points, out_dir=out_dir, fps=fps, scale=save_scale)

    meta = transform.band_meta(device=device)
    low_mask, high_mask = make_low_high_masks(meta=meta, kt_threshold=kt_threshold, ks_min_replace=ks_min_replace)

    ref_video = trace_points[-1].video[0].detach().cpu()
    low_ref = band_vector(transform=transform, video=ref_video, mask=low_mask.cpu())
    high_ref = band_vector(transform=transform, video=ref_video, mask=high_mask.cpu())

    curve_points = []
    for pt in trace_points:
        vid = pt.video[0].detach().cpu()
        low_vec = band_vector(transform=transform, video=vid, mask=low_mask.cpu())
        high_vec = band_vector(transform=transform, video=vid, mask=high_mask.cpu())
        curve_points.append(
            {
                "tau": float(pt.tau),
                "tag": str(pt.tag),
                "low_cos_to_final": float(cosine(low_vec, low_ref)),
                "high_cos_to_final": float(cosine(high_vec, high_ref)),
            }
        )

    curve_png = out_dir / "cos_curve_to_final.png"
    save_cosine_curve_png(
        points=curve_points,
        series={
            "low_cos_to_final": "low_cos_to_final",
            "high_cos_to_final": "high_cos_to_final",
        },
        out_png=curve_png,
        title="Standard Inference: low/high cosine to final sample",
    )

    payload = {
        "trace_points": [
            {
                "tau": float(pt.tau),
                "tag": str(pt.tag),
            }
            for pt in trace_points
        ],
        "saved_items": [x.__dict__ for x in saved],
        "cos_curve": curve_points,
        "curve_png": str(curve_png),
        "partition": {
            "kt_threshold": float(kt_threshold),
            "ks_min_replace": float(ks_min_replace),
        },
        "save_scale": float(save_scale),
    }
    (out_dir / "trace_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard G-STVK-Flow inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-scale", type=float, default=1.0, help="Output upsample factor for visualization/export")

    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-percent", type=float, default=10.0, help="Save trajectory near every N percent in tau")
    parser.add_argument("--trace-kt-threshold", type=float, default=0.55)
    parser.add_argument("--trace-ks-min-replace", type=float, default=0.15)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, schedule = _load_model(args.checkpoint, args.config, device)

    steps = args.steps if args.steps is not None else cfg.inference.steps
    solver = args.solver if args.solver is not None else cfg.inference.solver

    shape = (
        1,
        cfg.data.in_channels,
        cfg.data.frames,
        cfg.data.image_size,
        cfg.data.image_size,
    )

    transform = Haar3DTransform(levels=cfg.transform.levels)

    if args.trace_dir is None:
        sample = sample_video(
            model=model,
            transform=transform,
            schedule=schedule,
            shape=shape,
            steps=steps,
            solver=solver,
            device=device,
            class_label=args.class_label,
            seed=args.seed,
        )
    else:
        trace_taus = build_trace_taus(step_percent=args.trace_percent)
        sample, traces = sample_video_with_trace(
            model=model,
            transform=transform,
            schedule=schedule,
            shape=shape,
            steps=steps,
            solver=solver,
            device=device,
            class_label=args.class_label,
            seed=args.seed,
            trace_taus=trace_taus,
        )
        traces.append(TracePoint(tau=1.0, tag="final", video=sample.detach().clone()))
        trace_points = _dedup_trace(traces)
        _save_trace_artifacts(
            trace_points=trace_points,
            out_dir=args.trace_dir,
            fps=cfg.inference.fps,
            transform=transform,
            device=device,
            kt_threshold=args.trace_kt_threshold,
            ks_min_replace=args.trace_ks_min_replace,
            save_scale=args.save_scale,
        )

    save_video_tensor(sample[0], args.out, fps=cfg.inference.fps, scale=args.save_scale)
    print(f"Saved sample to {args.out}")
    if args.trace_dir is not None:
        print(f"Saved trace artifacts to {args.trace_dir}")


if __name__ == "__main__":
    main()




