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
from g_stvk_flow.engine import TracePoint, sample_video_disentangled, sample_video_disentangled_with_trace
from g_stvk_flow.models import STVKFlowModel
from g_stvk_flow.transforms import Haar3DTransform, SAASchedule
from g_stvk_flow.utils import (
    band_vector,
    build_trace_taus,
    cosine,
    load_checkpoint,
    make_low_high_masks,
    save_anchor_compare_panel,
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


def _find_trace(points: list[TracePoint], tag: str) -> TracePoint | None:
    for pt in points:
        if pt.tag == tag:
            return pt
    return None


def _save_trace_artifacts(
    trace_points: list[TracePoint],
    out_dir: Path,
    fps: int,
    transform: Haar3DTransform,
    device: torch.device,
    kt_threshold: float,
    ks_min_replace: float,
    edit_weights: torch.Tensor,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = save_trace_videos(trace_points=trace_points, out_dir=out_dir, fps=fps)

    anchor_pre = _find_trace(trace_points, "anchor_pre_edit")
    anchor_post = _find_trace(trace_points, "anchor_post_edit")
    if anchor_pre is None or anchor_post is None:
        raise RuntimeError("Missing anchor_pre_edit or anchor_post_edit in trace points")

    anchor_png = out_dir / "anchor_pre_post_compare.png"
    save_anchor_compare_panel(
        pre_video=anchor_pre.video[0].detach().cpu(),
        post_video=anchor_post.video[0].detach().cpu(),
        out_png=anchor_png,
    )

    meta = transform.band_meta(device=device)
    low_mask, high_mask = make_low_high_masks(meta=meta, kt_threshold=kt_threshold, ks_min_replace=ks_min_replace)

    low_pre = band_vector(transform=transform, video=anchor_pre.video[0].detach().cpu(), mask=low_mask.cpu())
    high_pre = band_vector(transform=transform, video=anchor_pre.video[0].detach().cpu(), mask=high_mask.cpu())
    low_post = band_vector(transform=transform, video=anchor_post.video[0].detach().cpu(), mask=low_mask.cpu())
    high_post = band_vector(transform=transform, video=anchor_post.video[0].detach().cpu(), mask=high_mask.cpu())

    curve_points = []
    for pt in trace_points:
        vid = pt.video[0].detach().cpu()
        low_vec = band_vector(transform=transform, video=vid, mask=low_mask.cpu())
        high_vec = band_vector(transform=transform, video=vid, mask=high_mask.cpu())
        curve_points.append(
            {
                "tau": float(pt.tau),
                "tag": str(pt.tag),
                "low_cos_to_anchor_pre": float(cosine(low_vec, low_pre)),
                "high_cos_to_anchor_pre": float(cosine(high_vec, high_pre)),
                "low_cos_to_anchor_post": float(cosine(low_vec, low_post)),
                "high_cos_to_anchor_post": float(cosine(high_vec, high_post)),
            }
        )

    curve_png = out_dir / "cos_curve_anchor_pre_post.png"
    save_cosine_curve_png(
        points=curve_points,
        series={
            "low->pre": "low_cos_to_anchor_pre",
            "high->pre": "high_cos_to_anchor_pre",
            "low->post": "low_cos_to_anchor_post",
            "high->post": "high_cos_to_anchor_post",
        },
        out_png=curve_png,
        title="Disentangled Trace: low/high cosine to anchor pre/post",
    )

    pre_post_l1 = float((anchor_pre.video[0].detach().cpu() - anchor_post.video[0].detach().cpu()).abs().mean().item())

    payload = {
        "trace_points": [
            {
                "tau": float(pt.tau),
                "tag": str(pt.tag),
            }
            for pt in trace_points
        ],
        "saved_items": [x.__dict__ for x in saved],
        "anchor_compare_png": str(anchor_png),
        "cos_curve": curve_points,
        "curve_png": str(curve_png),
        "anchor_edit_l1": pre_post_l1,
        "partition": {
            "kt_threshold": float(kt_threshold),
            "ks_min_replace": float(ks_min_replace),
        },
        "edit_weights": [float(x) for x in edit_weights.detach().cpu().tolist()],
    }
    (out_dir / "trace_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Disentangled G-STVK-Flow inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    parser.add_argument("--anchor", type=float, default=0.35)
    parser.add_argument("--kt-threshold", type=float, default=0.55)
    parser.add_argument("--ks-min-replace", type=float, default=0.15)
    parser.add_argument("--kt-softness", type=float, default=None)
    parser.add_argument("--ks-softness", type=float, default=None)
    parser.add_argument("--path-softness", type=float, default=None)
    parser.add_argument("--content-label", type=int, default=None)
    parser.add_argument("--motion-label", type=int, default=None)
    parser.add_argument("--reference-pt", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-percent", type=float, default=10.0, help="Save trajectory near every N percent in tau")
    parser.add_argument("--trace-dense-window", type=float, default=0.05, help="Add anchor+-window trace taus")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, schedule = _load_model(args.checkpoint, args.config, device)

    steps = args.steps if args.steps is not None else cfg.inference.steps
    solver = args.solver if args.solver is not None else cfg.inference.solver

    kt_softness = args.kt_softness if args.kt_softness is not None else cfg.inference.kt_softness
    ks_softness = args.ks_softness if args.ks_softness is not None else cfg.inference.ks_softness
    path_softness = args.path_softness if args.path_softness is not None else cfg.inference.path_softness

    shape = (
        1,
        cfg.data.in_channels,
        cfg.data.frames,
        cfg.data.image_size,
        cfg.data.image_size,
    )

    reference_video = None
    if args.reference_pt is not None:
        payload = load_checkpoint(args.reference_pt, map_location="cpu")
        ref = payload["video"].float() if isinstance(payload, dict) and "video" in payload else payload.float()
        if ref.ndim == 4:
            ref = ref.unsqueeze(0)
        reference_video = ref

    transform = Haar3DTransform(levels=cfg.transform.levels)

    if args.trace_dir is None:
        sample = sample_video_disentangled(
            model=model,
            transform=transform,
            schedule=schedule,
            shape=shape,
            steps=steps,
            solver=solver,
            device=device,
            anchor=args.anchor,
            kt_threshold=args.kt_threshold,
            ks_min_replace=args.ks_min_replace,
            kt_softness=kt_softness,
            ks_softness=ks_softness,
            path_softness=path_softness,
            class_label_content=args.content_label,
            class_label_motion=args.motion_label,
            reference_video=reference_video,
            seed=args.seed,
        )
    else:
        trace_taus = build_trace_taus(
            step_percent=args.trace_percent,
            anchor=args.anchor,
            dense_window=args.trace_dense_window,
        )
        sample, traces, edit_weights, _meta = sample_video_disentangled_with_trace(
            model=model,
            transform=transform,
            schedule=schedule,
            shape=shape,
            steps=steps,
            solver=solver,
            device=device,
            anchor=args.anchor,
            kt_threshold=args.kt_threshold,
            ks_min_replace=args.ks_min_replace,
            kt_softness=kt_softness,
            ks_softness=ks_softness,
            path_softness=path_softness,
            class_label_content=args.content_label,
            class_label_motion=args.motion_label,
            reference_video=reference_video,
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
            kt_threshold=args.kt_threshold,
            ks_min_replace=args.ks_min_replace,
            edit_weights=edit_weights,
        )

    save_video_tensor(sample[0], args.out, fps=cfg.inference.fps)
    print(f"Saved disentangled sample to {args.out}")
    if args.trace_dir is not None:
        print(f"Saved trace artifacts to {args.trace_dir}")


if __name__ == "__main__":
    main()

