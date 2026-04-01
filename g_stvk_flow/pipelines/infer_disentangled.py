from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from g_stvk_flow.gstvk import Haar3DTransform
from g_stvk_flow.gstvk.anchor_edit import resolve_anchor
from g_stvk_flow.pipelines.common import load_model_schedule_cfg
from g_stvk_flow.pipelines.inference_core import (
    TracePoint,
    sample_video_disentangled,
    sample_video_disentangled_with_trace,
)
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
    save_scale: float,
    anchor_info: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = save_trace_videos(trace_points=trace_points, out_dir=out_dir, fps=fps, scale=save_scale)

    anchor_pre = _find_trace(trace_points, "anchor_pre_edit")
    anchor_post = _find_trace(trace_points, "anchor_post_edit")
    if anchor_pre is None or anchor_post is None:
        raise RuntimeError("Missing anchor_pre_edit or anchor_post_edit in trace points")

    anchor_png = out_dir / "anchor_pre_post_compare.png"
    save_anchor_compare_panel(
        pre_video=anchor_pre.video[0].detach().cpu(),
        post_video=anchor_post.video[0].detach().cpu(),
        out_png=anchor_png,
        scale=save_scale,
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
        "save_scale": float(save_scale),
        "anchor_selection": anchor_info,
    }
    (out_dir / "trace_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Disentangled g-STVK inference pipeline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    parser.add_argument("--anchor", type=str, default="0.35", help="Anchor tau in (0,1), or 'auto' for grid search")
    parser.add_argument("--anchor-grid", type=str, default="0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--anchor-low-min", type=float, default=0.45)
    parser.add_argument("--anchor-min-edit-mass", type=float, default=0.15)
    parser.add_argument("--kt-threshold", type=float, default=0.55)
    parser.add_argument("--ks-min-replace", type=float, default=0.15)
    parser.add_argument("--kt-softness", type=float, default=None)
    parser.add_argument("--ks-softness", type=float, default=None)
    parser.add_argument("--path-softness", type=float, default=None)
    parser.add_argument("--content-label", type=int, default=None)
    parser.add_argument("--motion-label", type=int, default=None)
    parser.add_argument("--reference-pt", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-scale", type=float, default=1.0)

    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-percent", type=float, default=10.0)
    parser.add_argument("--trace-dense-window", type=float, default=0.05)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, schedule = load_model_schedule_cfg(args.checkpoint, args.config, device=device)

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

    anchor, anchor_info = resolve_anchor(
        anchor_arg=args.anchor,
        schedule=schedule,
        transform=transform,
        device=device,
        kt_threshold=args.kt_threshold,
        ks_min_replace=args.ks_min_replace,
        kt_softness=kt_softness,
        ks_softness=ks_softness,
        path_softness=path_softness,
        anchor_grid=args.anchor_grid,
        anchor_low_min=args.anchor_low_min,
        anchor_min_edit_mass=args.anchor_min_edit_mass,
    )
    print(f"Using anchor={anchor:.4f} (mode={anchor_info['mode']})")

    if args.trace_dir is None:
        sample = sample_video_disentangled(
            model=model,
            transform=transform,
            schedule=schedule,
            shape=shape,
            steps=steps,
            solver=solver,
            device=device,
            anchor=anchor,
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
            anchor=anchor,
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
            anchor=anchor,
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
            save_scale=args.save_scale,
            anchor_info=anchor_info,
        )

    save_video_tensor(sample[0], args.out, fps=cfg.inference.fps, scale=args.save_scale)
    print(f"Saved disentangled sample to {args.out}")
    if args.trace_dir is not None:
        print(f"Saved trace artifacts to {args.trace_dir}")


if __name__ == "__main__":
    main()
