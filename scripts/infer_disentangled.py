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


def _find_trace(points: list[TracePoint], tag: str) -> TracePoint | None:
    for pt in points:
        if pt.tag == tag:
            return pt
    return None


def _parse_anchor_grid(text: str | None) -> list[float]:
    default = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    if text is None or text.strip() == "":
        return default

    vals: list[float] = []
    for tok in text.split(","):
        s = tok.strip()
        if s == "":
            continue
        try:
            v = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid anchor-grid value: {s}") from exc
        if 0.0 < v < 1.0:
            vals.append(v)

    vals = sorted(set(float(v) for v in vals))
    if not vals:
        raise ValueError("anchor-grid must include at least one value in (0,1)")
    return vals


def _resolve_anchor(
    anchor_arg: str,
    schedule: SAASchedule,
    transform: Haar3DTransform,
    device: torch.device,
    kt_threshold: float,
    ks_min_replace: float,
    kt_softness: float,
    ks_softness: float,
    path_softness: float,
    anchor_grid: str | None,
    anchor_low_min: float,
    anchor_min_edit_mass: float,
) -> tuple[float, dict]:
    mode = anchor_arg.strip().lower()
    if mode != "auto":
        try:
            anchor = float(anchor_arg)
        except ValueError as exc:
            raise ValueError(f"Invalid --anchor value: {anchor_arg}. Use float in (0,1) or 'auto'.") from exc
        if not (0.0 < anchor < 1.0):
            raise ValueError(f"--anchor must be in (0,1), got {anchor}")
        return anchor, {"mode": "fixed", "selected_anchor": float(anchor)}

    with torch.no_grad():
        meta = transform.band_meta(device=device)
        low_mask, high_mask = make_low_high_masks(meta=meta, kt_threshold=kt_threshold, ks_min_replace=ks_min_replace)

        if not bool(high_mask.any()):
            high_mask = torch.ones_like(high_mask, dtype=torch.bool)
        if not bool(low_mask.any()):
            low_mask = torch.ones_like(low_mask, dtype=torch.bool)

        candidates = _parse_anchor_grid(anchor_grid)
        rows: list[dict] = []

        for a in candidates:
            tau = torch.tensor([float(a)], device=device, dtype=meta.ks.dtype)
            lam, _lam_dot, _state = schedule.lambda_and_derivative(tau=tau, ks=meta.ks, kt=meta.kt)
            lam_anchor = lam[0]

            edit_w = schedule.build_edit_weights(
                ks=meta.ks,
                kt=meta.kt,
                tau_anchor=float(a),
                kt_threshold=kt_threshold,
                ks_min_replace=ks_min_replace,
                kt_softness=kt_softness,
                ks_softness=ks_softness,
                path_softness=path_softness,
            )

            low_lock = float(lam_anchor[low_mask].mean().item())
            high_open = float((1.0 - lam_anchor[high_mask]).mean().item())
            edit_mass = float(edit_w[high_mask].mean().item())

            # Higher score means: low band is already stabilized, high band still open and editable.
            score = 0.5 * edit_mass + 0.3 * high_open + 0.2 * low_lock
            feasible = (low_lock >= float(anchor_low_min)) and (edit_mass >= float(anchor_min_edit_mass))

            rows.append(
                {
                    "anchor": float(a),
                    "low_lock": low_lock,
                    "high_open": high_open,
                    "edit_mass": edit_mass,
                    "score": float(score),
                    "feasible": bool(feasible),
                }
            )

        feasible_rows = [r for r in rows if bool(r["feasible"])]
        pick_pool = feasible_rows if feasible_rows else rows
        best = max(pick_pool, key=lambda r: float(r["score"]))

        info = {
            "mode": "auto",
            "selected_anchor": float(best["anchor"]),
            "selected_stats": {
                "low_lock": float(best["low_lock"]),
                "high_open": float(best["high_open"]),
                "edit_mass": float(best["edit_mass"]),
                "score": float(best["score"]),
                "feasible": bool(best["feasible"]),
            },
            "constraints": {
                "anchor_low_min": float(anchor_low_min),
                "anchor_min_edit_mass": float(anchor_min_edit_mass),
            },
            "candidates": rows,
            "has_feasible": bool(len(feasible_rows) > 0),
        }
        return float(best["anchor"]), info


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
    parser = argparse.ArgumentParser(description="Disentangled G-STVK-Flow inference")
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
    parser.add_argument("--save-scale", type=float, default=1.0, help="Output upsample factor for visualization/export")

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

    anchor, anchor_info = _resolve_anchor(
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

