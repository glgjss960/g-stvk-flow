from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.config import load_config
from g_stvk_flow.engine import sample_video_disentangled
from g_stvk_flow.models import STVKFlowModel
from g_stvk_flow.transforms import Haar3DTransform, SAASchedule
from g_stvk_flow.utils import load_checkpoint


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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


def _load_model(checkpoint: Path, cfg: object, device: torch.device) -> tuple[torch.nn.Module, SAASchedule]:
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
        raise KeyError("Checkpoint missing schedule. Use G-STVK-Flow checkpoints that include schedule state.")
    schedule.load_state_dict(ckpt["schedule"])

    model.eval()
    schedule.eval()
    return model, schedule


def _motion_energy(v: torch.Tensor) -> float:
    if v.shape[1] < 2:
        return 0.0
    return float((v[:, 1:] - v[:, :-1]).abs().mean().item())


def _band_vector(transform: Haar3DTransform, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    coeffs, _ = transform.forward(v.unsqueeze(0))
    flat = transform.flatten(coeffs)
    parts: List[torch.Tensor] = []
    for i, band in enumerate(flat):
        if bool(mask[i].item()):
            parts.append(band.reshape(-1))
    if not parts:
        return torch.zeros(1, dtype=v.dtype)
    return torch.cat(parts, dim=0)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
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


def _sample_pair(
    model: torch.nn.Module,
    transform: Haar3DTransform,
    schedule: SAASchedule,
    cfg: object,
    device: torch.device,
    content_label: int,
    motion_label: int,
    seed: int,
    steps: int,
    solver: str,
    anchor: float,
    kt_threshold: float,
    ks_min_replace: float,
    kt_softness: float,
    ks_softness: float,
    path_softness: float,
) -> torch.Tensor:
    shape = (1, cfg.data.in_channels, cfg.data.frames, cfg.data.image_size, cfg.data.image_size)
    return sample_video_disentangled(
        model=model,
        transform=transform,
        schedule=schedule,
        shape=shape,
        steps=steps,
        solver=solver,
        device=device,
        anchor=anchor,
        kt_threshold=kt_threshold,
        ks_min_replace=ks_min_replace,
        kt_softness=kt_softness,
        ks_softness=ks_softness,
        path_softness=path_softness,
        class_label_content=content_label,
        class_label_motion=motion_label,
        reference_video=None,
        seed=seed,
    )[0].detach().cpu()


def _direction_a_pair_metrics(
    model: torch.nn.Module,
    transform: Haar3DTransform,
    schedule: SAASchedule,
    cfg: object,
    device: torch.device,
    content_label: int,
    motion_a: int,
    motion_b: int,
    seeds: Sequence[int],
    steps: int,
    solver: str,
    anchor: float,
    kt_threshold: float,
    ks_min_replace: float,
    kt_softness: float,
    ks_softness: float,
    path_softness: float,
    th_low_preserve: float,
    th_high_change: float,
    th_motion_gap_change: float,
) -> dict:
    meta = transform.band_meta(device=device)
    low_mask = (meta.kt < kt_threshold) | (meta.ks < ks_min_replace)
    high_mask = ~low_mask

    low_cos: List[float] = []
    high_cos: List[float] = []
    motion_gap: List[float] = []

    for seed in seeds:
        va = _sample_pair(
            model=model,
            transform=transform,
            schedule=schedule,
            cfg=cfg,
            device=device,
            content_label=content_label,
            motion_label=motion_a,
            seed=seed,
            steps=steps,
            solver=solver,
            anchor=anchor,
            kt_threshold=kt_threshold,
            ks_min_replace=ks_min_replace,
            kt_softness=kt_softness,
            ks_softness=ks_softness,
            path_softness=path_softness,
        )
        vb = _sample_pair(
            model=model,
            transform=transform,
            schedule=schedule,
            cfg=cfg,
            device=device,
            content_label=content_label,
            motion_label=motion_b,
            seed=seed,
            steps=steps,
            solver=solver,
            anchor=anchor,
            kt_threshold=kt_threshold,
            ks_min_replace=ks_min_replace,
            kt_softness=kt_softness,
            ks_softness=ks_softness,
            path_softness=path_softness,
        )

        low_a = _band_vector(transform, va, low_mask.cpu())
        low_b = _band_vector(transform, vb, low_mask.cpu())
        high_a = _band_vector(transform, va, high_mask.cpu())
        high_b = _band_vector(transform, vb, high_mask.cpu())

        low_cos.append(_cos(low_a, low_b))
        high_cos.append(_cos(high_a, high_b))
        motion_gap.append(abs(_motion_energy(va) - _motion_energy(vb)))

    pass_count = 0
    for l, h, g in zip(low_cos, high_cos, motion_gap):
        if l >= th_low_preserve and h <= th_high_change and g >= th_motion_gap_change:
            pass_count += 1

    return {
        "direction": "A_low_preserve_high_change",
        "content_label": int(content_label),
        "motion_a": int(motion_a),
        "motion_b": int(motion_b),
        "num_seeds": len(seeds),
        "low_band_cos_mean": float(statistics.mean(low_cos)),
        "high_band_cos_mean": float(statistics.mean(high_cos)),
        "motion_energy_gap_mean": float(statistics.mean(motion_gap)),
        "pass_rate": float(pass_count / max(len(low_cos), 1)),
    }


def _direction_b_pair_metrics(
    model: torch.nn.Module,
    transform: Haar3DTransform,
    schedule: SAASchedule,
    cfg: object,
    device: torch.device,
    motion_label: int,
    content_a: int,
    content_b: int,
    seeds: Sequence[int],
    steps: int,
    solver: str,
    anchor: float,
    kt_threshold: float,
    ks_min_replace: float,
    kt_softness: float,
    ks_softness: float,
    path_softness: float,
    th_high_preserve: float,
    th_low_change: float,
    th_motion_gap_keep: float,
) -> dict:
    meta = transform.band_meta(device=device)
    low_mask = (meta.kt < kt_threshold) | (meta.ks < ks_min_replace)
    high_mask = ~low_mask

    low_cos: List[float] = []
    high_cos: List[float] = []
    motion_gap: List[float] = []

    for seed in seeds:
        va = _sample_pair(
            model=model,
            transform=transform,
            schedule=schedule,
            cfg=cfg,
            device=device,
            content_label=content_a,
            motion_label=motion_label,
            seed=seed,
            steps=steps,
            solver=solver,
            anchor=anchor,
            kt_threshold=kt_threshold,
            ks_min_replace=ks_min_replace,
            kt_softness=kt_softness,
            ks_softness=ks_softness,
            path_softness=path_softness,
        )
        vb = _sample_pair(
            model=model,
            transform=transform,
            schedule=schedule,
            cfg=cfg,
            device=device,
            content_label=content_b,
            motion_label=motion_label,
            seed=seed,
            steps=steps,
            solver=solver,
            anchor=anchor,
            kt_threshold=kt_threshold,
            ks_min_replace=ks_min_replace,
            kt_softness=kt_softness,
            ks_softness=ks_softness,
            path_softness=path_softness,
        )

        low_a = _band_vector(transform, va, low_mask.cpu())
        low_b = _band_vector(transform, vb, low_mask.cpu())
        high_a = _band_vector(transform, va, high_mask.cpu())
        high_b = _band_vector(transform, vb, high_mask.cpu())

        low_cos.append(_cos(low_a, low_b))
        high_cos.append(_cos(high_a, high_b))
        motion_gap.append(abs(_motion_energy(va) - _motion_energy(vb)))

    pass_count = 0
    for l, h, g in zip(low_cos, high_cos, motion_gap):
        if h >= th_high_preserve and l <= th_low_change and g <= th_motion_gap_keep:
            pass_count += 1

    return {
        "direction": "B_high_preserve_low_change",
        "motion_label": int(motion_label),
        "content_a": int(content_a),
        "content_b": int(content_b),
        "num_seeds": len(seeds),
        "low_band_cos_mean": float(statistics.mean(low_cos)),
        "high_band_cos_mean": float(statistics.mean(high_cos)),
        "motion_energy_gap_mean": float(statistics.mean(motion_gap)),
        "pass_rate": float(pass_count / max(len(low_cos), 1)),
    }


def _mean(items: List[float]) -> float:
    return float(statistics.mean(items)) if items else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Bidirectional disentanglement evaluation for G-STVK-Flow")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--content-labels", type=str, required=True, help="Comma list, e.g. 0,1,2")
    p.add_argument("--motion-labels", type=str, required=True, help="Comma list, e.g. 3,4,5")
    p.add_argument("--num-seeds", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=123)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    p.add_argument("--anchor", type=float, default=0.35)
    p.add_argument("--kt-threshold", type=float, default=0.55)
    p.add_argument("--ks-min-replace", type=float, default=0.15)
    p.add_argument("--kt-softness", type=float, default=None)
    p.add_argument("--ks-softness", type=float, default=None)
    p.add_argument("--path-softness", type=float, default=None)

    p.add_argument("--th-low-preserve", type=float, default=0.90)
    p.add_argument("--th-high-change", type=float, default=0.95)
    p.add_argument("--th-motion-gap-change", type=float, default=0.005)
    p.add_argument("--th-high-preserve", type=float, default=0.90)
    p.add_argument("--th-low-change", type=float, default=0.97)
    p.add_argument("--th-motion-gap-keep", type=float, default=0.005)

    p.add_argument("--out-json", type=Path, default=Path("outputs/eval_disentangle_bidirectional.json"))
    args = p.parse_args()

    cfg = load_config(args.config.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, schedule = _load_model(args.checkpoint, cfg, device)
    transform = Haar3DTransform(levels=cfg.transform.levels)

    contents = _parse_int_list(args.content_labels)
    motions = _parse_int_list(args.motion_labels)
    if len(motions) < 2:
        raise ValueError("motion-labels requires at least two labels")
    if len(contents) < 2:
        raise ValueError("content-labels requires at least two labels")

    seeds = [args.seed_start + i for i in range(args.num_seeds)]
    steps = args.steps if args.steps is not None else cfg.inference.steps
    solver = args.solver if args.solver is not None else cfg.inference.solver
    kt_softness = args.kt_softness if args.kt_softness is not None else cfg.inference.kt_softness
    ks_softness = args.ks_softness if args.ks_softness is not None else cfg.inference.ks_softness
    path_softness = args.path_softness if args.path_softness is not None else cfg.inference.path_softness

    direction_a_pairs: List[dict] = []
    for c in contents:
        for ma, mb in itertools.combinations(motions, 2):
            direction_a_pairs.append(
                _direction_a_pair_metrics(
                    model=model,
                    transform=transform,
                    schedule=schedule,
                    cfg=cfg,
                    device=device,
                    content_label=c,
                    motion_a=ma,
                    motion_b=mb,
                    seeds=seeds,
                    steps=steps,
                    solver=solver,
                    anchor=args.anchor,
                    kt_threshold=args.kt_threshold,
                    ks_min_replace=args.ks_min_replace,
                    kt_softness=kt_softness,
                    ks_softness=ks_softness,
                    path_softness=path_softness,
                    th_low_preserve=args.th_low_preserve,
                    th_high_change=args.th_high_change,
                    th_motion_gap_change=args.th_motion_gap_change,
                )
            )

    direction_b_pairs: List[dict] = []
    for m in motions:
        for ca, cb in itertools.combinations(contents, 2):
            direction_b_pairs.append(
                _direction_b_pair_metrics(
                    model=model,
                    transform=transform,
                    schedule=schedule,
                    cfg=cfg,
                    device=device,
                    motion_label=m,
                    content_a=ca,
                    content_b=cb,
                    seeds=seeds,
                    steps=steps,
                    solver=solver,
                    anchor=args.anchor,
                    kt_threshold=args.kt_threshold,
                    ks_min_replace=args.ks_min_replace,
                    kt_softness=kt_softness,
                    ks_softness=ks_softness,
                    path_softness=path_softness,
                    th_high_preserve=args.th_high_preserve,
                    th_low_change=args.th_low_change,
                    th_motion_gap_keep=args.th_motion_gap_keep,
                )
            )

    a_summary = {
        "num_pairs": len(direction_a_pairs),
        "low_band_cos_mean": _mean([x["low_band_cos_mean"] for x in direction_a_pairs]),
        "high_band_cos_mean": _mean([x["high_band_cos_mean"] for x in direction_a_pairs]),
        "motion_energy_gap_mean": _mean([x["motion_energy_gap_mean"] for x in direction_a_pairs]),
        "pass_rate_mean": _mean([x["pass_rate"] for x in direction_a_pairs]),
    }
    b_summary = {
        "num_pairs": len(direction_b_pairs),
        "low_band_cos_mean": _mean([x["low_band_cos_mean"] for x in direction_b_pairs]),
        "high_band_cos_mean": _mean([x["high_band_cos_mean"] for x in direction_b_pairs]),
        "motion_energy_gap_mean": _mean([x["motion_energy_gap_mean"] for x in direction_b_pairs]),
        "pass_rate_mean": _mean([x["pass_rate"] for x in direction_b_pairs]),
    }

    payload = {
        "direction_A_low_preserve_high_change": {
            "summary": a_summary,
            "pairs": direction_a_pairs,
        },
        "direction_B_high_preserve_low_change": {
            "summary": b_summary,
            "pairs": direction_b_pairs,
        },
        "recommended_thresholds": {
            "direction_A": {
                "low_band_cos_mean": ">= 0.90",
                "high_band_cos_mean": "<= 0.95",
                "motion_energy_gap_mean": ">= 0.005",
                "pass_rate_mean": ">= 0.60",
            },
            "direction_B": {
                "high_band_cos_mean": ">= 0.90",
                "low_band_cos_mean": "<= 0.97",
                "motion_energy_gap_mean": "<= 0.005",
                "pass_rate_mean": ">= 0.60",
            },
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()


