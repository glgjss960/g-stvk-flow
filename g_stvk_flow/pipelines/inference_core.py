from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from g_stvk_flow.fm_core import GSTVKSolver, GSTVKVelocityWrapper
from g_stvk_flow.gstvk import BandMeta, Haar3DTransform, SAASchedule
from g_stvk_flow.gstvk.anchor_edit import apply_anchor_edit


@dataclass
class TracePoint:
    tau: float
    tag: str
    video: torch.Tensor  # [B,C,T,H,W]


def _capture_index_map(
    tau_start: float,
    tau_end: float,
    steps: int,
    trace_taus: Optional[list[float]],
) -> dict[int, float]:
    if not trace_taus or steps <= 0:
        return {}

    span = float(tau_end - tau_start)
    if abs(span) < 1e-12:
        return {}

    mapping: dict[int, tuple[float, float]] = {}
    # idx -> (requested_tau, distance)
    for tau in trace_taus:
        tau_f = float(tau)
        if tau_f < min(tau_start, tau_end) - 1e-6 or tau_f > max(tau_start, tau_end) + 1e-6:
            continue
        ratio = (tau_f - tau_start) / span
        idx = int(round(ratio * steps))
        idx = max(0, min(steps, idx))
        tau_actual = float(tau_start + span * (idx / float(steps)))
        dist = abs(tau_actual - tau_f)

        if idx not in mapping or dist < mapping[idx][1]:
            mapping[idx] = (tau_f, dist)

    return {idx: req_tau for idx, (req_tau, _d) in mapping.items()}


@torch.no_grad()
def _integrate(
    model: torch.nn.Module,
    psi: torch.Tensor,
    tau_start: float,
    tau_end: float,
    steps: int,
    solver: str,
    schedule: SAASchedule,
    band_meta: BandMeta,
    class_labels: Optional[torch.Tensor] = None,
    trace_taus: Optional[list[float]] = None,
    trace_tag: str = "ode",
) -> tuple[torch.Tensor, list[TracePoint]]:
    if steps <= 0:
        return psi, []

    time_grid = torch.linspace(tau_start, tau_end, steps + 1, device=psi.device, dtype=psi.dtype)
    capture_map = _capture_index_map(tau_start=tau_start, tau_end=tau_end, steps=steps, trace_taus=trace_taus)

    velocity = GSTVKVelocityWrapper(
        model=model,
        schedule=schedule,
        ks=band_meta.ks.to(device=psi.device, dtype=psi.dtype),
        kt=band_meta.kt.to(device=psi.device, dtype=psi.dtype),
    )
    sampler = GSTVKSolver(velocity_model=velocity)

    if capture_map:
        traj = sampler.sample(
            x_init=psi,
            time_grid=time_grid,
            method=solver,
            return_intermediates=True,
            class_labels=class_labels,
        )

        traces: list[TracePoint] = []
        for idx in sorted(capture_map.keys()):
            traces.append(
                TracePoint(
                    tau=float(time_grid[idx].item()),
                    tag=trace_tag,
                    video=traj[idx].detach().clone(),
                )
            )
        return traj[-1], traces

    out = sampler.sample(
        x_init=psi,
        time_grid=time_grid,
        method=solver,
        return_intermediates=False,
        class_labels=class_labels,
    )
    return out, []


@torch.no_grad()
def sample_video_with_trace(
    model: torch.nn.Module,
    transform: Haar3DTransform,
    schedule: SAASchedule,
    shape: tuple[int, int, int, int, int],
    steps: int,
    solver: str,
    device: torch.device,
    class_label: Optional[int] = None,
    seed: Optional[int] = None,
    trace_taus: Optional[list[float]] = None,
) -> tuple[torch.Tensor, list[TracePoint]]:
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        psi = torch.randn(shape, generator=g, device=device)
    else:
        psi = torch.randn(shape, device=device)

    labels = None
    if class_label is not None:
        labels = torch.full((shape[0],), int(class_label), dtype=torch.long, device=device)

    band_meta = transform.band_meta(device=device)
    out, traces = _integrate(
        model=model,
        psi=psi,
        tau_start=0.0,
        tau_end=1.0,
        steps=steps,
        solver=solver,
        schedule=schedule,
        band_meta=band_meta,
        class_labels=labels,
        trace_taus=trace_taus,
        trace_tag="standard",
    )
    return out, traces


@torch.no_grad()
def sample_video(
    model: torch.nn.Module,
    transform: Haar3DTransform,
    schedule: SAASchedule,
    shape: tuple[int, int, int, int, int],
    steps: int,
    solver: str,
    device: torch.device,
    class_label: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    out, _ = sample_video_with_trace(
        model=model,
        transform=transform,
        schedule=schedule,
        shape=shape,
        steps=steps,
        solver=solver,
        device=device,
        class_label=class_label,
        seed=seed,
        trace_taus=None,
    )
    return out


@torch.no_grad()
def sample_video_disentangled_with_trace(
    model: torch.nn.Module,
    transform: Haar3DTransform,
    schedule: SAASchedule,
    shape: tuple[int, int, int, int, int],
    steps: int,
    solver: str,
    device: torch.device,
    anchor: float,
    kt_threshold: float,
    ks_min_replace: float,
    kt_softness: float,
    ks_softness: float,
    path_softness: float,
    class_label_content: Optional[int] = None,
    class_label_motion: Optional[int] = None,
    reference_video: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    trace_taus: Optional[list[float]] = None,
) -> tuple[torch.Tensor, list[TracePoint], torch.Tensor, BandMeta]:
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        psi0 = torch.randn(shape, generator=g, device=device)
    else:
        psi0 = torch.randn(shape, device=device)

    content_labels = None
    if class_label_content is not None:
        content_labels = torch.full((shape[0],), int(class_label_content), dtype=torch.long, device=device)

    motion_labels = content_labels
    if class_label_motion is not None:
        motion_labels = torch.full((shape[0],), int(class_label_motion), dtype=torch.long, device=device)

    steps_a = max(1, int(round(steps * anchor)))
    steps_b = max(1, steps - steps_a)

    band_meta = transform.band_meta(device=device)

    trace_taus_a = None
    trace_taus_c = None
    if trace_taus:
        trace_taus_a = [float(t) for t in trace_taus if float(t) <= float(anchor) + 1e-8]
        trace_taus_c = [float(t) for t in trace_taus if float(t) > float(anchor) + 1e-8]

    # Stage A: content anchoring
    psi_anchor, trace_a = _integrate(
        model=model,
        psi=psi0,
        tau_start=0.0,
        tau_end=anchor,
        steps=steps_a,
        solver=solver,
        schedule=schedule,
        band_meta=band_meta,
        class_labels=content_labels,
        trace_taus=trace_taus_a,
        trace_tag="stage_a",
    )

    # Anchor edit in GSTVK band space.
    psi_edit, edit_weights, meta = apply_anchor_edit(
        transform=transform,
        schedule=schedule,
        psi_anchor=psi_anchor,
        anchor=anchor,
        kt_threshold=kt_threshold,
        ks_min_replace=ks_min_replace,
        kt_softness=kt_softness,
        ks_softness=ks_softness,
        path_softness=path_softness,
        reference_video=reference_video,
    )

    trace_special = [
        TracePoint(tau=float(anchor), tag="anchor_pre_edit", video=psi_anchor.detach().clone()),
        TracePoint(tau=float(anchor), tag="anchor_post_edit", video=psi_edit.detach().clone()),
    ]

    # Stage C: motion injection
    psi_final, trace_c = _integrate(
        model=model,
        psi=psi_edit,
        tau_start=anchor,
        tau_end=1.0,
        steps=steps_b,
        solver=solver,
        schedule=schedule,
        band_meta=band_meta,
        class_labels=motion_labels,
        trace_taus=trace_taus_c,
        trace_tag="stage_c",
    )

    traces = trace_a + trace_special + trace_c
    return psi_final, traces, edit_weights.detach().clone(), meta


@torch.no_grad()
def sample_video_disentangled(
    model: torch.nn.Module,
    transform: Haar3DTransform,
    schedule: SAASchedule,
    shape: tuple[int, int, int, int, int],
    steps: int,
    solver: str,
    device: torch.device,
    anchor: float,
    kt_threshold: float,
    ks_min_replace: float,
    kt_softness: float,
    ks_softness: float,
    path_softness: float,
    class_label_content: Optional[int] = None,
    class_label_motion: Optional[int] = None,
    reference_video: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    out, _tr, _w, _meta = sample_video_disentangled_with_trace(
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
        class_label_content=class_label_content,
        class_label_motion=class_label_motion,
        reference_video=reference_video,
        seed=seed,
        trace_taus=None,
    )
    return out
