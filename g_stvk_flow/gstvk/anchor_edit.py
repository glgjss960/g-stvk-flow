from __future__ import annotations

from typing import Optional

import torch

from g_stvk_flow.gstvk.haar3d import BandMeta, Haar3DTransform
from g_stvk_flow.gstvk.scheduler import SAASchedule
from g_stvk_flow.utils import make_low_high_masks


def parse_anchor_grid(text: str | None) -> list[float]:
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


def resolve_anchor(
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

        candidates = parse_anchor_grid(anchor_grid)
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


@torch.no_grad()
def apply_anchor_edit(
    transform: Haar3DTransform,
    schedule: SAASchedule,
    psi_anchor: torch.Tensor,
    anchor: float,
    kt_threshold: float,
    ks_min_replace: float,
    kt_softness: float,
    ks_softness: float,
    path_softness: float,
    reference_video: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, BandMeta]:
    z_anchor, meta = transform.forward(psi_anchor)

    if reference_video is None:
        ref_video = torch.randn_like(psi_anchor)
    else:
        ref_video = reference_video.to(device=psi_anchor.device, dtype=psi_anchor.dtype)
        if ref_video.shape[0] == 1 and psi_anchor.shape[0] > 1:
            ref_video = ref_video.repeat(psi_anchor.shape[0], 1, 1, 1, 1)
        if tuple(ref_video.shape) != tuple(psi_anchor.shape):
            raise ValueError(
                f"reference_video shape must be {tuple(psi_anchor.shape)} or batch=1 variant, got {tuple(ref_video.shape)}"
            )

    z_new, _ = transform.forward(ref_video)

    edit_weights = schedule.build_edit_weights(
        ks=meta.ks,
        kt=meta.kt,
        tau_anchor=anchor,
        kt_threshold=kt_threshold,
        ks_min_replace=ks_min_replace,
        kt_softness=kt_softness,
        ks_softness=ks_softness,
        path_softness=path_softness,
    )

    flat_anchor = transform.flatten(z_anchor)
    flat_new = transform.flatten(z_new)

    flat_edit = []
    for i, (band_anchor, band_new) in enumerate(zip(flat_anchor, flat_new)):
        w = edit_weights[i].to(device=band_anchor.device, dtype=band_anchor.dtype)
        view_shape = [1] + [1] * (band_anchor.ndim - 1)
        w_view = w.view(view_shape)
        flat_edit.append((1.0 - w_view) * band_anchor + w_view * band_new)

    z_edit = transform.unflatten_like(z_anchor, flat_edit)
    psi_edit = transform.inverse(z_edit)
    return psi_edit, edit_weights.detach().clone(), meta
