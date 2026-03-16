from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.config import load_config
from g_stvk_flow.engine import sample_video_disentangled
from g_stvk_flow.models import STVKFlowModel
from g_stvk_flow.transforms import Haar3DTransform, SAASchedule
from g_stvk_flow.utils import load_checkpoint, save_video_tensor


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

    save_video_tensor(sample[0], args.out, fps=cfg.inference.fps)
    print(f"Saved disentangled sample to {args.out}")


if __name__ == "__main__":
    main()
