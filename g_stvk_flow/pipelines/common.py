from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from g_stvk_flow.backbone import STVKFlowModel
from g_stvk_flow.config import Config, load_config
from g_stvk_flow.gstvk import Haar3DTransform, SAASchedule, STVKInterpolant
from g_stvk_flow.utils import load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_config(config_path: Path) -> tuple[Config, Path]:
    path = config_path.resolve()
    cfg = load_config(path)
    return cfg, path.parent


def resolve_path(config_root: Path, path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (config_root / p)


def build_model(cfg: Config, device: torch.device, num_classes_override: Optional[int] = None) -> STVKFlowModel:
    num_classes = cfg.model.num_classes if num_classes_override is None else int(num_classes_override)
    return STVKFlowModel(
        in_channels=cfg.data.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        cond_dim=cfg.model.cond_dim,
        phase_dim=cfg.model.phase_dim,
        num_classes=num_classes,
        dropout=cfg.model.dropout,
        backbone=getattr(cfg.model, "backbone", "opensora_dit"),
        hidden_size=getattr(cfg.model, "hidden_size", 512),
        depth=getattr(cfg.model, "depth", 8),
        num_heads=getattr(cfg.model, "num_heads", 8),
        mlp_ratio=getattr(cfg.model, "mlp_ratio", 4.0),
        patch_size_t=getattr(cfg.model, "patch_size_t", 1),
        patch_size_h=getattr(cfg.model, "patch_size_h", 2),
        patch_size_w=getattr(cfg.model, "patch_size_w", 2),
    ).to(device)


def build_schedule(cfg: Config, device: torch.device) -> SAASchedule:
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
        integration_grid_size=getattr(cfg.flow, "integration_grid_size", 129),
        rate_floor=getattr(cfg.flow, "rate_floor", 1e-4),
        lambda_replace_thr=getattr(cfg.flow, "lambda_replace_thr", 0.55),
        tail_start=getattr(cfg.flow, "tail_start", 0.85),
    ).to(device)


def build_transform(cfg: Config) -> Haar3DTransform:
    return Haar3DTransform(levels=cfg.transform.levels)


def build_interpolant(cfg: Config, device: torch.device) -> STVKInterpolant:
    transform = build_transform(cfg)
    schedule = build_schedule(cfg, device)
    return STVKInterpolant(transform=transform, schedule=schedule)


def load_model_schedule_cfg(
    checkpoint: Path,
    cfg_path: Path,
    device: torch.device,
    num_classes_override: Optional[int] = None,
) -> tuple[STVKFlowModel, Config, SAASchedule]:
    cfg, _ = resolve_config(cfg_path)
    model = build_model(cfg=cfg, device=device, num_classes_override=num_classes_override)
    schedule = build_schedule(cfg=cfg, device=device)

    ckpt = load_checkpoint(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    if not (isinstance(ckpt, dict) and "schedule" in ckpt):
        raise KeyError("Checkpoint does not contain 'schedule'.")
    schedule.load_state_dict(ckpt["schedule"])

    model.eval()
    schedule.eval()
    return model, cfg, schedule
