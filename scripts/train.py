from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.config import load_config
from g_stvk_flow.data import CachedVideoDataset
from g_stvk_flow.engine import train_loop
from g_stvk_flow.models import STVKFlowModel
from g_stvk_flow.transforms import Haar3DTransform, SAASchedule, STVKInterpolant
from g_stvk_flow.utils import load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G-STVK-Flow")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    config_root = config_path.parent

    def resolve_path(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (config_root / pp)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = resolve_path(cfg.data.manifest_train)
    val_manifest = resolve_path(cfg.data.manifest_val)

    train_ds = CachedVideoDataset(train_manifest)
    val_ds = CachedVideoDataset(val_manifest) if val_manifest.exists() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

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

    transform = Haar3DTransform(levels=cfg.transform.levels)
    schedule = SAASchedule(
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
    interpolant = STVKInterpolant(transform=transform, schedule=schedule)

    params = list(model.parameters()) + list(schedule.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    start_epoch = 0
    if args.resume is not None and args.resume.exists():
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "schedule" in ckpt:
            schedule.load_state_dict(ckpt["schedule"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    run_dir = resolve_path(cfg.run.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loop(
        model=model,
        interpolant=interpolant,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()

