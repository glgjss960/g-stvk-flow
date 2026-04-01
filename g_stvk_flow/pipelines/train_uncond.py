from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from g_stvk_flow.data import CachedVideoDataset
from g_stvk_flow.pipelines.common import (
    build_model,
    build_schedule,
    build_transform,
    resolve_config,
    resolve_path,
    set_seed,
)
from g_stvk_flow.pipelines.train_core import train_loop
from g_stvk_flow.gstvk import STVKInterpolant
from g_stvk_flow.utils import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train g-STVK (unconditional)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg, config_root = resolve_config(args.config)
    cfg.model.num_classes = 0
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = resolve_path(config_root, cfg.data.manifest_train)
    val_manifest = resolve_path(config_root, cfg.data.manifest_val)

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

    model = build_model(cfg=cfg, device=device, num_classes_override=0)
    transform = build_transform(cfg)
    schedule = build_schedule(cfg=cfg, device=device)
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

    run_dir = resolve_path(config_root, cfg.run.output_dir)
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
