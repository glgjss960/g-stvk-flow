from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from g_stvk_flow.data import CachedVideoDataset
from g_stvk_flow.kflow import FixedBandPath, SeparableHaarVideoDecomposer
from g_stvk_flow.losses import BandwiseDiffusionObjective, VanillaDiffusionObjective
from g_stvk_flow.models.stdit_band import StageABandVideoModel
from g_stvk_flow.models.vae_wrapper import VideoVAEWrapper
from g_stvk_flow.utils import load_checkpoint, save_checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, maybe_path: str | None) -> Path | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else (root / p)


def _dtype_from_string(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _get_data_hw(data_cfg: dict[str, Any]) -> tuple[int, int]:
    h = data_cfg.get("image_height", data_cfg.get("image_size", None))
    w = data_cfg.get("image_width", data_cfg.get("image_size", None))
    if h is None or w is None:
        raise ValueError("data.image_height/image_width or data.image_size must be provided")
    return int(h), int(w)


def _build_model(cfg: dict[str, Any], num_bands: int, device: torch.device) -> StageABandVideoModel:
    model_cfg = cfg["model"]
    return StageABandVideoModel(
        in_channels=int(cfg["data"]["in_channels"]),
        num_bands=int(num_bands),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        hidden_size=int(model_cfg.get("hidden_size", 512)),
        depth=int(model_cfg.get("depth", 8)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        patch_size_t=int(model_cfg.get("patch_size_t", 1)),
        patch_size_h=int(model_cfg.get("patch_size_h", 2)),
        patch_size_w=int(model_cfg.get("patch_size_w", 2)),
        num_classes=int(model_cfg.get("num_classes", 0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_band_embed=bool(model_cfg.get("use_band_embed", True)),
        grad_checkpoint=bool(model_cfg.get("grad_checkpoint", False)),
    ).to(device)


def _print_model_stats(model: StageABandVideoModel) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = sum(p.numel() for p in model.backbone.parameters())
    print(f"Model params total={total:,} trainable={trainable:,} backbone={backbone:,}")
    print(
        "Approx model memory (weights only): "
        f"fp16/bf16={total * 2 / (1024 ** 3):.2f} GiB, fp32={total * 4 / (1024 ** 3):.2f} GiB"
    )


def _build_loader(
    dataset: CachedVideoDataset,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    device: torch.device,
    train_cfg: dict[str, Any],
) -> DataLoader:
    pin_memory_default = device.type == "cuda"
    pin_memory = bool(train_cfg.get("pin_memory", pin_memory_default))
    shuffle = bool(train_cfg.get("shuffle_train", True)) if is_train else bool(train_cfg.get("shuffle_val", False))
    drop_last = bool(train_cfg.get("drop_last_train", True)) if is_train else bool(train_cfg.get("drop_last_val", False))
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = train_cfg.get("prefetch_factor", None)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None and int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**kwargs)


def _run_eval(
    model: StageABandVideoModel,
    vae: VideoVAEWrapper,
    loader: DataLoader,
    device: torch.device,
    train_mode: str,
    fixed_path: FixedBandPath,
    band_objective: BandwiseDiffusionObjective,
    vanilla_objective: VanillaDiffusionObjective,
    amp_enabled: bool,
    raw_frames: int,
    raw_h: int,
    raw_w: int,
    in_channels: int,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            labels = batch["label"].to(device)

            if video.ndim != 5:
                raise ValueError(f"Expected input video shape [B,C,T,H,W], got {tuple(video.shape)}")
            if video.shape[2] != raw_frames or video.shape[3] != raw_h or video.shape[4] != raw_w:
                raise ValueError(
                    "Batch video shape mismatches config: "
                    f"got (T,H,W)=({video.shape[2]},{video.shape[3]},{video.shape[4]}), "
                    f"expected ({raw_frames},{raw_h},{raw_w})"
                )

            z = vae.encode(video)
            if z.shape[1] != in_channels:
                raise ValueError(
                    f"Encoded latent channels mismatch: z has C={z.shape[1]}, expected data.in_channels={in_channels}"
                )

            t = torch.rand(z.shape[0], device=device, dtype=z.dtype)

            if train_mode == "bandwise":
                band_name = random.choice(list(fixed_path.order))
                state = band_objective.build_state(clean_latent=z, t=t, band_name=band_name)
                band_id = torch.full((z.shape[0],), fixed_path.index_of(band_name), dtype=torch.long, device=device)
                with autocast(enabled=amp_enabled):
                    pred = model(x=state.x_t, t=t, band_id=band_id, class_labels=labels)
                    loss = band_objective.compute_loss(pred, state)
            else:
                x_t, target = vanilla_objective.build_state(clean_latent=z, t=t)
                band_id = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
                with autocast(enabled=amp_enabled):
                    pred = model(x=x_t, t=t, band_id=band_id, class_labels=labels)
                    loss = vanilla_objective.compute_loss(pred, target)

            losses.append(float(loss.item()))

    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phase-A bandwise video prototype")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg_root = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    _set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = _resolve(cfg_root, cfg["run"]["output_dir"])
    assert run_dir is not None
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    train_manifest = _resolve(cfg_root, data_cfg["manifest_train"])
    val_manifest = _resolve(cfg_root, data_cfg.get("manifest_val"))
    if train_manifest is None:
        raise ValueError("manifest_train is required")

    train_ds = CachedVideoDataset(train_manifest)
    val_ds = CachedVideoDataset(val_manifest) if (val_manifest is not None and val_manifest.exists()) else None

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 2))

    train_loader = _build_loader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,
        device=device,
        train_cfg=train_cfg,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = _build_loader(
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=False,
            device=device,
            train_cfg=train_cfg,
        )

    decomp_mode = str(cfg["decomposition"].get("mode", "spatial_temporal"))
    decomposer = SeparableHaarVideoDecomposer(mode=decomp_mode)
    fixed_path = FixedBandPath(
        band_names=decomposer.band_names,
        path_name=str(cfg["path"].get("name", "A")),
    )

    model = _build_model(cfg=cfg, num_bands=len(decomposer.band_names), device=device)
    _print_model_stats(model)

    vae_cfg = cfg.get("vae", {})
    vae = VideoVAEWrapper(
        backend=str(vae_cfg.get("backend", "identity")),
        open_sora_root=_resolve(cfg_root, vae_cfg.get("open_sora_root")),
        pretrained_path=str(_resolve(cfg_root, vae_cfg.get("pretrained_path"))) if vae_cfg.get("pretrained_path") else None,
        device=device,
        dtype=_dtype_from_string(vae_cfg.get("dtype", "float32")),
        freeze=bool(vae_cfg.get("freeze", True)),
        sample_posterior=bool(vae_cfg.get("sample_posterior", False)),
    ).to(device)

    in_channels = int(data_cfg["in_channels"])
    if vae.latent_channels is not None and in_channels != int(vae.latent_channels):
        raise ValueError(
            f"data.in_channels={in_channels} mismatches VAE latent_channels={vae.latent_channels}. "
            "Use the VAE latent channel count in config."
        )

    raw_frames = int(data_cfg["frames"])
    raw_h, raw_w = _get_data_hw(data_cfg)
    raw_fps = data_cfg.get("fps", None)
    latent_t, latent_h, latent_w = vae.get_latent_size(raw_frames, raw_h, raw_w)
    fps_msg = f", fps={raw_fps}" if raw_fps is not None else ""
    print(
        f"Raw clip shape (T,H,W)=({raw_frames},{raw_h},{raw_w}){fps_msg} -> "
        f"Latent shape (T,H,W)=({latent_t},{latent_h},{latent_w})"
    )

    if decomp_mode in {"spatial_temporal", "temporal_only"} and latent_t % 2 != 0:
        raise ValueError(
            f"Latent temporal length must be even for temporal Haar split, got T={latent_t}. "
            "Adjust data.frames (raw clip length)."
        )
    if decomp_mode in {"spatial_temporal", "spatial_only"} and ((latent_h % 2 != 0) or (latent_w % 2 != 0)):
        raise ValueError(
            f"Latent spatial size must be even for spatial Haar split, got (H,W)=({latent_h},{latent_w}). "
            "Adjust data.image_height/image_width."
        )

    train_mode = str(train_cfg.get("mode", "bandwise")).lower().strip()
    if train_mode not in {"bandwise", "vanilla"}:
        raise ValueError(f"Unsupported training.mode={train_mode}, expected bandwise or vanilla")

    flow_root = _resolve(cfg_root, train_cfg.get("flow_matching_root"))
    band_objective = BandwiseDiffusionObjective(
        decomposer=decomposer,
        fixed_path=fixed_path,
        target_type=str(train_cfg.get("target_type", "epsilon")),
        scheduler_name=str(train_cfg.get("scheduler", "cosine")),
        flow_matching_root=flow_root,
        band_weights=train_cfg.get("band_weights", None),
    )
    vanilla_objective = VanillaDiffusionObjective(
        target_type=str(train_cfg.get("target_type", "epsilon")),
        scheduler_name=str(train_cfg.get("scheduler", "cosine")),
        flow_matching_root=flow_root,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    scaler = GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    start_epoch = 0
    global_step = 0
    if args.resume is not None and args.resume.exists():
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {args.resume} at epoch={start_epoch}, step={global_step}")

    epochs = int(train_cfg.get("epochs", 100))
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_loader, desc=f"stageA epoch {epoch}", leave=False)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            video = batch["video"].to(device)
            labels = batch["label"].to(device)

            if video.ndim != 5:
                raise ValueError(f"Expected input video shape [B,C,T,H,W], got {tuple(video.shape)}")
            if video.shape[2] != raw_frames or video.shape[3] != raw_h or video.shape[4] != raw_w:
                raise ValueError(
                    "Batch video shape mismatches config: "
                    f"got (T,H,W)=({video.shape[2]},{video.shape[3]},{video.shape[4]}), "
                    f"expected ({raw_frames},{raw_h},{raw_w})"
                )

            z = vae.encode(video)
            if z.shape[1] != in_channels:
                raise ValueError(
                    f"Encoded latent channels mismatch: z has C={z.shape[1]}, expected data.in_channels={in_channels}"
                )

            t = torch.rand(z.shape[0], device=device, dtype=z.dtype)

            if train_mode == "bandwise":
                band_name = random.choice(list(fixed_path.order))
                state = band_objective.build_state(clean_latent=z, t=t, band_name=band_name)
                band_id = torch.full((z.shape[0],), fixed_path.index_of(band_name), dtype=torch.long, device=device)

                with autocast(enabled=amp_enabled):
                    pred = model(x=state.x_t, t=t, band_id=band_id, class_labels=labels)
                    loss = band_objective.compute_loss(pred, state)
            else:
                x_t, target = vanilla_objective.build_state(clean_latent=z, t=t)
                band_id = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
                with autocast(enabled=amp_enabled):
                    pred = model(x=x_t, t=t, band_id=band_id, class_labels=labels)
                    loss = vanilla_objective.compute_loss(pred, target)

            loss_for_backward = loss / float(grad_accum_steps)
            scaler.scale(loss_for_backward).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                running += float(loss.item())
                if global_step % log_every == 0:
                    pbar.set_postfix({"loss": f"{running / log_every:.6f}", "mode": train_mode})
                    running = 0.0

        val_loss = None
        if val_loader is not None:
            val_loss = _run_eval(
                model=model,
                vae=vae,
                loader=val_loader,
                device=device,
                train_mode=train_mode,
                fixed_path=fixed_path,
                band_objective=band_objective,
                vanilla_objective=vanilla_objective,
                amp_enabled=amp_enabled,
                raw_frames=raw_frames,
                raw_h=raw_h,
                raw_w=raw_w,
                in_channels=in_channels,
            )
            print(f"[epoch {epoch}] val_loss={val_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "val_loss": val_loss,
            }
            save_checkpoint(state, ckpt_dir / "last.pt")
            save_checkpoint(state, ckpt_dir / f"epoch_{epoch + 1:04d}.pt")


if __name__ == "__main__":
    main()
