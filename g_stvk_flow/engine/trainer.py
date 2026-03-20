from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from g_stvk_flow.config import Config
from g_stvk_flow.transforms import GSTVKInterpolant
from g_stvk_flow.utils import save_checkpoint


def _evaluate(
    model: torch.nn.Module,
    interpolant: GSTVKInterpolant,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch["video"].to(device)
            labels = batch["label"].to(device)
            tau = torch.rand(x.size(0), device=device)
            eps = torch.randn_like(x)

            sample = interpolant.build(x_data=x, eps=eps, tau=tau)
            with autocast(enabled=use_amp):
                pred = model(
                    sample.psi_tau,
                    tau=tau,
                    class_labels=labels,
                    phase_features=sample.phase_features,
                )
                loss = F.mse_loss(pred, sample.v_target)
            losses.append(float(loss.item()))

    model.train()
    return sum(losses) / max(len(losses), 1)


def _regularization_loss(cfg: Config, interpolant: GSTVKInterpolant, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    meta = interpolant.transform.band_meta(device=device)
    reg_terms = interpolant.schedule.regularization_terms(ks=meta.ks, kt=meta.kt)

    loss = (
        cfg.train.reg_endpoint * reg_terms["endpoint"]
        + cfg.train.reg_coverage * reg_terms["coverage"]
        + cfg.train.reg_spread * reg_terms["spread"]
        + cfg.train.reg_smooth * reg_terms["smooth"]
        + cfg.train.reg_mono * reg_terms["mono"]
    )

    metrics = {
        "reg_endpoint": float(reg_terms["endpoint"].item()),
        "reg_coverage": float(reg_terms["coverage"].item()),
        "reg_spread": float(reg_terms["spread"].item()),
        "reg_smooth": float(reg_terms["smooth"].item()),
        "reg_mono": float(reg_terms["mono"].item()),
    }
    return loss, metrics


def train_loop(
    model: torch.nn.Module,
    interpolant: GSTVKInterpolant,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    run_dir: str | Path,
    start_epoch: int = 0,
) -> None:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler(enabled=cfg.train.amp and device.type == "cuda")
    global_step = 0

    for epoch in range(start_epoch, cfg.train.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        running_fm = 0.0
        running_total = 0.0

        for batch in pbar:
            x = batch["video"].to(device)
            labels = batch["label"].to(device)

            tau = torch.rand(x.size(0), device=device)
            eps = torch.randn_like(x)
            sample = interpolant.build(x_data=x, eps=eps, tau=tau)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.train.amp and device.type == "cuda"):
                pred = model(
                    sample.psi_tau,
                    tau=tau,
                    class_labels=labels,
                    phase_features=sample.phase_features,
                )
                fm_loss = F.mse_loss(pred, sample.v_target)

            reg_loss = torch.zeros((), device=device)
            reg_metrics: dict[str, float] = {}
            if cfg.train.reg_every > 0 and (global_step % cfg.train.reg_every == 0):
                reg_loss, reg_metrics = _regularization_loss(cfg=cfg, interpolant=interpolant, device=device)

            loss = fm_loss + reg_loss

            scaler.scale(loss).backward()

            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running_fm += float(fm_loss.item())
            running_total += float(loss.item())
            global_step += 1

            if global_step % cfg.train.log_every == 0:
                payload = {
                    "fm": f"{running_fm / cfg.train.log_every:.6f}",
                    "total": f"{running_total / cfg.train.log_every:.6f}",
                }
                if reg_metrics:
                    payload["reg_ep"] = f"{reg_metrics['reg_endpoint']:.4f}"
                    payload["reg_cov"] = f"{reg_metrics['reg_coverage']:.4f}"
                    payload["reg_mon"] = f"{reg_metrics['reg_mono']:.4f}"
                pbar.set_postfix(payload)
                running_fm = 0.0
                running_total = 0.0

        val_loss = None
        if val_loader is not None:
            val_loss = _evaluate(
                model=model,
                interpolant=interpolant,
                loader=val_loader,
                device=device,
                use_amp=cfg.train.amp and device.type == "cuda",
            )
            print(f"[epoch {epoch}] val_fm_loss={val_loss:.6f}")

        if (epoch + 1) % cfg.train.save_every == 0:
            state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "schedule": interpolant.schedule.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "val_loss": val_loss,
                "global_step": global_step,
            }
            save_checkpoint(state, ckpt_dir / "last.pt")
            save_checkpoint(state, ckpt_dir / f"epoch_{epoch + 1:04d}.pt")
