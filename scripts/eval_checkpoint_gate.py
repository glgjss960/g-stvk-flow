from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.config import load_config
from g_stvk_flow.data import CachedVideoDataset
from g_stvk_flow.models import STVKFlowModel
from g_stvk_flow.transforms import Haar3DTransform, SAASchedule, STVKInterpolant
from g_stvk_flow.engine import sample_video
from g_stvk_flow.utils import load_checkpoint


def _resolve(cfg_path: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (cfg_path.parent / pp)


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


def _val_fm_loss(
    model: torch.nn.Module,
    interpolant: STVKInterpolant,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    vals: List[float] = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            x = batch["video"].to(device)
            labels = batch["label"].to(device)
            tau = torch.rand(x.size(0), device=device)
            eps = torch.randn_like(x)
            sample = interpolant.build(x_data=x, eps=eps, tau=tau)
            pred = model(sample.psi_tau, tau=tau, class_labels=labels, phase_features=sample.phase_features)
            vals.append(float(F.mse_loss(pred, sample.v_target).item()))
    return float(sum(vals) / max(len(vals), 1))


def _spatial_corr(v: torch.Tensor) -> float:
    # v: [C,T,H,W] in [-1,1]
    x = v[:, :, :, :-1]
    y = v[:, :, :, 1:]
    num = (x * y).mean()
    den = (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-8)
    return float((num / den).item())


def _temporal_corr(v: torch.Tensor) -> float:
    if v.shape[1] < 2:
        return 0.0
    x = v[:, :-1]
    y = v[:, 1:]
    num = (x * y).mean()
    den = (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-8)
    return float((num / den).item())


def _temporal_diff(v: torch.Tensor) -> float:
    if v.shape[1] < 2:
        return 0.0
    return float((v[:, 1:] - v[:, :-1]).abs().mean().item())


def _video_std(v: torch.Tensor) -> float:
    return float(v.std().item())


def _judge(summary: Dict[str, float]) -> Dict[str, object]:
    rules = {
        "val_fm_loss": summary["val_fm_loss"] <= 0.08,
        "video_std": 0.15 <= summary["video_std_mean"] <= 0.80,
        "spatial_corr": summary["spatial_corr_mean"] >= 0.12,
        "temporal_corr": summary["temporal_corr_mean"] >= 0.05,
        "temporal_diff": summary["temporal_diff_mean"] >= 0.02,
    }
    passed = all(rules.values())
    return {
        "passed": passed,
        "rules": rules,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Gate checkpoint for standard G-STVK-Flow inference")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-val-batches", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--class-label", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    p.add_argument("--seed-start", type=int, default=123)
    p.add_argument("--out-json", type=Path, default=Path("outputs/eval_gate.json"))
    args = p.parse_args()

    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, schedule = _load_model(args.checkpoint, cfg, device)

    transform = Haar3DTransform(levels=cfg.transform.levels)
    interpolant = STVKInterpolant(transform=transform, schedule=schedule)

    val_manifest = args.val_manifest if args.val_manifest is not None else _resolve(cfg_path, cfg.data.manifest_val)
    val_ds = CachedVideoDataset(val_manifest)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_fm = _val_fm_loss(model=model, interpolant=interpolant, loader=val_loader, device=device, max_batches=args.max_val_batches)

    steps = args.steps if args.steps is not None else cfg.inference.steps
    solver = args.solver if args.solver is not None else cfg.inference.solver

    shape = (1, cfg.data.in_channels, cfg.data.frames, cfg.data.image_size, cfg.data.image_size)

    stds: List[float] = []
    scorrs: List[float] = []
    tcorrs: List[float] = []
    tdiffs: List[float] = []

    for i in range(args.num_samples):
        s = args.seed_start + i
        vid = sample_video(
            model=model,
            transform=transform,
            schedule=schedule,
            shape=shape,
            steps=steps,
            solver=solver,
            device=device,
            class_label=args.class_label,
            seed=s,
        )[0].detach().cpu()

        stds.append(_video_std(vid))
        scorrs.append(_spatial_corr(vid))
        tcorrs.append(_temporal_corr(vid))
        tdiffs.append(_temporal_diff(vid))

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_samples": int(args.num_samples),
        "val_fm_loss": float(val_fm),
        "video_std_mean": float(statistics.mean(stds)),
        "video_std_std": float(statistics.pstdev(stds) if len(stds) > 1 else 0.0),
        "spatial_corr_mean": float(statistics.mean(scorrs)),
        "temporal_corr_mean": float(statistics.mean(tcorrs)),
        "temporal_diff_mean": float(statistics.mean(tdiffs)),
    }

    gate = _judge(summary)
    payload = {
        "summary": summary,
        "gate": gate,
        "recommended_thresholds": {
            "val_fm_loss": "<= 0.08 (good: <= 0.05)",
            "video_std_mean": "[0.15, 0.80]",
            "spatial_corr_mean": ">= 0.12",
            "temporal_corr_mean": ">= 0.05",
            "temporal_diff_mean": ">= 0.02",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
