from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import List

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


def _predict_label(classifier: torch.jit.ScriptModule, video: torch.Tensor, device: torch.device) -> int:
    # video: [C,T,H,W] in [-1,1]
    x = video.unsqueeze(0).to(device)
    with torch.no_grad():
        out = classifier(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    pred = int(out.argmax(dim=1).item())
    return pred


def main() -> None:
    p = argparse.ArgumentParser(description="Semantic motion controllability eval with external classifier")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--motion-clf-ts", type=Path, required=True, help="TorchScript classifier path")
    p.add_argument("--content-labels", type=str, required=True)
    p.add_argument("--motion-labels", type=str, required=True)
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=123)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    p.add_argument("--anchor", type=float, default=0.35)
    p.add_argument("--kt-threshold", type=float, default=0.55)
    p.add_argument("--ks-min-replace", type=float, default=0.15)
    p.add_argument("--kt-softness", type=float, default=None)
    p.add_argument("--ks-softness", type=float, default=None)
    p.add_argument("--path-softness", type=float, default=None)
    p.add_argument("--out-json", type=Path, default=Path("outputs/eval_disentangle_semantic.json"))
    args = p.parse_args()

    cfg = load_config(args.config.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, schedule = _load_model(args.checkpoint, cfg, device)
    transform = Haar3DTransform(levels=cfg.transform.levels)

    motion_clf = torch.jit.load(str(args.motion_clf_ts), map_location=device)
    motion_clf.eval()

    contents = _parse_int_list(args.content_labels)
    motions = _parse_int_list(args.motion_labels)

    steps = args.steps if args.steps is not None else cfg.inference.steps
    solver = args.solver if args.solver is not None else cfg.inference.solver
    kt_softness = args.kt_softness if args.kt_softness is not None else cfg.inference.kt_softness
    ks_softness = args.ks_softness if args.ks_softness is not None else cfg.inference.ks_softness
    path_softness = args.path_softness if args.path_softness is not None else cfg.inference.path_softness

    shape = (1, cfg.data.in_channels, cfg.data.frames, cfg.data.image_size, cfg.data.image_size)
    seeds = [args.seed_start + i for i in range(args.num_seeds)]

    rows = []
    correct = 0
    total = 0

    for c, m in itertools.product(contents, motions):
        hit = 0
        for s in seeds:
            vid = sample_video_disentangled(
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
                class_label_content=c,
                class_label_motion=m,
                reference_video=None,
                seed=s,
            )[0].detach()

            pred = _predict_label(motion_clf, vid, device)
            ok = int(pred == m)
            hit += ok
            correct += ok
            total += 1

        rows.append(
            {
                "content_label": int(c),
                "motion_label": int(m),
                "acc": float(hit / max(len(seeds), 1)),
            }
        )

    macro = float(statistics.mean([r["acc"] for r in rows])) if rows else 0.0
    micro = float(correct / max(total, 1))

    payload = {
        "summary": {
            "macro_motion_acc": macro,
            "micro_motion_acc": micro,
            "num_cases": len(rows),
            "num_samples": total,
        },
        "by_case": rows,
        "recommended_thresholds": {
            "macro_motion_acc": ">= 0.35 (usable), >= 0.50 (strong)",
            "micro_motion_acc": ">= 0.35",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()


