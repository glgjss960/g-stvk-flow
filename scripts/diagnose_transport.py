from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g_stvk_flow.config import load_config
from g_stvk_flow.models import STVKFlowModel
from g_stvk_flow.transforms import Haar3DTransform, SAASchedule
from g_stvk_flow.utils import load_checkpoint


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


def _state_num_classes(state: object) -> int | None:
    if not isinstance(state, dict):
        return None
    w = state.get("class_embed.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    return None


def _load_model_and_schedule(checkpoint: Path, config: Path, device: torch.device) -> tuple[STVKFlowModel, SAASchedule, object]:
    cfg = load_config(config)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    cfg_num_classes = int(cfg.model.num_classes)
    ckpt_num_classes = _state_num_classes(state)

    if ckpt_num_classes is None and cfg_num_classes > 0:
        model_num_classes = 0
        print(
            "[warn] checkpoint has no class_embed.weight but config sets model.num_classes>0; "
            "forcing num_classes=0 for compatible loading."
        )
    elif ckpt_num_classes is not None and ckpt_num_classes != cfg_num_classes:
        model_num_classes = ckpt_num_classes
        print(
            f"[warn] config model.num_classes={cfg_num_classes} mismatches checkpoint class_embed={ckpt_num_classes}; "
            f"using checkpoint value {ckpt_num_classes}."
        )
    else:
        model_num_classes = cfg_num_classes

    model = STVKFlowModel(
        in_channels=cfg.data.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        cond_dim=cfg.model.cond_dim,
        phase_dim=cfg.model.phase_dim,
        num_classes=model_num_classes,
        dropout=cfg.model.dropout,
    ).to(device)

    schedule = _build_schedule(cfg, device)

    model.load_state_dict(state)
    if not (isinstance(ckpt, dict) and "schedule" in ckpt):
        raise KeyError("Checkpoint does not contain 'schedule'.")
    schedule.load_state_dict(ckpt["schedule"])

    model.eval()
    schedule.eval()
    return model, schedule, cfg


def _rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt().item())


def _quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.quantile(x.float().reshape(-1), torch.tensor(q, device=x.device)).item())


def _entropy_normalized(weights: torch.Tensor) -> float:
    # weights: [N], non-negative
    w = weights.float().clamp_min(0.0)
    s = w.sum().item()
    n = int(w.numel())
    if n <= 1 or s <= 0.0:
        return 0.0
    p = (w / w.sum()).clamp_min(1e-12)
    ent = float((-p * p.log()).sum().item())
    return ent / math.log(float(n))


@torch.no_grad()
def diagnose_transport(
    model: STVKFlowModel,
    schedule: SAASchedule,
    cfg: object,
    device: torch.device,
    steps: int,
    solver: str,
    class_label: int | None,
    seed: int | None,
) -> dict[str, Any]:
    shape = (
        1,
        cfg.data.in_channels,
        cfg.data.frames,
        cfg.data.image_size,
        cfg.data.image_size,
    )

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        psi = torch.randn(shape, generator=g, device=device)
    else:
        psi = torch.randn(shape, device=device)

    labels = None
    if class_label is not None:
        labels = torch.full((shape[0],), int(class_label), dtype=torch.long, device=device)

    transform = Haar3DTransform(levels=cfg.transform.levels)
    meta = transform.band_meta(device=device)
    ks = meta.ks.to(device)
    kt = meta.kt.to(device)

    taus = torch.linspace(0.0, 1.0, steps + 1, device=device)
    rows: list[dict[str, float]] = []

    m_t_list: list[float] = []
    ratio_list: list[float] = []

    for i in range(steps + 1):
        t0 = taus[i]
        tau_vec0 = torch.full((shape[0],), float(t0.item()), device=device)

        lam, lam_dot, _state = schedule.lambda_and_derivative(tau=tau_vec0, ks=ks, kt=kt)
        lam_mean = float(lam.mean().item())
        lam_p05 = _quantile(lam, 0.05)
        lam_p95 = _quantile(lam, 0.95)

        lam_dot_abs = lam_dot.abs()
        lam_dot_abs_mean = float(lam_dot_abs.mean().item())
        lam_dot_abs_p95 = _quantile(lam_dot_abs, 0.95)
        lam_dot_abs_max = float(lam_dot_abs.max().item())

        phase0 = schedule.phase_features_from_tau(tau=tau_vec0, ks=ks, kt=kt)
        v0 = model(psi, tau=tau_vec0, class_labels=labels, phase_features=phase0)

        psi_rms = _rms(psi)
        v_rms = _rms(v0)
        ratio = float(v_rms / max(psi_rms, 1e-12))

        rows.append(
            {
                "tau": float(t0.item()),
                "lambda_mean": lam_mean,
                "lambda_p05": lam_p05,
                "lambda_p95": lam_p95,
                "lambda_dot_abs_mean": lam_dot_abs_mean,
                "lambda_dot_abs_p95": lam_dot_abs_p95,
                "lambda_dot_abs_max": lam_dot_abs_max,
                "psi_rms": psi_rms,
                "v_rms": v_rms,
                "v_over_psi": ratio,
            }
        )

        m_t_list.append(lam_dot_abs_mean)
        ratio_list.append(ratio)

        if i == steps:
            break

        t1 = taus[i + 1]
        dt = t1 - t0
        if solver.lower() == "heun":
            psi_pred = psi + dt * v0
            tau_vec1 = torch.full((shape[0],), float(t1.item()), device=device)
            phase1 = schedule.phase_features_from_tau(tau=tau_vec1, ks=ks, kt=kt)
            v1 = model(psi_pred, tau=tau_vec1, class_labels=labels, phase_features=phase1)
            psi = psi + 0.5 * dt * (v0 + v1)
        else:
            psi = psi + dt * v0

    m_t = torch.tensor(m_t_list, dtype=torch.float32)
    ratio_t = torch.tensor(ratio_list, dtype=torch.float32)

    active_frac_02 = float((m_t > 0.2).float().mean().item())
    active_frac_05 = float((m_t > 0.5).float().mean().item())
    entropy_norm = _entropy_normalized(m_t)

    ratio_mean = float(ratio_t.mean().item())
    ratio_p90 = _quantile(ratio_t, 0.90)
    ratio_max = float(ratio_t.max().item())

    lamdot_mean = float(m_t.mean().item())
    lamdot_p90 = _quantile(m_t, 0.90)
    lamdot_max = float(m_t.max().item())

    collapse_rules = {
        "rule_ratio_mean_lt_0.02": ratio_mean < 0.02,
        "rule_ratio_p90_lt_0.05": ratio_p90 < 0.05,
        "rule_lamdot_entropy_lt_0.30": entropy_norm < 0.30,
        "rule_lamdot_active_frac05_lt_0.10": active_frac_05 < 0.10,
    }
    collapse_score = int(sum(int(v) for v in collapse_rules.values()))
    likely_collapse = bool(collapse_score >= 2)

    summary = {
        "steps": int(steps),
        "solver": str(solver),
        "ratio_mean": ratio_mean,
        "ratio_p90": ratio_p90,
        "ratio_max": ratio_max,
        "lamdot_mean": lamdot_mean,
        "lamdot_p90": lamdot_p90,
        "lamdot_max": lamdot_max,
        "lamdot_entropy_normalized": entropy_norm,
        "lamdot_active_fraction_gt_0.2": active_frac_02,
        "lamdot_active_fraction_gt_0.5": active_frac_05,
        "likely_transport_collapse": likely_collapse,
        "collapse_score": collapse_score,
        "collapse_rules": collapse_rules,
    }

    return {
        "summary": summary,
        "rows": rows,
    }


def _print_table(rows: list[dict[str, float]], max_rows: int = 1000) -> None:
    show = rows[:max_rows]
    header = (
        "idx  tau     lambda_mean  |lambda_dot|_mean  |lambda_dot|_p95   "
        "psi_rms   v_rms    ||v||/||psi||"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(show):
        print(
            f"{i:>3d}  {r['tau']:.3f}   {r['lambda_mean']:.4f}       "
            f"{r['lambda_dot_abs_mean']:.4f}            {r['lambda_dot_abs_p95']:.4f}        "
            f"{r['psi_rms']:.4f}  {r['v_rms']:.4f}   {r['v_over_psi']:.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose schedule/velocity transport strength in G-STVK-Flow")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    p.add_argument("--class-label", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-json", type=Path, default=Path("outputs/diagnose_transport.json"))
    p.add_argument("--out-csv", type=Path, default=Path("outputs/diagnose_transport.csv"))
    p.add_argument("--max-print-rows", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, schedule, cfg = _load_model_and_schedule(args.checkpoint, args.config, device)

    if args.class_label is not None and int(getattr(model, "num_classes", 0)) <= 0:
        raise ValueError("Checkpoint is unconditional (no class embedding), but --class-label was provided.")

    steps = int(args.steps if args.steps is not None else cfg.inference.steps)
    solver = str(args.solver if args.solver is not None else cfg.inference.solver)

    payload = diagnose_transport(
        model=model,
        schedule=schedule,
        cfg=cfg,
        device=device,
        steps=steps,
        solver=solver,
        class_label=args.class_label,
        seed=args.seed,
    )

    _print_table(payload["rows"], max_rows=max(1, int(args.max_print_rows)))
    print("\nSummary:")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=True))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    cols = [
        "tau",
        "lambda_mean",
        "lambda_p05",
        "lambda_p95",
        "lambda_dot_abs_mean",
        "lambda_dot_abs_p95",
        "lambda_dot_abs_max",
        "psi_rms",
        "v_rms",
        "v_over_psi",
    ]
    lines = [",".join(cols)]
    for r in payload["rows"]:
        lines.append(
            ",".join(
                [
                    f"{float(r[c]):.8f}" for c in cols
                ]
            )
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV : {args.out_csv}")


if __name__ == "__main__":
    main()


