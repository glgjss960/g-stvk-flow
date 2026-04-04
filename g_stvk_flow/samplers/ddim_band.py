from __future__ import annotations

import sys
from pathlib import Path

import torch

from g_stvk_flow.kflow.path_scheduler import FixedBandPath


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_flow_matching_import(flow_matching_root: str | Path | None) -> None:
    root = Path(flow_matching_root) if flow_matching_root is not None else (_repo_root() / "flow_matching")
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"flow_matching root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


class _SchedulerFactory:
    def __init__(self, scheduler_name: str, flow_matching_root: str | Path | None) -> None:
        _ensure_flow_matching_import(flow_matching_root)
        from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, LinearVPScheduler, VPScheduler

        name = str(scheduler_name).lower().strip()
        if name == "condot":
            self.scheduler = CondOTScheduler()
        elif name == "cosine":
            self.scheduler = CosineScheduler()
        elif name == "linear_vp":
            self.scheduler = LinearVPScheduler()
        elif name == "vp":
            self.scheduler = VPScheduler()
        else:
            raise ValueError(f"Unknown scheduler_name={scheduler_name}")


class StageABandSampler:
    """
    Fixed-path phase-A sampler.

    For epsilon / clean targets: deterministic DDIM-style update in band space.
    For velocity targets: explicit Euler update in band space.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decomposer,
        fixed_path: FixedBandPath,
        target_type: str = "epsilon",
        scheduler_name: str = "cosine",
        flow_matching_root: str | Path | None = None,
    ) -> None:
        self.model = model
        self.decomposer = decomposer
        self.fixed_path = fixed_path
        self.target_type = str(target_type).lower().strip()
        if self.target_type not in {"epsilon", "velocity", "clean"}:
            raise ValueError(f"Unknown target_type={target_type}")

        self.scheduler = _SchedulerFactory(
            scheduler_name=scheduler_name,
            flow_matching_root=flow_matching_root,
        ).scheduler

    def _scheduler_values(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.scheduler(t)
        return out.alpha_t, out.sigma_t

    def sample(
        self,
        shape: tuple[int, int, int, int, int],
        steps_per_band: int,
        device: torch.device,
        class_labels: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        if steps_per_band < 2:
            raise ValueError("steps_per_band must be >= 2")

        bsz = int(shape[0])
        if class_labels is not None and class_labels.shape[0] != bsz:
            raise ValueError(f"class_labels batch mismatch: expected {bsz}, got {class_labels.shape[0]}")

        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            noise_latent = torch.randn(shape, generator=gen, device=device)
        else:
            noise_latent = torch.randn(shape, device=device)

        placeholder_bands = self.decomposer.forward(noise_latent)
        done_bands: dict[str, torch.Tensor] = {}
        band_to_id = {name: i for i, name in enumerate(self.fixed_path.order)}

        t_eps = max(1.0 / float(steps_per_band), 1e-3)
        t_grid = torch.linspace(t_eps, 1.0, steps_per_band, device=device)

        for band_name in self.fixed_path.order:
            current = placeholder_bands[band_name].clone()
            band_id = torch.full((bsz,), band_to_id[band_name], dtype=torch.long, device=device)

            for i in range(steps_per_band - 1):
                t_now = t_grid[i]
                t_next = t_grid[i + 1]
                dt = t_next - t_now

                state_bands: dict[str, torch.Tensor] = {}
                for name in self.fixed_path.order:
                    if name in done_bands:
                        state_bands[name] = done_bands[name]
                    elif name == band_name:
                        state_bands[name] = current
                    else:
                        state_bands[name] = placeholder_bands[name]

                x_t = self.decomposer.inverse(state_bands)
                t_batch = torch.full((bsz,), float(t_now.item()), device=device, dtype=x_t.dtype)
                pred_latent = self.model(x=x_t, t=t_batch, band_id=band_id, class_labels=class_labels)
                pred_band = self.decomposer.forward(pred_latent)[band_name]

                if self.target_type == "velocity":
                    current = current + dt * pred_band
                    continue

                alpha_t, sigma_t = self._scheduler_values(t_batch)
                alpha_next, sigma_next = self._scheduler_values(
                    torch.full((bsz,), float(t_next.item()), device=device, dtype=x_t.dtype)
                )

                view_shape = [bsz] + [1] * (current.ndim - 1)
                alpha_t = alpha_t.view(view_shape).clamp_min(1e-6)
                sigma_t = sigma_t.view(view_shape)
                alpha_next = alpha_next.view(view_shape)
                sigma_next = sigma_next.view(view_shape)

                if self.target_type == "epsilon":
                    clean_hat = (current - sigma_t * pred_band) / alpha_t
                else:
                    clean_hat = pred_band

                if self.target_type == "clean":
                    noise_hat = (current - alpha_t * clean_hat) / sigma_t.clamp_min(1e-6)
                    current = sigma_next * noise_hat + alpha_next * clean_hat
                else:
                    current = sigma_next * pred_band + alpha_next * clean_hat

            done_bands[band_name] = current

        return self.decomposer.inverse(done_bands)
