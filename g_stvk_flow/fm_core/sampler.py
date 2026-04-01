from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from g_stvk_flow.fm_core.bootstrap import bootstrap_flow_matching_path
from g_stvk_flow.gstvk import SAASchedule

bootstrap_flow_matching_path()

try:
    from flow_matching.solver.ode_solver import ODESolver as _FMODESolver
except Exception:
    _FMODESolver = None

try:
    from flow_matching.utils import ModelWrapper as _FMModelWrapper
except Exception:
    _FMModelWrapper = None


class _FallbackModelWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


_BaseModelWrapper = _FMModelWrapper if _FMModelWrapper is not None else _FallbackModelWrapper


class GSTVKVelocityWrapper(_BaseModelWrapper):
    """
    Wraps a velocity model into the `(x, t, **extras)` interface used by flow_matching
    solvers while injecting g-STVK phase features from scheduler metadata.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: SAASchedule,
        ks: torch.Tensor,
        kt: torch.Tensor,
    ) -> None:
        super().__init__(model)
        self.schedule = schedule
        self.register_buffer("ks", ks.detach().clone(), persistent=False)
        self.register_buffer("kt", kt.detach().clone(), persistent=False)

    @staticmethod
    def _expand_time(t: torch.Tensor | float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(t, float):
            return torch.full((batch,), float(t), device=device, dtype=dtype)
        if t.ndim == 0:
            return t.to(device=device, dtype=dtype).repeat(batch)
        if t.ndim == 1 and t.shape[0] == 1:
            return t.to(device=device, dtype=dtype).repeat(batch)
        if t.ndim == 1 and t.shape[0] == batch:
            return t.to(device=device, dtype=dtype)
        raise ValueError(f"Unsupported time shape {tuple(t.shape)} for batch={batch}")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        class_labels: Optional[torch.Tensor] = None,
        phase_features: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        tau = self._expand_time(t=t, batch=x.shape[0], device=x.device, dtype=x.dtype)

        if phase_features is None:
            phase_features = self.schedule.phase_features_from_tau(
                tau=tau,
                ks=self.ks.to(device=x.device, dtype=x.dtype),
                kt=self.kt.to(device=x.device, dtype=x.dtype),
            )
        else:
            phase_features = phase_features.to(device=x.device, dtype=x.dtype)

        labels = None
        if class_labels is not None:
            labels = class_labels.to(device=x.device)

        return self.model(x=x, tau=tau, class_labels=labels, phase_features=phase_features)


class GSTVKSolver(nn.Module):
    """
    Flow-matching style ODE sampler abstraction.

    If official flow_matching ODESolver is available (and torchdiffeq installed),
    it can be used for non-Heun methods; otherwise a local explicit integrator is used.
    """

    def __init__(self, velocity_model: GSTVKVelocityWrapper) -> None:
        super().__init__()
        self.velocity_model = velocity_model
        self._official_solver = _FMODESolver(velocity_model=velocity_model) if _FMODESolver is not None else None

    def _explicit_sample(
        self,
        x_init: torch.Tensor,
        time_grid: torch.Tensor,
        method: str,
        return_intermediates: bool,
        class_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        method_l = method.lower()
        if method_l not in {"euler", "heun"}:
            raise ValueError(f"Unsupported explicit solver method: {method}")

        x = x_init
        intermediates = [x] if return_intermediates else None

        for i in range(time_grid.shape[0] - 1):
            t0 = time_grid[i]
            t1 = time_grid[i + 1]
            dt = t1 - t0

            v0 = self.velocity_model(x=x, t=t0, class_labels=class_labels)
            if method_l == "heun":
                x_pred = x + dt * v0
                v1 = self.velocity_model(x=x_pred, t=t1, class_labels=class_labels)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * v0

            if intermediates is not None:
                intermediates.append(x)

        if intermediates is None:
            return x
        return torch.stack(intermediates, dim=0)

    def sample(
        self,
        x_init: torch.Tensor,
        time_grid: torch.Tensor,
        method: str = "heun",
        return_intermediates: bool = False,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if time_grid.ndim != 1 or time_grid.numel() < 2:
            raise ValueError(f"time_grid must be 1D with >=2 entries, got {tuple(time_grid.shape)}")

        # Heun is not part of torchdiffeq's canonical method list used by flow_matching,
        # so we keep a local explicit implementation for exact parity with existing scripts.
        if self._official_solver is not None and method.lower() != "heun":
            return self._official_solver.sample(
                x_init=x_init,
                step_size=None,
                method=method,
                time_grid=time_grid,
                return_intermediates=return_intermediates,
                class_labels=class_labels,
            )

        return self._explicit_sample(
            x_init=x_init,
            time_grid=time_grid,
            method=method,
            return_intermediates=return_intermediates,
            class_labels=class_labels,
        )

