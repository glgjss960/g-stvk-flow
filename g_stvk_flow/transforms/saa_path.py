from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeometricPathState:
    gamma_s: torch.Tensor
    gamma_t: torch.Tensor
    radius: torch.Tensor
    a_s: torch.Tensor
    a_t: torch.Tensor
    delta: torch.Tensor
    distance: torch.Tensor
    score: torch.Tensor


class GeometricPathScheduler(nn.Module):
    """
    Learnable geometric path on the (k_s, k_t) plane.

    The front is defined by a moving anisotropic ball:
      score = (radius - distance) / delta
      lambda = sigmoid(score)
    """

    def __init__(
        self,
        num_knots: int,
        delta_min: float,
        delta_max: float,
        radius_min: float,
        radius_max: float,
        anisotropy_min: float,
        derivative_eps: float,
        delta_hidden_dim: int,
        spread_temperature: float,
        reg_grid_size: int,
    ) -> None:
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if not (0.0 < delta_min < delta_max):
            raise ValueError("Require 0 < delta_min < delta_max")

        self.num_knots = int(num_knots)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.anisotropy_min = float(anisotropy_min)
        self.derivative_eps = float(derivative_eps)
        self.spread_temperature = float(spread_temperature)
        self.reg_grid_size = int(reg_grid_size)

        # Monotone curves are parameterized by positive increments on knot segments.
        self.gamma_s_logits = nn.Parameter(torch.zeros(self.num_knots))
        self.gamma_t_logits = nn.Parameter(torch.zeros(self.num_knots))
        self.radius_logits = nn.Parameter(torch.zeros(self.num_knots))

        # Positive anisotropy scales.
        self.log_a_s = nn.Parameter(torch.tensor(0.0))
        self.log_a_t = nn.Parameter(torch.tensor(0.0))

        hidden = max(8, int(delta_hidden_dim))
        self.delta_head = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _monotone_curve(self, logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        increments = F.softplus(logits) + 1e-6
        values = torch.cat([
            torch.zeros(1, device=logits.device, dtype=logits.dtype),
            torch.cumsum(increments, dim=0),
        ])
        values = values / values[-1].clamp_min(1e-8)

        tau = tau.clamp(0.0, 1.0)
        pos = tau * self.num_knots
        idx0 = pos.floor().long().clamp(min=0, max=self.num_knots - 1)
        frac = (pos - idx0.to(pos.dtype)).clamp(0.0, 1.0)

        v0 = values.index_select(0, idx0)
        v1 = values.index_select(0, idx0 + 1)
        return v0 + (v1 - v0) * frac

    def _path_state(self, tau: torch.Tensor, ks: torch.Tensor, kt: torch.Tensor) -> GeometricPathState:
        bsz = tau.shape[0]
        k = ks.shape[0]

        gamma_s = self._monotone_curve(self.gamma_s_logits, tau)
        gamma_t = self._monotone_curve(self.gamma_t_logits, tau)
        radius01 = self._monotone_curve(self.radius_logits, tau)
        radius = self.radius_min + (self.radius_max - self.radius_min) * radius01

        a_s_scalar = self.anisotropy_min + F.softplus(self.log_a_s)
        a_t_scalar = self.anisotropy_min + F.softplus(self.log_a_t)
        a_s = torch.ones_like(tau) * a_s_scalar
        a_t = torch.ones_like(tau) * a_t_scalar

        ks_b = ks[None, :].to(device=tau.device, dtype=tau.dtype).expand(bsz, k)
        kt_b = kt[None, :].to(device=tau.device, dtype=tau.dtype).expand(bsz, k)

        ds = (ks_b - gamma_s[:, None]) / a_s[:, None].clamp_min(1e-6)
        dt = (kt_b - gamma_t[:, None]) / a_t[:, None].clamp_min(1e-6)
        distance = torch.sqrt(ds.square() + dt.square() + 1e-8)

        tau_b = tau[:, None].expand(bsz, k)
        feat = torch.stack([ks_b, kt_b, ks_b * kt_b, tau_b], dim=-1)
        delta_raw = self.delta_head(feat).squeeze(-1)
        delta = self.delta_min + (self.delta_max - self.delta_min) * torch.sigmoid(delta_raw)

        score = (radius[:, None] - distance) / delta.clamp_min(1e-6)
        return GeometricPathState(
            gamma_s=gamma_s,
            gamma_t=gamma_t,
            radius=radius,
            a_s=a_s,
            a_t=a_t,
            delta=delta,
            distance=distance,
            score=score,
        )

    def _lambda_only(self, tau: torch.Tensor, ks: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
        state = self._path_state(tau=tau, ks=ks, kt=kt)
        return torch.sigmoid(state.score)

    def lambda_and_derivative(
        self,
        tau: torch.Tensor,
        ks: torch.Tensor,
        kt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, GeometricPathState]:
        state = self._path_state(tau=tau, ks=ks, kt=kt)
        lam = torch.sigmoid(state.score)

        eps = max(self.derivative_eps, 1e-4)
        tau_plus = (tau + eps).clamp(0.0, 1.0)
        tau_minus = (tau - eps).clamp(0.0, 1.0)

        lam_plus = self._lambda_only(tau=tau_plus, ks=ks, kt=kt)
        lam_minus = self._lambda_only(tau=tau_minus, ks=ks, kt=kt)
        lam_dot = (lam_plus - lam_minus) / (2.0 * eps)

        return lam, lam_dot, state

    def phase_features(
        self,
        lambda_k: torch.Tensor,
        lambda_dot: torch.Tensor,
        state: GeometricPathState,
        ks: torch.Tensor,
        kt: torch.Tensor,
    ) -> torch.Tensor:
        weights = lambda_dot.abs() + 1e-8
        den = weights.sum(dim=1, keepdim=True)

        phi_ks = (weights * ks[None, :]).sum(dim=1, keepdim=True) / den
        phi_kt = (weights * kt[None, :]).sum(dim=1, keepdim=True) / den
        lam_mean = lambda_k.mean(dim=1, keepdim=True)
        transition_mass = (lambda_k * (1.0 - lambda_k)).mean(dim=1, keepdim=True)
        dot_mass = lambda_dot.abs().mean(dim=1, keepdim=True)
        delta_mean = state.delta.mean(dim=1, keepdim=True)

        return torch.cat(
            [
                phi_ks,
                phi_kt,
                lam_mean,
                transition_mass,
                dot_mass,
                state.gamma_s[:, None],
                state.gamma_t[:, None],
                state.radius[:, None],
                state.a_s[:, None],
                state.a_t[:, None],
                delta_mean,
            ],
            dim=1,
        )

    def phase_features_from_tau(self, tau: torch.Tensor, ks: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
        lam, lam_dot, state = self.lambda_and_derivative(tau=tau, ks=ks, kt=kt)
        return self.phase_features(lambda_k=lam, lambda_dot=lam_dot, state=state, ks=ks, kt=kt)

    def regularization_terms(self, ks: torch.Tensor, kt: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = max(5, self.reg_grid_size)
        tau_grid = torch.linspace(0.0, 1.0, grid, device=ks.device, dtype=ks.dtype)

        lam, lam_dot, state = self.lambda_and_derivative(tau=tau_grid, ks=ks, kt=kt)

        endpoint = lam[0].square().mean() + (1.0 - lam[-1]).square().mean()

        dt = 1.0 / float(grid - 1)
        coverage = (lam_dot.abs().sum(dim=0) * dt - 1.0).square().mean()

        temp = max(self.spread_temperature, 1e-4)
        align = torch.softmax(-state.score.abs() / temp, dim=0)
        tau_star = (align * tau_grid[:, None]).sum(dim=0)
        tau_sorted = torch.sort(tau_star).values
        k = ks.shape[0]
        target = (torch.arange(k, device=ks.device, dtype=ks.dtype) + 0.5) / float(k)
        spread = (tau_sorted - target).square().mean()

        def second_diff(x: torch.Tensor) -> torch.Tensor:
            return x[2:] - 2.0 * x[1:-1] + x[:-2]

        smooth_path = second_diff(state.gamma_s).square().mean()
        smooth_path = smooth_path + second_diff(state.gamma_t).square().mean()
        smooth_path = smooth_path + second_diff(state.radius).square().mean()
        smooth_delta = second_diff(state.delta.mean(dim=1)).square().mean()
        smooth = smooth_path + smooth_delta

        return {
            "endpoint": endpoint,
            "coverage": coverage,
            "spread": spread,
            "smooth": smooth,
        }

    def build_edit_weights(
        self,
        ks: torch.Tensor,
        kt: torch.Tensor,
        tau_anchor: float,
        kt_threshold: float,
        ks_min_replace: float,
        kt_softness: float,
        ks_softness: float,
        path_softness: float,
    ) -> torch.Tensor:
        tau = torch.tensor([float(tau_anchor)], device=ks.device, dtype=ks.dtype)
        _, _, state = self.lambda_and_derivative(tau=tau, ks=ks, kt=kt)

        dist = state.distance[0]
        radius = state.radius[0]

        kt_term = torch.sigmoid((kt - float(kt_threshold)) / max(float(kt_softness), 1e-4))
        ks_term = torch.sigmoid((ks - float(ks_min_replace)) / max(float(ks_softness), 1e-4))
        path_term = torch.sigmoid((dist - radius) / max(float(path_softness), 1e-4))

        return (kt_term * ks_term * path_term).clamp(0.0, 1.0)


# Backward-compatible alias for earlier code paths.
SAASchedule = GeometricPathScheduler

