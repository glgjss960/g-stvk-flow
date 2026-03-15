from __future__ import annotations

from dataclasses import dataclass

import torch

from .haar3d import Haar3DTransform
from .saa_path import GeometricPathScheduler


@dataclass
class InterpolantOutput:
    psi_tau: torch.Tensor
    v_target: torch.Tensor
    phase_features: torch.Tensor


class GSTVKInterpolant:
    def __init__(
        self,
        transform: Haar3DTransform,
        schedule: GeometricPathScheduler,
        whiten_eps: float = 1e-5,
    ) -> None:
        self.transform = transform
        self.schedule = schedule
        self.whiten_eps = float(whiten_eps)

    def _band_std(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.ndim))
        var = x.detach().pow(2).mean(dim=dims, keepdim=True)
        return (var + self.whiten_eps).sqrt()

    def build(self, x_data: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor) -> InterpolantOutput:
        z_data, band_meta = self.transform.forward(x_data)
        z_noise, _ = self.transform.forward(eps)

        lam, lam_dot, state = self.schedule.lambda_and_derivative(
            tau=tau,
            ks=band_meta.ks,
            kt=band_meta.kt,
        )

        flat_data = self.transform.flatten(z_data)
        flat_noise = self.transform.flatten(z_noise)

        flat_tau = []
        flat_u = []
        for i, (band_data, band_noise) in enumerate(zip(flat_data, flat_noise)):
            view_shape = [band_data.shape[0]] + [1] * (band_data.ndim - 1)
            lam_i = lam[:, i].to(device=band_data.device, dtype=band_data.dtype).view(view_shape)
            lam_dot_i = lam_dot[:, i].to(device=band_data.device, dtype=band_data.dtype).view(view_shape)

            std_data = self._band_std(band_data)
            std_noise = self._band_std(band_noise)

            x_hat = band_data / std_data
            n_hat = band_noise / std_noise

            # Variance-preserving harmonic bridge in whitened coordinates.
            a = torch.sin(0.5 * torch.pi * lam_i)
            b = torch.cos(0.5 * torch.pi * lam_i)

            z_hat = a * x_hat + b * n_hat
            u_hat = lam_dot_i * (0.5 * torch.pi) * (b * x_hat - a * n_hat)

            z_tau_i = std_data * z_hat
            u_i = std_data * u_hat

            flat_tau.append(z_tau_i)
            flat_u.append(u_i)

        z_tau = self.transform.unflatten_like(z_data, flat_tau)
        u_hat = self.transform.unflatten_like(z_data, flat_u)

        psi_tau = self.transform.inverse(z_tau)
        v_target = self.transform.inverse(u_hat)

        phase_features = self.schedule.phase_features(
            lambda_k=lam,
            lambda_dot=lam_dot,
            state=state,
            ks=band_meta.ks,
            kt=band_meta.kt,
        )

        return InterpolantOutput(
            psi_tau=psi_tau,
            v_target=v_target,
            phase_features=phase_features,
        )


# Backward-compatible alias.
STVKInterpolant = GSTVKInterpolant
