from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .band_defs import FOUR_BAND_NAMES, SPATIAL_ONLY_BANDS, TEMPORAL_ONLY_BANDS, VANILLA_BAND
from .haar1d_time import forward_temporal_haar, inverse_temporal_haar
from .haar2d import forward_spatial_haar, inverse_spatial_haar


class SeparableHaarVideoDecomposer(nn.Module):
    """
    Phase-A separable Haar decomposition for latent videos.

    Modes:
    - spatial_temporal: 2D spatial + 1D temporal -> four semantic bands
        ls_lt, ls_ht, hs_lt, hs_ht
    - spatial_only: 2D spatial only -> ls, hs
    - temporal_only: 1D temporal only -> lt, ht
    - none: no decomposition, single band "full"
    """

    def __init__(self, mode: str = "spatial_temporal") -> None:
        super().__init__()
        mode = str(mode).lower().strip()
        valid = {"spatial_temporal", "spatial_only", "temporal_only", "none"}
        if mode not in valid:
            raise ValueError(f"Unsupported mode={mode}, expected one of {sorted(valid)}")
        self.mode = mode

    @property
    def band_names(self) -> tuple[str, ...]:
        if self.mode == "spatial_temporal":
            return FOUR_BAND_NAMES
        if self.mode == "spatial_only":
            return SPATIAL_ONLY_BANDS
        if self.mode == "temporal_only":
            return TEMPORAL_ONLY_BANDS
        return VANILLA_BAND

    @staticmethod
    def _split_high_spatial(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[1] % 3 != 0:
            raise ValueError(
                "High-spatial tensor must have channel count divisible by 3, "
                f"got C={x.shape[1]}"
            )
        c = x.shape[1] // 3
        return x[:, 0:c], x[:, c : 2 * c], x[:, 2 * c : 3 * c]

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 5:
            raise ValueError(f"Expected latent [B,C,T,H,W], got {tuple(z.shape)}")

        if self.mode == "none":
            return {"full": z}

        if self.mode == "temporal_only":
            lt, ht = forward_temporal_haar(z)
            return {"lt": lt, "ht": ht}

        ll, lh, hl, hh = forward_spatial_haar(z)
        if self.mode == "spatial_only":
            hs = torch.cat([lh, hl, hh], dim=1)
            return {"ls": ll, "hs": hs}

        ll_lt, ll_ht = forward_temporal_haar(ll)
        lh_lt, lh_ht = forward_temporal_haar(lh)
        hl_lt, hl_ht = forward_temporal_haar(hl)
        hh_lt, hh_ht = forward_temporal_haar(hh)

        hs_lt = torch.cat([lh_lt, hl_lt, hh_lt], dim=1)
        hs_ht = torch.cat([lh_ht, hl_ht, hh_ht], dim=1)

        return {
            "ls_lt": ll_lt,
            "ls_ht": ll_ht,
            "hs_lt": hs_lt,
            "hs_ht": hs_ht,
        }

    def inverse(self, bands: Dict[str, torch.Tensor]) -> torch.Tensor:
        expected = set(self.band_names)
        keys = set(bands.keys())
        if keys != expected:
            raise KeyError(f"Band keys mismatch, expected={sorted(expected)} got={sorted(keys)}")

        if self.mode == "none":
            return bands["full"]

        if self.mode == "temporal_only":
            return inverse_temporal_haar(bands["lt"], bands["ht"])

        if self.mode == "spatial_only":
            lh, hl, hh = self._split_high_spatial(bands["hs"])
            return inverse_spatial_haar(bands["ls"], lh, hl, hh)

        lh_lt, hl_lt, hh_lt = self._split_high_spatial(bands["hs_lt"])
        lh_ht, hl_ht, hh_ht = self._split_high_spatial(bands["hs_ht"])

        ll = inverse_temporal_haar(bands["ls_lt"], bands["ls_ht"])
        lh = inverse_temporal_haar(lh_lt, lh_ht)
        hl = inverse_temporal_haar(hl_lt, hl_ht)
        hh = inverse_temporal_haar(hh_lt, hh_ht)

        return inverse_spatial_haar(ll, lh, hl, hh)
