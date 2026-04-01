from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

SQRT2 = 2.0 ** 0.5


@dataclass
class BandMeta:
    ks: torch.Tensor  # [K]
    kt: torch.Tensor  # [K]
    names: List[str]


@dataclass
class PyramidCoeffs:
    # details are ordered from coarse to fine, each tensor is [B,C,7,T,H,W]
    approx: torch.Tensor
    details: List[torch.Tensor]



def _split_axis(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[dim]
    if n % 2 != 0:
        raise ValueError(f"Dimension {dim} with size {n} must be even for Haar transform.")

    even_idx = torch.arange(0, n, 2, device=x.device)
    odd_idx = even_idx + 1
    even = x.index_select(dim, even_idx)
    odd = x.index_select(dim, odd_idx)

    low = (even + odd) / SQRT2
    high = (even - odd) / SQRT2
    return low, high



def _merge_axis(low: torch.Tensor, high: torch.Tensor, dim: int) -> torch.Tensor:
    even = (low + high) / SQRT2
    odd = (low - high) / SQRT2

    out_shape = list(low.shape)
    out_shape[dim] = out_shape[dim] * 2
    out = torch.empty(out_shape, device=low.device, dtype=low.dtype)

    even_slice = [slice(None)] * out.ndim
    odd_slice = [slice(None)] * out.ndim
    even_slice[dim] = slice(0, None, 2)
    odd_slice[dim] = slice(1, None, 2)
    out[tuple(even_slice)] = even
    out[tuple(odd_slice)] = odd
    return out


class Haar3DTransform:
    """
    Multi-level separable 3D Haar transform for video tensors [B,C,T,H,W].

    The returned coefficient pyramid follows Mallat decomposition:
    recursively decompose the low-low-low branch, keep 7 detail bands per level.
    """

    _detail_codes: Sequence[Tuple[int, int, int]] = (
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    )

    def __init__(self, levels: int = 2) -> None:
        if levels <= 0:
            raise ValueError("levels must be >= 1")
        self.levels = int(levels)
        self._ks_cpu, self._kt_cpu, self._names = self._build_meta_cpu(self.levels)

    @property
    def band_count(self) -> int:
        return 1 + 7 * self.levels

    def band_meta(self, device: torch.device) -> BandMeta:
        return BandMeta(
            ks=self._ks_cpu.to(device),
            kt=self._kt_cpu.to(device),
            names=list(self._names),
        )

    def _check_dims(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got shape={tuple(x.shape)}")

        factor = 2 ** self.levels
        t, h, w = x.shape[2], x.shape[3], x.shape[4]
        if (t % factor) or (h % factor) or (w % factor):
            raise ValueError(
                f"T/H/W must be divisible by 2**levels={factor}, got T={t}, H={h}, W={w}."
            )

    @staticmethod
    def _split_to_8(x: torch.Tensor) -> List[torch.Tensor]:
        t_low, t_high = _split_axis(x, dim=2)
        bands: List[torch.Tensor] = []
        for t_band in (t_low, t_high):
            h_low, h_high = _split_axis(t_band, dim=3)
            for h_band in (h_low, h_high):
                w_low, w_high = _split_axis(h_band, dim=4)
                bands.extend([w_low, w_high])
        return bands

    @staticmethod
    def _merge_from_8(bands: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(bands) != 8:
            raise ValueError(f"Expected 8 bands, got {len(bands)}")

        idx = 0
        t_parts = []
        for _ in range(2):
            h_parts = []
            for _ in range(2):
                w_low = bands[idx]
                w_high = bands[idx + 1]
                idx += 2
                h_parts.append(_merge_axis(w_low, w_high, dim=4))
            t_parts.append(_merge_axis(h_parts[0], h_parts[1], dim=3))

        return _merge_axis(t_parts[0], t_parts[1], dim=2)

    def _decompose_recursive(self, x: torch.Tensor, depth: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bands8 = self._split_to_8(x)
        low = bands8[0]
        detail = torch.stack(list(bands8[1:]), dim=2)  # [B,C,7,T,H,W]

        if depth == 1:
            return low, [detail]

        low_deep, details = self._decompose_recursive(low, depth - 1)
        details.append(detail)  # coarse -> fine ordering
        return low_deep, details

    def forward(self, x: torch.Tensor) -> Tuple[PyramidCoeffs, BandMeta]:
        self._check_dims(x)
        approx, details = self._decompose_recursive(x, self.levels)
        coeffs = PyramidCoeffs(approx=approx, details=details)
        return coeffs, self.band_meta(x.device)

    def inverse(self, coeffs: PyramidCoeffs) -> torch.Tensor:
        if len(coeffs.details) != self.levels:
            raise ValueError(f"Expected {self.levels} detail levels, got {len(coeffs.details)}")

        x = coeffs.approx
        for detail in coeffs.details:
            if detail.ndim != 6 or detail.shape[2] != 7:
                raise ValueError(f"Detail tensor must be [B,C,7,T,H,W], got {tuple(detail.shape)}")
            bands8 = [x] + [detail[:, :, i] for i in range(7)]
            x = self._merge_from_8(bands8)
        return x

    def flatten(self, coeffs: PyramidCoeffs) -> List[torch.Tensor]:
        flat: List[torch.Tensor] = [coeffs.approx]
        for detail in coeffs.details:
            flat.extend([detail[:, :, i] for i in range(7)])
        return flat

    def unflatten_like(self, template: PyramidCoeffs, flat: Sequence[torch.Tensor]) -> PyramidCoeffs:
        expected = 1 + 7 * len(template.details)
        if len(flat) != expected:
            raise ValueError(f"Expected {expected} flattened bands, got {len(flat)}")

        idx = 1
        details: List[torch.Tensor] = []
        for detail in template.details:
            parts = [flat[idx + i] for i in range(7)]
            idx += 7
            details.append(torch.stack(parts, dim=2).to(dtype=detail.dtype, device=detail.device))

        return PyramidCoeffs(approx=flat[0], details=details)

    @classmethod
    def _build_meta_cpu(cls, levels: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        ks: List[float] = [0.0]
        kt: List[float] = [0.0]
        names: List[str] = [f"L{levels}_LLL"]

        for order_idx in range(levels):
            wavelet_level = levels - order_idx
            if levels == 1:
                level_scale = 1.0
            else:
                level_scale = float(order_idx) / float(levels - 1)

            for code in cls._detail_codes:
                t_h, h_h, w_h = code
                spatial_orientation = 0.5 * float(h_h + w_h)
                temporal_orientation = float(t_h)

                # Blend level hierarchy with orientation so scales are ordered but direction-sensitive.
                k_s = 0.65 * level_scale + 0.35 * spatial_orientation
                k_t = 0.65 * level_scale + 0.35 * temporal_orientation

                ks.append(float(max(0.0, min(1.0, k_s))))
                kt.append(float(max(0.0, min(1.0, k_t))))
                names.append(
                    f"L{wavelet_level}_"
                    f"{'H' if t_h else 'L'}"
                    f"{'H' if h_h else 'L'}"
                    f"{'H' if w_h else 'L'}"
                )

        return torch.tensor(ks, dtype=torch.float32), torch.tensor(kt, dtype=torch.float32), names
