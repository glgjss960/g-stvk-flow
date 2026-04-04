from __future__ import annotations

import torch

_SQRT2 = 2.0 ** 0.5


def _split_axis(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[dim]
    if n % 2 != 0:
        raise ValueError(f"Axis {dim} with size {n} must be even for Haar split.")

    even_idx = torch.arange(0, n, 2, device=x.device)
    odd_idx = even_idx + 1
    even = x.index_select(dim, even_idx)
    odd = x.index_select(dim, odd_idx)
    low = (even + odd) / _SQRT2
    high = (even - odd) / _SQRT2
    return low, high


def _merge_axis(low: torch.Tensor, high: torch.Tensor, dim: int) -> torch.Tensor:
    even = (low + high) / _SQRT2
    odd = (low - high) / _SQRT2

    out_shape = list(low.shape)
    out_shape[dim] *= 2
    out = torch.empty(out_shape, device=low.device, dtype=low.dtype)

    even_slice = [slice(None)] * out.ndim
    odd_slice = [slice(None)] * out.ndim
    even_slice[dim] = slice(0, None, 2)
    odd_slice[dim] = slice(1, None, 2)

    out[tuple(even_slice)] = even
    out[tuple(odd_slice)] = odd
    return out


def forward_spatial_haar(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        video: [B,C,T,H,W]

    Returns:
        ll, lh, hl, hh: each [B,C,T,H/2,W/2]
    """
    if video.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")

    low_h, high_h = _split_axis(video, dim=3)

    ll, lh = _split_axis(low_h, dim=4)
    hl, hh = _split_axis(high_h, dim=4)
    return ll, lh, hl, hh


def inverse_spatial_haar(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    low_h = _merge_axis(ll, lh, dim=4)
    high_h = _merge_axis(hl, hh, dim=4)
    return _merge_axis(low_h, high_h, dim=3)
