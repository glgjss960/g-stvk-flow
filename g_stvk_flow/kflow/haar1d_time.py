from __future__ import annotations

import torch

_SQRT2 = 2.0 ** 0.5


def forward_temporal_haar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: [B,C,T,H,W]

    Returns:
        low_t, high_t: [B,C,T/2,H,W]
    """
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")

    t = x.shape[2]
    if t % 2 != 0:
        raise ValueError(f"Temporal length {t} must be even for Haar split.")

    even_idx = torch.arange(0, t, 2, device=x.device)
    odd_idx = even_idx + 1
    even = x.index_select(2, even_idx)
    odd = x.index_select(2, odd_idx)
    low = (even + odd) / _SQRT2
    high = (even - odd) / _SQRT2
    return low, high


def inverse_temporal_haar(low_t: torch.Tensor, high_t: torch.Tensor) -> torch.Tensor:
    even = (low_t + high_t) / _SQRT2
    odd = (low_t - high_t) / _SQRT2

    out_shape = list(low_t.shape)
    out_shape[2] *= 2
    out = torch.empty(out_shape, device=low_t.device, dtype=low_t.dtype)

    out[:, :, 0::2] = even
    out[:, :, 1::2] = odd
    return out
