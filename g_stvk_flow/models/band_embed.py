from __future__ import annotations

import torch
import torch.nn as nn


class BandEmbedding(nn.Module):
    def __init__(self, num_bands: int, dim: int) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {num_bands}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        self.num_bands = int(num_bands)
        self.dim = int(dim)
        self.embedding = nn.Embedding(self.num_bands, self.dim)

    def forward(self, band_id: torch.Tensor) -> torch.Tensor:
        if band_id.ndim == 0:
            band_id = band_id[None]
        if band_id.dtype != torch.long:
            band_id = band_id.long()
        return self.embedding(band_id)
