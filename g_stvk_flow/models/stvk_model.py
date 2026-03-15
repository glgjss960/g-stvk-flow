from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import ScalarEmbedding, VectorEmbedding
from .unet3d import UNet3D


class STVKFlowModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        cond_dim: int,
        phase_dim: int = 11,
        num_classes: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.phase_dim = phase_dim
        self.num_classes = num_classes

        self.t_embed = ScalarEmbedding(cond_dim)
        self.phase_embed = VectorEmbedding(phase_dim, cond_dim)
        self.class_embed = nn.Embedding(num_classes, cond_dim) if num_classes > 0 else None

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.backbone = UNet3D(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            cond_dim=cond_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        phase_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = x.shape[0]
        if phase_features is None:
            phase_features = torch.zeros(bsz, self.phase_dim, device=x.device, dtype=tau.dtype)

        cond = self.t_embed(tau) + self.phase_embed(phase_features)

        if self.class_embed is not None:
            if class_labels is None:
                class_labels = torch.zeros(bsz, dtype=torch.long, device=x.device)
            cond = cond + self.class_embed(class_labels)

        cond = self.cond_proj(cond)
        return self.backbone(x, cond)

