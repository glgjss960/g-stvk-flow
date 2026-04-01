from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import ScalarEmbedding, VectorEmbedding
from .video_dit import OpenSoraStyleDiTBackbone
from .unet3d import UNet3D


class STVKFlowModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 96,
        channel_mults: list[int] | None = None,
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        phase_dim: int = 11,
        num_classes: int = 0,
        dropout: float = 0.0,
        backbone: str = "opensora_dit",
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size_t: int = 1,
        patch_size_h: int = 2,
        patch_size_w: int = 2,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.phase_dim = int(phase_dim)
        self.num_classes = int(num_classes)
        self.backbone_type = str(backbone).lower()

        self.t_embed = ScalarEmbedding(self.cond_dim)
        self.phase_embed = VectorEmbedding(self.phase_dim, self.cond_dim)
        self.class_embed = nn.Embedding(self.num_classes, self.cond_dim) if self.num_classes > 0 else None

        if channel_mults is None:
            channel_mults = [1, 2, 4]

        if self.backbone_type in {"unet", "unet3d"}:
            backbone_cond_dim = self.cond_dim
            self.backbone = UNet3D(
                in_channels=in_channels,
                out_channels=in_channels,
                base_channels=base_channels,
                channel_mults=channel_mults,
                num_res_blocks=num_res_blocks,
                cond_dim=backbone_cond_dim,
                dropout=dropout,
            )
        else:
            backbone_cond_dim = int(hidden_size)
            self.backbone = OpenSoraStyleDiTBackbone(
                in_channels=in_channels,
                out_channels=in_channels,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                patch_size=(patch_size_t, patch_size_h, patch_size_w),
                dropout=dropout,
            )

        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, backbone_cond_dim),
            nn.SiLU(),
            nn.Linear(backbone_cond_dim, backbone_cond_dim),
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

        if self.backbone_type in {"unet", "unet3d"}:
            return self.backbone(x, cond)
        return self.backbone(x, cond)


