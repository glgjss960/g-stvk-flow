from __future__ import annotations

import torch
import torch.nn as nn

from g_stvk_flow.backbone.embeddings import ScalarEmbedding
from g_stvk_flow.backbone.video_dit import OpenSoraStyleDiTBackbone
from g_stvk_flow.models.band_embed import BandEmbedding


class StageABandVideoModel(nn.Module):
    """
    Lightweight phase-A model:
    - Open-Sora-style Video DiT backbone
    - timestep embedding + optional discrete band embedding
    - optional class embedding
    """

    def __init__(
        self,
        in_channels: int,
        num_bands: int,
        cond_dim: int = 256,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size_t: int = 1,
        patch_size_h: int = 2,
        patch_size_w: int = 2,
        num_classes: int = 0,
        dropout: float = 0.0,
        use_band_embed: bool = True,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.use_band_embed = bool(use_band_embed)

        self.time_embed = ScalarEmbedding(embed_dim=self.cond_dim)
        self.band_embed = BandEmbedding(num_bands=int(num_bands), dim=self.cond_dim)
        self.class_embed = nn.Embedding(int(num_classes), self.cond_dim) if int(num_classes) > 0 else None

        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.backbone = OpenSoraStyleDiTBackbone(
            in_channels=int(in_channels),
            out_channels=int(in_channels),
            hidden_size=int(hidden_size),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            patch_size=(int(patch_size_t), int(patch_size_h), int(patch_size_w)),
            dropout=float(dropout),
            grad_checkpoint=bool(grad_checkpoint),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        band_id: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = x.shape[0]
        if t.ndim == 0:
            t = t.repeat(bsz)
        elif t.ndim == 1 and t.shape[0] == 1 and bsz > 1:
            t = t.repeat(bsz)

        cond = self.time_embed(t)
        if self.use_band_embed:
            cond = cond + self.band_embed(band_id)

        if self.class_embed is not None:
            if class_labels is None:
                class_labels = torch.zeros((bsz,), dtype=torch.long, device=x.device)
            cond = cond + self.class_embed(class_labels.long())

        cond = self.cond_proj(cond)
        return self.backbone(x=x, cond=cond)
