from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _build_1d_sincos(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros(length, 0, device=device, dtype=dtype)

    half = dim // 2
    if half == 0:
        return torch.zeros(length, dim, device=device, dtype=dtype)

    positions = torch.arange(length, device=device, dtype=torch.float32)
    inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1))
    sinusoid = positions[:, None] * inv_freq[None, :]
    emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)
    if emb.shape[1] < dim:
        emb = torch.cat([emb, torch.zeros(length, dim - emb.shape[1], device=device, dtype=torch.float32)], dim=-1)
    return emb[:, :dim].to(dtype=dtype)


def _build_2d_sincos(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dim_h = dim // 2
    dim_w = dim - dim_h

    emb_h = _build_1d_sincos(height, dim_h, device=device, dtype=dtype)  # [H, Dh]
    emb_w = _build_1d_sincos(width, dim_w, device=device, dtype=dtype)   # [W, Dw]

    emb_h = emb_h[:, None, :].expand(height, width, dim_h)
    emb_w = emb_w[None, :, :].expand(height, width, dim_w)
    emb = torch.cat([emb_h, emb_w], dim=-1)
    return emb.reshape(height * width, dim)


class AdaLNDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond).chunk(6, dim=-1)

        attn_in = _modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out

        mlp_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class AdaLNFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


class OpenSoraStyleDiTBackbone(nn.Module):
    """
    Lightweight Open-Sora/DiT-style video backbone with AdaLN conditioning.

    Inputs:
      - x:    [B,C,T,H,W]
      - cond: [B,D]
    Output:
      - v:    [B,C,T,H,W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values (t,h,w), got {patch_size}")

        pt, ph, pw = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
        if pt <= 0 or ph <= 0 or pw <= 0:
            raise ValueError(f"Invalid patch_size={patch_size}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_size = int(hidden_size)
        self.patch_size = (pt, ph, pw)

        self.patch_embed = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNDiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        patch_volume = pt * ph * pw
        self.final = AdaLNFinalLayer(hidden_size=self.hidden_size, out_dim=self.out_channels * patch_volume)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0.0)

        for block in self.blocks:
            nn.init.constant_(block.ada_ln[-1].weight, 0.0)
            nn.init.constant_(block.ada_ln[-1].bias, 0.0)

        nn.init.constant_(self.final.ada_ln[-1].weight, 0.0)
        nn.init.constant_(self.final.ada_ln[-1].bias, 0.0)
        nn.init.constant_(self.final.proj.weight, 0.0)
        nn.init.constant_(self.final.proj.bias, 0.0)

    def _add_positional_embeddings(self, tokens: torch.Tensor, tp: int, hp: int, wp: int) -> torch.Tensor:
        bsz, n_tokens, dim = tokens.shape
        if n_tokens != tp * hp * wp:
            raise ValueError(
                f"Token shape mismatch: got n={n_tokens}, expected tp*hp*wp={tp * hp * wp}"
            )

        spatial = _build_2d_sincos(height=hp, width=wp, dim=dim, device=tokens.device, dtype=tokens.dtype)
        temporal = _build_1d_sincos(length=tp, dim=dim, device=tokens.device, dtype=tokens.dtype)

        x = tokens.view(bsz, tp, hp * wp, dim)
        x = x + spatial.unsqueeze(0).unsqueeze(1)
        x = x.permute(0, 2, 1, 3)
        x = x + temporal.unsqueeze(0).unsqueeze(1)
        return x.permute(0, 2, 1, 3).reshape(bsz, n_tokens, dim)

    def _unpatchify(self, tokens: torch.Tensor, tp: int, hp: int, wp: int) -> torch.Tensor:
        bsz = tokens.shape[0]
        pt, ph, pw = self.patch_size

        x = tokens.view(bsz, tp, hp, wp, pt, ph, pw, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(bsz, self.out_channels, tp * pt, hp * ph, wp * pw)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")

        pt, ph, pw = self.patch_size
        _, _, t, h, w = x.shape
        if (t % pt) != 0 or (h % ph) != 0 or (w % pw) != 0:
            raise ValueError(
                f"Input shape {(t, h, w)} must be divisible by patch_size={self.patch_size}"
            )

        feat = self.patch_embed(x)
        tp, hp, wp = feat.shape[2], feat.shape[3], feat.shape[4]

        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B,N,D]
        tokens = self._add_positional_embeddings(tokens=tokens, tp=tp, hp=hp, wp=wp)

        for block in self.blocks:
            tokens = block(tokens, cond)

        tokens = self.final(tokens, cond)
        out = self._unpatchify(tokens=tokens, tp=tp, hp=hp, wp=wp)
        return out
