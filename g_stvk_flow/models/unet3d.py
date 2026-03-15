from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F



def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class FiLMResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))

        return h + self.skip(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_mults: List[int],
        num_res_blocks: int,
        cond_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channel_mults) < 2:
            raise ValueError("channel_mults must have at least 2 stages")

        stage_channels = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv3d(in_channels, stage_channels[0], kernel_size=3, padding=1)

        self.down_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = stage_channels[0]
        for i, out_ch in enumerate(stage_channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(FiLMResBlock3D(in_ch, out_ch, cond_dim, dropout=dropout))
                in_ch = out_ch
            self.down_stages.append(blocks)

            if i < len(stage_channels) - 1:
                self.downsamples.append(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=(1, 2, 2), padding=1))
            else:
                self.downsamples.append(nn.Identity())

        bottleneck_ch = stage_channels[-1]
        self.mid1 = FiLMResBlock3D(bottleneck_ch, bottleneck_ch, cond_dim, dropout=dropout)
        self.mid2 = FiLMResBlock3D(bottleneck_ch, bottleneck_ch, cond_dim, dropout=dropout)

        self.up_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        rev_stage_channels = list(reversed(stage_channels))
        cur_ch = rev_stage_channels[0]
        for i, skip_ch in enumerate(rev_stage_channels):
            blocks = nn.ModuleList()
            merge_in = cur_ch + skip_ch
            blocks.append(FiLMResBlock3D(merge_in, skip_ch, cond_dim, dropout=dropout))
            for _ in range(num_res_blocks - 1):
                blocks.append(FiLMResBlock3D(skip_ch, skip_ch, cond_dim, dropout=dropout))
            self.up_stages.append(blocks)
            cur_ch = skip_ch

            if i < len(rev_stage_channels) - 1:
                next_ch = rev_stage_channels[i + 1]
                self.upsamples.append(
                    nn.ConvTranspose3d(cur_ch, next_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
                )
                cur_ch = next_ch
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = _group_norm(stage_channels[0])
        self.out_conv = nn.Conv3d(stage_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        skips = []

        for blocks, down in zip(self.down_stages, self.downsamples):
            for block in blocks:
                x = block(x, cond)
            skips.append(x)
            x = down(x)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        for blocks, up in zip(self.up_stages, self.upsamples):
            skip = skips.pop()
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, cond)
            x = up(x)

        x = F.silu(self.out_norm(x))
        return self.out_conv(x)
