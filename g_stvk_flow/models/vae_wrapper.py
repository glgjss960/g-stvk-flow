from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_open_sora_path(open_sora_root: str | Path | None) -> Path:
    root = Path(open_sora_root) if open_sora_root is not None else (_repo_root() / "Open-Sora")
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Open-Sora root not found: {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


class IdentityVideoVAE(nn.Module):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    @staticmethod
    def get_latent_size(input_size: list[int]) -> list[int]:
        return [int(v) for v in input_size]

    @property
    def latent_channels(self) -> int | None:
        return None


class OpenSoraHunyuanVAE(nn.Module):
    def __init__(
        self,
        pretrained_path: str | None,
        open_sora_root: str | Path | None,
        device: torch.device,
        dtype: torch.dtype,
        freeze: bool = True,
        sample_posterior: bool = False,
    ) -> None:
        super().__init__()
        _ensure_open_sora_path(open_sora_root)

        from opensora.models.hunyuan_vae.autoencoder_kl_causal_3d import CausalVAE3D_HUNYUAN

        self.sample_posterior = bool(sample_posterior)
        self.model = CausalVAE3D_HUNYUAN(
            from_pretrained=pretrained_path,
            device_map=device,
            torch_dtype=dtype,
        )
        self.model.eval()

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _take_tensor(x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        if hasattr(x, "latent_dist"):
            latent_dist = x.latent_dist
            if hasattr(latent_dist, "mode"):
                return latent_dist.mode()
            if hasattr(latent_dist, "sample"):
                return latent_dist.sample()
        if hasattr(x, "sample") and torch.is_tensor(x.sample):
            return x.sample
        if isinstance(x, (tuple, list)) and x:
            return OpenSoraHunyuanVAE._take_tensor(x[0])
        raise TypeError(f"Cannot extract tensor from type {type(x)}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.model.encode(x)
        if hasattr(enc, "latent_dist") and self.sample_posterior:
            return enc.latent_dist.sample()
        return self._take_tensor(enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        dec = self.model.decode(z)
        return self._take_tensor(dec)

    def get_latent_size(self, input_size: list[int]) -> list[int]:
        return [int(v) for v in self.model.get_latent_size(input_size)]

    @property
    def latent_channels(self) -> int:
        return int(getattr(self.model, "z_channels", 0))


class VideoVAEWrapper(nn.Module):
    """
    Phase-A VAE wrapper.

    backends:
    - identity
    - opensora_hunyuan
    """

    def __init__(
        self,
        backend: str = "identity",
        open_sora_root: str | Path | None = None,
        pretrained_path: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        freeze: bool = True,
        sample_posterior: bool = False,
    ) -> None:
        super().__init__()
        backend = str(backend).lower().strip()
        self.backend = backend

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if backend == "identity":
            self.vae = IdentityVideoVAE()
        elif backend == "opensora_hunyuan":
            self.vae = OpenSoraHunyuanVAE(
                pretrained_path=pretrained_path,
                open_sora_root=open_sora_root,
                device=device,
                dtype=dtype,
                freeze=freeze,
                sample_posterior=sample_posterior,
            )
        else:
            raise ValueError(f"Unsupported VAE backend: {backend}")

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(video)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)

    def get_latent_size(self, frames: int, height: int, width: int) -> tuple[int, int, int]:
        if hasattr(self.vae, "get_latent_size"):
            t, h, w = self.vae.get_latent_size([int(frames), int(height), int(width)])
            return int(t), int(h), int(w)
        return int(frames), int(height), int(width)

    @property
    def latent_channels(self) -> int | None:
        if hasattr(self.vae, "latent_channels"):
            v = self.vae.latent_channels
            if v is None:
                return None
            return int(v)
        return None
