from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from g_stvk_flow.kflow.path_scheduler import FixedBandPath


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_flow_matching_import(flow_matching_root: str | Path | None) -> None:
    root = Path(flow_matching_root) if flow_matching_root is not None else (_repo_root() / "flow_matching")
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"flow_matching root not found: {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


@dataclass
class BandwiseTrainingState:
    x_t: torch.Tensor
    target_band: torch.Tensor
    band_name: str


class _AffinePathFactory:
    def __init__(self, scheduler_name: str, flow_matching_root: str | Path | None) -> None:
        _ensure_flow_matching_import(flow_matching_root)

        from flow_matching.path.affine import AffineProbPath
        from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, LinearVPScheduler, VPScheduler

        name = str(scheduler_name).lower().strip()
        if name == "condot":
            scheduler = CondOTScheduler()
        elif name == "cosine":
            scheduler = CosineScheduler()
        elif name == "linear_vp":
            scheduler = LinearVPScheduler()
        elif name == "vp":
            scheduler = VPScheduler()
        else:
            raise ValueError(f"Unknown scheduler_name={scheduler_name}")

        self.path = AffineProbPath(scheduler=scheduler)


class BandwiseDiffusionObjective:
    """
    Phase-A training objective:
    - sample current band with flow_matching AffineProbPath
    - compose partial state with fixed path
    - supervise only the current band
    """

    def __init__(
        self,
        decomposer,
        fixed_path: FixedBandPath,
        target_type: str = "epsilon",
        scheduler_name: str = "cosine",
        flow_matching_root: str | Path | None = None,
        band_weights: Dict[str, float] | None = None,
    ) -> None:
        self.decomposer = decomposer
        self.fixed_path = fixed_path
        self.target_type = str(target_type).lower().strip()
        if self.target_type not in {"epsilon", "velocity", "clean"}:
            raise ValueError(f"Unknown target_type={target_type}")

        self.path = _AffinePathFactory(
            scheduler_name=scheduler_name,
            flow_matching_root=flow_matching_root,
        ).path

        self.band_weights = dict(band_weights or {})

    def _sample_noisy_target(self, clean_band: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean_band)
        sample = self.path.sample(x_0=noise, x_1=clean_band, t=t)

        if self.target_type == "epsilon":
            target = noise
        elif self.target_type == "velocity":
            target = sample.dx_t
        else:
            target = clean_band

        return sample.x_t, target

    def build_state(self, clean_latent: torch.Tensor, t: torch.Tensor, band_name: str) -> BandwiseTrainingState:
        clean_bands = self.decomposer.forward(clean_latent)
        band_name = str(band_name)
        if band_name not in clean_bands:
            raise KeyError(f"Unknown band_name={band_name}, available={sorted(clean_bands.keys())}")

        current_noisy, target = self._sample_noisy_target(clean_bands[band_name], t=t)
        current_idx = self.fixed_path.index_of(band_name)

        state_bands: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.fixed_path.order):
            if idx < current_idx:
                state_bands[name] = clean_bands[name]
            elif idx == current_idx:
                state_bands[name] = current_noisy
            else:
                state_bands[name] = torch.randn_like(clean_bands[name])

        x_t = self.decomposer.inverse(state_bands)
        return BandwiseTrainingState(x_t=x_t, target_band=target, band_name=band_name)

    def extract_band(self, tensor: torch.Tensor, band_name: str) -> torch.Tensor:
        return self.decomposer.forward(tensor)[band_name]

    def compute_loss(self, pred_latent: torch.Tensor, state: BandwiseTrainingState) -> torch.Tensor:
        pred_band = self.extract_band(pred_latent, state.band_name)
        weight = float(self.band_weights.get(state.band_name, 1.0))
        return F.mse_loss(pred_band, state.target_band) * weight


class VanillaDiffusionObjective:
    """
    Vanilla latent baseline without decomposition.
    """

    def __init__(
        self,
        target_type: str = "epsilon",
        scheduler_name: str = "cosine",
        flow_matching_root: str | Path | None = None,
    ) -> None:
        self.target_type = str(target_type).lower().strip()
        if self.target_type not in {"epsilon", "velocity", "clean"}:
            raise ValueError(f"Unknown target_type={target_type}")

        self.path = _AffinePathFactory(
            scheduler_name=scheduler_name,
            flow_matching_root=flow_matching_root,
        ).path

    def build_state(self, clean_latent: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean_latent)
        sample = self.path.sample(x_0=noise, x_1=clean_latent, t=t)

        if self.target_type == "epsilon":
            target = noise
        elif self.target_type == "velocity":
            target = sample.dx_t
        else:
            target = clean_latent

        return sample.x_t, target

    @staticmethod
    def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)
