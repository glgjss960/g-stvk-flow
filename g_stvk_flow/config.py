from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class RunConfig:
    name: str
    output_dir: str


@dataclass
class DataConfig:
    cache_dir: str
    manifest_train: str
    manifest_val: str
    frames: int
    image_size: int
    in_channels: int


@dataclass
class TransformConfig:
    levels: int


@dataclass
class ModelConfig:
    # Legacy UNet knobs (still supported when backbone=unet3d).
    base_channels: int = 96
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 2

    # Shared conditioning knobs.
    cond_dim: int = 256
    phase_dim: int = 11
    num_classes: int = 0
    dropout: float = 0.0

    # Open-Sora-style DiT knobs (default backend).
    backbone: str = "opensora_dit"
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    patch_size_t: int = 1
    patch_size_h: int = 2
    patch_size_w: int = 2


@dataclass
class FlowConfig:
    num_knots: int
    delta_min: float
    delta_max: float
    radius_min: float
    radius_max: float
    anisotropy_min: float
    derivative_eps: float
    delta_hidden_dim: int
    spread_temperature: float
    reg_grid_size: int
    integration_grid_size: int = 129
    rate_floor: float = 1e-4
    lambda_replace_thr: float = 0.55
    tail_start: float = 0.85


@dataclass
class TrainConfig:
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    amp: bool
    log_every: int
    save_every: int
    reg_endpoint: float
    reg_coverage: float
    reg_spread: float
    reg_smooth: float
    reg_every: int
    reg_mono: float = 0.0
    reg_tail: float = 0.0
    reg_end_slope: float = 0.0


@dataclass
class InferenceConfig:
    steps: int
    solver: str
    fps: int
    kt_softness: float
    ks_softness: float
    path_softness: float


@dataclass
class Config:
    seed: int
    run: RunConfig
    data: DataConfig
    transform: TransformConfig
    model: ModelConfig
    flow: FlowConfig
    train: TrainConfig
    inference: InferenceConfig


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> Config:
    raw = _load_yaml(path)

    flow_raw = dict(raw["flow"])
    flow_raw.setdefault("integration_grid_size", 129)
    flow_raw.setdefault("rate_floor", 1e-4)
    flow_raw.setdefault("lambda_replace_thr", 0.55)
    flow_raw.setdefault("tail_start", 0.85)

    train_raw = dict(raw["train"])
    train_raw.setdefault("reg_mono", 0.0)
    train_raw.setdefault("reg_tail", 0.0)
    train_raw.setdefault("reg_end_slope", 0.0)

    model_raw = dict(raw.get("model", {}))
    model_defaults = {
        "base_channels": 96,
        "channel_mults": [1, 2, 4],
        "num_res_blocks": 2,
        "cond_dim": 256,
        "phase_dim": 11,
        "num_classes": 0,
        "dropout": 0.0,
        "backbone": "opensora_dit",
        "hidden_size": 512,
        "depth": 8,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "patch_size_t": 1,
        "patch_size_h": 2,
        "patch_size_w": 2,
    }
    for key, value in model_defaults.items():
        model_raw.setdefault(key, value)

    return Config(
        seed=int(raw["seed"]),
        run=RunConfig(**raw["run"]),
        data=DataConfig(**raw["data"]),
        transform=TransformConfig(**raw["transform"]),
        model=ModelConfig(**model_raw),
        flow=FlowConfig(**flow_raw),
        train=TrainConfig(**train_raw),
        inference=InferenceConfig(**raw["inference"]),
    )
