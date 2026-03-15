from __future__ import annotations

from dataclasses import dataclass
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
    base_channels: int
    channel_mults: List[int]
    num_res_blocks: int
    cond_dim: int
    phase_dim: int
    num_classes: int
    dropout: float


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
    return Config(
        seed=int(raw["seed"]),
        run=RunConfig(**raw["run"]),
        data=DataConfig(**raw["data"]),
        transform=TransformConfig(**raw["transform"]),
        model=ModelConfig(**raw["model"]),
        flow=FlowConfig(**raw["flow"]),
        train=TrainConfig(**raw["train"]),
        inference=InferenceConfig(**raw["inference"]),
    )
