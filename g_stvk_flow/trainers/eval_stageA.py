from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from g_stvk_flow.kflow import FixedBandPath, SeparableHaarVideoDecomposer
from g_stvk_flow.models.stdit_band import StageABandVideoModel
from g_stvk_flow.models.vae_wrapper import VideoVAEWrapper
from g_stvk_flow.samplers import StageABandSampler
from g_stvk_flow.utils import load_checkpoint, save_video_tensor


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, maybe_path: str | None) -> Path | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else (root / p)


def _dtype_from_string(name: str) -> torch.dtype:
    name = str(name).lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _get_data_hw(data_cfg: dict[str, Any]) -> tuple[int, int]:
    h = data_cfg.get("image_height", data_cfg.get("image_size", None))
    w = data_cfg.get("image_width", data_cfg.get("image_size", None))
    if h is None or w is None:
        raise ValueError("data.image_height/image_width or data.image_size must be provided")
    return int(h), int(w)


def _build_model(cfg: dict[str, Any], num_bands: int, device: torch.device) -> StageABandVideoModel:
    model_cfg = cfg["model"]
    return StageABandVideoModel(
        in_channels=int(cfg["data"]["in_channels"]),
        num_bands=int(num_bands),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        hidden_size=int(model_cfg.get("hidden_size", 512)),
        depth=int(model_cfg.get("depth", 8)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        patch_size_t=int(model_cfg.get("patch_size_t", 1)),
        patch_size_h=int(model_cfg.get("patch_size_h", 2)),
        patch_size_w=int(model_cfg.get("patch_size_w", 2)),
        num_classes=int(model_cfg.get("num_classes", 0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_band_embed=bool(model_cfg.get("use_band_embed", True)),
        grad_checkpoint=bool(model_cfg.get("grad_checkpoint", False)),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate phase-A bandwise video prototype")
    parser.add_argument("--checkpoint", type=Path, default=None, help="override sampling.checkpoint from config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="override sampling.out from config")
    parser.add_argument("--steps-per-band", type=int, default=None)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ablate-band", type=str, default=None)
    parser.add_argument("--ablate-scale", type=float, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--save-scale", type=float, default=None)
    parser.add_argument("--decode-dtype", type=str, default=None, help="override sampling.decode_dtype from config")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg_root = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    sampling_cfg = cfg.get("sampling", {})
    ckpt_path = args.checkpoint if args.checkpoint is not None else _resolve(cfg_root, sampling_cfg.get("checkpoint"))
    out_path = args.out if args.out is not None else _resolve(cfg_root, sampling_cfg.get("out"))
    if ckpt_path is None:
        raise ValueError("checkpoint path must be provided via --checkpoint or sampling.checkpoint in config")
    if out_path is None:
        raise ValueError("output path must be provided via --out or sampling.out in config")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decomp_mode = str(cfg["decomposition"].get("mode", "spatial_temporal"))
    decomposer = SeparableHaarVideoDecomposer(mode=decomp_mode)
    fixed_path = FixedBandPath(
        band_names=decomposer.band_names,
        path_name=str(cfg["path"].get("name", "A")),
    )

    model = _build_model(cfg=cfg, num_bands=len(decomposer.band_names), device=device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    vae_cfg = cfg.get("vae", {})
    vae = VideoVAEWrapper(
        backend=str(vae_cfg.get("backend", "identity")),
        open_sora_root=_resolve(cfg_root, vae_cfg.get("open_sora_root")),
        pretrained_path=str(_resolve(cfg_root, vae_cfg.get("pretrained_path"))) if vae_cfg.get("pretrained_path") else None,
        device=device,
        dtype=_dtype_from_string(vae_cfg.get("dtype", "float32")),
        freeze=True,
        sample_posterior=bool(vae_cfg.get("sample_posterior", False)),
    ).to(device)

    in_channels = int(cfg["data"]["in_channels"])
    if vae.latent_channels is not None and in_channels != int(vae.latent_channels):
        raise ValueError(
            f"data.in_channels={in_channels} mismatches VAE latent_channels={vae.latent_channels}."
        )

    flow_root = _resolve(cfg_root, cfg["training"].get("flow_matching_root"))
    sampler = StageABandSampler(
        model=model,
        decomposer=decomposer,
        fixed_path=fixed_path,
        target_type=str(cfg["training"].get("target_type", "epsilon")),
        scheduler_name=str(cfg["training"].get("scheduler", "cosine")),
        flow_matching_root=flow_root,
    )

    steps_per_band = int(args.steps_per_band if args.steps_per_band is not None else sampling_cfg.get("steps_per_band", 24))
    seed = args.seed if args.seed is not None else sampling_cfg.get("seed", None)
    class_label = args.class_label if args.class_label is not None else sampling_cfg.get("class_label", None)
    ablate_band = args.ablate_band if args.ablate_band is not None else sampling_cfg.get("ablate_band", None)
    ablate_scale = float(args.ablate_scale if args.ablate_scale is not None else sampling_cfg.get("ablate_scale", 1.0))
    decode_dtype = _dtype_from_string(
        args.decode_dtype
        if args.decode_dtype is not None
        else sampling_cfg.get("decode_dtype", vae_cfg.get("dtype", "float32"))
    )

    raw_frames = int(cfg["data"]["frames"])
    raw_h, raw_w = _get_data_hw(cfg["data"])
    latent_t, latent_h, latent_w = vae.get_latent_size(raw_frames, raw_h, raw_w)
    shape = (
        1,
        int(in_channels),
        int(latent_t),
        int(latent_h),
        int(latent_w),
    )

    labels = None
    if class_label is not None:
        labels = torch.full((1,), int(class_label), dtype=torch.long, device=device)

    with torch.no_grad():
        latent = sampler.sample(
            shape=shape,
            steps_per_band=steps_per_band,
            device=device,
            class_labels=labels,
            seed=(int(seed) if seed is not None else None),
        )

        if ablate_band is not None:
            bands = decomposer.forward(latent)
            if ablate_band not in bands:
                raise KeyError(f"Unknown band={ablate_band}, available={sorted(bands.keys())}")
            bands[ablate_band] = bands[ablate_band] + ablate_scale * torch.randn_like(bands[ablate_band])
            latent = decomposer.inverse(bands)

        latent = latent.to(device=device, dtype=decode_dtype)
        video = vae.decode(latent)

    fps_default = cfg.get("data", {}).get("fps", 8)
    fps = int(args.fps if args.fps is not None else sampling_cfg.get("fps", fps_default))
    save_scale = float(args.save_scale if args.save_scale is not None else sampling_cfg.get("save_scale", 1.0))
    save_video_tensor(video[0], out_path, fps=fps, scale=save_scale)
    print(f"Saved phase-A sample to {out_path}")


if __name__ == "__main__":
    main()
