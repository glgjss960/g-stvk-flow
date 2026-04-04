# g-STVK-Flow

Geometric Spatio-Temporal Variance-preserving K-Flow for conditional video generation.

This repository is now reorganized into the requested 4-layer structure.

## 4-layer architecture

- `g_stvk_flow/backbone/`
  - Open-Sora style Video-DiT backbone (`video_dit.py`)
  - Velocity model wrapper (`velocity_model.py`)
  - Legacy UNet fallback (`unet3d.py`)
- `g_stvk_flow/fm_core/`
  - flow_matching-style path abstraction (`path.py`)
  - velocity objective wiring + ODE sampler (`sampler.py`)
- `g_stvk_flow/gstvk/`
  - `haar3d.py`
  - `scheduler.py`
  - `interpolant.py`
  - `anchor_edit.py`
- `g_stvk_flow/pipelines/`
  - `train_uncond.py`
  - `train_class_cond.py`
  - `train_t2v.py`
  - `infer_disentangled.py`
  - shared runtime cores: `train_core.py`, `inference_core.py`, `common.py`

Compatibility re-export layers are kept in `models/`, `fm/`, `transforms/`, `engine/` so old imports continue to work.

## Environment

```bash
cd g-stvk-flow
pip install -r requirements.txt
```

## Data preprocessing

```bash
python scripts/preprocess_videos.py \
  --raw-dir /path/to/raw_videos \
  --out-dir /path/to/data_cache \
  --frames 16 \
  --size 128 \
  --stride 8 \
  --train-ratio 0.9
```

## Training pipelines

Class-conditional:

```bash
python scripts/train_class_cond.py --config configs/default.yaml
```

Unconditional:

```bash
python scripts/train_uncond.py --config configs/default.yaml
```

T2V scaffold pipeline:

```bash
python scripts/train_t2v.py --config configs/default.yaml
```

Legacy alias (still available):

```bash
python scripts/train.py --config configs/default.yaml
```

## Disentangled inference pipeline

```bash
python scripts/infer_disentangled.py \
  --checkpoint /path/to/checkpoints/last.pt \
  --config configs/default.yaml \
  --out /path/to/outputs/disentangled.mp4 \
  --steps 60 \
  --solver heun \
  --content-label 0 \
  --motion-label 10 \
  --anchor auto
```

## Standard inference and evaluation

- `scripts/infer.py`
- `scripts/eval_checkpoint_gate.py`
- `scripts/eval_disentangle_intrinsic.py`
- `scripts/eval_disentangle_semantic.py`
- `scripts/eval_disentangle_bidirectional.py`
- `scripts/diagnose_transport.py`

Use `--help` for full arguments.

## Phase-A (MVP) implementation

This branch now includes a dedicated phase-A pipeline that matches `phaseA.pdf`:

- latent pipeline through `VideoVAEWrapper` (supports `identity` and `opensora_hunyuan` backends)
- separable Haar decomposition: `2D spatial + 1D temporal`
- fixed hand-crafted path (`Path A` / `Path B`)
- lightweight band-aware conditioning (`band_id` embedding only)
- flow-matching based affine path noise construction (from local `flow_matching/`)

### Train phase-A

```bash
python scripts/train_stageA.py --config configs/train_stageA.yaml
```

### Train vanilla baseline

```bash
python scripts/train_stageA.py --config configs/baseline_vanilla.yaml
```

### Sample / eval

```bash
python scripts/eval_stageA.py \
  --checkpoint /path/to/checkpoints/last.pt \
  --config configs/train_stageA.yaml \
  --out /path/to/output/sample.mp4
```

### Band ablation

```bash
python scripts/ablate_band.py \
  --checkpoint /path/to/checkpoints/last.pt \
  --config configs/train_stageA.yaml \
  --out /path/to/output/ablate_hs_ht.mp4 \
  --band hs_ht \
  --scale 1.0
```
