# g-STVK-Flow

Geometric Spatio-Temporal Variance-preserving K-Flow for conditional video generation.

This implementation is organized as a three-layer bridge, matching your requested engineering route:

- Open-Sora style backbone: video DiT-style transformer (`g_stvk_flow/models/opensora_dit.py`).
- flow_matching style FM abstraction: `ProbPath`-like path sampling + `Solver`-like ODE sampling (`g_stvk_flow/fm/`).
- g-STVK method layer: geometric scheduler + Haar-band bridge/interpolant (`g_stvk_flow/transforms/`).

## Project layout

- `g_stvk_flow/transforms/`: g-STVK scheduler/interpolant and Haar3D band ops.
- `g_stvk_flow/fm/`: flow_matching-style path/sampler bridge.
- `g_stvk_flow/models/`: velocity model (`STVKFlowModel`) with Open-Sora style DiT default.
- `g_stvk_flow/engine/`: training and inference pipeline.
- `scripts/preprocess_videos.py`: raw video -> cached clip tensors.
- `scripts/train.py`: training.
- `scripts/infer.py`: standard generation.
- `scripts/infer_disentangled.py`: two-stage anchor disentangled generation.

## Environment

```bash
cd g-stvk-flow
pip install -r requirements.txt
```

Notes:
- `torchdiffeq` is optional. If available, non-Heun sampler methods can use official `flow_matching` ODE solver backend.
- Default `heun` sampling always works through local explicit solver implementation.

## Data preprocessing

Expected raw directory format:

```text
raw_videos/
  class_a/
    xxx.mp4
  class_b/
    yyy.avi
```

Run preprocessing:

```bash
python scripts/preprocess_videos.py \
  --raw-dir /path/to/raw_videos \
  --out-dir /path/to/data_cache \
  --frames 16 \
  --size 128 \
  --stride 8 \
  --train-ratio 0.9
```

Outputs:

- `clips/*.pt`
- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `class_to_idx.json`
- `stats.json`

## Configure training

Edit `configs/default.yaml`:

- `data.manifest_train` / `data.manifest_val`: training/validation manifests.
- `model.backbone`: `opensora_dit` (default) or `unet3d`.
- `model.num_classes`: class count (for UCF101 set to `101`).
- `run.output_dir`: output directory.

## Train

```bash
python scripts/train.py --config configs/default.yaml
```

Checkpoints:

- `runs/g_stvk_flow/checkpoints/last.pt`
- `runs/g_stvk_flow/checkpoints/epoch_xxxx.pt`

Resume:

```bash
python scripts/train.py --config configs/default.yaml --resume /path/to/last.pt
```

## Standard inference

```bash
python scripts/infer.py \
  --checkpoint /path/to/runs/g_stvk_flow/checkpoints/last.pt \
  --config configs/default.yaml \
  --out /path/to/outputs/sample.mp4 \
  --steps 60 \
  --solver heun \
  --class-label 0 \
  --seed 123
```

## Disentangled inference

```bash
python scripts/infer_disentangled.py \
  --checkpoint /path/to/runs/g_stvk_flow/checkpoints/last.pt \
  --config configs/default.yaml \
  --out /path/to/outputs/disentangled.mp4 \
  --steps 60 \
  --solver heun \
  --content-label 0 \
  --motion-label 10 \
  --anchor 0.35 \
  --kt-threshold 0.55 \
  --ks-min-replace 0.15 \
  --seed 123
```

Optional soft-gate controls:

- `--kt-softness`
- `--ks-softness`
- `--path-softness`

You can also pass `--reference-pt /path/to/reference.pt`.

Reference file format:

- `Tensor[C,T,H,W]` or `Tensor[1,C,T,H,W]`
- or a dict containing key `video`.

## Practical defaults for UCF101

- `transform.levels: 2`
- `flow.num_knots: 8`
- `flow.delta_min/delta_max: 0.04/0.20`
- `train.reg_endpoint/reg_coverage/reg_spread/reg_smooth: 0.05/0.02/0.02/0.001`
- disentangled inference: `anchor=0.35`, `kt-threshold=0.55`, `ks-min-replace=0.15`

## Evaluation scripts

Provided scripts:

- `scripts/eval_checkpoint_gate.py`
- `scripts/eval_disentangle_intrinsic.py`
- `scripts/eval_disentangle_semantic.py`
- `scripts/eval_disentangle_bidirectional.py`
- `scripts/diagnose_transport.py`

Use `--help` on each script for complete arguments.
