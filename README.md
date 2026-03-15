# G-STVK-Flow

Geometric Spatio-Temporal Variance-preserving K-Flow for conditional video generation.

This implementation focuses on three core upgrades:

- Geometric band path: a learnable front on `(k_s, k_t)` with monotone motion.
- Path regularization: endpoint, coverage, spread, and smoothness losses.
- Whitened harmonic bridge: variance-preserving interpolation in whitened K-space.

## Project layout

- `g_stvk_flow/`: library package.
- `scripts/preprocess_videos.py`: raw video -> cached clip tensors.
- `scripts/train.py`: training.
- `scripts/infer.py`: standard sampling.
- `scripts/infer_disentangled.py`: content-motion disentangled sampling.
- `configs/default.yaml`: default configuration.

## Environment

```bash
cd g-stvk-flow
pip install -r requirements.txt
```

## 1) Data preprocessing

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

## 2) Configure training

Edit `configs/default.yaml`:

- `data.manifest_train` / `data.manifest_val`: point to your manifests.
- `model.num_classes`: set to your class count (for UCF101 use `101`).
- `run.output_dir`: training output directory.

## 3) Train

```bash
python scripts/train.py --config configs/default.yaml
```

Checkpoints are saved to:

- `runs/g_stvk_flow/checkpoints/last.pt`
- `runs/g_stvk_flow/checkpoints/epoch_xxxx.pt`

Notes:

- Checkpoint includes both `model` and `schedule` state.
- Resume training:

```bash
python scripts/train.py --config configs/default.yaml --resume /path/to/last.pt
```

## 4) Standard inference

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

## 5) Disentangled inference

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
