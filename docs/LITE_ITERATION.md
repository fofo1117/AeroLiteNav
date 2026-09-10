# AeroVLA-Lite: stable LLM-free iteration

This version starts from the first non-autoregressive no-LLM policy. It does
not reuse the collapsed axis-AR decoder or text action generation path.

## Design

- Paired MobileCLIP-S1 visual/text backbone: 84,969,729 total parameters
  (21,541,632 visual and 63,428,096 text), replacing the 700M+ frozen towers;
  the complete policy has 95,484,075 parameters, remaining below 100M.
- Three observations per sample, with front and downward cameras processed
  separately through a shared 256px encoder. No vertical mosaic is used.
- The last 8x8 MobileCLIP feature map is adaptively pooled to 4x4 per view.
  View, spatial, and temporal embeddings keep these tokens distinguishable.
- Four `d_model=384` fusion blocks. Visual tokens first exchange spatial and
  temporal information, then explicitly query the 64-token text plus direction
  memory through cross-attention.
- Direct non-autoregressive heads: ordinal forward/down, 99-way categorical yaw
  with Gaussian label smoothing, and an independent LAND head.
- LAND uses a four-frame soft terminal ramp, positive weight 25, and focal
  gamma 2. The released split has 19,403 effective positive samples after this
  expansion, for an effective negative/positive ratio of about 20.15.
- The final MobileCLIP visual stage and final two text layers are fine-tuned at
  `2e-5`; new policy layers use `2e-4`. Base weights omitted from saved policy
  checkpoints are reloaded from the local asset directory.
  The default trainable split is 17,366,656 backbone plus 10,514,346 policy
  parameters; incremental checkpoints contain about 27.9M parameters.

The current JSON contains expert RGB observations and actions but no depth,
clearance, or collision labels. Consequently this iteration adds temporal
state and improves action modeling, but does not claim explicit collision
supervision. That requires a separate depth/collision dataset or closed-loop
DAgger collection.

Closed-loop evaluation additionally enables a 3 m onboard-depth safety shield:
when the central front depth is unsafe it caps forward speed and, for an
otherwise nearly straight command, turns toward the more open image half. Set
`AEROVLA_SAFETY_DEPTH_M=0` to report the unshielded policy separately.

## Offline assets

The CPU node has downloaded all new model artifacts into the shared project:

| Asset | Path | SHA-256 |
|---|---|---|
| MobileCLIP-S1 OpenCLIP | `pretrained/mobileclip-s1-openclip/open_clip_model.safetensors` | `4c44cc904a37444bed045b0498c9917711e5c9076aff3988f3946326238b959e` |
| DINOv2-S/14 backup | `pretrained/dinov2-vits14/model.safetensors` | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |

The required new Python packages are also cached under `offline_wheels/`:

```bash
pip install --no-index --find-links ./offline_wheels \
  open_clip_torch==2.32.0 timm==1.0.20 ftfy==6.3.1
```

MobileCLIP-S1 requires timm 1.x for the `fastvit_mci1` implementation. The
downloaded OpenCLIP 2.32.0 remains compatible with the project's existing
PyTorch stack and does not require network access at runtime.

## Training

From the repository root on the GPU node:

```bash
bash scripts/train_lite.sh
```

The defaults are tuned for two H200 GPUs: micro-batch 32 per device,
accumulation 1, and effective global batch 64. TF32 matrix multiplication,
BF16 training, fused AdamW, pinned memory, 16 workers per rank, and persistent workers are
enabled. NCCL P2P/IB are enabled so same-node H200 training can use its fast
GPU interconnect:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
AEROVLA_NUM_GPUS=2 \
bash scripts/train_lite.sh
```

If memory is insufficient, lower the per-device micro-batch; accumulation stays
at 1 unless explicitly overridden:

```bash
bash scripts/train_lite.sh --micro-batch-size 16 --gradient-accumulation-steps 1
```

## Evaluation

```bash
AEROVLA_MODEL_DIR=./checkpoints/aero_vla_lite_s1_t3_d384_l4 \
AEROVLA_TASK_ID=seen_valset/NYCEnvironmentMegapa \
bash scripts/eval_lite.sh
```

The default command evaluates policy plus safety shield. For the pure learned
policy ablation:

```bash
AEROVLA_SAFETY_DEPTH_M=0 bash scripts/eval_lite.sh

Calibrate `--land_threshold` on validation data rather than changing it on the
test split. Besides SR/OSR/collision rate, monitor per-axis bin histograms,
yaw entropy, LAND precision/recall, and predicted forward speed conditional on
large absolute yaw. These reveal collapse before a full closed-loop run ends.
