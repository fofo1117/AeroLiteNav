# AeroLiteNav: LLM-free UAV navigation

This project keeps AeroVLA's released, geometrically filtered training split
and its frozen DINOv2 + SigLIP image tower, but removes LLaMA-2, LoRA and
autoregressive action tokens.

## Architecture

- Frozen OpenVLA DINOv2-L + SigLIP-So400m image features (2176 channels).
- Frozen `google/siglip-so400m-patch14-224` text tower (1152 channels). Only the
  target description is encoded. The paired checkpoint natively supports 16
  text positions, so longer descriptions are truncated to 16 tokens.
- The fuzzy direction phrase is parsed into exactly seven classes and represented
  The released terminal-frame prompt sometimes has no direction phrase; those
  7,611 samples map to `straight ahead` rather than adding an eighth class.
  by a learned lookup embedding; it is not passed to the text encoder.
- A retrained MLP projects visual tokens to `d_model=768`.
- A 6-layer, 12-head, pre-norm Transformer encoder fuses a CLS token, 256 visual
  tokens, up to 16 text tokens and one direction token.
- Three monotonic cumulative-link ordinal heads predict the original 99 bins for
  forward/down/yaw. A separate binary head predicts LAND.

Action ranges and LAND labels are unchanged: forward `[0, 5]`, down `[-5, 5]`,
yaw `[-1.1, 1.1]`, with both the last and penultimate frames labeled LAND.
`data/aerovla_train_dataset.json` is consumed as released, so the upstream
geometric consistency filtering (about 4% removed) is preserved exactly.

## Offline assets

All paths below are in the shared repository and require no network access:

- `openvla-7b/vision_backbone.safetensors`: DINOv2 + SigLIP image weights
  extracted from the existing OpenVLA checkpoint (685 tensors, 1,461,899,064 bytes).
- `pretrained/siglip-so400m-patch14-224-text/`: text-only SigLIP model and tokenizer.
- `pretrained/siglip-so400m-patch14-224/`: original pinned Hugging Face snapshot,
  revision `945aa7089e54a4085410d83f36691100e5cebcf2`. Its model SHA-256 is
  `30a00c26a045c84a0bd14adc94278d7f1cc941c8bd67e5fa50f7bed34c34e60d`.

The full paired snapshot is retained for provenance; normal training reads the
smaller text-only directory.

If the GPU environment is missing timm, install the bundled wheel offline:

```bash
pip install --no-index offline_wheels/timm-0.9.10-py3-none-any.whl
```

## Training on the GPU node

From the repository root:

```bash
bash scripts/train_nollm.sh
```

The launcher forces Hugging Face offline mode. Defaults use BF16, five epochs,
micro-batch 4 and gradient accumulation 4. With eight GPUs this gives global
batch 128, matching the released training launcher's effective batch size.
Override settings with CLI flags, for example:

```bash
bash scripts/train_nollm.sh \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --output-dir ./checkpoints/aero_vla_nollm_d768_l6
```

LAND uses unweighted BCE by default (`--land-pos-weight 1`). Pass
`--land-pos-weight 0` to derive negative/positive weighting from the split, but
that changes more than the decoder and is not recommended for the strict first
ablation.

Checkpoints contain only trainable modules. The frozen towers are reloaded from
the offline asset paths stored in the model config, avoiding multi-gigabyte
duplication at every save step.

## Closed-loop evaluation

After training, select a split/map with the same environment variables used by
the original evaluation script:

```bash
AEROVLA_MODEL_DIR=./checkpoints/aero_vla_nollm_d768_l6 \
AEROVLA_TASK_ID=seen_valset/NYCEnvironmentMegapa \
AEROVLA_GPU_ID=0 \
bash scripts/eval_nollm.sh
```

The evaluation wrapper dequantizes ordinal predictions with the original ranges
and stops when the LAND probability is at least `0.5`. Use
`--land_threshold VALUE` after the shell command to calibrate that threshold on
a validation split without retraining.

