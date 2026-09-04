# AeroVLA Cross-Attention ablation

This is the strict third ablation for answering: **what is lost when the LLM
transformer stack is removed?** It is intentionally not the final edge model.

## Controlled variables

Kept identical to the LLM baseline:

- released geometrically filtered JSON split and front/down mosaic;
- frozen OpenVLA DINOv2-L + SigLIP-So400m vision tower;
- LLaMA tokenizer and its frozen pretrained token embedding table;
- complete `Fly ... and find the target ... Action:` prompt;
- 99-bin text action representation, `LAND`, EOS, teacher forcing, and
  next-token cross entropy only on the action suffix;
- greedy autoregressive decoding and the existing text action parser.

Changed:

- the 7B LLaMA transformer stack is replaced by six compact pre-norm decoder
  layers (`d_model=768`), each with causal text self-attention followed by
  cross-attention to all visual patch tokens.

Unlike the earlier no-LLM variants, this experiment does not introduce a
SigLIP text encoder, seven-way direction ID, independent ordinal/categorical
action heads, axis teacher forcing, or a separate LAND threshold.

## Train and evaluate

```bash
bash scripts/train_crossattn.sh

AEROVLA_MODEL_DIR=./checkpoints/aero_vla_crossattn_d768_l6 \
AEROVLA_TASK_ID=seen_valset/NYCEnvironmentMegapa \
AEROVLA_GPU_ID=0 \
bash scripts/eval_crossattn.sh
```

The default global batch (64), learning rate (`2e-4`), and five epochs match
the current LLM training script. Checkpoints omit the frozen vision tower and
token embedding; both are reloaded from `openvla_path`.

## Interpretation

Compare sequence-level action validity in addition to SR/OSR. Recommended
diagnostics are invalid-output rate, per-axis bin MAE, LAND precision/recall,
and SR conditional on OSR. If OSR remains high but SR is low, navigation and
visual grounding are present while stopping/terminal recognition is the main
deficit. If both OSR and action-token validity collapse, first verify
optimization and lexical conditioning before attributing the result to absent
LLM reasoning.

The frozen OpenVLA vision tower is still large. This version isolates the LLM
ablation cleanly; a later edge-deployment stage should distill or replace the
vision tower only after this diagnostic comparison is stable.
