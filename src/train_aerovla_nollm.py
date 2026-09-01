"""Train the LLM-free AeroVLA ablation on the released filtered dataset."""

from __future__ import annotations

import argparse
import json
import os

import transformers
from transformers import AutoImageProcessor, AutoTokenizer, TrainingArguments

from aerovla_nollm_dataset import AeroVLANoLLMDataset, NoLLMCollator
from aerovla_nollm_model import AeroVLANoLLMConfig, AeroVLANoLLMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla-path", default="./openvla-7b")
    parser.add_argument("--siglip-text-path", default="./pretrained/siglip-so400m-patch14-224-text")
    parser.add_argument("--data-root", default="./dataset_raw")
    parser.add_argument("--split-json", default="./data/aerovla_train_dataset.json")
    parser.add_argument("--output-dir", default="./checkpoints/aero_vla_nollm_yawcat_xattn_ar_d768_l6")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64,
                        help="Target global batch; used when gradient accumulation is 0.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0,
                        help="Explicit accumulation steps; 0 derives it from global batch and WORLD_SIZE.")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--dim-feedforward", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-bins", type=int, default=99)
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--yaw-head-type", choices=("ordinal", "categorical"), default="categorical")
    parser.add_argument("--yaw-label-smoothing-sigma", type=float, default=1.5)
    parser.add_argument("--spatial-cross-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-autoregression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-embedding-dim", type=int, default=64)
    parser.add_argument("--land-pos-weight", type=float, default=1.0,
                        help="BCE positive weight; <=0 computes neg/pos from the training JSON.")
    return parser.parse_args()


def infer_land_pos_weight(path: str) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    positives = sum(bool(x["is_last_step"] or x["is_penultimate"]) for x in samples)
    negatives = len(samples) - positives
    if positives == 0:
        raise ValueError("Training split contains no LAND-positive samples")
    value = negatives / positives
    print(f"[NoLLM] LAND positives={positives}, negatives={negatives}, pos_weight={value:.4f}")
    return value


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    land_pos_weight = args.land_pos_weight if args.land_pos_weight > 0 else infer_land_pos_weight(args.split_json)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.gradient_accumulation_steps > 0:
        gradient_accumulation_steps = args.gradient_accumulation_steps
    else:
        denominator = args.micro_batch_size * world_size
        if args.global_batch_size < denominator or args.global_batch_size % denominator != 0:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} must be a positive multiple of "
                f"micro_batch_size({args.micro_batch_size}) * WORLD_SIZE({world_size})={denominator}"
            )
        gradient_accumulation_steps = args.global_batch_size // denominator
    effective_global_batch = args.micro_batch_size * world_size * gradient_accumulation_steps
    print(
        f"[NoLLM] batch: per_device={args.micro_batch_size}, world_size={world_size}, "
        f"accumulation={gradient_accumulation_steps}, global={effective_global_batch}"
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.openvla_path, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    text_tokenizer = AutoTokenizer.from_pretrained(args.siglip_text_path, local_files_only=True)
    config = AeroVLANoLLMConfig(
        openvla_path=args.openvla_path,
        siglip_text_path=args.siglip_text_path,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_bins=args.num_bins,
        max_text_tokens=args.max_text_tokens,
        land_pos_weight=land_pos_weight,
        yaw_head_type=args.yaw_head_type,
        yaw_label_smoothing_sigma=args.yaw_label_smoothing_sigma,
        use_spatial_cross_attention=args.spatial_cross_attention,
        use_axis_autoregression=args.axis_autoregression,
        action_embedding_dim=args.action_embedding_dim,
    )
    model = AeroVLANoLLMModel(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[NoLLM] Trainable parameters: {trainable:,}; frozen parameters: {frozen:,}")

    dataset = AeroVLANoLLMDataset(args.data_root, args.split_json, image_processor, args.num_bins)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        dataloader_num_workers=args.workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        num_train_epochs=args.epochs,
        bf16=True,
        fp16=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=NoLLMCollator(text_tokenizer, args.max_text_tokens),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    text_tokenizer.save_pretrained(args.output_dir)
    print(f"[NoLLM] Training finished; checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
