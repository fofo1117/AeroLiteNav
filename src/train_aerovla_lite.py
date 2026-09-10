"""Train the stable MobileCLIP-based AeroVLA LLM-free iteration."""

from __future__ import annotations

import argparse
import os

import open_clip
import torch
import transformers
from transformers import TrainingArguments

from aerovla_lite_dataset import AeroVLALiteCollator, AeroVLALiteDataset, build_mobileclip_transform
from aerovla_lite_model import AeroVLALiteConfig, AeroVLALiteModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobileclip-path", default="./pretrained/mobileclip-s1-openclip")
    parser.add_argument("--data-root", default="./dataset_raw")
    parser.add_argument("--split-json", default="./data/aerovla_train_dataset.json")
    parser.add_argument("--output-dir", default="./checkpoints/aero_vla_lite_s1_t3_d384_l4")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-bins", type=int, default=99)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--land-soft-frames", type=int, default=4)
    parser.add_argument("--land-pos-weight", type=float, default=25.0)
    parser.add_argument("--land-focal-gamma", type=float, default=2.0)
    parser.add_argument("--yaw-label-smoothing-sigma", type=float, default=1.5)
    parser.add_argument("--unfreeze-visual-stages", type=int, default=1)
    parser.add_argument("--unfreeze-text-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.gradient_accumulation_steps > 0:
        accumulation = args.gradient_accumulation_steps
    else:
        denominator = args.micro_batch_size * world_size
        if args.global_batch_size < denominator or args.global_batch_size % denominator:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} must be a positive multiple of "
                f"micro_batch_size({args.micro_batch_size}) * WORLD_SIZE({world_size})={denominator}"
            )
        accumulation = args.global_batch_size // denominator
    print(
        f"[AeroVLALite] batch per_device={args.micro_batch_size}, world_size={world_size}, "
        f"accumulation={accumulation}, global={args.micro_batch_size * world_size * accumulation}"
    )

    config = AeroVLALiteConfig(
        mobileclip_path=args.mobileclip_path,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_bins=args.num_bins,
        max_text_tokens=args.max_text_tokens,
        history_frames=args.history_frames,
        yaw_label_smoothing_sigma=args.yaw_label_smoothing_sigma,
        land_pos_weight=args.land_pos_weight,
        land_focal_gamma=args.land_focal_gamma,
        unfreeze_visual_stages=args.unfreeze_visual_stages,
        unfreeze_text_layers=args.unfreeze_text_layers,
    )
    model = AeroVLALiteModel(config)
    tokenizer = open_clip.get_tokenizer(config.mobileclip_model, context_length=config.max_text_tokens)
    dataset = AeroVLALiteDataset(
        args.data_root,
        args.split_json,
        build_mobileclip_transform(),
        num_bins=args.num_bins,
        history_frames=args.history_frames,
        land_soft_frames=args.land_soft_frames,
    )
    backbone_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("mobileclip.") and parameter.requires_grad
    ]
    policy_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("mobileclip.") and parameter.requires_grad
    ]
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(
        f"[AeroVLALite] trainable backbone={sum(p.numel() for p in backbone_parameters):,}, "
        f"policy={sum(p.numel() for p in policy_parameters):,}, frozen={frozen:,}"
    )
    optimizer = torch.optim.AdamW(
        (
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": policy_parameters, "lr": args.learning_rate},
        ),
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.workers > 0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        bf16=True,
        fp16=False,
        report_to="tensorboard",
        tf32=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AeroVLALiteCollator(tokenizer),
        optimizers=(optimizer, None),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    print(f"[AeroVLALite] Training finished; checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
