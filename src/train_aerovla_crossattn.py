"""Train the minimal-change Cross-Attention replacement for AeroVLA's LLM."""

from __future__ import annotations

import argparse
import os

import torch
import transformers
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments

from aerovla_crossattn_model import AeroVLACrossAttentionConfig, AeroVLACrossAttentionModel
from aerovla_dataset import AeroVLADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla-path", default="./openvla-7b")
    parser.add_argument("--data-root", default="./dataset_raw")
    parser.add_argument("--split-json", default="./data/aerovla_train_dataset.json")
    parser.add_argument("--output-dir", default="./checkpoints/aero_vla_crossattn_d768_l6")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--dim-feedforward", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    return parser.parse_args()


class CausalActionCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[dict]) -> dict:
        batch = self.tokenizer(
            [sample["text"] for sample in samples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = batch.input_ids.clone()
        for row, sample in enumerate(samples):
            prompt_ids = self.tokenizer(
                sample["prompt_only"], add_special_tokens=True, truncation=True, max_length=self.max_length
            ).input_ids
            labels[row, : len(prompt_ids)] = -100
        labels[batch.attention_mask == 0] = -100
        return {
            "pixel_values": torch.stack([sample["image"] for sample in samples]),
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "labels": labels,
        }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    tokenizer = AutoTokenizer.from_pretrained(args.openvla_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.openvla_path, trust_remote_code=True, local_files_only=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    denominator = args.micro_batch_size * world_size
    accumulation = args.gradient_accumulation_steps
    if accumulation <= 0:
        if args.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive when accumulation is derived automatically")
        accumulation = max(1, round(args.global_batch_size / denominator))
        effective_batch = denominator * accumulation
        if effective_batch != args.global_batch_size:
            print(
                f"[CrossAttn] requested global_batch={args.global_batch_size} is not divisible by "
                f"micro_batch({args.micro_batch_size}) * WORLD_SIZE({world_size})={denominator}; "
                f"using accumulation={accumulation}, effective global_batch={effective_batch}."
            )

    embedding = _embedding_shape(args.openvla_path)
    config = AeroVLACrossAttentionConfig(
        openvla_path=args.openvla_path,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_text_tokens=args.max_text_tokens,
        vocab_size=len(tokenizer),
        embedding_dim=embedding[1],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AeroVLACrossAttentionModel(config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(f"[CrossAttn] trainable={trainable:,}; frozen={frozen:,}; global_batch={denominator * accumulation}")
    dataset = AeroVLADataset(args.data_root, args.split_json, tokenizer, processor)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=accumulation,
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
        data_collator=CausalActionCollator(tokenizer, args.max_text_tokens),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def _embedding_shape(openvla_path: str) -> tuple[int, int]:
    from aerovla_crossattn_model import _checkpoint_tensor

    return tuple(_checkpoint_tensor(openvla_path, "language_model.model.embed_tokens.weight").shape)


if __name__ == "__main__":
    main()
