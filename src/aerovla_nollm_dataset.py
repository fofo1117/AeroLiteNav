"""Dataset and prompt parsing for the LLM-free AeroVLA ablation."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from src.aerovla_nollm_model import ACTION_STATS
except ModuleNotFoundError:  # Direct execution via ``python src/train_*.py``.
    from aerovla_nollm_model import ACTION_STATS


DIRECTION_NAMES = (
    "straight ahead",
    "forward-right",
    "to your right",
    "to your right rear",
    "forward-left",
    "to your left",
    "to your left rear",
)
DIRECTION_TO_ID = {name: idx for idx, name in enumerate(DIRECTION_NAMES)}
_INSTRUCTION_RE = re.compile(
    r"^(?:<image>\s*)?Fly(?:\s+(?P<direction>straight ahead|forward-right|to your right rear|to your right|forward-left|to your left rear|to your left))?\s+and find the target\.\s*(?P<description>.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_instruction(instruction: str) -> tuple[int, str]:
    match = _INSTRUCTION_RE.match(instruction.strip())
    if match is None:
        raise ValueError(f"Instruction does not match AeroVLA template: {instruction!r}")
    direction = (match.group("direction") or "straight ahead").lower()
    return DIRECTION_TO_ID[direction], match.group("description").strip()


class AeroVLANoLLMDataset(Dataset):
    def __init__(self, data_root: str, split_json: str, image_processor, num_bins: int = 99) -> None:
        self.data_root = data_root
        self.image_processor = image_processor
        self.num_bins = num_bins
        with open(split_json, "r", encoding="utf-8") as handle:
            self.samples = json.load(handle)
        print(f"[NoLLM Dataset] Loaded {len(self.samples)} geometrically filtered samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mosaic(self, sample: dict) -> Image.Image:
        views = []
        trajectory = os.path.join(self.data_root, sample["traj_rel_dir"])
        for camera in ("frontcamera", "downcamera"):
            path = os.path.join(trajectory, camera, sample["img_name"])
            with Image.open(path) as image:
                views.append(image.convert("RGB").resize((224, 224), resample=Image.BICUBIC))
        mosaic = Image.new("RGB", (224, 448))
        mosaic.paste(views[0], (0, 0))
        mosaic.paste(views[1], (0, 224))
        return mosaic

    def _quantize(self, value: float, axis: str) -> int:
        stats = ACTION_STATS[axis]
        normalized = (np.clip(value, stats["min"], stats["max"]) - stats["min"]) / (
            stats["max"] - stats["min"]
        )
        return int(normalized * (self.num_bins - 1))

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        direction_id, description = parse_instruction(sample["instruction"])
        pixels = self.image_processor(images=self._load_mosaic(sample), return_tensors="pt")["pixel_values"][0]
        label = sample["label"]
        return {
            "pixel_values": pixels,
            "description": description,
            "direction_id": direction_id,
            "action_bins": torch.tensor(
                [
                    self._quantize(label["fwd"], "forward"),
                    self._quantize(label["down"], "down"),
                    self._quantize(label["yaw"], "yaw"),
                ],
                dtype=torch.long,
            ),
            "land_label": float(sample["is_last_step"] or sample["is_penultimate"]),
        }


class NoLLMCollator:
    def __init__(self, tokenizer, max_text_tokens: int = 16) -> None:
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, samples: list[dict]) -> dict:
        text = self.tokenizer(
            [sample["description"] for sample in samples],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
            "input_ids": text.input_ids,
            "attention_mask": text.attention_mask,
            "direction_ids": torch.tensor([sample["direction_id"] for sample in samples], dtype=torch.long),
            "action_bins": torch.stack([sample["action_bins"] for sample in samples]),
            "land_labels": torch.tensor([sample["land_label"] for sample in samples]),
        }

