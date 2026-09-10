"""Sequence dataset for the lightweight, LLM-free AeroVLA policy."""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

try:
    from src.aerovla_nollm_dataset import DIRECTION_TO_ID, parse_instruction
    from src.aerovla_nollm_model import ACTION_STATS
except ModuleNotFoundError:  # Direct execution via ``python src/train_*.py``.
    from aerovla_nollm_dataset import DIRECTION_TO_ID, parse_instruction
    from aerovla_nollm_model import ACTION_STATS


MOBILECLIP_MEAN = (0.0, 0.0, 0.0)
MOBILECLIP_STD = (1.0, 1.0, 1.0)


def build_mobileclip_transform(image_size: int = 256):
    """Deterministic transform matching the downloaded OpenCLIP config."""
    return Compose(
        (
            Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(MOBILECLIP_MEAN, MOBILECLIP_STD),
        )
    )


class AeroVLALiteDataset(Dataset):
    """Return short dual-view histories and dense, soft terminal targets.

    History is built only within a trajectory. Early samples repeat the first
    available frame, so training and closed-loop inference have identical shapes.
    """

    def __init__(
        self,
        data_root: str,
        split_json: str,
        image_transform,
        num_bins: int = 99,
        history_frames: int = 3,
        land_soft_frames: int = 4,
    ) -> None:
        self.data_root = data_root
        self.image_transform = image_transform
        self.num_bins = num_bins
        self.history_frames = history_frames
        with open(split_json, "r", encoding="utf-8") as handle:
            self.samples = json.load(handle)
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        if land_soft_frames < 1:
            raise ValueError("land_soft_frames must be positive")

        self.history_indices: list[list[int]] = []
        recent: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=history_frames))
        trajectories: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(self.samples):
            trajectory = sample["traj_rel_dir"]
            recent[trajectory].append(index)
            indices = list(recent[trajectory])
            self.history_indices.append([indices[0]] * (history_frames - len(indices)) + indices)
            trajectories[trajectory].append(index)

        self.land_targets = np.asarray(
            [float(x["is_last_step"] or x["is_penultimate"]) for x in self.samples], dtype=np.float32
        )
        for indices in trajectories.values():
            terminal_positions = [i for i, index in enumerate(indices) if self.samples[index]["is_last_step"]]
            if not terminal_positions:
                continue
            terminal = terminal_positions[-1]
            start = max(0, terminal - land_soft_frames + 1)
            window = indices[start : terminal + 1]
            ramp = np.linspace(1.0 / len(window), 1.0, len(window), dtype=np.float32)
            self.land_targets[window] = np.maximum(self.land_targets[window], ramp)
        print(
            f"[AeroVLALite Dataset] Loaded {len(self.samples)} samples, "
            f"history={history_frames}, effective LAND positives={self.land_targets.sum():.1f}."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_views(self, sample: dict) -> torch.Tensor:
        trajectory = os.path.join(self.data_root, sample["traj_rel_dir"])
        views = []
        for camera in ("frontcamera", "downcamera"):
            path = os.path.join(trajectory, camera, sample["img_name"])
            with Image.open(path) as image:
                views.append(self.image_transform(image.convert("RGB")))
        return torch.stack(views)  # [view, channel, height, width]

    def _quantize(self, value: float, axis: str) -> int:
        stats = ACTION_STATS[axis]
        normalized = (np.clip(value, stats["min"], stats["max"]) - stats["min"]) / (
            stats["max"] - stats["min"]
        )
        return int(normalized * (self.num_bins - 1))

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        direction_id, description = parse_instruction(sample["instruction"])
        label = sample["label"]
        return {
            "pixel_values": torch.stack(
                [self._load_views(self.samples[i]) for i in self.history_indices[index]]
            ),  # [time, view, channel, height, width]
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
            "land_label": float(self.land_targets[index]),
        }


class AeroVLALiteCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, samples: list[dict]) -> dict:
        input_ids = self.tokenizer([sample["description"] for sample in samples])
        return {
            "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
            "input_ids": input_ids,
            "direction_ids": torch.tensor([sample["direction_id"] for sample in samples], dtype=torch.long),
            "action_bins": torch.stack([sample["action_bins"] for sample in samples]),
            "land_labels": torch.tensor([sample["land_label"] for sample in samples]),
        }
