"""Closed-loop evaluation wrapper for the LLM-free AeroVLA ablation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transformers import AutoImageProcessor, AutoTokenizer

from src.aerovla_nollm_dataset import DIRECTION_TO_ID
from src.aerovla_nollm_model import ACTION_STATS, AeroVLANoLLMModel
from src.model_wrapper.base_model import BaseModelWrapper


class AeroVLANoLLMWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        del data_args
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AeroVLANoLLMModel.from_pretrained(model_args.model_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model.config.openvla_path, trust_remote_code=True, local_files_only=True, use_fast=False
        )
        tokenizer_path = self.model.config.siglip_text_path
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.model.to(self.device).eval()
        self.num_bins = self.model.config.num_bins
        self.max_text_tokens = self.model.config.max_text_tokens
        self.land_threshold = getattr(model_args, "land_threshold", 0.5)

    @staticmethod
    def get_semantic_direction(curr_state, target_pos) -> str:
        position = np.asarray(curr_state["position"])
        raw_quaternion = curr_state["orientation"]
        quaternion = (
            [raw_quaternion.get("x", 0), raw_quaternion.get("y", 0), raw_quaternion.get("z", 0), raw_quaternion.get("w", 1)]
            if isinstance(raw_quaternion, dict)
            else raw_quaternion
        )
        body_vector = R.from_quat(quaternion).inv().apply(np.asarray(target_pos) - position)
        angle = np.degrees(np.arctan2(body_vector[1], body_vector[0]))
        if -15 <= angle <= 15:
            return "straight ahead"
        if 15 < angle <= 60:
            return "forward-right"
        if 60 < angle <= 120:
            return "to your right"
        if 120 < angle <= 180:
            return "to your right rear"
        if -60 <= angle < -15:
            return "forward-left"
        if -120 <= angle < -60:
            return "to your left"
        return "to your left rear"

    @staticmethod
    def _target_description(instruction: str) -> str:
        if "degrees from you." in instruction:
            text = instruction.split("degrees from you.", 1)[1]
        else:
            text = instruction
        return text.split(" Please control", 1)[0].strip()

    def prepare_inputs(self, episodes, target_positions, instructions=None):
        pixels, descriptions, directions = [], [], []
        for index, episode in enumerate(episodes):
            state = episode[-1]["sensors"]["state"]
            rgb = episode[-1]["rgb"]
            front = Image.fromarray(cv2.cvtColor(rgb[0], cv2.COLOR_BGR2RGB)).resize((224, 224), Image.BICUBIC)
            down = Image.fromarray(cv2.cvtColor(rgb[4], cv2.COLOR_BGR2RGB)).resize((224, 224), Image.BICUBIC)
            mosaic = Image.new("RGB", (224, 448))
            mosaic.paste(front, (0, 0))
            mosaic.paste(down, (0, 224))
            pixels.append(self.image_processor(images=mosaic, return_tensors="pt")["pixel_values"][0])
            descriptions.append(self._target_description(instructions[index]))
            direction = self.get_semantic_direction(state, target_positions[index])
            directions.append(DIRECTION_TO_ID[direction])
        text = self.text_tokenizer(
            descriptions,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            "pixel_values": torch.stack(pixels),
            "input_ids": text.input_ids,
            "attention_mask": text.attention_mask,
            "direction_ids": torch.tensor(directions, dtype=torch.long),
        }
        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        return {key: value.to(self.device) for key, value in inputs.items()}, None

    @staticmethod
    def _dequantize(value: int, axis: str, num_bins: int) -> float:
        stats = ACTION_STATS[axis]
        return value / (num_bins - 1) * (stats["max"] - stats["min"]) + stats["min"]

    def run(self, inputs, episodes, rot_to_targets):
        del episodes, rot_to_targets
        bins, land_probability = self.model.predict_actions(**inputs)
        actions = []
        for row in bins.cpu().tolist():
            actions.append({
                "fwd": self._dequantize(row[0], "forward", self.num_bins),
                "down": self._dequantize(row[1], "down", self.num_bins),
                "yaw": self._dequantize(row[2], "yaw", self.num_bins),
            })
        return actions, (land_probability >= self.land_threshold).cpu().tolist()

