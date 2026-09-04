"""Closed-loop wrapper for the minimal-change Cross-Attention ablation."""

from __future__ import annotations

import re
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transformers import AutoImageProcessor, AutoTokenizer

from src.aerovla_crossattn_model import AeroVLACrossAttentionModel
from src.model_wrapper.base_model import BaseModelWrapper


class AeroVLACrossAttentionWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        del data_args
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AeroVLACrossAttentionModel.from_pretrained(model_args.model_path, local_files_only=True)
        asset_path = self.model.config.openvla_path
        self.tokenizer = AutoTokenizer.from_pretrained(asset_path, trust_remote_code=True, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.image_processor = AutoImageProcessor.from_pretrained(
            asset_path, trust_remote_code=True, local_files_only=True, use_fast=False
        )
        self.model.to(self.device).eval()
        self.num_bins = 99
        self.action_stats = {
            "forward": {"min": 0.0, "max": 5.0},
            "down": {"min": -5.0, "max": 5.0},
            "yaw": {"min": -1.1, "max": 1.1},
        }

    @staticmethod
    def get_semantic_direction(curr_state, target_pos) -> str:
        position = np.asarray(curr_state["position"])
        raw = curr_state["orientation"]
        quaternion = (
            [raw.get("x", 0), raw.get("y", 0), raw.get("z", 0), raw.get("w", 1)]
            if isinstance(raw, dict) else raw
        )
        vector = R.from_quat(quaternion).inv().apply(np.asarray(target_pos) - position)
        if np.linalg.norm(vector[:2]) < 0.01:
            return ""
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        if -15 <= angle <= 15: return "straight ahead "
        if 15 < angle <= 60: return "forward-right "
        if 60 < angle <= 120: return "to your right "
        if 120 < angle <= 180: return "to your right rear "
        if -60 <= angle < -15: return "forward-left "
        if -120 <= angle < -60: return "to your left "
        return "to your left rear "

    @staticmethod
    def _description(instruction: str) -> str:
        if "degrees from you." in instruction:
            instruction = instruction.split("degrees from you.", 1)[1]
        return instruction.split(" Please control", 1)[0].strip()

    def prepare_inputs(self, episodes, target_positions, instructions=None):
        prompts, pixels = [], []
        for index, episode in enumerate(episodes):
            state, rgb = episode[-1]["sensors"]["state"], episode[-1]["rgb"]
            front = Image.fromarray(cv2.cvtColor(rgb[0], cv2.COLOR_BGR2RGB)).resize((224, 224), Image.BICUBIC)
            down = Image.fromarray(cv2.cvtColor(rgb[4], cv2.COLOR_BGR2RGB)).resize((224, 224), Image.BICUBIC)
            mosaic = Image.new("RGB", (224, 448))
            mosaic.paste(front, (0, 0)); mosaic.paste(down, (0, 224))
            pixels.append(self.image_processor(images=mosaic, return_tensors="pt")["pixel_values"][0])
            direction = self.get_semantic_direction(state, target_positions[index])
            prompts.append(f"<image>\nFly {direction}and find the target. {self._description(instructions[index])}\nAction: ")
        tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.model.config.max_text_tokens - 20,
        )
        dtype = next(parameter.dtype for parameter in self.model.parameters() if parameter.is_floating_point())
        inputs = dict(tokens)
        inputs["pixel_values"] = torch.stack(pixels).to(dtype)
        return {key: value.to(self.device) for key, value in inputs.items()}, None

    def run(self, inputs, episodes, rot_to_targets):
        del episodes, rot_to_targets
        generated = self.model.generate_actions(**inputs, max_new_tokens=20)
        actions, stops = [], []
        for ids in generated:
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            action = self._parse_action(text)
            actions.append(action)
            stops.append("LAND" in text or all(abs(action[key]) < 0.01 for key in ("fwd", "down", "yaw")))
        return actions, stops

    def _parse_action(self, text: str) -> dict[str, float]:
        matches = re.findall(r"\d+", text.split("Action:")[-1])
        values = (0.0, 0.0, 0.0)
        if len(matches) >= 3:
            bins = [max(0, min(self.num_bins - 1, value)) for value in map(int, matches[-3:])]
            decoded = []
            for value, axis in zip(bins, ("forward", "down", "yaw")):
                stats = self.action_stats[axis]
                decoded.append(value / (self.num_bins - 1) * (stats["max"] - stats["min"]) + stats["min"])
            values = tuple(decoded)
        return {"fwd": values[0], "down": values[1], "yaw": values[2]}
