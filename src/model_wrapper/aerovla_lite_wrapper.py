"""Closed-loop evaluation wrapper for the MobileCLIP AeroVLA policy."""

from __future__ import annotations

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from src.aerovla_lite_dataset import build_mobileclip_transform
from src.aerovla_lite_model import AeroVLALiteModel
from src.aerovla_nollm_dataset import DIRECTION_TO_ID
from src.aerovla_nollm_model import ACTION_STATS
from src.model_wrapper.base_model import BaseModelWrapper


class AeroVLALiteWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        del data_args
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AeroVLALiteModel.from_pretrained(model_args.model_path, local_files_only=True)
        self.tokenizer = open_clip.get_tokenizer(
            self.model.config.mobileclip_model, context_length=self.model.config.max_text_tokens
        )
        self.image_transform = build_mobileclip_transform()
        self.model.to(self.device).eval()
        self.history_frames = self.model.config.history_frames
        self.num_bins = self.model.config.num_bins
        self.land_threshold = getattr(model_args, "land_threshold", 0.5)
        self.safety_depth_threshold_m = getattr(model_args, "safety_depth_threshold_m", 0.0)

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
        text = instruction.split("degrees from you.", 1)[-1]
        return text.split(" Please control", 1)[0].strip()

    def _episode_history(self, episode) -> torch.Tensor:
        frames = list(episode[-self.history_frames :])
        frames = [frames[0]] * (self.history_frames - len(frames)) + frames
        history = []
        for frame in frames:
            rgb = frame["rgb"]
            views = []
            for camera in (0, 4):
                image = Image.fromarray(cv2.cvtColor(rgb[camera], cv2.COLOR_BGR2RGB))
                views.append(self.image_transform(image))
            history.append(torch.stack(views))
        return torch.stack(history)

    def prepare_inputs(self, episodes, target_positions, instructions=None):
        pixels, descriptions, directions = [], [], []
        for index, episode in enumerate(episodes):
            pixels.append(self._episode_history(episode))
            descriptions.append(self._target_description(instructions[index]))
            state = episode[-1]["sensors"]["state"]
            direction = self.get_semantic_direction(state, target_positions[index])
            directions.append(DIRECTION_TO_ID[direction])
        inputs = {
            "pixel_values": torch.stack(pixels),
            "input_ids": self.tokenizer(descriptions),
            "direction_ids": torch.tensor(directions, dtype=torch.long),
        }
        return {key: value.to(self.device) for key, value in inputs.items()}, None

    @staticmethod
    def _dequantize(value: int, axis: str, num_bins: int) -> float:
        stats = ACTION_STATS[axis]
        return value / (num_bins - 1) * (stats["max"] - stats["min"]) + stats["min"]

    def run(self, inputs, episodes, rot_to_targets):
        del rot_to_targets
        bins, land_probability = self.model.predict_actions(**inputs)
        actions = [
            {
                "fwd": self._dequantize(row[0], "forward", self.num_bins),
                "down": self._dequantize(row[1], "down", self.num_bins),
                "yaw": self._dequantize(row[2], "yaw", self.num_bins),
            }
            for row in bins.cpu().tolist()
        ]
        if self.safety_depth_threshold_m > 0:
            for action, episode in zip(actions, episodes):
                depth_m = np.asarray(episode[-1]["depth"][0], dtype=np.float32) / 255.0 * 100.0
                height, width = depth_m.shape
                center = depth_m[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
                if np.percentile(center, 20) >= self.safety_depth_threshold_m:
                    continue
                action["fwd"] = min(action["fwd"], 0.5)
                if abs(action["yaw"]) < 0.15:
                    left = np.median(depth_m[:, : width // 2])
                    right = np.median(depth_m[:, width // 2 :])
                    action["yaw"] = 0.55 if right >= left else -0.55
        return actions, (land_probability >= self.land_threshold).cpu().tolist()
