"""LLM-free AeroVLA ablation model.

The frozen DINOv2 + SigLIP image backbone is loaded directly from the local
OpenVLA checkpoint.  A frozen SigLIP text tower, a learned seven-way direction
embedding and a trainable Transformer encoder replace LLaMA-2.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from timm.models.vision_transformer import LayerScale
from transformers import PretrainedConfig, PreTrainedModel, SiglipTextModel
from transformers.utils import ModelOutput


ACTION_STATS = {
    "forward": {"min": 0.0, "max": 5.0},
    "down": {"min": -5.0, "max": 5.0},
    "yaw": {"min": -1.1, "max": 1.1},
}


def _unpack_tuple(fn):
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, (tuple, list)) else result

    return wrapped


def _layer_scale_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def _patch_layer_scale(module: LayerScale) -> None:
    module.scale_factor = nn.Parameter(module.gamma.clone())
    module.forward = _layer_scale_forward.__get__(module, LayerScale)
    del module.gamma


class DinoSiglipVisionBackbone(nn.Module):
    """The exact fused 224px image tower used by openvla-7b."""

    def __init__(self) -> None:
        super().__init__()
        self.featurizer = timm.create_model(
            "vit_large_patch14_reg4_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=224,
            weight_init="skip",
        )
        self.featurizer.forward = _unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.fused_featurizer = timm.create_model(
            "vit_so400m_patch14_siglip_224",
            pretrained=False,
            num_classes=0,
            img_size=224,
            weight_init="skip",
        )
        self.fused_featurizer.forward = _unpack_tuple(
            partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
        )
        for child in self.modules():
            if isinstance(child, LayerScale):
                _patch_layer_scale(child)
        self.embed_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dino_pixels, siglip_pixels = torch.split(pixel_values, [3, 3], dim=1)
        dino = self.featurizer(dino_pixels)
        siglip = self.fused_featurizer(siglip_pixels)
        return torch.cat((dino, siglip), dim=-1)


def load_openvla_vision_weights(module: nn.Module, openvla_path: str) -> None:
    """Load only vision tensors, never materializing the 7B language model."""
    extracted = os.path.join(openvla_path, "vision_backbone.safetensors")
    state: Dict[str, torch.Tensor] = {}
    if os.path.isfile(extracted):
        with safe_open(extracted, framework="pt", device="cpu") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
    else:
        index_path = os.path.join(openvla_path, "model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        keys_by_shard: Dict[str, list[str]] = {}
        prefix = "vision_backbone."
        for full_key, shard in weight_map.items():
            if full_key.startswith(prefix):
                keys_by_shard.setdefault(shard, []).append(full_key)
        for shard, full_keys in keys_by_shard.items():
            with safe_open(os.path.join(openvla_path, shard), framework="pt", device="cpu") as handle:
                for full_key in full_keys:
                    state[full_key[len(prefix) :]] = handle.get_tensor(full_key)
    # Extracted files already use keys without the OpenVLA module prefix.
    state = {key.removeprefix("vision_backbone."): value for key, value in state.items()}
    missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Vision checkpoint mismatch: missing={missing}, unexpected={unexpected}")


class AeroVLANoLLMConfig(PretrainedConfig):
    model_type = "aerovla_nollm"

    def __init__(
        self,
        openvla_path: str = "./openvla-7b",
        siglip_text_path: str = "./pretrained/siglip-so400m-patch14-224-text",
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        num_bins: int = 99,
        max_text_tokens: int = 16,
        max_fusion_tokens: int = 384,
        land_pos_weight: float = 1.0,
        ordinal_loss_weight: float = 1.0,
        land_loss_weight: float = 1.0,
        frozen_backbone_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.openvla_path = openvla_path
        self.siglip_text_path = siglip_text_path
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_bins = num_bins
        self.max_text_tokens = max_text_tokens
        self.max_fusion_tokens = max_fusion_tokens
        self.land_pos_weight = land_pos_weight
        self.ordinal_loss_weight = ordinal_loss_weight
        self.land_loss_weight = land_loss_weight
        self.frozen_backbone_dtype = frozen_backbone_dtype
        self.action_stats = ACTION_STATS


class MonotonicOrdinalHead(nn.Module):
    """Cumulative-link head whose learned thresholds remain ordered."""

    def __init__(self, d_model: int, num_bins: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-3.0))
        initial_step = 6.0 / max(num_bins - 2, 1)
        self.threshold_deltas_raw = nn.Parameter(
            torch.full((num_bins - 2,), math.log(math.expm1(initial_step)))
        )

    def thresholds(self) -> torch.Tensor:
        if self.threshold_deltas_raw.numel() == 0:
            return self.threshold_start[None]
        deltas = F.softplus(self.threshold_deltas_raw)
        return torch.cat((self.threshold_start[None], self.threshold_start + torch.cumsum(deltas, dim=0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # P(y > k) decreases as k increases.
        return self.score(x) - self.thresholds()


@dataclass
class AeroVLANoLLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    forward_logits: Optional[torch.Tensor] = None
    down_logits: Optional[torch.Tensor] = None
    yaw_logits: Optional[torch.Tensor] = None
    land_logits: Optional[torch.Tensor] = None


class AeroVLANoLLMModel(PreTrainedModel):
    config_class = AeroVLANoLLMConfig
    base_model_prefix = "aerovla_nollm"
    _keys_to_ignore_on_load_missing = [r"vision_backbone\..*", r"text_backbone\..*"]

    def __init__(self, config: AeroVLANoLLMConfig, initialize_backbones: bool = True) -> None:
        super().__init__(config)
        frozen_dtype = getattr(torch, config.frozen_backbone_dtype)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(frozen_dtype)
        try:
            self.vision_backbone = DinoSiglipVisionBackbone()
        finally:
            torch.set_default_dtype(previous_dtype)
        if initialize_backbones:
            load_openvla_vision_weights(self.vision_backbone, config.openvla_path)
        self.text_backbone = SiglipTextModel.from_pretrained(
            config.siglip_text_path, local_files_only=True, torch_dtype=frozen_dtype
        )

        vision_dim = self.vision_backbone.embed_dim
        text_dim = self.text_backbone.config.hidden_size
        self.vision_projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.d_model),
        )
        self.text_projector = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, config.d_model))
        self.direction_embedding = nn.Embedding(7, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_fusion_tokens, config.d_model))
        self.modality_embedding = nn.Embedding(4, config.d_model)  # CLS, vision, text, direction

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(config.d_model))
        self.forward_head = MonotonicOrdinalHead(config.d_model, config.num_bins)
        self.down_head = MonotonicOrdinalHead(config.d_model, config.num_bins)
        self.yaw_head = MonotonicOrdinalHead(config.d_model, config.num_bins)
        self.land_head = nn.Linear(config.d_model, 1)
        self._init_trainable_parameters()
        self.freeze_backbones()

    def _init_trainable_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.direction_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def freeze_backbones(self) -> None:
        for backbone in (self.vision_backbone, self.text_backbone):
            backbone.requires_grad_(False)
            backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_backbone.eval()
        self.text_backbone.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Keep checkpoints small: frozen towers are referenced, not duplicated."""
        complete = super().state_dict(*args, **kwargs)
        frozen_prefixes = ("vision_backbone.", "text_backbone.")
        return {key: value for key, value in complete.items() if not key.startswith(frozen_prefixes)}

    @staticmethod
    def _ordinal_targets(labels: torch.Tensor, num_bins: int, dtype: torch.dtype) -> torch.Tensor:
        thresholds = torch.arange(num_bins - 1, device=labels.device)
        return (labels[:, None] > thresholds[None, :]).to(dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        direction_ids: torch.Tensor,
        action_bins: Optional[torch.Tensor] = None,
        land_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AeroVLANoLLMOutput:
        del kwargs
        with torch.no_grad():
            visual = self.vision_backbone(pixel_values)
            text = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Projectors remain FP32 outside autocast; cast frozen features at the boundary.
        visual = self.vision_projector(visual.to(self.vision_projector[1].weight.dtype))
        text = self.text_projector(text.to(self.text_projector[1].weight.dtype))

        batch_size = visual.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1) + self.modality_embedding.weight[0]
        visual = visual + self.modality_embedding.weight[1]
        text = text + self.modality_embedding.weight[2]
        direction = self.direction_embedding(direction_ids)[:, None] + self.modality_embedding.weight[3]
        tokens = torch.cat((cls, visual, text, direction), dim=1)
        if tokens.shape[1] > self.config.max_fusion_tokens:
            raise ValueError(f"Fusion sequence {tokens.shape[1]} exceeds max_fusion_tokens={self.config.max_fusion_tokens}")
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]

        visual_mask = torch.ones((batch_size, 1 + visual.shape[1]), dtype=torch.bool, device=tokens.device)
        direction_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device)
        valid_mask = torch.cat((visual_mask, attention_mask.bool(), direction_mask), dim=1)
        fused = self.fusion_transformer(tokens, src_key_padding_mask=~valid_mask)[:, 0]

        logits = (self.forward_head(fused), self.down_head(fused), self.yaw_head(fused))
        land_logits = self.land_head(fused).squeeze(-1)
        loss = None
        if action_bins is not None and land_labels is not None:
            ordinal_loss = sum(
                F.binary_cross_entropy_with_logits(
                    axis_logits,
                    self._ordinal_targets(action_bins[:, axis], self.config.num_bins, axis_logits.dtype),
                )
                for axis, axis_logits in enumerate(logits)
            ) / 3.0
            pos_weight = torch.as_tensor(self.config.land_pos_weight, device=land_logits.device, dtype=land_logits.dtype)
            land_loss = F.binary_cross_entropy_with_logits(
                land_logits, land_labels.to(land_logits.dtype), pos_weight=pos_weight
            )
            loss = self.config.ordinal_loss_weight * ordinal_loss + self.config.land_loss_weight * land_loss
        return AeroVLANoLLMOutput(
            loss=loss,
            forward_logits=logits[0],
            down_logits=logits[1],
            yaw_logits=logits[2],
            land_logits=land_logits,
        )

    @torch.no_grad()
    def predict_actions(self, **inputs):
        output = self(**inputs)
        bins = torch.stack(
            tuple((torch.sigmoid(x) >= 0.5).sum(dim=-1) for x in (
                output.forward_logits, output.down_logits, output.yaw_logits
            )),
            dim=-1,
        )
        return bins, torch.sigmoid(output.land_logits)

