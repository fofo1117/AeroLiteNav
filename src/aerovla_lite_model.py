"""Stable lightweight AeroVLA iteration built on MobileCLIP-S1.

The model keeps v0's direct action heads, but replaces the 700M+ visual/text
towers with an 85M paired encoder, models a short dual-view history, and makes
visual tokens explicitly query the language/direction memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils import ModelOutput

try:
    from src.aerovla_nollm_model import ACTION_STATS, MonotonicOrdinalHead
except ModuleNotFoundError:
    from aerovla_nollm_model import ACTION_STATS, MonotonicOrdinalHead


class AeroVLALiteConfig(PretrainedConfig):
    model_type = "aerovla_lite"

    def __init__(
        self,
        mobileclip_path: str = "./pretrained/mobileclip-s1-openclip",
        mobileclip_model: str = "MobileCLIP-S1",
        d_model: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        num_bins: int = 99,
        max_text_tokens: int = 64,
        history_frames: int = 3,
        spatial_grid_size: int = 4,
        yaw_label_smoothing_sigma: float = 1.5,
        land_pos_weight: float = 25.0,
        land_focal_gamma: float = 2.0,
        unfreeze_visual_stages: int = 1,
        unfreeze_text_layers: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if d_model % num_heads:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if not 1 <= max_text_tokens <= 77:
            raise ValueError("MobileCLIP-S1 max_text_tokens must be in [1, 77]")
        self.mobileclip_path = mobileclip_path
        self.mobileclip_model = mobileclip_model
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_bins = num_bins
        self.max_text_tokens = max_text_tokens
        self.history_frames = history_frames
        self.spatial_grid_size = spatial_grid_size
        self.yaw_label_smoothing_sigma = yaw_label_smoothing_sigma
        self.land_pos_weight = land_pos_weight
        self.land_focal_gamma = land_focal_gamma
        self.unfreeze_visual_stages = unfreeze_visual_stages
        self.unfreeze_text_layers = unfreeze_text_layers
        self.action_stats = ACTION_STATS


@dataclass
class AeroVLALiteOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    forward_logits: Optional[torch.Tensor] = None
    down_logits: Optional[torch.Tensor] = None
    yaw_logits: Optional[torch.Tensor] = None
    land_logits: Optional[torch.Tensor] = None


class VisualLanguageBlock(nn.Module):
    """Pre-norm visual self-attention followed by language cross-attention."""

    def __init__(self, config: AeroVLALiteConfig) -> None:
        super().__init__()
        self.visual_norm = nn.LayerNorm(config.d_model)
        self.visual_attention = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.language_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.language_attention = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, visual: torch.Tensor, memory: torch.Tensor, memory_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.visual_norm(visual)
        attended, _ = self.visual_attention(normalized, normalized, normalized, need_weights=False)
        visual = visual + self.dropout(attended)
        attended, _ = self.language_attention(
            self.language_norm(visual),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        visual = visual + self.dropout(attended)
        return visual + self.dropout(self.ffn(self.ffn_norm(visual)))


class AeroVLALiteModel(PreTrainedModel):
    config_class = AeroVLALiteConfig
    base_model_prefix = "aerovla_lite"
    _keys_to_ignore_on_load_missing = [r"mobileclip\..*"]

    @classmethod
    def get_init_context(cls, is_quantized: bool, _is_ds_init_called: bool):
        # OpenCLIP moves its freshly-created module to CPU while constructing it,
        # which is incompatible with Transformers' default meta-device context.
        del is_quantized, _is_ds_init_called
        return [no_init_weights()]

    def __init__(self, config: AeroVLALiteConfig, initialize_backbone: bool = True) -> None:
        super().__init__(config)
        checkpoint = f"{config.mobileclip_path}/open_clip_model.safetensors" if initialize_backbone else None
        self.mobileclip = open_clip.create_model(
            config.mobileclip_model,
            pretrained=checkpoint,
            device="cpu",
        )
        visual_dim = self.mobileclip.visual.trunk.feature_info[-1]["num_chs"]
        text_dim = self.mobileclip.text.ln_final.normalized_shape[0]
        self.visual_projector = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, config.d_model))
        self.text_projector = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, config.d_model))
        self.direction_embedding = nn.Embedding(7, config.d_model)
        self.view_embedding = nn.Embedding(2, config.d_model)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, config.history_frames, 1, config.d_model))
        tokens_per_view = config.spatial_grid_size**2
        self.spatial_embedding = nn.Parameter(torch.zeros(1, 1, 2 * tokens_per_view, config.d_model))
        self.layers = nn.ModuleList(VisualLanguageBlock(config) for _ in range(config.num_layers))
        self.policy_query = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.policy_attention = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.policy_norm = nn.LayerNorm(config.d_model)
        self.forward_head = MonotonicOrdinalHead(config.d_model, config.num_bins)
        self.down_head = MonotonicOrdinalHead(config.d_model, config.num_bins)
        self.yaw_head = nn.Linear(config.d_model, config.num_bins)
        self.land_head = nn.Linear(config.d_model, 1)
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        nn.init.trunc_normal_(self.policy_query, std=0.02)
        nn.init.normal_(self.direction_embedding.weight, std=0.02)
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        self._configure_backbone_training()

    def _configure_backbone_training(self) -> None:
        self.mobileclip.requires_grad_(False)
        stages = self.mobileclip.visual.trunk.stages
        visual_count = min(self.config.unfreeze_visual_stages, len(stages))
        for stage in list(stages)[len(stages) - visual_count :]:
            stage.requires_grad_(True)
        blocks = self.mobileclip.text.transformer.resblocks
        text_count = min(self.config.unfreeze_text_layers, len(blocks))
        for block in list(blocks)[len(blocks) - text_count :]:
            block.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mobileclip.eval()
        if mode:
            stages = self.mobileclip.visual.trunk.stages
            for stage in list(stages)[len(stages) - self.config.unfreeze_visual_stages :]:
                stage.train()
            blocks = self.mobileclip.text.transformer.resblocks
            for block in list(blocks)[len(blocks) - self.config.unfreeze_text_layers :]:
                block.train()
        return self

    def state_dict(self, *args, **kwargs):
        complete = super().state_dict(*args, **kwargs)
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        visual_prefixes = tuple(
            f"mobileclip.visual.trunk.stages.{index}."
            for index in range(
                len(self.mobileclip.visual.trunk.stages) - self.config.unfreeze_visual_stages,
                len(self.mobileclip.visual.trunk.stages),
            )
        )
        text_prefixes = tuple(
            f"mobileclip.text.transformer.resblocks.{index}."
            for index in range(
                len(self.mobileclip.text.transformer.resblocks) - self.config.unfreeze_text_layers,
                len(self.mobileclip.text.transformer.resblocks),
            )
        )
        kept = {}
        for key, value in complete.items():
            if not key.startswith("mobileclip.") or key in trainable or key.startswith(visual_prefixes + text_prefixes):
                kept[key] = value
        return kept

    @staticmethod
    def _ordinal_targets(labels: torch.Tensor, num_bins: int, dtype: torch.dtype) -> torch.Tensor:
        thresholds = torch.arange(num_bins - 1, device=labels.device)
        return (labels[:, None] > thresholds[None, :]).to(dtype)

    @staticmethod
    def _ordinal_prediction(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) >= 0.5).sum(dim=-1)

    @staticmethod
    def _gaussian_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, sigma: float) -> torch.Tensor:
        bins = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        if sigma <= 0:
            return F.cross_entropy(logits, labels)
        targets = torch.exp(-0.5 * ((bins[None] - labels[:, None].to(logits.dtype)) / sigma) ** 2)
        targets = targets / targets.sum(dim=-1, keepdim=True)
        return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _encode_visual(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, time, views, channels, height, width = pixel_values.shape
        if time != self.config.history_frames or views != 2:
            raise ValueError(
                f"Expected [B,{self.config.history_frames},2,C,H,W], got {tuple(pixel_values.shape)}"
            )
        flat = pixel_values.reshape(batch * time * views, channels, height, width)
        features = self.mobileclip.visual.forward_intermediates(
            flat, indices=[-1], normalize_intermediates=True, output_fmt="NCHW"
        )["image_intermediates"][-1]
        features = F.adaptive_avg_pool2d(
            features, (self.config.spatial_grid_size, self.config.spatial_grid_size)
        )
        features = features.flatten(2).transpose(1, 2)
        features = self.visual_projector(features)
        tokens_per_view = self.config.spatial_grid_size**2
        features = features.reshape(batch, time, views, tokens_per_view, self.config.d_model)
        view_ids = torch.arange(views, device=features.device)
        features = features + self.view_embedding(view_ids)[None, None, :, None]
        features = features.reshape(batch, time, views * tokens_per_view, self.config.d_model)
        return features + self.temporal_embedding + self.spatial_embedding

    def _encode_memory(self, input_ids: torch.Tensor, direction_ids: torch.Tensor):
        text = self.mobileclip.text.forward_intermediates(
            input_ids, indices=[-1], normalize_intermediates=True, output_fmt="NLC"
        )["text_intermediates"][-1]
        text = self.text_projector(text)
        direction = self.direction_embedding(direction_ids)[:, None]
        memory = torch.cat((text, direction), dim=1)
        valid = torch.cat(
            (input_ids.ne(0), torch.ones((input_ids.shape[0], 1), dtype=torch.bool, device=input_ids.device)),
            dim=1,
        )
        return memory, ~valid

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        action_bins: Optional[torch.Tensor] = None,
        land_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AeroVLALiteOutput:
        del kwargs
        visual = self._encode_visual(pixel_values)
        batch = visual.shape[0]
        visual = visual.flatten(1, 2)
        memory, memory_padding_mask = self._encode_memory(input_ids, direction_ids)
        for layer in self.layers:
            visual = layer(visual, memory, memory_padding_mask)
        query = self.policy_query.expand(batch, -1, -1)
        state, _ = self.policy_attention(query, self.policy_norm(visual), self.policy_norm(visual), need_weights=False)
        state = self.policy_norm(state[:, 0])
        forward_logits = self.forward_head(state)
        down_logits = self.down_head(state)
        yaw_logits = self.yaw_head(state)
        land_logits = self.land_head(state).squeeze(-1)

        loss = None
        if action_bins is not None and land_labels is not None:
            forward_loss = F.binary_cross_entropy_with_logits(
                forward_logits, self._ordinal_targets(action_bins[:, 0], self.config.num_bins, forward_logits.dtype)
            )
            down_loss = F.binary_cross_entropy_with_logits(
                down_logits, self._ordinal_targets(action_bins[:, 1], self.config.num_bins, down_logits.dtype)
            )
            yaw_loss = self._gaussian_cross_entropy(
                yaw_logits, action_bins[:, 2], self.config.yaw_label_smoothing_sigma
            )
            labels = land_labels.to(land_logits.dtype)
            weights = 1.0 + labels * (self.config.land_pos_weight - 1.0)
            bce = F.binary_cross_entropy_with_logits(land_logits, labels, reduction="none")
            probability = torch.sigmoid(land_logits)
            p_t = probability * labels + (1.0 - probability) * (1.0 - labels)
            land_loss = (weights * (1.0 - p_t).pow(self.config.land_focal_gamma) * bce).mean()
            loss = (forward_loss + down_loss + yaw_loss) / 3.0 + land_loss
        return AeroVLALiteOutput(
            loss=loss,
            forward_logits=forward_logits,
            down_logits=down_logits,
            yaw_logits=yaw_logits,
            land_logits=land_logits,
        )

    @torch.no_grad()
    def predict_actions(self, **inputs):
        output = self(**inputs)
        bins = torch.stack(
            (
                self._ordinal_prediction(output.forward_logits),
                self._ordinal_prediction(output.down_logits),
                output.yaw_logits.argmax(dim=-1),
            ),
            dim=-1,
        )
        return bins, torch.sigmoid(output.land_logits)
