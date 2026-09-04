"""A minimal-contrast AeroVLA ablation that replaces only the LLM layers.

The model retains OpenVLA's frozen vision tower, LLaMA tokenizer/embedding
table, original prompt, and autoregressive action-token objective.  A compact
causal decoder with cross-attention to visual patches replaces the 7B LLaMA
transformer stack.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

try:
    from src.aerovla_nollm_model import DinoSiglipVisionBackbone, load_openvla_vision_weights
except ModuleNotFoundError:
    from aerovla_nollm_model import DinoSiglipVisionBackbone, load_openvla_vision_weights


def _checkpoint_tensor(openvla_path: str, key: str) -> torch.Tensor:
    index_path = os.path.join(openvla_path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as handle:
        shard = json.load(handle)["weight_map"][key]
    with safe_open(os.path.join(openvla_path, shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


class AeroVLACrossAttentionConfig(PretrainedConfig):
    model_type = "aerovla_crossattn"

    def __init__(
        self,
        openvla_path: str = "./openvla-7b",
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_text_tokens: int = 256,
        vocab_size: int = 32064,
        embedding_dim: int = 4096,
        pad_token_id: int = 32000,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        frozen_backbone_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if d_model % num_heads:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.openvla_path = openvla_path
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_text_tokens = max_text_tokens
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.frozen_backbone_dtype = frozen_backbone_dtype


@dataclass
class AeroVLACrossAttentionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CrossAttentionDecoderLayer(nn.Module):
    """Pre-norm causal decoder layer with explicit visual cross-attention."""

    def __init__(self, config: AeroVLACrossAttentionConfig) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(config.d_model)
        self.self_attention = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = nn.MultiheadAttention(
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
        self,
        hidden: torch.Tensor,
        visual: torch.Tensor,
        causal_mask: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        normalized = self.self_norm(hidden)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout(attended)
        normalized = self.cross_norm(hidden)
        attended, _ = self.cross_attention(normalized, visual, visual, need_weights=False)
        hidden = hidden + self.dropout(attended)
        return hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))


class AeroVLACrossAttentionModel(PreTrainedModel):
    config_class = AeroVLACrossAttentionConfig
    base_model_prefix = "aerovla_crossattn"
    _keys_to_ignore_on_load_missing = [r"vision_backbone\..*", r"token_embedding\..*"]

    def __init__(self, config: AeroVLACrossAttentionConfig, initialize_backbones: bool = True) -> None:
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

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, config.pad_token_id)
        if initialize_backbones:
            pretrained = _checkpoint_tensor(
                config.openvla_path, "language_model.model.embed_tokens.weight"
            )
            if pretrained.shape[0] < config.vocab_size or pretrained.shape[1] != config.embedding_dim:
                raise RuntimeError(
                    f"Token embedding mismatch: checkpoint={tuple(pretrained.shape)}, "
                    f"configured={tuple(self.token_embedding.weight.shape)}"
                )
            # Match the LLM baseline resize to the real tokenizer vocabulary; the checkpoint has padded rows.
            self.token_embedding.weight = nn.Parameter(pretrained[: config.vocab_size], requires_grad=False)

        self.text_projector = nn.Sequential(
            nn.LayerNorm(config.embedding_dim), nn.Linear(config.embedding_dim, config.d_model)
        )
        vision_dim = self.vision_backbone.embed_dim
        self.vision_projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.d_model),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_text_tokens, config.d_model))
        self.layers = nn.ModuleList(CrossAttentionDecoderLayer(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.freeze_backbones()

    def freeze_backbones(self) -> None:
        self.vision_backbone.requires_grad_(False).eval()
        self.token_embedding.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_backbone.eval()
        return self

    def state_dict(self, *args, **kwargs):
        complete = super().state_dict(*args, **kwargs)
        frozen_prefixes = ("vision_backbone.", "token_embedding.")
        return {key: value for key, value in complete.items() if not key.startswith(frozen_prefixes)}

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AeroVLACrossAttentionOutput:
        del kwargs
        if pixel_values is None:
            raise ValueError("pixel_values are required for the Cross-Attention ablation")
        if input_ids.shape[1] > self.config.max_text_tokens:
            raise ValueError(
                f"Text sequence {input_ids.shape[1]} exceeds max_text_tokens={self.config.max_text_tokens}"
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        with torch.no_grad():
            visual = self.vision_backbone(pixel_values)
            token_embeddings = self.token_embedding(input_ids)
        visual = self.vision_projector(visual.to(self.vision_projector[1].weight.dtype))
        hidden = self.text_projector(token_embeddings.to(self.text_projector[1].weight.dtype))
        hidden = hidden + self.position_embedding[:, : hidden.shape[1]]
        causal_mask = torch.triu(
            torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=hidden.device), diagonal=1
        )
        padding_mask = ~attention_mask.bool()
        for layer in self.layers:
            hidden = layer(hidden, visual, causal_mask, padding_mask)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return AeroVLACrossAttentionOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding; short UAV outputs make KV caching unnecessary."""
        generated = input_ids
        mask = attention_mask
        # Encode the two camera views once per control step, not once per generated token.
        visual = self.vision_backbone(pixel_values)
        visual = self.vision_projector(visual.to(self.vision_projector[1].weight.dtype))
        for _ in range(max_new_tokens):
            token_embeddings = self.token_embedding(generated)
            hidden = self.text_projector(token_embeddings.to(self.text_projector[1].weight.dtype))
            hidden = hidden + self.position_embedding[:, : hidden.shape[1]]
            causal_mask = torch.triu(
                torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=hidden.device), diagonal=1
            )
            for layer in self.layers:
                hidden = layer(hidden, visual, causal_mask, ~mask.bool())
            logits = self.lm_head(self.final_norm(hidden))
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
            if torch.all(next_token.squeeze(1).eq(self.config.eos_token_id)):
                break
        return generated
