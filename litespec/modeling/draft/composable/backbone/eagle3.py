"""Single-layer EAGLE3 with differentiable, functional TTT KV history.

The input fusion, decoder equations and training branch mask follow DeepSpec
005e03b / SpecForge (Apache-2.0). No mutable inference cache enters checkpoint
recomputation. Physical TTT slots and logical RoPE positions are independent.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half,
)

from ...dflash import _rope_config
from ...flex_attention import compile_friendly_flex_attention, create_flex_block_mask, flex_kernel_options


def training_mask(valid, length, step, *, block=False, backend="TRITON"):
    """First chunk: causal prefix; later chunks: same-anchor diagonal only."""
    lengths = valid.long().sum(-1) - step

    def mask_mod(b, h, q, k):
        prefix = (k < length) & (k <= q) & (k < lengths[b])
        branch = (k >= length) & (k % length == q)
        return (q < lengths[b]) & (prefix | branch)

    if block:
        return create_flex_block_mask(mask_mod, B=valid.shape[0], H=1, Q_LEN=length,
                                     KV_LEN=length * (step + 1), device=str(valid.device), backend=backend)
    q = torch.arange(length, device=valid.device)[None, :, None]
    k = torch.arange(length * (step + 1), device=valid.device)[None, None, :]
    return mask_mod(torch.arange(valid.shape[0], device=valid.device)[:, None, None],
                    0, q, k)[:, None]


@dataclass
class Eagle3State:
    """Request-local draft state; target KV is owned by the verifier."""
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    hidden: torch.Tensor | None = None

    @property
    def length(self):
        return 0 if self.key is None else self.key.shape[-2]

    def crop(self, length):
        if not 0 <= length <= self.length:
            raise ValueError("Invalid EAGLE3 cache crop")
        if self.key is not None:
            self.key = self.key[..., :length, :]
            self.value = self.value[..., :length, :]
        self.hidden = None


class Eagle3Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        profile = config.composable_config["backbone"]["layers"][0]["attention"]
        self.heads = profile["num_attention_heads"]
        self.kv_heads = profile["num_key_value_heads"]
        self.head_dim = profile["head_dim"]
        self.backend = profile["backend"]
        self.flex_backend = profile.get("flex_backend", "TRITON")
        width = config.hidden_size
        bias = config.attention_bias
        self.q_proj = nn.Linear(2 * width, self.heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(2 * width, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(2 * width, self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.heads * self.head_dim, width, bias=bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.hidden_norm = Qwen3RMSNorm(width, eps=config.rms_norm_eps)
        self.input_layernorm = Qwen3RMSNorm(width, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(width, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)
        rope_config = deepcopy(config)
        rope_config.head_dim = self.head_dim
        self.rotary_emb = Qwen3RotaryEmbedding(_rope_config(rope_config, "gqa"))

    def forward(self, hidden, embeds, positions, key, value, mask, use_flex=False):
        batch, length, _ = hidden.shape
        fused = torch.cat((self.input_layernorm(embeds), self.hidden_norm(hidden)), -1)
        q = self.q_norm(self.q_proj(fused).view(batch, length, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(fused).view(batch, length, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(fused).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(hidden, positions)
        cos, sin = cos[:, None], sin[:, None]
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        key, value = torch.cat((key, k), -2), torch.cat((value, v), -2)
        if use_flex:
            attended = compile_friendly_flex_attention(
                q, key.contiguous(), value.contiguous(), block_mask=mask, enable_gqa=True,
                kernel_options=flex_kernel_options(self.flex_backend))
        else:
            # Explicit mask also works for cached multi-token rebuilds.
            groups = self.heads // self.kv_heads
            k_all, v_all = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
            if self.backend == "eager":
                scores = (q.float() @ k_all.float().transpose(-1, -2)) * self.head_dim ** -0.5
                probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
                attended = probabilities.to(v_all.dtype) @ v_all
            else:
                attended = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        output = hidden + self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))
        output = output + self.mlp(self.post_attention_layernorm(output))
        return output, key, value


class Eagle3Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = Eagle3Decoder(config)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ttt_length = config.block_size - 1
        self.gradient_checkpointing = False

    def empty_kv(self, hidden):
        shape = (hidden.shape[0], self.layer.kv_heads, 0, self.layer.head_dim)
        return hidden.new_empty(shape), hidden.new_empty(shape)

    def ttt(self, hidden, input_ids, valid, embed_tokens, anchor_positions):
        """All L queries run at every depth; only output gathering samples anchors."""
        length = input_ids.shape[1]
        key, value = self.empty_kv(hidden)
        selected = []
        positions = torch.arange(length, device=hidden.device)[None].expand(hidden.shape[0], -1)
        ids = input_ids
        for step in range(self.ttt_length):
            ids = torch.cat((ids[:, 1:], torch.zeros_like(ids[:, :1])), 1)
            use_flex = self.layer.backend == "flex_attention"
            mask = training_mask(valid, length, step, block=use_flex, backend=self.layer.flex_backend)
            # Freeze loop bindings: checkpoint recomputation occurs after this loop.
            def operation(h, e, p, k, v, mask=mask, use_flex=use_flex):
                return self.layer(h, e, p, k, v, mask, use_flex)
            args = (hidden, embed_tokens(ids), positions + step, key, value)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden, key, value = checkpoint(operation, *args, use_reentrant=False)
            else:
                hidden, key, value = operation(*args)
            indices = (anchor_positions - 1).clamp_min(0)
            selected.append(hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1])))
        return torch.stack(selected, 2)

    def extend(self, hidden, token_embeddings, state):
        """Pure decoder step, then publish a new request-local cache state."""
        if hidden.shape[1] == 0 or hidden.shape != token_embeddings.shape:
            raise ValueError("EAGLE3 extension requires matching nonempty hidden and embedding tensors")
        previous = state.length
        key, value = self.empty_kv(hidden) if state.key is None else (state.key, state.value)
        positions = torch.arange(previous, previous + hidden.shape[1], device=hidden.device)[None]
        mask = (torch.arange(previous + hidden.shape[1], device=hidden.device)[None, :]
                <= positions[0, :, None])[None, None]
        output, key, value = self.layer(hidden, token_embeddings, positions, key, value, mask)
        state.key, state.value, state.hidden = key, value, output[:, -1:]
        return state.hidden
