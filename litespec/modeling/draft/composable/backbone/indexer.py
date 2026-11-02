"""One differentiable lightning scorer per anchor, shared across draft layers."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import IndexerAux, IndexerKeyState, SelectionView


def partial_rope(x, positions, width, theta):
    if not width:
        return x
    inv = theta ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = positions.float()[..., None] * inv
    while angles.ndim < x.ndim:
        angles = angles.unsqueeze(-2)
    first, second = x[..., :width // 2].float(), x[..., width // 2:width].float()
    rotated = torch.cat((first * angles.cos() - second * angles.sin(),
                         second * angles.cos() + first * angles.sin()), -1)
    return torch.cat((rotated.to(x.dtype), x[..., width:]), -1)


class AnchorIndexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recipe = dict(config.composable_config["indexer"])
        self.heads = self.recipe["num_query_heads"]
        self.dim = self.recipe["head_dim"]
        self.rope_dim = self.recipe["rope_head_dim"]
        self.topk = self.recipe["index_topk"]
        self.chunk_size = self.recipe["anchor_chunk_size"]
        self.theta = self.recipe.get("rope_theta", config.rope_theta)
        self.block_size = config.block_size
        self.query_projection = nn.Linear(getattr(config, "indexer_source_width", config.num_attention_heads * config.head_dim),
                                          self.heads * self.dim, bias=False)
        self.key_projection = nn.Linear(len(config.composable_config["context"]["target_layer_ids"])
                                        * config.hidden_size, self.dim, bias=False)
        self.key_norm = nn.LayerNorm(self.dim)
        self.weight_projection = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.fingerprint = tuple(config.composable_config["context"]["target_layer_ids"]) + (
            self.heads, self.dim, self.rope_dim, self.theta,
        )

    def forward(self, inputs, first_layer):
        n, a = inputs.anchor_positions.shape
        raw = inputs.target_hidden_states.flatten(2).detach()
        s = raw.size(1)
        positions = inputs.context_position_ids
        if positions is None:
            positions = torch.arange(s, device=raw.device)[None].expand(n, -1)
        valid = inputs.context_valid_mask
        if valid is None:
            valid = torch.ones_like(positions, dtype=torch.bool)
        documents = inputs.document_ids
        if documents is None:
            documents = torch.zeros_like(positions)
        keys = self.key_norm(self.key_projection(raw))
        keys = partial_rope(keys, positions, self.rope_dim, self.theta)
        if inputs.draft_state is not None:
            fingerprint = self.fingerprint + (id(self),) + tuple(p._version for p in self.parameters())
            old = inputs.draft_state.indexer_state
            if old is not None:
                if old.fingerprint != fingerprint:
                    raise ValueError("Indexer cache source/model fingerprint changed")
                if s and old.positions.numel() and (
                    positions[:, :1] <= old.positions.masked_fill(~old.valid, -1).amax(-1, keepdim=True)
                ).any():
                    raise ValueError("Indexer cache requires strictly new confirmed positions")
                keys = torch.cat((old.keys, keys), 1)
                positions = torch.cat((old.positions, positions), 1)
                valid = torch.cat((old.valid, valid), 1)
                documents = torch.cat((old.document_ids, documents), 1)
            valid = valid & (positions < inputs.anchor_positions[:, :1])
            inputs.draft_state.indexer_state = IndexerKeyState(
                keys.detach(), positions, documents, valid, fingerprint,
            )
        # Detach AFTER the shared source projections; indexer loss owns only its parameters.
        anchor = inputs.draft_embeddings.reshape(n, a, self.block_size, -1)[:, :, 0]
        u = first_layer.input_norm(anchor)
        attention = first_layer.self_attn
        source = attention.q_norm(attention.q_proj(u).reshape(n, a, -1, attention.head_dim))
        query = self.query_projection(source.flatten(-2).detach()).reshape(n, a, self.heads, self.dim)
        query = partial_rope(query, inputs.anchor_positions, self.rope_dim, self.theta)
        weights = self.weight_projection(u.detach()).float() / math.sqrt(self.heads)
        anchor_docs = inputs.anchor_document_ids
        if anchor_docs is None:
            anchor_docs = torch.zeros_like(inputs.anchor_positions)
        dense = self.recipe.get("stage", "sparse") == "warmup"
        width = keys.size(1) if dense else min(self.topk, keys.size(1))
        indices, selected_valid, logits, eligible = [], [], [], []
        for start in range(0, a, self.chunk_size):
            end = min(start + self.chunk_size, a)
            score = torch.einsum("nahd,nsd->nahs", query[:, start:end].float(), keys.float())
            score = (F.relu(score / math.sqrt(self.dim)) * weights[:, start:end, :, None]).sum(-2)
            legal = (positions[:, None] < inputs.anchor_positions[:, start:end, None])
            legal = legal & valid[:, None] & (documents[:, None] == anchor_docs[:, start:end, None])
            legal = legal & inputs.block_keep_mask[:, start:end, None]
            if dense:
                ids = torch.arange(width, device=raw.device)[None, None].expand(n, end-start, -1)
            else:
                # Stable sorting preserves absolute-position tie order without changing scores.
                order = positions.argsort(dim=-1, stable=True)[:, None].expand(n, end-start, -1)
                ranked = score.detach().masked_fill(~legal, -torch.inf).gather(-1, order)
                ids = order.gather(-1, ranked.argsort(dim=-1, descending=True, stable=True)[..., :width])
                ids = ids.gather(-1, positions[:, None].expand(n,end-start,-1).gather(-1,ids).argsort(dim=-1,stable=True))
            selected_valid.append(legal.gather(-1, ids))
            indices.append(ids)
            logits.append(score.gather(-1, ids))
            supervised = inputs.block_keep_mask[:, start:end]
            if inputs.prediction_mask is not None:
                supervised = supervised & (inputs.prediction_mask[:, start:end] > 0).any(-1)
            eligible.append(supervised & legal.any(-1))
        ids = torch.cat(indices, 1)
        return SelectionView(
            ids, positions[:, None].expand(n,a,-1).gather(-1,ids),
            torch.cat(selected_valid, 1), torch.cat(logits, 1), positions, documents, valid,
            torch.cat(eligible, 1), self.recipe.get("execution", "gather"), self.chunk_size,
        )

    @staticmethod
    def auxiliary(selection, teacher):
        scores, valid = selection.scores, selection.selected_valid
        eligible = selection.anchor_valid
        if not scores.size(-1):
            return IndexerAux(scores.sum() * 0, eligible.sum().float())
        safe = scores.masked_fill(~valid, -torch.inf)
        safe = torch.where(valid.any(-1, keepdim=True), safe, torch.zeros_like(safe))
        logq = safe.log_softmax(-1).masked_fill(~valid, 0)
        teacher = teacher.detach()
        divergence = (teacher * (teacher.clamp_min(1e-30).log() - logq)).sum(-1)
        return IndexerAux((divergence * eligible).sum(), eligible.sum().float())
