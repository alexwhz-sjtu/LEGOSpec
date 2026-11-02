# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Adapted from Transformers Qwen3.5, aad13b87ed59f2afcfaebc985f403301887a35fc.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""GDN context writes and identity-transition block reads (not softmax attention).

Q/Z come from draft blocks; K/V/a/b come from confirmed target context. Q and
context use independent causal depthwise convolutions. Read-only events reuse
the official chunk recurrence with exactly zero beta and log-decay.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..gated_delta_state import GDNLayerState


def l2norm(x):
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def reference_event_scan(q, k, v, g, beta, cu_seqlens_cpu, initial_state=None):
    """Small, differentiable FP32 oracle; never a silent CUDA training fallback."""
    outputs, states = [], []
    boundaries = cu_seqlens_cpu.tolist()
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (
            v.new_zeros(v.size(2), k.size(-1), v.size(-1), dtype=torch.float32)
            if initial_state is None else initial_state[seq].float()
        )
        for t in range(start, end):
            qt, kt = l2norm(q[0, t]), l2norm(k[0, t])
            repeat = v.size(2) // k.size(2)
            if repeat != 1:
                qt = qt.repeat_interleave(repeat, 0)
                kt = kt.repeat_interleave(repeat, 0)
            state = state * g[0, t].float().exp()[:, None, None]
            old = (state * kt[:, :, None]).sum(-2)
            delta = beta[0, t].float()[:, None] * (v[0, t].float() - old)
            state = state + kt[:, :, None] * delta[:, None, :]
            outputs.append((state * (qt / math.sqrt(q.size(-1)))[:, :, None]).sum(-2))
        states.append(state)
    return torch.stack(outputs)[None].to(q.dtype), torch.stack(states)


class GDNRMSNormGated(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * F.silu(z.float())).to(dtype)


class GDNContextMemoryMixer(nn.Module):
    def __init__(self, config, recipe):
        super().__init__()
        self.hk = recipe["linear_num_key_heads"]
        self.hv = recipe["linear_num_value_heads"]
        self.dk = recipe["linear_key_head_dim"]
        self.dv = recipe["linear_value_head_dim"]
        self.q_kernel = recipe["query_conv_kernel_size"]
        self.q_mode = recipe.get("query_conv_mode", "block")
        self.context_kernel = recipe["context_conv_kernel_size"]
        self.backend = recipe["backend"]
        self.key_dim, self.value_dim = self.hk * self.dk, self.hv * self.dv
        self.in_proj_q = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.in_proj_kv = nn.Linear(config.hidden_size, self.key_dim + self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.hv, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.hv, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.query_conv = nn.Conv1d(
            self.key_dim, self.key_dim, self.q_kernel, groups=self.key_dim, bias=False
        )
        channels = self.key_dim + self.value_dim
        self.context_conv = nn.Conv1d(
            channels, channels, self.context_kernel, groups=channels, bias=False
        )
        self.A_log = nn.Parameter(torch.empty(self.hv))
        self.dt_bias = nn.Parameter(torch.ones(self.hv))
        self.norm = GDNRMSNormGated(self.dv, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.reset_gdn_parameters()

    def reset_gdn_parameters(self):
        with torch.no_grad():
            self.dt_bias.fill_(1)
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(0, 16).clamp_min_(1e-6).log_())

    def context_convolution(self, raw, layout, previous):
        """Pack real context only; document loops do not replicate prefixes."""
        if layout.dense_shape is not None:
            # Fixed-capacity rows are compacted on device. Invalid rows follow
            # all real context, so causal convolution cannot contaminate it.
            if previous is not None:
                raise ValueError("Dense GDN events cannot update a persistent cache")
            n, _ = layout.dense_shape
            width = raw.size(0) // n
            if not width:
                return raw + self.context_conv.weight.sum() * 0, None
            value = raw.reshape(n, width, -1).transpose(1, 2)
            value = F.silu(self.context_conv(F.pad(value, (self.context_kernel - 1, 0))))
            return value.transpose(1, 2).reshape_as(raw), None
        outputs, tails, offset = [], [], 0
        history = self.context_kernel - 1
        for sample, length in zip(layout.samples, layout.context_lengths):
            part = raw[offset:offset + length].T[None]
            tail = (
                part.new_zeros(1, raw.size(-1), history)
                if previous is None or previous.kv_conv_state is None
                else previous.kv_conv_state[sample:sample + 1].to(part)
            )
            joined = torch.cat((tail, part), -1)
            if length:
                value = F.silu(self.context_conv(joined)).squeeze(0).T
            else:
                value = raw[offset:offset] + self.context_conv.weight.sum() * 0
            outputs.append(value)
            if history:
                tails.append(joined[0, :, -history:])
            offset += length
        value = torch.cat(outputs) if outputs else raw + self.context_conv.weight.sum() * 0
        return value, torch.stack(tails) if tails else None

    def query_prefix(self, raw_context, inputs, previous=None, positions=None):
        """Gather exact confirmed positions on device, independently for each anchor.

        Cached Q values are pre-convolution projections, with the same layer's
        input norm and Wq as the slots. They never include draft slot values.
        """
        n, s, d = raw_context.shape
        anchors = inputs.anchor_positions if positions is None else positions
        h = self.q_kernel - 1
        wanted = anchors[..., None] + torch.arange(-h, 0, device=anchors.device)
        valid = inputs.context_valid_mask
        valid = torch.ones(n, s, device=anchors.device, dtype=torch.bool) if valid is None else valid.bool()
        if inputs.context_position_ids is None and previous is None and inputs.document_ids is None and inputs.anchor_document_ids is None:
            if not s:
                return raw_context.new_zeros(n, anchors.size(1), h, d)
            idx = wanted.clamp(0, s - 1)
            rows = torch.arange(n, device=anchors.device)[:, None, None]
            found = (wanted >= 0) & (wanted < s) & valid[rows, idx]
            return raw_context[rows, idx].masked_fill(~found[..., None], 0)
        cp = inputs.context_position_ids
        cp = torch.arange(s, device=anchors.device).expand(n, -1) if cp is None else cp
        docs = inputs.document_ids
        docs = torch.zeros_like(cp) if docs is None else docs
        ad = inputs.anchor_document_ids
        ad = torch.zeros_like(anchors) if ad is None else ad
        values = raw_context
        if previous is not None:
            # A commit is contiguous; the new anchor follows all supplied context.
            start = inputs.anchor_positions[:, :1] - valid.sum(-1, keepdim=True)
            hp = start + torch.arange(-h, 0, device=cp.device)
            values = torch.cat((previous.q_conv_state.transpose(1, 2).to(values), values), 1)
            cp = torch.cat((hp, cp), 1)
            docs = torch.cat((ad[:, :1].expand(-1, h), docs), 1)
            valid = torch.cat((hp >= 0, valid), 1)
        if not cp.size(1):
            return raw_context.new_zeros(n, anchors.size(1), h, d)
        stride = torch.cat((cp.flatten(), anchors.flatten(), cp.new_zeros(1))).amax().clamp_min(0) + 2
        keys = (docs * stride + cp).masked_fill(~valid, torch.iinfo(torch.long).max)
        keys, order = keys.sort(dim=-1, stable=True)
        targets = ad[..., None] * stride + wanted
        indices = torch.searchsorted(keys.contiguous(), targets.flatten(1).contiguous()).clamp_max(keys.size(1) - 1)
        found = keys.gather(1, indices) == targets.flatten(1)
        found = found.reshape_as(wanted) & (wanted >= 0)
        indices = order.gather(1, indices)
        gathered = values.gather(1, indices[..., None].expand(-1, -1, d)).reshape(n, anchors.size(1), h, d)
        return gathered.masked_fill(~found[..., None], 0)

    def forward(self, hidden, context, layout, block_size, previous=None, *, inputs=None, query_context=None):
        n, length, _ = hidden.shape
        a = length // block_size
        raw_q = self.in_proj_q(hidden).reshape(n * a, block_size, self.key_dim).transpose(1, 2)
        q_tail = None
        if self.q_mode == "confirmed_history":
            if inputs is None or query_context is None:
                raise ValueError("Historical Q convolution requires context and ModelInputs")
            raw_context_q = self.in_proj_q(query_context)
            prefix = self.query_prefix(raw_context_q, inputs, previous)
            joined = torch.cat((prefix.reshape(n * a, self.q_kernel - 1, self.key_dim).transpose(1, 2), raw_q), -1)
            q = F.silu(self.query_conv(joined))
            if previous is not None:
                q_tail = self.query_prefix(raw_context_q, inputs, previous, inputs.anchor_positions[:, :1])[:, 0].transpose(1, 2).detach()
        else:
            q = F.silu(self.query_conv(F.pad(raw_q, (self.q_kernel - 1, 0))))
        q = q.transpose(1, 2).reshape(n * length, self.hk, self.dk)
        z = self.in_proj_z(hidden).reshape(n, length, self.hv, self.dv)
        c = context.flatten(0, 1).index_select(0, layout.context_indices)
        if layout.context_keep is not None:
            c = c.masked_fill(~layout.context_keep[:, None], 0)
        raw_kv = self.in_proj_kv(c)
        mixed, tails = self.context_convolution(raw_kv, layout, previous)
        k, v = mixed.split((self.key_dim, self.value_dim), -1)
        k, v = k.reshape(-1, self.hk, self.dk), v.reshape(-1, self.hv, self.dv)
        beta = self.in_proj_b(c).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(c).float() + self.dt_bias.float())
        if layout.context_keep is not None:
            beta = beta.masked_fill(~layout.context_keep[:, None], 0)
            g = g.masked_fill(~layout.context_keep[:, None], 0)
        # index_copy is differentiable, including empty source indices. Read
        # gates MUST stay zero: never activate/sigmoid these events in-kernel.
        e = layout.num_events
        if e:
            qe = q.new_zeros(1, e, self.hk, self.dk).index_copy(
                1, layout.read_events, q.index_select(0, layout.query_indices)[None]
            )
            ke = k.new_zeros(1, e, self.hk, self.dk).index_copy(1, layout.write_events, k[None])
            ve = v.new_zeros(1, e, self.hv, self.dv).index_copy(1, layout.write_events, v[None])
            ge = g.new_zeros(1, e, self.hv).index_copy(1, layout.write_events, g[None])
            be = beta.new_zeros(1, e, self.hv).index_copy(1, layout.write_events, beta[None])
            initial = None if previous is None else previous.recurrent_state
            if self.backend == "reference" or (self.backend == "auto" and not hidden.is_cuda):
                output, final = reference_event_scan(qe, ke, ve, ge, be, layout.cu_seqlens_cpu, initial)
            else:
                if not hidden.is_cuda:
                    raise ValueError("GDN FLA backend requires CUDA; use backend=reference on CPU")
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule

                dense = layout.dense_shape
                def scan_shape(value):
                    return value if dense is None else value.reshape(*dense, *value.shape[2:])
                output, final = chunk_gated_delta_rule(
                    *map(scan_shape, (qe, ke, ve, ge, be)),
                    initial_state=initial, output_final_state=previous is not None,
                    use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=False,
                    use_gate_in_kernel=False, state_v_first=False,
                    cu_seqlens=None if dense is not None else layout.cu_seqlens,
                    cu_seqlens_cpu=None if dense is not None else layout.cu_seqlens_cpu,
                )
                if dense is not None:
                    output = output.reshape(1, e, self.hv, self.dv)
            selected = output[0].index_select(0, layout.read_events)
            if layout.query_keep is not None:
                selected = selected.masked_fill(~layout.query_keep[:, None, None], 0)
            out = v.new_zeros(n * length, self.hv, self.dv).index_copy(0, layout.query_indices, selected)
        else:
            out = z.flatten(0, 1) * 0 + (q.sum() + mixed.sum() + beta.sum() + g.sum()) * 0
            final = None
        out = self.norm(out.reshape(n, length, self.hv, self.dv), z)
        result = self.out_proj(out.flatten(-2))
        state = None
        if previous is not None:
            state = GDNLayerState(final.float().detach(), None if tails is None else tails.detach(), self.context_kernel, q_tail)
        return result, state
