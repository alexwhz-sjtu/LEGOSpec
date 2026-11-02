"""Confirmed-context GDN writes and independent, read-only draft queries."""

import torch
from sglang.kernels.ops.attention.legospec_gdn import (
    legospec_gdn_gated_norm,
    legospec_gdn_read,
)
from sglang.kernels.ops.attention.legospec_gdn_mixed import (
    cache_gdn_chunk,
    mixed_gdn_read,
    pack_gdn_events,
    rewind_gdn_partial_chunk,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.mem_cache.kv.router import STATE_PART
from sglang.srt.mem_cache.legospec_gdn_pool import GDNMixedLayout
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.models.dflash import DFlashMLP
from torch import nn


@torch.compile(dynamic=True, options={"emulate_precision_casts": True})
def _gates(a, b, a_log, dt_bias):
    return -a_log.float().exp() * torch.nn.functional.softplus(
        a.float() + dt_bias.float()
    ), b.sigmoid()


class _GatedNorm(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(128))
        self.eps = eps

    def forward(self, x, z):
        return legospec_gdn_gated_norm(x, z, self.weight, self.eps)


class LEGOSpecGDNMixer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        recipe = config.composable_config["attention_configs"]["gdn"]
        expected = dict(
            context_conv_kernel_size=4,
            query_conv_kernel_size=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            query_conv_mode="block",
        )
        if any(recipe.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Unaudited LEGOSpec GDN geometry: {recipe}")
        for name, width in (
            ("q", 2048),
            ("kv", 6144),
            ("a", 32),
            ("b", 32),
            ("z", 4096),
        ):
            setattr(
                self,
                f"in_proj_{name}",
                ReplicatedLinear(config.hidden_size, width, bias=False),
            )
        self.query_conv = nn.Conv1d(2048, 2048, 2, groups=2048, bias=False)
        self.context_conv = nn.Conv1d(6144, 6144, 4, groups=6144, bias=False)
        self.A_log = nn.Parameter(torch.empty(32))
        self.dt_bias = nn.Parameter(torch.empty(32))
        self.norm = _GatedNorm(config.rms_norm_eps)
        self.out_proj = ReplicatedLinear(4096, config.hidden_size, bias=False)
        self._query_layouts = {}
        self._mixed_layouts = {}
        self.qwen35_mixed_precision = True

    def forward(self, hidden, batch):
        bs = hidden.shape[0] // 8
        cu = self._query_layouts.get(bs)
        if cu is None:
            cu = torch.arange(bs + 1, device=hidden.device, dtype=torch.int32) * 8
            self._query_layouts[bs] = cu
        raw, _ = self.in_proj_q(hidden)
        q = causal_conv1d_fn(
            raw.T,
            self.query_conv.weight[:, 0],
            None,
            None,
            cu,
            [8] * bs,
            activation="silu",
            round_before_activation=True,
            fp32_product=True,
        ).T.contiguous()
        z, _ = self.in_proj_z(hidden)
        pool = get_forward_context().attn_backend.token_to_kv_pool
        state_view = pool.layer_router.read(self.layer_id, STATE_PART).view
        if self.qwen35_mixed_precision:
            from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

            cache = state_view.read_cache(self.layer_id)
            q = l2norm_fwd(q.view(bs, 8, 16, 128))
            readout = mixed_gdn_read(
                q,
                state_view.state(self.layer_id),
                cache.entry_state,
                cache.key,
                cache.transformed_value,
                cache.cumulative_gate,
                cache.count,
                batch.req_pool_indices[:bs],
            )
        else:
            readout = legospec_gdn_read(
                q, state_view.state(self.layer_id), batch.req_pool_indices[:bs]
            )
        normalized = self.norm(readout, z.view_as(readout))
        return self.out_proj(normalized.flatten(0, 1).flatten(1))[0]

    def commit(self, context, pool, layout, indices, commit_lens):
        raw, _ = self.in_proj_kv(context)
        a, _ = self.in_proj_a(context)
        b, _ = self.in_proj_b(context)
        g, beta = _gates(a, b, self.A_log, self.dt_bias)
        g, beta = g.contiguous(), beta.contiguous()
        state, tail = pool.state(self.layer_id), pool.tail(self.layer_id)
        if layout.lengths_cpu is not None:
            mixed = causal_conv1d_fn(
                raw.T,
                self.context_conv.weight[:, 0],
                None,
                tail,
                layout.cu_seqlens,
                layout.lengths_cpu,
                cache_indices=indices,
                has_initial_state=torch.ones_like(indices, dtype=torch.bool),
                activation="silu",
                round_before_activation=True,
                fp32_product=True,
            ).T.contiguous()
            if self.qwen35_mixed_precision:
                self._mixed_update(mixed, g, beta, pool, layout, indices, commit_lens)
            else:
                self._prefill_state(mixed, g, beta, state, layout, indices)
        else:
            bs = indices.numel()
            mixed = (
                causal_conv1d_update(
                    raw.view(bs, 8, 6144).transpose(1, 2),
                    tail,
                    self.context_conv.weight[:, 0],
                    activation="silu",
                    conv_state_indices=indices,
                    valid_token_lens=commit_lens,
                    round_before_activation=True,
                    fp32_product=True,
                )
                .transpose(1, 2)
                .contiguous()
            )
            if self.qwen35_mixed_precision:
                self._mixed_update(
                    mixed.flatten(0, 1), g, beta, pool, layout, indices, commit_lens
                )
                return
            k, v = mixed.split((2048, 4096), -1)
            g = g.view(bs, 8, 32)
            beta = beta.view(bs, 8, 32)
            from sglang.kernels.ops.attention.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule_update,
            )

            k = k.contiguous().view(bs, 8, 16, 128)
            v = v.contiguous().view(bs, 8, 32, 128)
            fused_recurrent_gated_delta_rule_update(
                k,
                k,
                v,
                g,
                beta,
                initial_state_source=state,
                initial_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                disable_output_calculation=True,
                valid_token_lens=commit_lens,
            )

    def _mixed_update(self, mixed, g, beta, pool, layout, indices, lengths):
        from sglang.kernels.ops.attention.fla.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h,
        )
        from sglang.kernels.ops.attention.fla.chunk_fwd import (
            chunk_gated_delta_rule_fwd_intra,
        )
        from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

        state, cache = pool.state(self.layer_id), pool.read_cache(self.layer_id)
        k, v = mixed.split((2048, 4096), -1)
        k = l2norm_fwd(k.contiguous().view(1, -1, 16, 128))
        v = v.contiguous().view(1, -1, 32, 128)
        short = layout.lengths_cpu is None
        if short:
            bs = indices.numel()
            event = self._mixed_layouts.get(bs)
            if event is None:
                event = GDNMixedLayout(
                    cu_seqlens=torch.arange(
                        bs + 1, device=mixed.device, dtype=torch.int32
                    )
                    * 80,
                    carry=torch.empty(bs, device=mixed.device, dtype=torch.int32),
                    read_chunks=torch.empty(bs, device=mixed.device, dtype=torch.int32),
                    total_events=bs * 80,
                    valid_event_lens=torch.empty(
                        bs, device=mixed.device, dtype=torch.int32
                    ),
                )
                self._mixed_layouts[bs] = event
            # Keep the canonical mixed-precision chunk across decode commits.
            # Replaying only its <=63 cached events preserves BF16 WY/readout
            # rounding; starting a fresh chunk at every commit does not.
            event.carry.copy_(
                torch.where(indices > 0, cache.count[indices.clamp_min(0)], 0)
            )
            event.read_chunks.copy_((event.carry + lengths) // 64)
            event.valid_event_lens.copy_(event.carry + lengths + 8)
        else:
            event = layout.mixed
            if event is None:
                raise ValueError("Mixed GDN prefill requires event layout")
        rewind_gdn_partial_chunk(state, cache.entry_state, indices, event.carry)
        pk, pv, pg, pb = pack_gdn_events(
            k,
            v,
            g,
            beta,
            indices,
            layout.cu_seqlens,
            event.cu_seqlens,
            event.carry,
            cache.key,
            cache.value,
            cache.gate,
            cache.beta,
            total_events=event.total_events,
            lengths=lengths,
        )
        if short:
            pk, pv, pg, pb = (x.view(bs, 80, *x.shape[2:]) for x in (pk, pv, pg, pb))
        cu = None if short else event.cu_seqlens
        gc = chunk_local_cumsum(
            pg,
            64,
            scale=1.4426950408889634,
            cu_seqlens=cu,
            chunk_indices=event.chunk_indices,
        )
        w, u, _ = chunk_gated_delta_rule_fwd_intra(
            pk,
            pv,
            gc,
            pb,
            cu_seqlens=cu,
            chunk_indices=event.chunk_indices,
            use_exp2=True,
        )
        chunk_gated_delta_rule_fwd_h(
            pk,
            w,
            u,
            gc,
            initial_state=state,
            initial_state_indices=indices,
            save_new_value=False,
            save_chunk_states=False,
            cu_seqlens=cu,
            chunk_indices=event.chunk_indices,
            chunk_offsets=event.chunk_offsets,
            use_exp2=True,
            track_state=cache.entry_state,
            track_chunk_idx=event.read_chunks,
            track_pool_index=True,
            tracked_values=cache.transformed_value,
            valid_token_lens=event.valid_event_lens,
        )
        cache_gdn_chunk(
            pk,
            pv,
            pg,
            pb,
            gc,
            indices,
            layout.cu_seqlens,
            event.cu_seqlens,
            event.carry,
            cache.key,
            cache.value,
            cache.gate,
            cache.beta,
            cache.cumulative_gate,
            cache.count,
            lengths=lengths,
        )

    @staticmethod
    def _prefill_state(mixed, g, beta, state, layout, indices):
        from sglang.kernels.ops.attention.fla.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h,
        )
        from sglang.kernels.ops.attention.fla.chunk_fwd import (
            chunk_gated_delta_rule_fwd_intra,
        )
        from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

        k, v = mixed.split((2048, 4096), -1)
        # Preserve the FP32 recurrence through the chunk intermediates too.
        k = l2norm_fwd(k.contiguous().view(1, -1, 16, 128), output_dtype=torch.float32)
        v = v.contiguous().view(1, -1, 32, 128).float()
        # Only the small scalar-gate prefix is wide. Its differences must retain
        # tiny decays even after -1000 gates; keys, values and state stay FP32.
        g = chunk_local_cumsum(
            g.view(1, -1, 32),
            64,
            cu_seqlens=layout.cu_seqlens,
            chunk_indices=layout.chunk_indices,
            output_dtype=torch.float64,
        )
        w, u, _ = chunk_gated_delta_rule_fwd_intra(
            k,
            v,
            g,
            beta.view(1, -1, 32).float(),
            cu_seqlens=layout.cu_seqlens,
            chunk_indices=layout.chunk_indices,
            use_fp32_math=True,
        )
        chunk_gated_delta_rule_fwd_h(
            k,
            w,
            u,
            g,
            initial_state=state,
            initial_state_indices=indices,
            save_new_value=False,
            save_chunk_states=False,
            cu_seqlens=layout.cu_seqlens,
            chunk_indices=layout.chunk_indices,
            chunk_offsets=layout.chunk_offsets,
            use_fp32_math=True,
        )


class LEGOSpecGDNLayer(nn.Module):
    def __init__(self, config, layer_id, norm_cls):
        super().__init__()
        self.attention_conv = self.mlp_conv = None
        self.block_size = 8
        self.position_embedding = nn.Embedding(8, config.hidden_size)
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mixer = LEGOSpecGDNMixer(config, layer_id)
        self.mlp = DFlashMLP(config)
        self.mlp.act_fn.hf_semantics = True
        operator = config.composable_config['backbone']['layers'][layer_id].get('local_operator', {'type': 'none'})
        if operator['type'] == 'dflash2_conv':
            from sglang.srt.models.dflash import DFlashGroupedConv
            def conv():
                return DFlashGroupedConv(config.hidden_size, 8, operator['kernel_size'], operator['group_size'], hf_semantics=True)
            self.attention_conv, self.mlp_conv = conv(), conv()
        elif operator['type'] != 'none':
            raise ValueError('Unsupported GDN local operator')

    def forward(self, positions, hidden, batch, residual=None, **kwargs):
        if not hidden.numel():
            return hidden, hidden if residual is None else residual
        if residual is not None:
            hidden = hidden + residual
        hidden = (
            hidden.view(-1, 8, hidden.shape[-1]) + self.position_embedding.weight
        ).flatten(0, 1)
        normalized = self.input_layernorm(hidden)
        kernel = None
        if self.attention_conv is not None:
            normalized, kernel = self.attention_conv.prepare(normalized)
        mixed = self.mixer(normalized, batch)
        if self.attention_conv is not None:
            mixed = self.attention_conv.finish(mixed, kernel)
        normed, residual = self.post_attention_layernorm(mixed, hidden)
        if self.mlp_conv is not None:
            normed, kernel = self.mlp_conv.prepare(normed)
        output = self.mlp(normed)
        if self.mlp_conv is not None:
            output = self.mlp_conv.finish(output, kernel)
        return output, residual
