"""Native LEGOSpec draft, using SGLang attention, MLP and proposal modules."""

from copy import deepcopy
from typing import Iterable

import torch
from sglang.kernels.ops.activation import dspark_rnn_update
from sglang.kernels.ops.gemm.legospec_linear import legospec_cublas_linear
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.legospec_linear import use_legospec_cublas
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlashDraftModel,
    DFlashGroupedConv,
)
from sglang.srt.models.dspark import (
    DSparkDraftMixin,
    RNNHead,
    gather_and_crop_vocab,
)


class LEGOSpecRMSNorm(RMSNorm):
    hf_semantics = True
    torch29_semantics = True

    def forward_cuda(
        self, x, residual=None, post_residual_addition=None, quant_linear=None
    ):
        from sglang.kernels.ops.layernorm.torch29_rmsnorm import torch29_rmsnorm

        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        return torch29_rmsnorm(x, self.weight, self.variance_epsilon, residual)


class LEGOSpecRNNHead(RNNHead):
    """DSpark head with a fused BF16 pointwise update; projections are shared."""

    def _rnn_step(self, state, prev_embeddings, hidden_states):
        raw = self.joint_proj(torch.cat((state, prev_embeddings, hidden_states), -1))
        new_state, latent = dspark_rnn_update(state, raw)
        return new_state, self.project_bias(latent)


class LEGOSpecDraftModel(DSparkDraftMixin, DFlashDraftModel):
    supports_pre_gather_target_hidden_projection = False
    context_linear = staticmethod(legospec_cublas_linear)

    def __init__(self, config, quant_config=None, prefix=""):
        if quant_config is not None:
            raise ValueError("LEGOSpec v1 requires unquantized BF16 weights")
        # Reuse the runtime methods, without DSpark's checkpoint/head policy.
        DFlashDraftModel.__init__(self, config, quant_config, prefix)
        recipe = config.composable_config
        self.num_stages = int(config.num_hidden_layers)
        self.gamma = self.block_size - 1
        self.sample_from_anchor = False
        self.confidence_head = None
        self.lm_head = None
        self.logits_mup_width_multiplier = None
        self._fused_kv_write_cache = None
        self._window_kv_write_cache = None
        self._base_logits_gatherer = None
        self.proposal_type = recipe["proposal"]["type"]
        self.markov_head = (
            LEGOSpecRNNHead(
                vocab_size=config.vocab_size,
                markov_rank=recipe["proposal"]["rank"],
                hidden_size=config.hidden_size,
            )
            if self.proposal_type == "rnn"
            else None
        )
        if self.proposal_type in ("selector", "residual_selector", "topk_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                state_rank=recipe["proposal"]["rank"],
                top_k=recipe["proposal"]["top_k"],
            )
        # HF constructs inverse frequencies on CPU before moving the model.
        # Keep this private: get_rope() may have returned a target-shared object.
        rotary = deepcopy(self.layers[0].self_attn.rotary_emb)
        inv = 1.0 / (
            float(config.rope_theta)
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32, device="cpu")
                / config.head_dim
            )
        )
        positions = torch.arange(
            rotary.max_position_embeddings,
            dtype=torch.float32,
            device=rotary.cos_sin_cache.device,
        )
        frequencies = torch.outer(positions, inv.to(positions.device))
        rotary.cos_sin_cache = torch.cat((frequencies.cos(), frequencies.sin()), -1)
        for layer_id, (layer, spec) in enumerate(
            zip(self.layers, recipe["backbone"]["layers"])
        ):
            if spec["attention"] == "gdn":
                if (
                    spec["access"] != "linear"
                ):
                    raise ValueError(
                        "LEGOSpec GDN requires linear access"
                    )
                from sglang.srt.layers.attention.legospec_gdn import LEGOSpecGDNLayer

                self.layers[layer_id] = LEGOSpecGDNLayer(
                    config, layer_id, LEGOSpecRMSNorm
                )
                continue
            layer.self_attn.rotary_emb = rotary
            layer.self_attn.context_linear = legospec_cublas_linear
            if spec["access"] not in ("dense", "sliding_window"):
                raise ValueError(f"Unsupported LEGOSpec access: {spec['access']}")
            if spec["attention"] != "gqa":
                raise ValueError(f"Unsupported LEGOSpec attention: {spec['attention']}")
            operator = spec["local_operator"]
            if operator["type"] == "dflash2_conv":

                def conv():
                    return DFlashGroupedConv(
                        config.hidden_size,
                        self.block_size,
                        operator["kernel_size"],
                        operator["group_size"],
                        hf_semantics=True,
                    )

                layer.attention_conv, layer.mlp_conv = conv(), conv()
            elif operator["type"] != "none":
                raise ValueError(f"Unsupported local operator: {operator}")
        # Preserve the frozen Torch 2.9 reduction tree as well as BF16 casts.
        for parent in list(self.modules()):
            for name, module in list(parent.named_children()):
                if type(module) is RMSNorm:
                    setattr(
                        parent,
                        name,
                        LEGOSpecRMSNorm(
                            module.hidden_size,
                            eps=module.variance_epsilon,
                            weight_dtype=module.weight.dtype,
                            cast_x_before_out_mul=True,
                            round_residual_before_norm=True,
                        ),
                    )
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.cast_x_before_out_mul = True
                module.round_residual_before_norm = True
            if isinstance(module, SiluAndMul):
                module.hf_semantics = True
        use_legospec_cublas(self)

    def _stacked_ctx_kv_params(self):
        # The generic fallback normalizes with a different BF16 rounding order.
        # The fused context writer below supports the HF contract directly.
        return None

    def context_kv_layers(self):
        specs = self.config.composable_config["backbone"]["layers"]
        return [
            layer
            for layer, spec in zip(self.layers, specs)
            if spec["attention"] == "gqa" and spec["access"] == "dense"
        ]

    def write_target_hidden_kv(
        self,
        *,
        target_hidden,
        pool,
        positions,
        cache_loc,
        cache_loc_2d=None,
        commit_lens=None,
        target_hidden_is_projected=False,
        state_slot=None,
        final_pos=None,
        gdn_layout=None,
    ):
        ctx = (
            target_hidden
            if target_hidden_is_projected
            else self.project_target_hidden(target_hidden)
        )
        super().write_target_hidden_kv(
            target_hidden=ctx,
            pool=pool,
            positions=positions,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
            target_hidden_is_projected=True,
        )
        window_layers = getattr(pool, "swa_layers", [])
        if window_layers:
            from sglang.kernels.ops.kvcache.legospec_ring import legospec_ring_locs
            from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
                fused_kv_norm_rope_write,
            )

            if state_slot is None:
                raise ValueError("LEGOSpec SWA commits require request slots")
            cached = self._window_kv_write_cache
            if cached is None or cached[0] != id(pool):
                bundle = self._build_fused_kv_write_bundle(
                    pool, [self.layers[i] for i in window_layers]
                )
                if bundle is None:
                    raise RuntimeError(
                        "LEGOSpec SWA requires the fused BF16 context writer"
                    )
                self._window_kv_write_cache = (id(pool), bundle)
            bundle = self._window_kv_write_cache[1]
            locs = legospec_ring_locs(
                state_slot,
                positions,
                final_pos,
                commit_lens,
                8 if cache_loc_2d is None else cache_loc_2d.shape[1],
            )
            fused_kv_norm_rope_write(
                bundle.project(ctx),
                bundle.meta,
                bundle.k_norm_weights,
                bundle.cos_sin,
                positions,
                locs,
                bundle.num_layers,
                bundle.kv_size,
                bundle.head_dim,
                bundle.eps,
                is_neox_style=bundle.is_neox_style,
                rotary_dim=bundle.rotary_dim,
                hf_semantics=True,
                torch29_semantics=True,
            )
        gdn_pool = getattr(pool, "gdn", None)
        if gdn_pool is not None:
            if gdn_layout is None:
                raise ValueError(
                    "GDN context writes require explicit confirmed-prefix layout"
                )
            from sglang.kernels.ops.attention.legospec_gdn import (
                legospec_gdn_advance,
                legospec_gdn_prepare,
            )

            gdn_pool.reset(gdn_layout)
            indices = legospec_gdn_prepare(
                gdn_layout.slots,
                gdn_layout.cu_seqlens,
                positions,
                gdn_pool.cursors,
                gdn_pool.generations,
                gdn_layout.generations,
                commit_lens,
            )
            for layer_id in pool.gdn_layers:
                self.layers[layer_id].mixer.commit(
                    ctx, gdn_pool, gdn_layout, indices, commit_lens
                )
            legospec_gdn_advance(
                gdn_layout.slots, gdn_layout.cu_seqlens, gdn_pool.cursors, commit_lens
            )

    def set_block_size(self, block_size):
        if int(block_size) != self.config.block_size:
            raise ValueError("LEGOSpec block size cannot override trained geometry")
        super().set_block_size(block_size)

    def post_load_weights(self):
        # Validated native recipes contain GQA/GDN, without MLA absorption.
        pass

    def compute_base_logits(self, hidden):
        if self.lm_head is None:
            raise RuntimeError("LEGOSpec requires the target's shared LM head")
        return gather_and_crop_vocab(
            legospec_cublas_linear(hidden, self.lm_head.weight), self.lm_head
        ), hidden

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        for label, module in (("embedding", embed_tokens), ("LM head", lm_head)):
            if tuple(module.weight.shape) != (
                self.config.vocab_size,
                self.config.hidden_size,
            ):
                raise ValueError(
                    f"LEGOSpec shared {label} has incompatible shape {module.weight.shape}"
                )
            if module.weight.dtype != torch.bfloat16:
                raise ValueError(f"LEGOSpec shared {label} must be BF16")
        super().attach_shared_modules(embed_tokens=embed_tokens, lm_head=lm_head)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Strict coverage includes every packed Q/K/V and gate/up shard."""
        params = dict(self.named_parameters())
        required = set()
        for name in params:
            if ".qkv_proj." in name:
                required.update((name, shard) for shard in ("q", "k", "v"))
            elif ".gate_up_proj." in name:
                required.update((name, shard) for shard in (0, 1))
            else:
                required.add((name, None))
        loaded = set()
        for source_name, weight in weights:
            name = source_name
            for source, dest in (
                ("backbone.layers.", "layers."),
                ("backbone.norm.", "norm."),
                ("context.projection.", "fc."),
                ("context.norm.", "hidden_norm."),
                ("proposal.markov_head.", "markov_head."),
                ("proposal.candidate_selector.", "candidate_selector."),
            ):
                if name.startswith(source):
                    name = dest + name[len(source) :]
                    break
            name = name.replace(".input_norm.", ".input_layernorm.").replace(
                ".post_attention_norm.", ".post_attention_layernorm."
            )
            shard = None
            for source, dest, part in (
                ("q_proj", "qkv_proj", "q"),
                ("k_proj", "qkv_proj", "k"),
                ("v_proj", "qkv_proj", "v"),
                ("gate_proj", "gate_up_proj", 0),
                ("up_proj", "gate_up_proj", 1),
            ):
                if f".{source}." in name:
                    name, shard = name.replace(f".{source}.", f".{dest}."), part
                    break
            key = (name, shard)
            if key not in required:
                raise ValueError(f"Unexpected LEGOSpec weight {source_name!r} -> {key}")
            if key in loaded:
                raise ValueError(f"Duplicate LEGOSpec weight {source_name!r} -> {key}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                if tuple(param.shape) != tuple(weight.shape):
                    raise ValueError(
                        f"Shape mismatch for {source_name}: {weight.shape} != {param.shape}"
                    )
                loader(param, weight)
            else:
                if shard in ("q", "k", "v"):
                    heads = (
                        self.config.num_attention_heads
                        if shard == "q"
                        else self.config.num_key_value_heads
                    )
                    width = heads * self.config.head_dim
                else:
                    width = self.config.intermediate_size
                expected = (width, *param.shape[1:])
                if tuple(weight.shape) != expected:
                    raise ValueError(
                        f"Shape mismatch for packed shard {source_name}: {tuple(weight.shape)} != {expected}"
                    )
                loader(param, weight, shard)
            loaded.add(key)
        if required != loaded:
            raise ValueError(
                f"Missing LEGOSpec weights: {sorted(required - loaded, key=str)}"
            )
        self.weight_coverage = {"num_loaded_shards": len(loaded), "complete": True}
        self.post_load_weights()


EntryClass = LEGOSpecDraftModel
