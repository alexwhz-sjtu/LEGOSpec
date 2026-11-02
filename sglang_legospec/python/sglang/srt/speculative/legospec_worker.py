"""LEGOSpec policy over the native DSpark V2 execution pipeline."""

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_schedule
from sglang.srt.speculative.dspark_components.dspark_config import DSparkRuntimeConfig
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.legospec_kv_inject import LEGOSpecKvInjector
from sglang.srt.speculative.legospec_proposal import (
    LEGOSpecProposalGraphs,
    LEGOSpecProposer,
)
from sglang.srt.speculative.legospec_verify import LEGOSpecVerifyExecutor


def handle_legospec_args(server_args):
    cfg = resolving_view(server_args)
    if not cfg.speculative_draft_model_path:
        raise ValueError("LEGOSPEC requires --speculative-draft-model-path")
    if not cfg.enable_deterministic_inference:
        raise ValueError(
            "LEGOSPEC v1 requires --enable-deterministic-inference so target AR "
            "and block verification use batch-invariant execution."
        )
    for backend in (cfg.prefill_attention_backend, cfg.decode_attention_backend):
        if (backend or cfg.attention_backend) != "fa3":
            raise ValueError(
                "LEGOSPEC v1 requires --attention-backend fa3 for the target "
                "and its AR control. FlashInfer FA2 target decode/verify "
                "rounding failed exact greedy agreement; draft GQA/SWA still "
                "use the independent FlashInfer paged backend."
            )
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        _ENABLE_MM_DEEPGEMM,
    )

    if _ENABLE_MM_DEEPGEMM:
        raise ValueError(
            "Set SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 before starting "
            "LEGOSPEC: target AR and verification must use the same invariant "
            "GEMM implementation across their different row counts."
        )
    if (cfg.tp_size, cfg.pp_size, cfg.dp_size) != (1, 1, 1):
        raise ValueError("LEGOSPEC v1 requires TP=PP=DP=1")
    if (
        cfg.device != "cuda"
        or cfg.quantization
        or cfg.speculative_draft_model_quantization
        or cfg.enable_lora
    ):
        raise ValueError("LEGOSPEC v1 requires CUDA, BF16 and no LoRA/quantization")
    if cfg.enable_fp32_lm_head or envs.SGLANG_DSPARK_FP32_LM_HEAD.get():
        raise ValueError("LEGOSPEC requires the trained BF16 LM-head rounding")
    if cfg.dtype not in ("auto", "bfloat16", "bf16"):
        raise ValueError("LEGOSPEC v1 requires BF16 model execution")
    if cfg.kv_cache_dtype not in (
        "auto",
        "bfloat16",
        "bf16",
    ) or cfg.speculative_draft_kv_cache_dtype not in (None, "auto", "bfloat16", "bf16"):
        raise ValueError("LEGOSPEC v1 requires BF16 target and draft KV caches")
    if cfg.enable_unified_memory:
        raise ValueError("LEGOSPEC v1 uses independent request-addressed draft pools")
    if cfg.enable_custom_logit_processor or cfg.enable_watermark:
        raise ValueError("LEGOSPEC v1 does not support custom logits or watermarking")
    if cfg.max_running_requests is not None and cfg.max_running_requests > 32:
        raise ValueError("LEGOSPEC v1 supports at most 32 running requests")
    if envs.SGLANG_SIMULATE_ACC_LEN.get() > 0:
        raise ValueError("LEGOSPEC validation and benchmarks require real acceptance")
    if (
        cfg.disaggregation_mode != "null"
        or cfg.enable_hierarchical_cache
        or cfg.hicache_storage_backend is not None
    ):
        raise ValueError("LEGOSPEC v1 does not support PD or HiCache")
    if envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
        raise ValueError("LEGOSPEC v1 requires SGLANG_RAGGED_VERIFY_MODE=static")
    if cfg.speculative_adaptive or cfg.speculative_use_block_verification:
        raise ValueError("LEGOSPEC v1 requires static chain rejection verification")
    if cfg.page_size not in (None, 1):
        raise ValueError("LEGOSPEC v1 requires page_size=1")
    if cfg.speculative_num_draft_tokens not in (None, 8):
        raise ValueError("LEGOSPEC requires eight query/verify tokens")
    declare_resolution(
        server_args,
        "handle_legospec_args",
        speculative_num_draft_tokens=8,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_draft_attention_backend="legospec",
        page_size=1,
        disable_radix_cache=True,
        sampling_backend="legospec",
        max_running_requests=cfg.max_running_requests or 32,
        random_seed=cfg.random_seed if cfg.random_seed is not None else 42,
    )


def resolve_legospec_runtime(
    *, draft_hf_config, speculative_num_draft_tokens, target_vocab_size
):
    if draft_hf_config.model_type != "legospec":
        raise ValueError("Export the trained checkpoint with export_legospec first")
    if speculative_num_draft_tokens != 8 or draft_hf_config.block_size != 8:
        raise ValueError("LEGOSpec checkpoint geometry requires block_size=8")
    if not 0 <= draft_hf_config.mask_token_id < target_vocab_size:
        raise ValueError("LEGOSpec mask token is outside the target embedding")
    return DSparkRuntimeConfig(
        gamma=7, verify_num_draft_tokens=8, mask_token_id=draft_hf_config.mask_token_id
    )


class LEGOSpecWorkerV2(DSparkWorkerV2):
    resolve_runtime_config = staticmethod(resolve_legospec_runtime)
    proposer_cls = LEGOSpecProposer
    verify_executor_cls = LEGOSpecVerifyExecutor
    kv_injector_cls = LEGOSpecKvInjector

    def __init__(self, server_args, gpu_id, nccl_port, target_worker, **kwargs):
        cfg = target_worker.model_runner.model_config.hf_config
        if (
            cfg.model_type != "qwen3"
            or cfg.hidden_size != 2560
            or cfg.num_hidden_layers != 36
            or cfg.vocab_size != 151936
        ):
            raise ValueError("LEGOSPEC v1 requires the unmodified Qwen3-4B target")
        super().__init__(server_args, gpu_id, nccl_port, target_worker, **kwargs)

    def dump_info_records(self):
        executor = self._verify_executor
        pool = self.draft_model_runner.token_to_kv_pool
        return dict(
            proposal_type=self.draft_model.proposal_type,
            target_batch_invariant=True,
            target_attention_backend=self.target_worker.model_runner.decode_attention_backend_str,
            draft_projection_backend="cublas-addmm-out",
            weight_coverage=self.draft_model.weight_coverage,
            accept_graph_replays=executor.graph_replay_ct,
            accept_graph_eager=executor.graph_eager_ct,
            context_graph_replays=self._kv_injector.graph_replay_ct,
            context_graph_eager=self._kv_injector.graph_eager_ct,
            proposal_graph_replays=self._draft_sampler.graph_replay_ct,
            proposal_graph_eager=self._draft_sampler.graph_eager_ct,
            dense_q_allocated=self._draft_sampler.sampled is not None,
            pool_type=type(pool).__name__,
            block_size=self.query_token_num,
            proposal_length=self.gamma,
            dense_kv_layers=len(pool.full_layers),
            swa_history_capacity=pool.ring_history if pool.swa_layers else 0,
            gdn_state_dtype=None if pool.gdn is None else str(pool.gdn.states.dtype),
            gdn_state_shape=None if pool.gdn is None else list(pool.gdn.states.shape),
            gdn_state_bytes=0 if pool.gdn is None else pool.gdn.get_bytes(),
            gdn_readout_mode=None if pool.gdn is None else "qwen35_mixed_chunk",
            gdn_read_cache_events=0 if pool.gdn is None else 64,
            gdn_generation_resets=0
            if pool.gdn is None
            else sum(pool.gdn.host_generations),
        )

    def _maybe_build_draft_sampler(self):
        # Backbone captures exclude the head; head graphs have their own mode key.
        return None

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        self._draft_sampler = LEGOSpecProposalGraphs(
            self.draft_model,
            max(
                get_exec().graph.cuda_graph_config.decode.bs
                or [get_schedule().max_running_requests or 32]
            ),
            self.device,
            self.target_worker.random_seed,
            noise=self._proposer.noise,
            tp_sync=self._tp_sync,
        )

        self._proposer.attach_draft_sampler(self._draft_sampler)
