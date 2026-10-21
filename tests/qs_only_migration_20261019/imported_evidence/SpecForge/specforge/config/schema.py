# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The typed run config behind the single ``specforge train`` entry point.

The schema intentionally describes a *run*, not a legacy Python script.  Model
assembly, prompt preparation, topology and the strategy-specific objective live
behind this one validated contract.  YAML or JSON on disk and dotted CLI
overrides both re-validate through the same schema.
"""

from __future__ import annotations

import copy
import json
import os
from typing import List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

# SGLang reserves one generated-token slot plus five internal slots, and its
# request validator rejects ``input_len >= context_len - 6``.  Accepting a
# prompt whose length is exactly ``data.max_length`` therefore needs 7 slots.
SGLANG_CAPTURE_CONTEXT_HEADROOM = 7


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfigModel):
    target_model_path: str
    #: Draft config as a local JSON file, a directory containing config.json,
    #: or a Hugging Face repository. EAGLE3, P-EAGLE, and DFlash can derive it
    #: from the target when omitted.
    draft_model_config: Optional[str] = None
    #: Load draft weights only. Unlike training.resume_from, this never restores
    #: optimizer/scheduler state, counters, data position, or RNG state.
    draft_checkpoint_path: Optional[str] = None
    #: Optional architecture override. Auto-generated defaults preserve the
    #: former trainers: EAGLE3=1, P-EAGLE=4, DFlash=1.
    draft_num_hidden_layers: Optional[int] = Field(default=None, gt=0)
    #: Optional DFlash/FlashMTP block-size override (auto-generated default: 16).
    draft_block_size: Optional[int] = Field(default=None, gt=0)
    #: FlashMTP history is intentionally kept to the token-window variant.
    flashmtp_history_mode: Literal["token"] = "token"
    #: Number of token positions in the FlashMTP sliding condition window.
    flashmtp_sliding_window_size: int = Field(default=1, gt=0)
    #: Supported serial-head profiles are markov/additive and rnn_easy/direct.
    flashmtp_markov_head_type: Literal["markov", "rnn_easy"] = "markov"
    flashmtp_markov_output_mode: Literal["additive", "direct"] = "additive"
    flashmtp_markov_rank: int = Field(default=64, gt=0)
    flashmtp_chs_num_layers: int = Field(default=7, ge=2)
    flashmtp_local_position: bool = False
    #: Wrap FlashMTP attention/MLP with v2swa grouped dynamic convolution.
    flashmtp_backbone_conv_enabled: bool = False
    #: Number of causal taps in each per-query-block convolution.
    flashmtp_conv_kernel_size: int = Field(default=2, gt=0)
    #: Adjacent hidden channels sharing one dynamic kernel delta.
    flashmtp_conv_group_size: int = Field(default=16, gt=0)
    #: Online capture uses SGLang: in-process for single-node colocated runs and
    #: external patched servers for multi-node disaggregated runs.
    target_backend: Literal["sglang"] = "sglang"
    #: Retained for config migration only. Colocated TP output sharding is
    #: automatic; external server capture transports complete feature records.
    shard_target_output: bool = False
    #: Input family consumed by the capture provider. Built-in algorithms support
    #: text only; unsupported modalities fail during application resolution.
    input_modality: str = "text"
    trust_remote_code: bool = False
    #: Enable Liger Qwen3 RMSNorm/SwiGLU kernels for DFlash. Requires ``specforge[liger]``.
    use_liger_kernel: bool = False
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    #: t2d/d2t vocab-mapping tensor file for the draft ("" = model has none).
    vocab_mapping_path: str = ""
    #: copy the frozen target embedding into the draft before training.
    load_target_embedding: bool = True
    #: aux hidden-state layer ids to capture (None = backend defaults).
    aux_hidden_state_layer_ids: Optional[List[int]] = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    cache_dir: Optional[str] = None
    #: DFlash-family and P-EAGLE mask token. ``None`` resolves it from the
    #: draft config, then the target tokenizer.
    mask_token_id: Optional[int] = None
    #: Explicit tokenizer padding ID for released targets whose tokenizer
    #: metadata omits it or whose padding row is outside the unpadded vocab.
    tokenizer_pad_token_id: Optional[int] = Field(default=None, ge=0)
    #: SGLang target-engine tuning. Ignored by hf/custom backends.
    sglang_attention_backend: str = "flashinfer"
    sglang_mem_fraction_static: float = Field(default=0.4, gt=0.0, le=1.0)
    #: Disable the radix cache by default for local capture. Hybrid targets
    #: such as Inkling require the radix tree and can opt back in explicitly.
    #: Disabling it is also required for hybrid linear-attention/Mamba targets
    #: on ROCm, whose mamba radix-cache ``extra_buffer`` strategy asserts
    #: CUDA/MUSA/NPU (FLA) at server init.
    sglang_disable_radix_cache: bool = True
    sglang_context_length: Optional[int] = Field(default=None, gt=0)
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = False
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_ep_size: int = Field(default=1, gt=0)
    sglang_max_running_requests: Optional[int] = Field(default=None, gt=0)
    sglang_max_total_tokens: Optional[int] = Field(default=None, gt=0)
    sglang_dp_size: Optional[int] = Field(default=None, gt=0)
    sglang_moe_a2a_backend: Optional[str] = None
    sglang_moe_runner_backend: Optional[str] = None
    sglang_page_size: Optional[int] = Field(default=None, gt=0)
    sglang_quantization: Optional[str] = None
    sglang_fp4_gemm_runner_backend: Optional[str] = None
    sglang_mamba_radix_cache_strategy: Optional[str] = None
    sglang_max_mamba_cache_size: Optional[int] = Field(default=None, gt=0)
    sglang_swa_full_tokens_ratio: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )
    sglang_mamba_full_memory_ratio: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def _validate_input_modality(self):
        if (
            not self.input_modality
            or self.input_modality.strip() != self.input_modality
            or any(character.isspace() for character in self.input_modality)
        ):
            raise ValueError(
                "model.input_modality must be a non-empty plugin identifier "
                "without whitespace"
            )
        valid_flashmtp_profiles = {
            ("markov", "additive"),
            ("rnn_easy", "direct"),
        }
        profile = (
            self.flashmtp_markov_head_type,
            self.flashmtp_markov_output_mode,
        )
        if profile not in valid_flashmtp_profiles:
            raise ValueError(
                "FlashMTP supports only markov/additive and rnn_easy/direct, "
                f"got {profile[0]}/{profile[1]}"
            )
        return self


class DataConfig(StrictConfigModel):
    #: online mode — raw conversation JSON/JSONL.
    train_data_path: str = ""
    #: online mode — already-tokenized JSONL records.
    prompts_path: str = ""
    #: offline mode — directory of precomputed hidden-state .ckpt files.
    hidden_states_path: str = ""
    #: Reserved migration field. Online evaluation is unsupported; keep empty.
    eval_data_path: str = ""
    #: offline evaluation — directory of precomputed hidden-state .ckpt files.
    eval_hidden_states_path: str = ""
    max_length: int = Field(default=2048, gt=0)
    chat_template: str = "llama3"
    is_preformatted: bool = False
    train_only_last_turn: bool = False
    build_dataset_num_proc: int = Field(default=8, gt=0)
    #: Ordered background feature-loader workers. ``None`` preserves the
    #: former strategy defaults (EAGLE/P-EAGLE=4, DFlash-family=8).
    dataloader_num_workers: Optional[int] = Field(default=None, ge=0)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    max_prompts: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _exactly_one_source(self):
        sources = [
            bool(self.train_data_path),
            bool(self.prompts_path),
            bool(self.hidden_states_path),
        ]
        if sum(sources) != 1:
            raise ValueError(
                "set exactly one of data.train_data_path (raw online data), "
                "data.prompts_path (pre-tokenized online data), or "
                "data.hidden_states_path (offline features)"
            )
        return self


class TrackingConfig(StrictConfigModel):
    """Optional experiment tracking behind the trainer's logger seam."""

    report_to: Literal["none", "wandb", "tensorboard", "swanlab", "mlflow"] = "none"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_key: Optional[str] = None
    wandb_offline: bool = False
    wandb_dir: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_name: Optional[str] = None
    swanlab_key: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None


class ProfilingConfig(StrictConfigModel):
    """Per-rank PyTorch trace window in completed optimizer steps."""

    enabled: bool = False
    start_step: int = Field(default=30, ge=0)
    num_steps: int = Field(default=4, gt=0)
    record_shapes: bool = False


class RuntimeConfig(StrictConfigModel):
    """Streaming bounds shared by unified disaggregated producer roles."""

    producer_lease: int = Field(default=8, gt=0)
    producer_concurrency: int = Field(default=1, gt=0)
    in_flight_high_watermark: int = Field(default=256, gt=0)
    in_flight_low_watermark: int = Field(default=192, ge=0)
    resident_high_watermark_bytes: Optional[int] = Field(default=None, gt=0)
    resident_low_watermark_bytes: Optional[int] = Field(default=None, ge=0)
    feature_store_max_resident_bytes: Optional[int] = Field(default=None, gt=0)
    feature_store_max_quarantined_bytes: int = Field(default=8 << 30, ge=0)

    @model_validator(mode="after")
    def _validate_watermarks(self):
        if self.in_flight_low_watermark > self.in_flight_high_watermark:
            raise ValueError(
                "runtime.in_flight_low_watermark must be <= "
                "runtime.in_flight_high_watermark"
            )
        if (
            self.resident_high_watermark_bytes is None
            and self.resident_low_watermark_bytes is not None
        ):
            raise ValueError(
                "runtime.resident_low_watermark_bytes requires "
                "runtime.resident_high_watermark_bytes"
            )
        if (
            self.resident_high_watermark_bytes is not None
            and self.resident_low_watermark_bytes is not None
            and self.resident_low_watermark_bytes > self.resident_high_watermark_bytes
        ):
            raise ValueError(
                "runtime.resident_low_watermark_bytes must be <= "
                "runtime.resident_high_watermark_bytes"
            )
        if (
            self.feature_store_max_resident_bytes is not None
            and self.resident_high_watermark_bytes is not None
            and self.feature_store_max_resident_bytes
            < self.resident_high_watermark_bytes
        ):
            raise ValueError(
                "runtime.feature_store_max_resident_bytes must be >= "
                "runtime.resident_high_watermark_bytes"
            )
        return self


class TrainerDeploymentConfig(StrictConfigModel):
    """Process topology used by the unified CLI launcher."""

    nnodes: int = Field(default=1, gt=0)
    nproc_per_node: int = Field(default=1, gt=0)
    #: Node-local identity. Shared multi-node configs normally provide this
    #: through ``specforge train --node-rank`` instead.
    node_rank: Optional[int] = Field(default=None, ge=0)
    master_addr: Optional[str] = None
    master_port: int = Field(default=29500, gt=0, le=65535)

    @model_validator(mode="after")
    def _validate_topology(self):
        if self.node_rank is not None and self.node_rank >= self.nnodes:
            raise ValueError(
                "deployment.trainer.node_rank must be smaller than "
                "deployment.trainer.nnodes"
            )
        if self.nnodes > 1 and not self.master_addr:
            raise ValueError(
                "deployment.trainer.master_addr is required when nnodes > 1"
            )
        return self


class DisaggregatedDeploymentConfig(StrictConfigModel):
    """Shared, non-secret topology for producer/consumer launch planning."""

    #: Attempt-scoped shared directory. The launcher derives refs, manifest,
    #: and lifecycle markers beneath it.
    control_dir: str
    #: Physical nodes hosting external target capture and the producer. This
    #: pool is disjoint from ``deployment.trainer.nnodes``.
    capture_nnodes: Optional[int] = Field(default=None, gt=0)
    #: Optional node-local root for the online consumer SQLite/WAL. When
    #: omitted, the historical control_dir-derived path remains in use.
    consumer_state_dir: Optional[str] = None
    #: Optional rank-0 HTTP relay for per-rank online inboxes.  This removes the
    #: shared-filesystem requirement between trainer nodes while keeping the
    #: authority-owned source channel and SQLite/WAL on trainer node 0.
    inbox_server_url: Optional[str] = None
    backend: Literal["shared_dir", "mooncake"]
    store_root: Optional[str] = None
    store_id: Optional[str] = None
    server_urls: List[str] = Field(default_factory=list)
    mooncake_metadata_server: Optional[str] = None
    mooncake_master_server_addr: Optional[str] = None
    mooncake_local_hostname: Optional[str] = None
    mooncake_protocol: Optional[str] = None
    mooncake_rdma_devices: Optional[str] = None
    #: Offline Mooncake producers own ingested feature objects until the
    #: consumer acknowledges them, so they require a positive segment. Online
    #: capture remains server-owned and the launcher forces client segments to
    #: zero for both SpecForge roles.
    producer_segment_size: Optional[int] = Field(default=None, gt=0)
    client_buffer_size: int = Field(default=256 << 20, gt=0)
    #: Consumer receive buffers for Mooncake ``get_into``: ``pageable`` allocates
    #: a fresh host tensor per feature (registered and unregistered around each
    #: read, then copied to the device on the training stream); ``pinned`` keeps
    #: a bounded pool of page-locked, once-registered host buffers and copies to
    #: the device on a side stream; ``cuda`` keeps the pool on the trainer device
    #: for device reads (Mooncake 0.3.x stages these through its client buffer).
    receive_buffers: Literal["pageable", "pinned", "cuda"] = "pinned"
    #: Retained receive-pool budget per rank; excludes output copies and overflow.
    receive_pool_bytes: int = Field(default=8 << 30, gt=0)
    idle_timeout_s: Optional[float] = Field(default=None, gt=0)
    peer_wait_timeout_s: Optional[float] = Field(default=None, gt=0)
    producer_hold_s: Optional[float] = Field(default=None, gt=0)
    #: SIGTERM-to-SIGKILL grace for supervisor teardown.
    #: Workers translate SIGTERM into cleanup (Mooncake drains, checkpoint
    #: flush, failure sentinels), so this window must cover that work.
    shutdown_grace_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _validate_store(self):
        if not self.control_dir:
            raise ValueError("deployment.disaggregated.control_dir must not be empty")
        if self.consumer_state_dir is not None and (
            not self.consumer_state_dir
            or self.consumer_state_dir.strip() != self.consumer_state_dir
        ):
            raise ValueError(
                "deployment.disaggregated.consumer_state_dir must be non-empty "
                "and must not contain surrounding whitespace"
            )
        if self.inbox_server_url is not None:
            parsed = urlparse(self.inbox_server_url)
            if (
                parsed.scheme != "http"
                or not parsed.hostname
                or parsed.port is None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in ("", "/")
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "deployment.disaggregated.inbox_server_url must be an "
                    "http://host:port origin without credentials, path, query, "
                    "or fragment"
                )
        if self.backend == "shared_dir" and not self.store_root:
            raise ValueError(
                "deployment.disaggregated.store_root is required for shared_dir"
            )
        return self


class DeploymentConfig(StrictConfigModel):
    """User-facing launch topology for ``specforge train``."""

    mode: Literal["local_colocated", "disaggregated"] = "local_colocated"
    trainer: TrainerDeploymentConfig = Field(default_factory=TrainerDeploymentConfig)
    disaggregated: Optional[DisaggregatedDeploymentConfig] = None

    @model_validator(mode="after")
    def _validate_mode(self):
        if self.mode == "disaggregated" and self.disaggregated is None:
            raise ValueError(
                "deployment.disaggregated is required when "
                "deployment.mode=disaggregated"
            )
        if self.mode != "disaggregated" and self.disaggregated is not None:
            raise ValueError(
                "deployment.disaggregated requires deployment.mode=disaggregated"
            )
        return self


class TrainingConfig(StrictConfigModel):
    strategy: str = "eagle3"
    num_epochs: int = Field(default=1, gt=0)
    max_steps: Optional[int] = Field(default=None, gt=0)
    total_steps: Optional[int] = Field(default=None, gt=0)
    batch_size: int = Field(default=1, gt=0)
    accumulation_steps: int = Field(default=1, gt=0)
    fsdp_sharding: Literal["SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"] = "SHARD_GRAD_OP"
    learning_rate: float = Field(default=1e-4, gt=0.0)
    lr_scheduler: Literal["cosine", "constant"] = "cosine"
    warmup_ratio: float = Field(default=0.015, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    #: Keep FP32 Adam masters and moments on CPU while the trainable draft
    #: remains on the accelerator.
    optimizer_cpu_offload: bool = False
    ttt_length: int = Field(default=7, gt=0)
    attention_backend: Literal["eager", "sdpa", "flex_attention", "fa", "usp"] = (
        "flex_attention"
    )
    #: In local online mode this is target-model TP. The draft remains FSDP over
    #: the full trainer world; disaggregated/offline consumers keep this at one.
    tp_size: int = Field(default=1, gt=0)
    sp_ulysses_size: int = Field(default=1, gt=0)
    sp_ring_size: int = Field(default=1, gt=0)
    dist_timeout: int = Field(default=10, gt=0)
    #: Acceptance-aware token objective. DFlash-family hard targets make
    #: ``alpha`` equivalent to CE; ``lambda`` mixes CE and TV.
    lk_loss_type: Optional[Literal["lambda", "alpha", "tv"]] = None
    kl_scale: float = 1.0
    kl_decay: float = 1.0
    #: Compute the teacher target_p, draft logits and loss only at supervised
    #: (loss-masked) positions instead of over the full sequence. Mathematically
    #: equivalent (the mean denominator is rescaled) and saves memory/compute on
    #: prompt-heavy data. Falls back to the full-length path for batch > 1 or when
    #: an lk_loss objective is used.
    trim_loss_positions: bool = False
    #: MTP token positions per lm_head + CE chunk (0 disables chunking).
    mtp_objective_chunk_size: int = Field(default=4096, ge=0)
    #: DFlash-family objective/model knobs.
    num_anchors: int = Field(default=512, gt=0)
    loss_decay_gamma: Optional[float] = None
    #: Anchor blocks per objective slice (0 = materialize all objective logits).
    objective_chunk_blocks: int = Field(default=128, ge=0)
    loss_type: Literal[
        "dflash",
        "dpace",
        "dpace-cumulative-confidence-only",
        "dpace-continuation-value-only",
    ] = "dflash"
    dpace_alpha: float = 0.5
    #: Weight of the top-k path-selector objective for DFlash2 drafts.
    dflash2_selector_loss_alpha: float = Field(default=1.0, ge=0.0)
    #: Fraction of optimizer steps that train only the DFlash2 base objective.
    dflash2_selector_warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Fraction of optimizer steps used to ramp the selector weight to its target.
    dflash2_selector_ramp_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Stop selector gradients at the unary/backbone boundary while preserving
    #: the primary DFlash/D-PACE/LK gradient path.
    dflash2_selector_stop_gradient: bool = False
    lambda_base_start: float = 1.0
    lambda_base_decay_ratio: float = 0.5
    dspark_ce_loss_alpha: float = 0.1
    dspark_l1_loss_alpha: float = 0.9
    dspark_confidence_head_alpha: float = 1.0
    #: FlashMTP serial-head objective weights and memory bounds.
    flashmtp_final_ce_weight: float = Field(default=1.0, ge=0.0)
    flashmtp_tv_loss_weight: float = Field(default=1.0, ge=0.0)
    flashmtp_base_lm_ce_weight: float = Field(default=0.0, ge=0.0)
    flashmtp_base_lm_ce_decay_gamma: Optional[float] = Field(default=None, gt=0.0)
    #: P-EAGLE COD sampling/model knobs.
    num_depths: int = Field(default=8, gt=0)
    down_sample_ratio: float = 0.8
    down_sample_ratio_min: float = 0.2
    norm_before_residual: Optional[bool] = None
    save_interval: int = Field(default=0, ge=0)
    #: Run a full evaluation pass every N optimizer steps (0 = disabled).
    eval_interval: int = Field(default=0, ge=0)
    log_interval: int = Field(default=50, gt=0)
    #: CheckpointManager rotation: keep the newest N checkpoints (0 = keep all).
    max_checkpoints: int = Field(default=0, ge=0)
    #: Offline EAGLE3 teacher projection without materializing full-vocab fp32
    #: logits. Exact, but trades additional head passes for lower peak memory.
    compact_teacher: bool = False
    compact_teacher_chunk_size: Optional[int] = Field(default=None, gt=0)
    #: resume target: a checkpoint dir / file:// URI / run root.
    resume_from: Optional[str] = None
    #: ``all`` is an offline colocated run. Online runs launch producer and
    #: consumer as separate ``specforge train`` processes with the same config
    #: and different roles.
    role: Literal["auto", "all", "producer", "consumer"] = "all"
    seed: int = 42
    #: Deterministic online prompt ordering. ``None`` preserves the historical
    #: behavior of using the run RNG seed for both model and prompt sampling.
    prompt_seed: Optional[int] = None
    #: Match ``DistributedSampler(drop_last=False)`` by padding the global
    #: prompt permutation to the DP world size instead of truncating its tail.
    prompt_sampler_drop_last: bool = True

    @model_validator(mode="after")
    def _validate_training_shape(self):
        if not 0.0 <= self.dpace_alpha <= 1.0:
            raise ValueError("training.dpace_alpha must be in [0, 1]")
        if not 0.0 < self.down_sample_ratio <= 1.0:
            raise ValueError("training.down_sample_ratio must be in (0, 1]")
        if not 0.0 < self.down_sample_ratio_min <= self.down_sample_ratio:
            raise ValueError(
                "training.down_sample_ratio_min must be in "
                "(0, training.down_sample_ratio]"
            )
        sp_size = self.sp_ulysses_size * self.sp_ring_size
        if self.attention_backend == "usp":
            if self.batch_size != 1:
                raise ValueError(
                    "USP attention currently requires training.batch_size=1"
                )
            if sp_size <= 1:
                raise ValueError(
                    "USP attention requires sp_ulysses_size * sp_ring_size > 1"
                )
        elif sp_size != 1:
            raise ValueError(
                "training.sp_ulysses_size/sp_ring_size require "
                "training.attention_backend=usp"
            )
        return self


def migrate_legacy_config(values: dict) -> dict:
    """Translate legacy deployment keys before constructing the domain config.

    Compatibility exists only at this raw loader boundary. The returned domain
    model never projects canonical deployment state back into ``training``.
    """

    if not isinstance(values, dict):
        return values
    raw = copy.deepcopy(values)
    deployment_payload = raw.get("deployment")
    if isinstance(deployment_payload, dict):
        disaggregated_payload = deployment_payload.get("disaggregated")
        if (
            isinstance(disaggregated_payload, dict)
            and "managed_local" in disaggregated_payload
        ):
            raise ValueError(
                "deployment.disaggregated.managed_local was removed: use "
                "deployment.mode=local_colocated for single-node online training, "
                "or configure external Mooncake capture with capture_nnodes for "
                "multi-node disaggregated training"
            )
    training = raw.get("training")
    if not isinstance(training, dict):
        return raw

    has_legacy_mode = "deployment_mode" in training
    has_legacy_urls = "server_urls" in training
    has_legacy_database = "metadata_db_path" in training
    if not (has_legacy_mode or has_legacy_urls or has_legacy_database):
        return raw

    legacy_mode = training.pop("deployment_mode", None)
    legacy_urls = training.pop("server_urls", None)
    legacy_database = training.pop("metadata_db_path", None)
    deployment = raw.setdefault("deployment", {})
    if not isinstance(deployment, dict):
        return raw

    canonical_mode = deployment.get("mode")
    if legacy_mode is not None:
        if canonical_mode is not None and canonical_mode != legacy_mode:
            raise ValueError(
                "deployment.mode conflicts with legacy training.deployment_mode"
            )
        deployment["mode"] = legacy_mode
    elif canonical_mode is None and (legacy_urls or legacy_database):
        deployment["mode"] = "disaggregated"

    if legacy_urls is not None or legacy_database is not None:
        disaggregated = deployment.setdefault("disaggregated", {})
        if not isinstance(disaggregated, dict):
            return raw
        if legacy_urls is not None:
            canonical_urls = disaggregated.get("server_urls")
            if canonical_urls is not None and canonical_urls != legacy_urls:
                raise ValueError(
                    "deployment.disaggregated.server_urls conflicts with legacy "
                    "training.server_urls"
                )
            disaggregated["server_urls"] = legacy_urls
        if legacy_database:
            if os.path.basename(legacy_database) != "consumer.sqlite":
                raise ValueError(
                    "legacy training.metadata_db_path must end in consumer.sqlite; "
                    "use deployment.disaggregated.consumer_state_dir instead"
                )
            state_dir = os.path.dirname(legacy_database) or "."
            canonical_state_dir = disaggregated.get("consumer_state_dir")
            if canonical_state_dir is not None and canonical_state_dir != state_dir:
                raise ValueError(
                    "deployment.disaggregated.consumer_state_dir conflicts with "
                    "legacy training.metadata_db_path"
                )
            disaggregated["consumer_state_dir"] = state_dir
    return raw


class Config(StrictConfigModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    run_id: str = "specforge-run"
    output_dir: str = "./output"

    @model_validator(mode="before")
    @classmethod
    def _default_role_for_deployment(cls, values):
        """Resolve the process-role default from the canonical topology."""
        if not isinstance(values, dict):
            return values
        raw = dict(values)
        deployment = dict(raw.get("deployment") or {})
        training = dict(raw.get("training") or {})
        mode = deployment.get("mode", "local_colocated")
        training.setdefault("role", "auto" if mode == "disaggregated" else "all")
        raw["training"] = training
        return raw

    @model_validator(mode="after")
    def _validate_run_structure(self):
        """Validate topology and cross-field shape without resolving algorithms."""
        mode = self.mode
        deployment = self.deployment.mode
        role = self.training.role

        if mode == "online" and self.model.target_backend != "sglang":
            raise ValueError(
                "online training uses SGLang target capture and "
                "requires model.target_backend=sglang"
            )
        if mode == "online" and deployment == "local_colocated":
            if self.deployment.trainer.nnodes != 1:
                raise ValueError(
                    "online local_colocated training requires "
                    "deployment.trainer.nnodes=1"
                )
            if self.data.dataloader_num_workers not in (None, 0):
                raise ValueError(
                    "online local_colocated capture runs synchronously on the "
                    "training rank and requires data.dataloader_num_workers=0 "
                    "or null"
                )
            if (
                self.model.sglang_enable_dp_attention
                or self.model.sglang_enable_dp_lm_head
                or self.model.sglang_dp_size not in (None, 1)
            ):
                raise ValueError(
                    "online local_colocated capture uses the SpecForge TP group "
                    "and does not support SGLang data parallelism"
                )
            if (
                self.model.sglang_ep_size > self.training.tp_size
                or self.training.tp_size % self.model.sglang_ep_size
            ):
                raise ValueError(
                    "model.sglang_ep_size must divide and not exceed "
                    "training.tp_size for online local_colocated capture"
                )
        if role != "all" and deployment != "disaggregated":
            raise ValueError(
                "training.role=auto/producer/consumer requires "
                "deployment.mode=disaggregated"
            )
        if deployment == "disaggregated" and role == "all":
            raise ValueError(
                "deployment.mode=disaggregated requires "
                "training.role=auto, producer, or consumer"
            )

        if (
            self.model.draft_checkpoint_path is not None
            and self.training.resume_from is not None
        ):
            raise ValueError(
                "model.draft_checkpoint_path (weights-only warm start) and "
                "training.resume_from (full resume) are mutually exclusive"
            )
        if self.training.role == "producer" and self.profiling.enabled:
            raise ValueError(
                "profiling.enabled applies to trainer roles, not a capture-only "
                "producer"
            )
        if self.data.eval_data_path:
            raise ValueError(
                "data.eval_data_path is unsupported; online evaluation is not "
                "supported by the current target-capture paths"
            )
        has_eval_source = bool(self.data.eval_hidden_states_path)
        has_eval_interval = self.training.eval_interval > 0
        if has_eval_source != has_eval_interval:
            raise ValueError(
                "an eval data source and training.eval_interval must be "
                "configured together"
            )
        if self.data.eval_hidden_states_path and mode != "offline":
            raise ValueError(
                "data.eval_hidden_states_path requires an offline training data source"
            )
        if (
            not self.training.compact_teacher
            and self.training.compact_teacher_chunk_size is not None
        ):
            raise ValueError(
                "training.compact_teacher_chunk_size requires "
                "training.compact_teacher=true"
            )
        if (
            mode == "online"
            and deployment == "disaggregated"
            and self.deployment.disaggregated is not None
            and self.deployment.disaggregated.backend != "mooncake"
        ):
            raise ValueError(
                "online disaggregated training requires "
                "deployment.disaggregated.backend=mooncake"
            )
        if mode == "online" and deployment == "disaggregated":
            if self.deployment.disaggregated.capture_nnodes is None:
                raise ValueError(
                    "online disaggregated training requires a positive "
                    "deployment.disaggregated.capture_nnodes; single-node "
                    "disaggregated execution is unsupported"
                )
        elif (
            self.deployment.disaggregated is not None
            and self.deployment.disaggregated.capture_nnodes is not None
        ):
            raise ValueError(
                "deployment.disaggregated.capture_nnodes is valid only for "
                "online disaggregated training"
            )
        consumer_state_dir = (
            self.deployment.disaggregated.consumer_state_dir
            if self.deployment.disaggregated is not None
            else None
        )
        inbox_server_url = (
            self.deployment.disaggregated.inbox_server_url
            if self.deployment.disaggregated is not None
            else None
        )
        if consumer_state_dir is not None:
            if mode != "online" or deployment != "disaggregated":
                raise ValueError(
                    "deployment.disaggregated.consumer_state_dir is valid only "
                    "for online disaggregated training"
                )
        if inbox_server_url is not None:
            if mode != "online" or deployment != "disaggregated":
                raise ValueError(
                    "deployment.disaggregated.inbox_server_url is valid only "
                    "for online disaggregated training"
                )
            if self.deployment.trainer.nnodes < 2:
                raise ValueError(
                    "deployment.disaggregated.inbox_server_url requires a "
                    "multi-node trainer"
                )
        if (
            mode == "online"
            and deployment == "disaggregated"
            and self.deployment.trainer.nnodes > 1
            and consumer_state_dir is None
        ):
            raise ValueError(
                "multi-node online consumers require an explicit node-local "
                "deployment.disaggregated.consumer_state_dir for SQLite/WAL"
            )
        if self.training.role == "producer" and self.training.resume_from is not None:
            raise ValueError("training.resume_from is valid only for a trainer role")
        if self.training.attention_backend == "usp":
            if mode != "offline":
                raise ValueError("USP attention currently requires offline features")
        if mode == "offline" and self.training.tp_size != 1:
            raise ValueError(
                "offline feature consumers do not implement trainer tensor "
                "parallelism; keep training.tp_size=1 so every non-SP rank "
                "receives its own data shard"
            )
        if (
            mode == "online"
            and (
                self.training.sp_ulysses_size != 1
                or self.training.sp_ring_size != 1
            )
        ):
            raise ValueError("online training requires both sequence-parallel sizes=1")
        if (
            mode == "online"
            and deployment == "disaggregated"
            and self.training.tp_size != 1
        ):
            raise ValueError(
                "the disaggregated online consumer uses every trainer rank for "
                "data parallelism; configure target TP on the external server and "
                "keep training.tp_size/sp sizes at 1"
            )
        return self

    @property
    def mode(self) -> str:
        return "offline" if self.data.hidden_states_path else "online"

    def validate_world_size(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}")
        tp_size = self.training.tp_size
        sp_size = self.training.sp_ulysses_size * self.training.sp_ring_size
        if world_size % tp_size:
            raise ValueError(
                f"world_size={world_size} must be divisible by "
                f"training.tp_size={tp_size}"
            )
        if world_size % sp_size:
            raise ValueError(
                f"world_size={world_size} must be divisible by draft sequence "
                f"parallel size {sp_size} "
                "(sp_ulysses_size * sp_ring_size)"
            )

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml

                raw = yaml.safe_load(f)
            else:
                raw = json.load(f)
        return cls.model_validate(migrate_legacy_config(raw))


def apply_overrides(config: Config, overrides: List[str]) -> Config:
    """Apply dotted ``section.field=value`` overrides, re-validating the result."""
    raw = config.model_dump()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form path=value")
        path, value = item.split("=", 1)
        node = raw
        keys = path.split(".")
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                raise ValueError(f"override path {path!r} does not exist")
            node = node[key]
        if keys[-1] not in node:
            raise ValueError(f"override path {path!r} does not exist")
        current = node[keys[-1]]
        if isinstance(current, (dict, list)) and value.lstrip().startswith(("[", "{")):
            import yaml

            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"override {path!r} contains an invalid structured value"
                ) from exc
        node[keys[-1]] = value  # pydantic coerces scalars on re-validation
    return Config.model_validate(raw)


def load_config(path: str, overrides: Optional[List[str]] = None) -> Config:
    config = Config.from_file(path)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrackingConfig",
    "ProfilingConfig",
    "RuntimeConfig",
    "TrainerDeploymentConfig",
    "DisaggregatedDeploymentConfig",
    "DeploymentConfig",
    "TrainingConfig",
    "Config",
    "migrate_legacy_config",
    "load_config",
    "apply_overrides",
]
