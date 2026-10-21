"""Built-in FlashMTP registration and executable providers."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.hidden_states_data import (
    NORMALIZER_ID,
    build_collator,
    build_offline_normalizer,
    build_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    TargetDerivedDraftDefaults,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens

ALGORITHM_NAME = "flashmtp"
DRAFT_ARCHITECTURE = "FlashMTPDraftModel"
COMPATIBLE_DRAFT_ARCHITECTURES = frozenset({DRAFT_ARCHITECTURE})


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import FlashMTPTrainStrategy

    return FlashMTPTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    method_config = dict(getattr(draft_model.config, "flashmtp_config", None) or {})
    return {
        "flashmtp_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "flashmtp_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "flashmtp_block_size": int(training_model.block_size),
        "flashmtp_mask_token_id": int(training_model.mask_token_id),
        "flashmtp_attention_backend": str(training_model.attention_backend),
        "flashmtp_num_anchors": int(training_model.num_anchors),
        "flashmtp_loss_decay_gamma": training_model.loss_decay_gamma,
        "flashmtp_history_mode": str(method_config["history_mode"]),
        "flashmtp_sliding_window_size": int(method_config["sliding_window_size"]),
        "flashmtp_markov_head_type": str(method_config["markov_head_type"]),
        "flashmtp_markov_output_mode": str(method_config["markov_output_mode"]),
        "flashmtp_markov_rank": int(method_config["markov_rank"]),
        "flashmtp_backbone_conv_enabled": bool(
            method_config["backbone_conv_enabled"]
        ),
        "flashmtp_conv_kernel_size": int(method_config["conv_kernel_size"]),
        "flashmtp_conv_group_size": int(method_config["conv_group_size"]),
        "flashmtp_final_ce_weight": float(training_model.final_ce_weight),
        "flashmtp_tv_loss_weight": float(training_model.tv_loss_weight),
        "flashmtp_base_lm_ce_weight": float(training_model.base_lm_ce_weight),
        "flashmtp_base_lm_ce_decay_gamma": (training_model.base_lm_ce_decay_gamma),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_flashmtp_model

    return build_flashmtp_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_flashmtp_capture_layers

    return resolve_flashmtp_capture_layers(config, draft_config, target_config)


def populate_target_defaults(payload, target_config, config):
    from specforge.algorithms.model_providers import populate_flashmtp_generated_config

    return populate_flashmtp_generated_config(payload, target_config, config)


def apply_draft_overrides(config, draft_config):
    from specforge.algorithms.model_providers import apply_flashmtp_overrides

    return apply_flashmtp_overrides(config, draft_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import flashmtp_min_loss_tokens

    return flashmtp_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import flashmtp_needs_input_tools

    return flashmtp_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = {"input_ids", "loss_mask", "hidden_states"}
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"num_hidden_layers", "block_size"},
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
                expected_auto_map_model="flashmtp.FlashMTPDraftModel",
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="qwen3",
                    num_hidden_layers=1,
                    populate=populate_target_defaults,
                ),
                apply_overrides=apply_draft_overrides,
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dflash",
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_offline_normalizer,
                build_collator=build_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation=None,
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
