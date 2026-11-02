"""Composable adapter: runtime/data only; model and loss math live in packages."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from litespec.core.composable.batch import (
    PreparedLogicalBatch,
    cpu_copy,
    prepare_layout,
    tensor_bytes,
)
from litespec.core.composable.objectives.selector import LossBranches
from litespec.modeling.draft.composable.schema_v3 import fields
from litespec.modeling.draft.composable.loss_config import LOSS_CONTRACT_VERSION, normalize_training_recipe
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.modeling.config_utils import load_text_model_config
from litespec.modeling.draft.composable import (
    ComposableDraftConfig,
    ComposableDraftModel,
)
from litespec.training.config import add_common_args, validate_common_args
from litespec.training.logging import get_tracker_class
from litespec.training.recipes._dflash_common import (
    DFlashFamilyTrainingAdapter,
    prepare_dflash_family_features,
)


def parse_args(mode=None, argv=None):
    parser = argparse.ArgumentParser(description="Train a composable parallel draft")
    add_common_args(parser)
    parser.set_defaults(attention_backend="sdpa", num_anchors=32)
    parser.add_argument("--draft-config", default="configs/composable_draft.json")
    parser.add_argument("--gdn-backend", choices=["auto", "fla", "reference"], default="auto")
    parser.add_argument("--init-from")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument(
        "--override-attention-backend", choices=["eager", "sdpa", "flex_attention"]
    )
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this many optimizer updates, preserving the full-epoch LR schedule.",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument(
        "--objective-chunk-blocks", type=int, default=512,
        help="Anchor blocks per loss chunk; 0 processes all anchors together.",
    )
    parser.add_argument("--window-buffer-mb", type=int, default=256)
    parser.add_argument("--complete-blocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anchor-sampling", choices=["sample_epoch", "legacy"],
                        default="sample_epoch",
                        help="Shared stateless anchors for OPD/SFT; legacy preserves historical RNG behavior.")
    parser.add_argument("--supervision-mode", choices=["sampled", "all"], default="sampled",
                        help="EAGLE3 loss starts: sampled anchors or every complete eligible block.")
    parser.add_argument("--opd-data-mode", choices=["teacher_forced", "sampled_anchor"], default="teacher_forced")
    parser.add_argument("--opd-target-chunk-anchors", type=int, default=0,
                        help="Sampled-anchor OPD currently requires 0: score all anchors together.")
    parser.add_argument("--bv-sampling-provenance", choices=["target_full_vocab_t1"])
    tokens = sys.argv[1:] if argv is None else argv
    removed = ("--selector-loss-alpha", "--selector-base-ce-alpha", "--selector-warmup-ratio", "--selector-ramp-ratio")
    if any(token.split("=", 1)[0] in removed for token in tokens):
        parser.error("Selector loss CLI parameters were removed; configure objective.base/final.weight instead")
    args = parser.parse_args(tokens)
    if args.anchor_sampling == "sample_epoch":
        args.complete_blocks = True
        if not any(token.split("=", 1)[0] == "--num-anchors" for token in tokens):
            args.num_anchors = 512
    if args.opd_data_mode == "sampled_anchor":
        if not any(token.split("=", 1)[0] == "--num-anchors" for token in tokens):
            args.num_anchors = 512
        if not any(token.split("=", 1)[0] == "--objective-chunk-blocks" for token in tokens):
            args.objective_chunk_blocks = 0
        if args.objective_chunk_blocks != 0 or args.opd_target_chunk_anchors != 0:
            parser.error("sampled_anchor processes all anchors; both chunk settings must be 0")
        if args.disaggregate or args.tp_size != 1:
            parser.error("sampled_anchor requires a collocated target, tp_size=1")
        if not (args.init_from or args.resume_from):
            parser.error("sampled_anchor requires --init-from or --resume-from")
    args.algorithm = "composable"
    validate_common_args(parser, args)
    if args.objective_chunk_blocks < 0 or args.window_buffer_mb <= 0:
        parser.error("chunk blocks must be nonnegative and window-buffer-mb positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    args.require_target_last_hidden = False
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


class ComposableTrainingAdapter(DFlashFamilyTrainingAdapter):
    def phases(self, args):
        from dataclasses import replace

        phases = super().phases(args)
        if args.max_steps is not None:
            phases[0] = replace(
                phases[0],
                max_batches=args.max_steps * args.accumulation_steps,
                schedule_full_epochs=True,
            )
        return phases

    def build_config(self, args, *, source_config=None, model_role=None):
        target = (
            load_text_model_config(
                args.target_model_path, trust_remote_code=args.trust_remote_code
            )
            if source_config is None
            else copy.deepcopy(source_config)
        )
        payload = json.loads(Path(args.draft_config).read_text())
        fields(payload, {"architectures", "block_size", "schema_version", "composable_config"}, "draft JSON")
        if payload.get("schema_version") != "composable_v3":
            raise ValueError("Fresh recipes require schema_version=composable_v3")
        if "architectures" in payload and payload["architectures"] != ["ComposableDraftModel"]:
            raise ValueError("architectures must be [ComposableDraftModel]")
        recipe = copy.deepcopy(payload["composable_config"])
        values = target.to_dict()
        target_depth = getattr(target, "num_target_layers", values["num_hidden_layers"])
        reference_attention = {key: getattr(target, key) for key in
                               ("num_attention_heads", "num_key_value_heads", "head_dim")}
        for key in ("model_type", "architectures", "auto_map", "layer_types", "composable_config",
                    "block_size", "num_target_layers", "schema_version", "execution_config", "target_attention",
                    "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            values.pop(key, None)
        values["num_hidden_layers"] = len(recipe["backbone"]["layers"])
        execution = {"attention_backend": args.override_attention_backend or args.attention_backend,
                     "flex_attention_backend": args.flex_attention_backend,
                     "gdn_backend": getattr(args, "gdn_backend", "auto")}
        config = ComposableDraftConfig(**values, composable_config=recipe,
            block_size=payload.get("block_size", args.block_size), num_target_layers=target_depth,
            schema_version="composable_v3", target_attention=reference_attention, execution_config=execution)
        if config.indexer_config is not None and not config.indexer_config["enabled"]:
            raise ValueError("DSA enabled=false is inference/reference only")
        if args.target_layer_ids:
            config.composable_config["context"]["target_layer_ids"] = [
                int(value) for value in args.target_layer_ids.split(",")
            ]
        config.validate_recipe()
        config.composable_config = normalize_training_recipe(config.composable_config)
        return config

    def initialize(self, session):
        args = session.args
        source = args.resume_from or args.init_from
        if args.resume_from and (getattr(session, 'resume_state', None) or {}).get('loss_contract_version') != LOSS_CONTRACT_VERSION:
            raise ValueError("Cannot resume a historical loss contract; use init_from with an explicitly migrated objective to start new training")
        if args.init_from and not args.resume_from:
            from safetensors.torch import load_file
            draft = self.build_model(args)
            state = load_file(str(Path(args.init_from) / "model.safetensors"))
            loading = draft.load_state_dict(state, strict=False)
            if loading.unexpected_keys or any(not key.startswith("backbone.indexer.") for key in loading.missing_keys):
                raise ValueError("Initialization checkpoint mismatch: " + str(loading))
            del state
        elif source:
            draft, loading = ComposableDraftModel.from_pretrained(
                source, torch_dtype=torch.bfloat16, output_loading_info=True
            )
            if any(
                loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
            ):
                raise ValueError(
                    "Composable checkpoint weights do not match its model recipe: "
                    + str(loading)
                )
            draft = draft.cuda()
        else:
            draft = self.build_model(args)
        draft.requires_grad_(True)
        if draft.is_eagle3 and args.opd_data_mode != "teacher_forced":
            raise ValueError("EAGLE3 requires teacher_forced TTT; sampled_anchor OPD is unsupported")
        if not draft.is_eagle3 and getattr(args, "supervision_mode", "sampled") != "sampled":
            raise ValueError("supervision_mode=all is currently EAGLE3-only")
        if draft.is_eagle3:
            args.complete_blocks = True
        draft.config.composable_config = normalize_training_recipe(draft.config.composable_config)
        loss_branches = LossBranches(draft.config.composable_config)
        if getattr(args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
            recipe = draft.config.composable_config
            if recipe['proposal']['type'] not in ('parallel', 'rnn') or recipe['objective']['type'] != 'opd_survival':
                raise ValueError("sampled_anchor requires Parallel/RNN with opd_survival")
            if float(draft.config.attention_dropout) != 0.:
                raise ValueError("sampled_anchor requires zero attention dropout")
        # Acceptance diagnostics are part of every composable training log,
        # including CE-only objectives. Teacher logits remain logging-only when
        # the objective itself is teacher-free (see ComposableTrainingModel).
        args.require_target_last_hidden = True
        indexer = draft.config.indexer_config
        if indexer is not None and indexer["stage"] == "warmup":
            draft.requires_grad_(False)
            draft.backbone.indexer.requires_grad_(True)
        draft.backbone.gradient_checkpointing = True
        session.draft = draft
        args.block_size = draft.block_size
        args.num_draft_layers = draft.config.num_hidden_layers
        args.target_layer_ids = ",".join(map(str, draft.target_layer_ids))
        if (
            loss_branches.needs_bv
            and args.bv_sampling_provenance != "target_full_vocab_t1"
        ):
            raise ValueError(
                "BV requires --bv-sampling-provenance target_full_vocab_t1; verify the dataset sampling provenance first"
            )

    def make_online(self, session):
        if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
            if session.args.target_model_backend == 'sglang':
                from litespec.modeling.target.sglang_anchor_opd import SGLangAnchorScorer
                session.anchor_scorer = SGLangAnchorScorer(session.target)
            else:
                from litespec.modeling.target.anchor_opd import Qwen3AnchorScorer
                session.anchor_scorer = Qwen3AnchorScorer(session.target.model)
        return ComposableTrainingModel(
            session.draft,
            session.components.lm_head,
            session.components.embed_tokens,
            session.args.mask_token_id,
            num_anchors=session.args.num_anchors,
            objective_chunk_blocks=session.args.objective_chunk_blocks,
            loss_process_group=session.process_group,
            opd_data_mode=getattr(session.args, "opd_data_mode", "teacher_forced"),
            complete_blocks=getattr(session.args, "complete_blocks", False),
            supervision_mode=getattr(session.args, "supervision_mode", "sampled"),
        )

    def target_proxies(self, session):
        proxies = super().target_proxies(session)
        for proxy in proxies:
            proxy.method_config_key = "composable_config"
        return proxies

    def packet_spec(self, session, phase):
        from litespec.core.composable.packets import ComposablePacketSpec

        base = super().packet_spec(session, phase)
        return ComposablePacketSpec(
            base.batch_size,
            base.max_length,
            base.context_width,
            base.hidden_size,
            base.include_target_last_hidden,
        )

    def empty_packet(self, spec):
        from litespec.core.composable.packets import ComposableBatchPacket

        return ComposableBatchPacket.empty(spec)

    def produce_packet(self, session, phase, values, batch_id):
        from litespec.core.composable.packets import ComposableBatchPacket

        packet = super().produce_packet(session, phase, values, batch_id)
        if packet is None:
            return None
        return ComposableBatchPacket(
            packet.input_ids,
            packet.loss_mask,
            packet.target_context_hidden,
            packet.target_last_hidden,
            values[1].bool(),
        )

    def log_metrics(self, session, reduced):
        if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
            if "rollout/accepted_length" not in reduced:
                raise RuntimeError("Sampled anchor OPD requires actual verifier acceptance metrics")
        required = {"hard_label/block_accepted_length", "expected_accept_len"}
        missing = required.difference(reduced)
        if missing:
            raise RuntimeError(
                "Composable training acceptance diagnostics missing: "
                + ", ".join(sorted(missing))
                + "; prepare target_last_hidden for metric collection"
            )
        return super().log_metrics(session, reduced)

    def training_metadata(self, session):
        metadata = {
            "loss_config": copy.deepcopy(
                session.draft.config.composable_config["objective"]
            ),
            "composable_recipe": copy.deepcopy(session.draft.config.composable_config),
            "loss_contract_version": LOSS_CONTRACT_VERSION,
            "objective_chunk_blocks": session.args.objective_chunk_blocks,
            "num_anchors": session.args.num_anchors,
            "anchor_sampling": getattr(session.args, "anchor_sampling", "legacy"),
            "anchor_sampling_version": ("sample_epoch_v1" if getattr(session.args, "anchor_sampling", "legacy") == "sample_epoch" else "legacy"),
            "anchor_seed": session.args.seed,
            "complete_blocks": getattr(session.args, "complete_blocks", False),
            "bv_sampling_provenance": session.args.bv_sampling_provenance,
        }
        if session.draft.is_eagle3:
            metadata.update(supervision_mode=getattr(session.args, "supervision_mode", "sampled"),
                            ttt_length=session.draft.ttt_length)
        branches = LossBranches(session.draft.config.composable_config)
        if any(getattr(o, 'needs_component_normalization', False)
               for o in branches.objectives.values()):
            metadata.update(opd_interpretation='offline_empirical_teacher_forced',
                opd_accepted_prefix='detached_greedy_match_to_fixed_labels',
                opd_reachability='min_accepted_plus_one_including_first_failure')
        if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
            metadata.update(opd_data_mode='sampled_anchor', opd_interpretation='current_policy_sampled_blocks',
                opd_accepted_prefix='probabilistic_rejection_sampling', opd_temperature=1.,
                opd_context_source='offline_hidden' if session.args.train_hidden_states_path else 'online',
                opd_target_chunk_anchors=0)
        return metadata

    def prepare_optimizer_window(self, session, first, iterator, count):
        if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
            from litespec.core.composable.anchor_opd_windows import prepare_anchor_opd_window
            return prepare_anchor_opd_window(self, session, first, iterator, count)
        recipe = session.draft.config.composable_config
        indexer = recipe.get("indexer")
        branches = LossBranches(recipe)
        if any(getattr(o, 'needs_component_normalization', False)
               for o in branches.objectives.values()):
            from litespec.core.composable.opd_windows import prepare_opd_window
            return prepare_opd_window(self, session, first, iterator, count)
        if recipe.get("proposal", {}).get("type") not in ("selector", "residual_selector", "topk_selector") and indexer is None:
            return None
        prepared, used = [], 0
        totals = torch.zeros(3, device="cuda", dtype=torch.float64)
        for index in range(count):
            raw = first if index == 0 else next(iterator)
            # Copy before advancing PacketReceiver: its slots are reusable.
            data = cpu_copy(raw)
            if session.topology is not None:
                ids, mask = data.input_ids, data.loss_mask
            else:
                ids, mask = data["input_ids"], data["loss_mask"]
                if session.tp_draft_rank is not None:
                    ids, mask = (
                        ids[session.tp_draft_rank : session.tp_draft_rank + 1],
                        mask[session.tp_draft_rank : session.tp_draft_rank + 1],
                    )
            layout = self.anchor_layout(session, data)
            if layout is None:
                layout = prepare_layout(
                    ids, mask, session.draft.block_size, session.args.num_anchors,
                    complete_blocks=branches.requires_complete_blocks,
                )
            denominators = branches.native_denominators(layout)
            primary = denominators["final"].item()
            if indexer is None:
                auxiliary = denominators["base"].item()
            else:
                context_valid = (data.context_valid_mask if session.topology is not None
                                 else data["attention_mask"])
                if session.tp_draft_rank is not None:
                    context_valid = context_valid[session.tp_draft_rank:session.tp_draft_rank+1]
                positions = torch.arange(ids.size(1))
                has_history = (context_valid[:,None].bool() &
                               (positions[None,None] < layout.anchor_positions[...,None])).any(-1)
                auxiliary = (layout.block_keep_mask & (layout.prediction_mask > 0).any(-1) & has_history).sum().item()
            totals += totals.new_tensor([ids.size(0), primary, auxiliary])
            item = PreparedLogicalBatch(data, layout, totals)
            size = tensor_bytes(data)
            if used + size > session.args.window_buffer_mb * 1024**2:
                item.spill(Path(session.args.cache_dir) / "composable_windows")
            else:
                used += size
            prepared.append(item)
        if dist.is_initialized():
            dist.all_reduce(totals, group=session.process_group)
        invalid = (totals[1] <= 0 if not branches.is_selector else any(
            totals[1 if name == 'final' else 2] <= 0 for name in branches.objectives))
        if invalid:
            for item in prepared:
                item.release()
            raise ValueError("Optimizer window has no supervised proposal positions")
        for item in prepared:
            item.window_denominators = totals
        return prepared

    def release_batch(self, data):
        if isinstance(data, PreparedLogicalBatch):
            data.release()

    def _step_values(self, session, data, microbatch_start=None, microbatch_end=None):
        args, draft = session.args, session.draft
        if session.topology is not None:
            input_ids = data.input_ids[microbatch_start:microbatch_end].cuda()
            attention_mask = data.context_valid_mask[
                microbatch_start:microbatch_end
            ].cuda()
            loss_mask = data.loss_mask[microbatch_start:microbatch_end].cuda()
            target_hidden_states = data.target_context_hidden[
                microbatch_start:microbatch_end
            ].cuda()
            last = data.target_last_hidden
            last = (
                None if last is None else last[microbatch_start:microbatch_end].cuda()
            )
        else:
            from litespec.training.setup import select_tp_rank_batch

            input_ids = data["input_ids"].cuda(non_blocking=True)
            attention_mask = data["attention_mask"].cuda(non_blocking=True)
            loss_mask = data["loss_mask"].cuda(non_blocking=True)
            target_output = (
                None
                if session.target is None
                else session.target.generate_training_data(
                    input_ids, attention_mask, loss_mask, return_logits=False
                )
            )
            target_hidden_states, last = prepare_dflash_family_features(
                data, target_output, draft, args.require_target_last_hidden
            )
            if session.tp_draft_rank is not None:
                input_ids, attention_mask, loss_mask, target_hidden_states, last = [
                    select_tp_rank_batch(value, session.tp_draft_rank)
                    for value in (
                        input_ids,
                        attention_mask,
                        loss_mask,
                        target_hidden_states,
                        last,
                    )
                ]
        return dict(input_ids=input_ids, target_hidden_states=target_hidden_states,
                    loss_mask=loss_mask, target_last_hidden_states=last,
                    context_valid_mask=attention_mask.bool())

    def anchor_layout(self, session, data, *, microbatch_start=None, microbatch_end=None):
        eagle3 = session.draft.is_eagle3
        supervision_mode = getattr(session.args, "supervision_mode", "sampled")
        if getattr(session.args, "anchor_sampling", "legacy") != "sample_epoch" and not (eagle3 and supervision_mode == "all"):
            return None
        from litespec.core.composable.anchor_sampling import prepare_sample_epoch_layout

        if session.topology is None:
            ids, mask, valid = (data[key] for key in ("input_ids", "loss_mask", "attention_mask"))
            indices = data.get("sample_index")
            epoch = int(data.get("_anchor_epoch", session.state.epoch))
            if session.tp_draft_rank is not None:
                start = session.tp_draft_rank
                ids, mask, valid = (value[start:start + 1] for value in (ids, mask, valid))
                if indices is not None:
                    indices = indices[start:start + 1]
        else:
            ids, mask, valid = data.input_ids, data.loss_mask, data.context_valid_mask
            indices = None
            epoch = session.state.epoch
        subset = slice(microbatch_start, microbatch_end)
        return prepare_sample_epoch_layout(ids[subset], mask[subset], valid[subset],
            sample_indices=None if indices is None else indices[subset],
            block_size=session.draft.block_size, num_anchors=session.args.num_anchors,
            seed=session.args.seed, epoch=epoch,
            supervision_mode=supervision_mode if eagle3 else "sampled",
            minimum_anchor=1 if eagle3 else 0)

    def _anchor_values(self, session, data):
        ids = data['input_ids'].cuda(non_blocking=True)
        hidden = last = None
        if session.args.train_hidden_states_path:
            hidden, last = prepare_dflash_family_features(data, None, session.draft, True)
        return dict(input_ids=ids, loss_mask=data['loss_mask'].cuda(non_blocking=True),
            target_hidden_states=hidden, target_last_hidden_states=last,
            context_valid_mask=data['attention_mask'].cuda(non_blocking=True).bool(),
            layout=self.anchor_layout(session, data))

    def compute_step(
        self,
        session,
        data,
        *,
        microbatch_start=None,
        microbatch_end=None,
        collect_detailed_metrics=False,
    ):
        if getattr(session.args, "opd_data_mode", "teacher_forced") == "sampled_anchor":
            if isinstance(data, PreparedLogicalBatch):
                values = data.materialize()
                record = values['sampled_opd_batch'].to('cuda')
                return session.fsdp_model(input_ids=values['input_ids'].cuda(non_blocking=True),
                    loss_mask=values['loss_mask'].cuda(non_blocking=True), target_hidden_states=None,
                    context_valid_mask=record.context_valid, sampled_opd_batch=record,
                    opd_global_denominators=data.opd_component_denominators,
                    opd_reference_samples=data.opd_reference_samples,
                    collect_detailed_metrics=collect_detailed_metrics)
            return session.fsdp_model(**self._anchor_values(session, data),
                                       opd_scorer=session.anchor_scorer,
                                       collect_detailed_metrics=collect_detailed_metrics)
        prepared = data if isinstance(data, PreparedLogicalBatch) else None
        data = prepared.materialize() if prepared is not None else data
        if prepared is not None and prepared.opd_inputs_prepared:
            values = {key: (None if value is None else
                           value[microbatch_start:microbatch_end].cuda(non_blocking=True))
                      for key, value in data.items()}
        else:
            values = self._step_values(session, data, microbatch_start, microbatch_end)
        layout = (None if prepared is None else
                  prepared.layout.batch_slice(microbatch_start, microbatch_end).to(
                      values['input_ids'].device))
        if prepared is None:
            layout = self.anchor_layout(session, data, microbatch_start=microbatch_start,
                                        microbatch_end=microbatch_end)
            if layout is not None:
                layout = layout.to(values['input_ids'].device)
        component = {}
        if prepared is not None and prepared.opd_metadata is not None:
            from litespec.training.checkpoint import restore_rng_state
            meta = prepared.opd_metadata[microbatch_start, microbatch_end]
            component = dict(
                opd_accepted_lengths={name: r.to(values['input_ids'].device)
                                      for name, r in meta.accepted_lengths.items()},
                opd_global_denominators=prepared.opd_component_denominators,
                opd_reference_samples=prepared.opd_reference_samples)
            restore_rng_state(meta.rng_state)
        return session.fsdp_model(
            **values, layout=layout,
            window_denominators=None if prepared is None else prepared.window_denominators,
            epoch_progress=session.state.phase_step / session.steps_per_epoch,
            collect_detailed_metrics=collect_detailed_metrics, **component)


ADAPTER = ComposableTrainingAdapter("composable", "ComposableDraftModel")
