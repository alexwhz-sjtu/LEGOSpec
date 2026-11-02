"""Lossless chain verification using the actual LEGOSpec proposal probabilities."""

import msgspec
import torch
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.runtime_context import get_exec, get_schedule
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor


class _SamplingFrame(msgspec.Struct):
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    is_all_greedy: bool
    is_any_greedy: bool
    need_top_p_sampling: bool
    need_top_k_sampling: bool
    need_min_p_sampling: bool = False


class _AcceptInputs:
    """Maximum-batch input storage shared by serial acceptance graphs.

    Like native static verify IO, only views depend on the capture shape.
    Sampling variants retain separate small outputs, not full-vocabulary inputs.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffers = {}

    def view(self, name, source, bs):
        if source is None:
            return None
        rows = source.shape[0]
        if not 0 < bs <= self.capacity or rows % bs:
            raise ValueError("Invalid LEGOSpec acceptance input batch geometry")
        shape = (self.capacity * (rows // bs), *source.shape[1:])
        buffer = self.buffers.get(name)
        if buffer is None:
            buffer = torch.empty(shape, dtype=source.dtype, device=source.device)
            self.buffers[name] = buffer
        elif (
            buffer.shape != shape
            or buffer.dtype != source.dtype
            or buffer.device != source.device
        ):
            raise ValueError(f"LEGOSpec acceptance input layout changed: {name}")
        return buffer[:rows]


class _AcceptFrame:
    def __init__(self, args, inputs):
        def view(name, source):
            return inputs.view(name, source, args["bs"])

        block = args["draft_block"]
        self.candidates = view("candidates", args["verify_ids_2d"])
        self.logits = view("logits", args["target_logits"])
        self.prefix = view("prefix", args["prefix_lens"])
        self.draft_tokens = view("draft_tokens", block.draft_tokens)
        sampling = args["sampling_info"]
        self.sampling = _SamplingFrame(
            temperatures=view("sampling_temperatures", sampling.temperatures),
            top_ps=view("top_ps", sampling.top_ps),
            top_ks=view("top_ks", sampling.top_ks),
            is_all_greedy=sampling.is_all_greedy,
            is_any_greedy=sampling.is_any_greedy,
            need_top_p_sampling=sampling.need_top_p_sampling,
            need_top_k_sampling=sampling.need_top_k_sampling,
        )
        self.block = DraftBlockResult(
            draft_tokens=self.draft_tokens,
            corrected_logits=None,
            greedy_mask=view("greedy_mask", block.greedy_mask),
            temperatures=view("block_temperatures", block.temperatures),
            greedy_draft=block.greedy_draft,
            proposal_probs=view("proposal_probs", block.proposal_probs),
            proposal_seeds=view("proposal_seeds", block.proposal_seeds),
            proposal_positions=view("proposal_positions", block.proposal_positions),
        )
        self.graph, self.outputs, self.warm = None, None, False

    def stage(self, args):
        self.candidates.copy_(args["verify_ids_2d"])
        self.draft_tokens.copy_(args["draft_tokens"])
        self.logits.copy_(args["target_logits"])
        self.prefix.copy_(args["prefix_lens"])
        for name in ("temperatures", "top_ps", "top_ks"):
            getattr(self.sampling, name).copy_(getattr(args["sampling_info"], name))
        for name in (
            "greedy_mask",
            "temperatures",
            "proposal_probs",
            "proposal_seeds",
            "proposal_positions",
        ):
            target = getattr(self.block, name)
            if target is not None:
                target.copy_(getattr(args["draft_block"], name))


class LEGOSpecVerifyExecutor(TargetVerifyExecutor):
    """Capture probability transforms, rejection, bonus and result construction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames = {}
        self.graph_replay_ct = self.graph_eager_ct = 0
        capacity = max(
            max(get_exec().graph.cuda_graph_config.decode.bs or [1]),
            get_schedule().max_running_requests or 32,
        )
        self.input_buffers = _AcceptInputs(capacity)
        _get_or_create_chain_verify_buffers(
            bs=capacity,
            draft_token_num=self.verify_num_draft_tokens,
            device=torch.device("cuda", torch.cuda.current_device()),
        )

    def accept_and_finalize(self, **args):
        self._last_candidates = args["verify_ids_2d"]
        sampling = args["sampling_info"]
        if (
            get_exec().graph.cuda_graph_config.decode.backend == "disabled"
            or sampling is None
            or args["folded_accept"]
        ):
            return super().accept_and_finalize(**args)
        key = (
            args["bs"],
            sampling.is_all_greedy,
            sampling.is_any_greedy,
            sampling.need_top_p_sampling,
            sampling.need_top_k_sampling,
        )
        frame = self.frames.get(key)
        if frame is None:
            frame = _AcceptFrame(args, self.input_buffers)
            self.frames[key] = frame
        frame.stage(args)
        static = dict(
            args,
            verify_ids_2d=frame.candidates,
            target_logits=frame.logits,
            prefix_lens=frame.prefix,
            draft_tokens=frame.draft_tokens,
            draft_block=frame.block,
            sampling_info=frame.sampling,
        )
        if not frame.warm:
            frame.outputs = super().accept_and_finalize(**static)
            frame.warm = True
            self.graph_eager_ct += 1
            return frame.outputs
        if frame.graph is None:
            frame.graph = torch.cuda.CUDAGraph()
            # Acceptance results survive into the next overlap iteration.
            # Keep the eager warmup's outputs outside the shared graph pool:
            # an earlier-captured backbone may overwrite allocator-free pool
            # blocks before the result copy or next-prefix consumer finishes.
            with (
                graph_pool_capture_scope(),
                torch.cuda.graph(
                    frame.graph,
                    pool=get_or_create_global_graph_memory_pool(torch.cuda),
                    stream=get_or_create_global_graph_capture_stream(),
                ),
            ):
                super().accept_and_finalize(**static, out=frame.outputs)
        with graph_pool_replay_scope():
            frame.graph.replay()
        self.graph_replay_ct += 1
        return frame.outputs

    def commit_hidden(self, **args):
        if args["run_compact"]:
            return super().commit_hidden(**args)
        hidden = args["logits_output"].hidden_states
        if hidden is None:
            raise RuntimeError("LEGOSpec commit requires verified target hidden states")
        self.kv_injector.commit_static_hidden(
            batch=args["batch"],
            verify_window=args["verify_window"],
            hidden=hidden,
            commit_lens=args["commit_lens"],
            bs=args["bs"],
            candidates=self._last_candidates,
        )
