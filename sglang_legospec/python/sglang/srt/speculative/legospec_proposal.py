"""Graph-compatible full-vocabulary proposals with authoritative final q."""

import torch
from sglang.kernels.ops.sampling.stable_top_p import stable_nucleus_probs
from sglang.kernels.ops.speculative.legospec import selector_full_vocab_row
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    DraftParams,
    DraftProposal,
    draft_markov_block,
    resolve_draft_params,
    select_draft_hidden_without_anchor,
)
from sglang.srt.speculative.legospec_rng import legospec_request_seed


@torch.compile(dynamic=True)
def _sample_q(probs, seeds, positions):
    return multinomial_with_seed(probs.log(), seeds, positions).view(-1).long()


class LEGOSpecDraftSampler:
    confidence_out = confidence_raw_out = None
    verify_ids = None

    def __init__(
        self, model, max_bs, device, greedy_draft, seed, *, noise=None, tp_sync=None
    ):
        self.model, self.max_bs, self.greedy_draft = model, max_bs, greedy_draft
        self.gamma = model.gamma
        self.seed = int(seed)
        self._seed_rows = None
        self.need_top_p = True
        self.noise, self.tp_sync = noise, tp_sync
        self.params = DraftParams(
            torch.ones(max_bs, device=device),
            torch.ones(max_bs, dtype=torch.bool, device=device),
            torch.zeros(max_bs, device=device),
        )
        self.top_ps = torch.ones(max_bs, device=device)
        self.seeds = torch.full((max_bs,), self.seed, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.out = torch.empty(max_bs * self.gamma, dtype=torch.int64, device=device)
        self.step_logits = None
        self.proposal_probs = (
            None
            if greedy_draft
            else torch.empty(
                max_bs,
                self.gamma,
                model.config.vocab_size,
                dtype=torch.float32,
                device=device,
            )
        )

    def stage_sampling_params(self, *, bs, sampling_info):
        resolve_draft_params(sampling_info=sampling_info, bs=bs, out=self.params)
        self.top_ps.fill_(1)
        self.need_top_p = (
            sampling_info is not None and sampling_info.need_top_p_sampling
        )
        if sampling_info is not None:
            self.top_ps[:bs].copy_(sampling_info.top_ps.view(-1)[:bs])

    def stage_request_seeds(self, reqs):
        rows = tuple((req.rid, req.sampling_params.sampling_seed) for req in reqs)
        if rows == self._seed_rows:
            return
        seeds = []
        for req in reqs:
            seeds.append(legospec_request_seed(req, self.seed) ^ 0x2D2A9C317E44B1F3)
        cpu = torch.tensor(seeds, dtype=torch.int64, pin_memory=True)
        self.seeds[: len(seeds)].copy_(cpu, non_blocking=True)
        self._seed_rows = rows

    def stage_positions(self, prefix_lens):
        self.positions.zero_()
        self.positions[: prefix_lens.numel()].copy_(prefix_lens)
        self.positions.mul_(8)

    @torch.inference_mode()
    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.model.block_size
        raw, hidden = select_draft_hidden_without_anchor(
            hidden_states, bs=bs, gamma=self.gamma
        )
        base, _ = self.model.compute_base_logits(raw)
        base = base.view(bs, self.gamma, -1)
        previous = input_ids.view(bs, self.model.block_size)[:, 0]
        if self.greedy_draft and self.model.proposal_type == "rnn":
            tokens, _ = draft_markov_block(
                markov_head=self.model.markov_head,
                base_logits=base,
                anchor_tokens=previous,
                draft_hidden=hidden,
                step_temperatures=self.params.step_temperatures[:bs],
                noise=self.noise,
                tp_sync=self.tp_sync,
                greedy_draft=True,
            )
            self.out[: bs * self.gamma].copy_(tokens.reshape(-1))
            return
        state = None
        is_selector = self.model.proposal_type in ("selector", "residual_selector", "topk_selector")
        candidate_only = (is_selector and self.model.config.composable_config['proposal']['selector_mode'] == 'candidate_only')
        if is_selector:
            selector = self.model.candidate_selector
            candidates = base.topk(selector.top_k, dim=-1).indices
            delta = selector.build_lattice(
                candidate_ids=candidates,
                unary_logits=base.new_zeros(candidates.shape),
                hidden_states=hidden,
                anchor_token_ids=previous,
            )
        else:
            state = self.model.markov_head.init_state(hidden)
        tokens = self.out[: bs * self.gamma].view(bs, self.gamma)
        for step in range(self.gamma):
            if is_selector:
                predecessors = (
                    previous[:, None].expand(-1, selector.top_k)
                    if step == 0
                    else candidates[:, step - 1]
                )
                if candidate_only:
                    rows = (predecessors == previous[:, None]).long().argmax(-1)
                    correction = delta[:, step].gather(1, rows[:, None, None].expand(-1, 1, selector.top_k)).squeeze(1)
                    logits = base[:, step].gather(-1, candidates[:, step]).float() + correction.float()
                else:
                    logits = selector_full_vocab_row(
                        base[:, step],
                        candidates[:, step],
                        predecessors,
                        previous,
                        delta[:, step],
                    )
            else:
                bias, state = self.model.markov_head.step_bias(
                    previous, hidden_states=hidden[:, step], state=state
                )
                logits = (base[:, step].float() + bias.float()).to(
                    self.model.markov_head.step_logits_dtype(base.dtype)
                )
            greedy = logits.argmax(-1)
            if self.greedy_draft:
                previous = candidates[:, step].gather(1, greedy[:, None]).squeeze(1) if candidate_only else greedy
            else:
                q = stable_nucleus_probs(
                    logits.float() / self.params.temperatures[:bs, None],
                    self.top_ps[:bs],
                    truncate=self.need_top_p,
                )
                one_hot = torch.zeros_like(q).scatter_(-1, greedy[:, None], 1)
                q = torch.where(self.params.greedy_mask[:bs, None], one_hot, q)
                selected = _sample_q(q, self.seeds[:bs], self.positions[:bs] + step)
                if candidate_only:
                    dense_q = torch.zeros_like(base[:, step], dtype=torch.float32).scatter(-1, candidates[:, step], q)
                    self.proposal_probs[:bs, step].copy_(dense_q)
                    previous = candidates[:, step].gather(1, selected[:, None]).squeeze(1)
                else:
                    self.proposal_probs[:bs, step].copy_(q)
                    previous = selected
            tokens[:, step].copy_(previous)

    def result(self, bs):
        return DraftBlockResult(
            draft_tokens=self.out[: bs * self.gamma].view(bs, self.gamma),
            corrected_logits=None,
            greedy_mask=self.params.greedy_mask[:bs],
            temperatures=self.params.temperatures[:bs],
            greedy_draft=self.greedy_draft,
            proposal_probs=None if self.greedy_draft else self.proposal_probs[:bs],
            proposal_seeds=self.seeds[:bs],
            proposal_positions=self.positions[:bs],
        )


class LEGOSpecProposer(DraftBlockProposer):
    def propose(
        self,
        *,
        batch,
        draft_input,
        verify_window,
        bs,
        device,
        target_model,
        sampling_info,
        anchor_prepared=False,
    ):
        sampler = self._draft_sampler
        if sampler is None:
            raise RuntimeError("LEGOSpec sampler was not initialized")
        fwd = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=self.draft_model.embed_tokens,
            draft_sampler=None,
            sampling_info=sampling_info,
            anchor_prepared=anchor_prepared,
        )
        block = sampler.propose(
            fwd=fwd, batch=batch, sampling_info=sampling_info, bs=bs
        )
        return DraftProposal(
            draft_block_ids=fwd.draft_block_ids[:, : self.gamma].contiguous(),
            draft_block=block,
            folded=True,
        )


class LEGOSpecProposalGraphs:
    """Fixed seven-step graphs, with no dense q for pure greedy batches."""

    def __init__(self, model, max_bs, device, seed, *, noise, tp_sync):
        self.options = dict(
            model=model,
            max_bs=max_bs,
            device=device,
            seed=seed,
            noise=noise,
            tp_sync=tp_sync,
        )
        self.greedy = LEGOSpecDraftSampler(**self.options, greedy_draft=True)
        self.sampled = None
        self.frames = {}
        self.graph_replay_ct = self.graph_eager_ct = 0

    @torch.inference_mode()
    def propose(self, *, fwd, batch, sampling_info, bs):
        greedy = sampling_info is None or sampling_info.is_all_greedy
        if not greedy and self.sampled is None:
            self.sampled = LEGOSpecDraftSampler(**self.options, greedy_draft=False)
        sampler = self.greedy if greedy else self.sampled
        sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)
        sampler.stage_request_seeds(batch.reqs)
        sampler.stage_positions(batch.seq_lens)
        if bs == 0:
            return sampler.result(0)
        key = (bs, greedy, sampler.need_top_p)
        frame = self.frames.get(key)
        if frame is None:
            h = fwd.draft_hidden_3d
            frame = _ProposalFrame(bs, h.shape[-1], h.device, h.dtype)
            self.frames[key] = frame
        # The sampler removes the unused anchor hidden before applying LM head.
        frame.hidden[:, 1:].copy_(fwd.draft_hidden_3d)
        frame.ids.copy_(fwd.draft_block_ids)

        def run():
            sampler(frame.hidden.flatten(0, 1), frame.ids.flatten())

        if (
            get_exec().graph.cuda_graph_config.decode.backend == "disabled"
            or not frame.warm
        ):
            run()
            frame.warm = True
            self.graph_eager_ct += 1
        else:
            if frame.graph is None:
                frame.graph = torch.cuda.CUDAGraph()
                with (
                    graph_pool_capture_scope(),
                    torch.cuda.graph(
                        frame.graph,
                        pool=get_or_create_global_graph_memory_pool(torch.cuda),
                        stream=get_or_create_global_graph_capture_stream(),
                    ),
                ):
                    run()
            with graph_pool_replay_scope():
                frame.graph.replay()
            self.graph_replay_ct += 1
        return sampler.result(bs)


class _ProposalFrame:
    def __init__(self, bs, width, device, dtype):
        self.hidden = torch.zeros((bs, 8, width), device=device, dtype=dtype)
        self.ids = torch.empty((bs, 8), device=device, dtype=torch.int64)
        self.graph, self.warm = None, False
