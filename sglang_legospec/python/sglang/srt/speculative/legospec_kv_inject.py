"""Native context placement, with explicit request addressing for stateful modules."""

import torch
from sglang.kernels.ops.speculative.legospec_commit import legospec_commit_lens
from sglang.srt.mem_cache.legospec_gdn_pool import GDNCommitLayout
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)


class LEGOSpecKvInjector(TargetHiddenKvInjector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commit_frames = {}
        self.prefill_layouts = {}
        self.graph_replay_ct = self.graph_eager_ct = 0

    @property
    def requires_state_slots(self):
        pool = self.draft_model_runner.token_to_kv_pool
        return bool(getattr(pool, "swa_layers", []) or getattr(pool, "gdn_layers", []))

    def place_prefill(self, batch):
        placement = super().place_prefill(batch)
        pool = self.draft_model_runner.token_to_kv_pool
        reset_rows = [i for i, prefix in enumerate(batch.prefix_lens) if prefix == 0]
        if reset_rows:
            rows = torch.tensor(reset_rows, dtype=torch.int64, pin_memory=True).to(
                batch.req_pool_indices.device, non_blocking=True
            )
            pool.commit_stopped.index_fill_(
                0, batch.req_pool_indices.index_select(0, rows).long(), 0
            )
        if pool.gdn is not None:
            self.prefill_layouts[id(placement)] = pool.gdn.prepare_prefill(
                batch.reqs,
                batch.prefix_lens,
                batch.extend_lens,
                batch.req_pool_indices.contiguous(),
            )
        return placement

    def inject_prefill(
        self, placement, *, target_hidden, token_indices, target_hidden_is_projected
    ):
        layout = self.prefill_layouts.pop(id(placement), None)
        if layout is None:
            return super().inject_prefill(
                placement,
                target_hidden=target_hidden,
                token_indices=token_indices,
                target_hidden_is_projected=target_hidden_is_projected,
            )
        if token_indices is not None:
            raise ValueError(
                "GDN requires every confirmed target-hidden row, without filtering"
            )
        self._inject(
            target_hidden=target_hidden,
            cache_loc=placement.cache_loc,
            positions=placement.positions,
            state_slot=placement.state_slot,
            final_pos=placement.final_pos,
            target_hidden_is_projected=target_hidden_is_projected,
            gdn_layout=layout,
        )

    def _inject(
        self,
        *,
        target_hidden,
        cache_loc,
        positions,
        cache_loc_2d=None,
        commit_lens=None,
        state_slot=None,
        final_pos=None,
        target_hidden_is_projected=False,
        swa_loc=None,
        gdn_layout=None,
    ):
        if target_hidden is None or target_hidden.numel() == 0:
            return
        n = positions.numel()
        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                target_hidden=target_hidden[:n],
                pool=self.draft_model_runner.token_to_kv_pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
                target_hidden_is_projected=target_hidden_is_projected,
                gdn_layout=gdn_layout,
            )

    def commit_static_hidden(
        self, *, batch, verify_window, hidden, commit_lens, bs, candidates
    ):
        stops = [_request_stop_ids(req) for req in batch.reqs[:bs]]
        capacity = max(1, max(map(len, stops), default=0))
        key = (bs, capacity)
        frame = self.commit_frames.get(key)
        if frame is None:
            frame = _CommitFrame(
                hidden,
                bs,
                self.verify_num_draft_tokens,
                capacity,
                self.draft_model_runner.token_to_kv_pool.gdn,
            )
            self.commit_frames[key] = frame
        frame.stage_limits(
            batch.reqs[:bs], stops, self.draft_model_runner.token_to_kv_pool.gdn
        )
        frame.candidates.copy_(candidates)
        frame.hidden.copy_(hidden.view_as(frame.hidden))
        frame.positions.copy_(verify_window.positions_2d)
        loc = self.ids_for(verify_window.iteration_ids)
        frame.locs.copy_(loc.view_as(frame.locs))
        frame.slots.copy_(batch.req_pool_indices[:bs, None])
        frame.lens.copy_(commit_lens)
        if (
            get_exec().graph.cuda_graph_config.decode.backend == "disabled"
            or not frame.warm
        ):
            self._commit_frame(frame)
            frame.warm = True
            self.graph_eager_ct += 1
            return
        if frame.graph is None:
            # Capture must not advance recurrent state. The first real eager
            # commit warmed the projection and writer; record only gated writes.
            frame.lens.zero_()
            frame.graph = torch.cuda.CUDAGraph()
            with (
                graph_pool_capture_scope(),
                torch.cuda.graph(
                    frame.graph,
                    pool=get_or_create_global_graph_memory_pool(torch.cuda),
                    stream=get_or_create_global_graph_capture_stream(),
                ),
            ):
                self._commit_frame(frame)
            frame.lens.copy_(commit_lens)
        with graph_pool_replay_scope():
            frame.graph.replay()
        self.graph_replay_ct += 1

    def _commit_frame(self, frame):
        legospec_commit_lens(
            frame.candidates,
            frame.positions,
            frame.slots,
            frame.lens,
            frame.limits,
            frame.stop_ids,
            frame.effective_lens,
            self.draft_model_runner.token_to_kv_pool.commit_stopped,
        )
        if frame.gdn_layout is not None:
            frame.gdn_layout.slots.copy_(frame.slots[:, 0])
        self._inject(
            target_hidden=frame.hidden,
            positions=frame.positions.flatten(),
            cache_loc=frame.locs.flatten(),
            cache_loc_2d=frame.locs,
            commit_lens=frame.effective_lens,
            state_slot=frame.slots.flatten(),
            gdn_layout=frame.gdn_layout,
        )


class _CommitFrame:
    def __init__(self, hidden, bs, width, stop_capacity, gdn_pool=None):
        self.hidden = torch.empty_like(hidden.view(bs * width, -1))
        options = dict(device=hidden.device, dtype=torch.int64)
        self.positions = torch.empty((bs, width), **options)
        self.locs = torch.empty_like(self.positions)
        self.slots = torch.empty_like(self.positions)
        self.lens = torch.zeros(bs, device=hidden.device, dtype=torch.int32)
        self.effective_lens = torch.empty_like(self.lens)
        self.candidates = torch.empty_like(self.positions)
        self.limits = torch.empty(bs, **options)
        self.stop_ids = torch.empty((bs, stop_capacity), **options)
        self._metadata_key = None
        self.graph, self.warm = None, False
        self.gdn_layout = (
            None
            if gdn_pool is None
            else GDNCommitLayout(
                slots=torch.empty(bs, **options),
                cu_seqlens=torch.arange(bs + 1, device=hidden.device, dtype=torch.int32)
                * width,
                generations=torch.empty(bs, **options),
            )
        )

    def stage_limits(self, reqs, stops, gdn_pool=None):
        limits = [
            len(r.origin_input_ids) + r.sampling_params.max_new_tokens for r in reqs
        ]
        generations = [] if gdn_pool is None else gdn_pool.generation_rows(reqs)
        key = (
            tuple(generations),
            tuple(
                (r.rid, limit, tuple(stop))
                for r, limit, stop in zip(reqs, limits, stops)
            ),
        )
        if key == self._metadata_key:
            return
        cpu_stops = torch.full(
            self.stop_ids.shape, -1, dtype=torch.int64, pin_memory=True
        )
        for row, ids in enumerate(stops):
            if ids:
                cpu_stops[row, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
        self.stop_ids.copy_(cpu_stops, non_blocking=True)
        self.limits.copy_(
            torch.tensor(limits, dtype=torch.int64, pin_memory=True), non_blocking=True
        )
        if self.gdn_layout is not None:
            self.gdn_layout.generations.copy_(
                torch.tensor(generations, dtype=torch.int64, pin_memory=True),
                non_blocking=True,
            )
        self._metadata_key = key


def _request_stop_ids(req):
    if req.sampling_params.ignore_eos:
        return []
    ids = set(req.sampling_params.stop_token_ids or ()) | set(req.eos_token_ids or ())
    if req.tokenizer is not None:
        if req.tokenizer.eos_token_id is not None:
            ids.add(req.tokenizer.eos_token_id)
        ids.update(req.tokenizer.additional_stop_token_ids or ())
    return sorted(ids)
