"""FlashInfer paged attention for block-anchored SWA and mixed GDN drafts."""

import torch
from sglang.kernels.ops.attention.legospec_metadata import fill_legospec_block_indices
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.runtime_context import get_buffer


class _BlockFrame:
    def __init__(self, bs, context_length, pool, config, workspace, window_backend):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self.full_tables = torch.zeros(
            (bs, context_length + 8), dtype=torch.int32, device="cuda"
        )
        self.full_indices = self.full_tables.flatten()
        self.full_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        self.window_tables = (
            torch.zeros((bs, 520), dtype=torch.int32, device="cuda")
            if pool.swa_layers
            else None
        )
        self.window_indices = self.window_tables.flatten() if pool.swa_layers else None
        self.window_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        self.q_indptr = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * 8
        self.last_page = torch.ones(bs, dtype=torch.int32, device="cuda")
        self.scratch = torch.zeros(bs * 8, dtype=torch.int64, device="cuda")
        self.q_cpu = (torch.arange(bs + 1, dtype=torch.int32) * 8).pin_memory()
        self.last_cpu = torch.ones(bs, dtype=torch.int32, pin_memory=True)

        def wrapper(indices, ptr, backend="cudnn"):
            return BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend=backend,
                use_cuda_graph=True,
                qo_indptr_buf=self.q_indptr,
                paged_kv_indptr_buf=ptr,
                paged_kv_indices_buf=indices,
                paged_kv_last_page_len_buf=self.last_page,
            )

        self.full = wrapper(self.full_indices, self.full_indptr)
        self.window = (
            wrapper(self.window_indices, self.window_indptr, window_backend)
            if pool.swa_layers
            else None
        )


class LEGOSpecAttnBackend(AttentionBackend):
    window_backend = "cudnn"
    physical_window_order = False

    def __init__(self, runner):
        super().__init__()
        self.token_to_kv_pool = runner.token_to_kv_pool
        self.req_to_token_pool = runner.req_to_token_pool
        self.runner_ids = runner.runner_ids
        self.config = runner.model_config.hf_config
        self.max_context = runner.model_config.context_len
        self.frames = {}
        self.active = None
        self.workspace = get_buffer(
            "legospec_attention_workspace",
            lambda: torch.empty(128 << 20, dtype=torch.uint8, device=runner.device),
        )

    def init_cuda_graph_state(self, max_bs, max_num_tokens, **kwargs):
        self.frames.clear()

    def get_cuda_graph_seq_len_fill_value(self):
        return 8

    def shared_read_ends(self, forward_mode):
        return SharedReadEnds.IN_REPLAY

    def init_forward_metadata_out_graph(self, batch, in_capture=False):
        bs = batch.batch_size
        if bs == 0:
            return
        frame = self.frames.get(bs)
        if frame is None:
            frame = _BlockFrame(
                bs,
                self.max_context,
                self.token_to_kv_pool,
                self.config,
                self.workspace,
                self.window_backend,
            )
            self.frames[bs] = frame
        # The proposer stages CPU lengths including the current eight queries.
        # This reads existing scheduler CPU metadata, never a GPU tensor.
        lengths = (
            [0] * bs
            if in_capture
            else [max(0, int(n) - 8) for n in batch.seq_lens_cpu[:bs].tolist()]
        )
        for wrapper, window in ((frame.full, False), (frame.window, True)):
            if wrapper is None:
                continue
            lens = [min(n, 512) + 8 if window else n + 8 for n in lengths]
            ptr = torch.tensor([0] + lens, dtype=torch.int32, pin_memory=True).cumsum(
                0, dtype=torch.int32
            )
            wrapper.plan(
                frame.q_cpu,
                ptr,
                frame.window_indices if window else frame.full_indices,
                frame.last_cpu,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                1,
                causal=False,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                non_blocking=True,
                block_tables=frame.window_tables if window else frame.full_tables,
                max_token_per_sequence=8,
                max_sequence_kv=520 if window else self.max_context + 8,
                seq_lens=torch.tensor(lens, dtype=torch.int32, pin_memory=True),
            )
        self.active = frame

    def init_forward_metadata_in_graph(self, batch):
        if batch.batch_size == 0:
            return
        frame = self.active
        fill_legospec_block_indices(
            batch.req_pool_indices,
            batch.seq_lens,
            self.req_to_token_pool.req_to_token,
            batch.out_cache_loc,
            frame.full_indptr,
            frame.window_indptr,
            frame.full_indices,
            frame.window_indices,
            frame.scratch,
            self.max_context,
            full_tables=frame.full_tables,
            window_tables=frame.window_tables,
            physical_window_order=self.physical_window_order,
        )

    def forward_extend(self, q, k, v, layer, batch, save_kv_cache=True):
        pool, frame = self.token_to_kv_pool, self.active
        window = layer.layer_id in pool._swa_indices
        if k is not None and save_kv_cache:
            pool.layer_router.layer(layer.layer_id).store(
                layer,
                KVWriteLoc(batch.out_cache_loc, frame.scratch, physical=True),
                k,
                v,
            )
        wrapper = frame.window if window else frame.full
        output = wrapper.forward(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            pool.layer_router.read(layer.layer_id).kv(),
            causal=False,
            window_left=-1,
            sm_scale=layer.scaling,
        )
        return output.flatten(1)
