"""Paged noncausal block indices; SWA history is anchored once per block."""

import triton
import triton.language as tl


@triton.jit
def _block_indices(
    REQS,
    LENS,
    TABLE,
    CURRENT,
    FULL_PTR,
    WINDOW_PTR,
    FULL_IDS,
    WINDOW_IDS,
    SCRATCH,
    FULL_TABLE,
    WINDOW_TABLE,
    FULL_WIDTH: tl.constexpr,
    TABLES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    PHYSICAL_WINDOW_ORDER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b, part = tl.program_id(0), tl.program_id(1)
    slot = tl.load(REQS + b).to(tl.int64)
    length = tl.load(LENS + b)
    pos = part * BLOCK + tl.arange(0, BLOCK)
    valid = pos < length + 8
    history = tl.load(TABLE + slot * TABLE_STRIDE + pos, (pos < length) & valid, 0)
    query = tl.load(CURRENT + b * 8 + pos - length, (pos >= length) & valid, 0)
    ids = tl.where(pos < length, history, query)
    offset = tl.load(FULL_PTR + b)
    if TABLES:
        tl.store(FULL_TABLE + b * FULL_WIDTH + pos, ids.to(tl.int32), valid)
    else:
        tl.store(FULL_IDS + offset + pos, ids.to(tl.int32), valid)
    if HAS_WINDOW:
        if part == 0:
            p = tl.arange(0, 1024)
            n = tl.minimum(length, 512)
            history_id = 1 + slot * 520 + (length - n + p) % 512
            if PHYSICAL_WINDOW_ORDER:
                history_id = 1 + slot * 520 + p
            scratch_id = 1 + slot * 520 + 512 + p - n
            win_id = tl.where(p < n, history_id, scratch_id)
            off = tl.load(WINDOW_PTR + b)
            if TABLES:
                tl.store(WINDOW_TABLE + b * 520 + p, win_id.to(tl.int32), p < n + 8)
            else:
                tl.store(WINDOW_IDS + off + p, win_id.to(tl.int32), p < n + 8)
            q = tl.arange(0, 8)
            tl.store(SCRATCH + b * 8 + q, 1 + slot * 520 + 512 + q)


def fill_legospec_block_indices(
    reqs,
    lens,
    table,
    current,
    full_indptr,
    window_indptr,
    full_indices,
    window_indices,
    scratch,
    max_context,
    full_tables=None,
    window_tables=None,
    physical_window_order=False,
):
    _block_indices[(reqs.numel(), triton.cdiv(max_context + 8, 1024))](
        reqs,
        lens,
        table,
        current,
        full_indptr,
        window_indptr,
        full_indices,
        window_indices,
        scratch,
        full_tables,
        window_tables,
        max_context + 8,
        full_tables is not None,
        table.stride(0),
        window_indices is not None,
        physical_window_order,
        1024,
    )
