"""Map confirmed absolute positions to bounded, request-local SWA rings."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ring_locs(
    SLOTS,
    POS,
    FINAL,
    LENS,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    HAS_LENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(POS + i, i < N, 0)
    slot = tl.load(SLOTS + i, i < N, -1)
    valid = (i < N) & (slot > 0) & (p >= 0)
    if HAS_FINAL:
        end = tl.load(FINAL + i, i < N, -1)
        valid &= (p > end - 512) & (p <= end)
    if HAS_LENS:
        length = tl.load(LENS + i // WIDTH, i < N, 0)
        valid &= i % WIDTH < length
    tl.store(OUT + i, tl.where(valid, 1 + slot * 520 + p % 512, -1), i < N)


def legospec_ring_locs(
    slots, positions, final_positions=None, commit_lens=None, width=8
):
    out = torch.empty_like(positions, dtype=torch.int64)
    if out.numel():
        _ring_locs[(triton.cdiv(out.numel(), 256),)](
            slots,
            positions,
            final_positions,
            commit_lens,
            out,
            out.numel(),
            width,
            final_positions is not None,
            commit_lens is not None,
            256,
        )
    return out
