"""Clip context commits to verified, in-budget tokens through the first EOS."""

import triton
import triton.language as tl


@triton.jit
def _commit_lens(
    IDS,
    POS,
    SLOTS,
    COUNTS,
    LIMITS,
    STOPS,
    OUT,
    TERMINATED,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    LATCH_STOP: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.arange(0, 8)
    stop = tl.arange(0, BLOCK)
    tokens = tl.load(IDS + b * 8 + c)
    stops = tl.load(STOPS + b * S + stop, stop < S, -1)
    matched = (
        tl.sum(
            ((tokens[:, None] == stops[None, :]) & (stops[None, :] >= 0)).to(tl.int32),
            1,
        )
        > 0
    )
    count = tl.load(COUNTS + b)
    first = tl.min(tl.where(matched & (c < count), c + 1, 8), 0)
    budget = tl.maximum(0, tl.load(LIMITS + b) - tl.load(POS + b * 8))
    slot = tl.load(SLOTS + b * 8)
    valid = slot > 0
    if LATCH_STOP:
        stopped = tl.load(TERMINATED + slot, valid, 0) != 0
        stop_reached = tl.sum((matched & (c < count)).to(tl.int32), 0) > 0
        terminal = stopped | ((count > 0) & (stop_reached | (count >= budget)))
        tl.store(TERMINATED + slot, terminal.to(tl.int32), valid)
        valid = valid & ~stopped
    tl.store(OUT + b, tl.where(valid, tl.minimum(count, tl.minimum(first, budget)), 0))


def legospec_commit_lens(
    candidates, positions, slots, counts, limits, stop_ids, out, terminated=None
):
    if counts.numel() == 0:
        return
    _commit_lens[(counts.numel(),)](
        candidates,
        positions,
        slots,
        counts,
        limits,
        stop_ids,
        out,
        terminated,
        stop_ids.shape[1],
        triton.next_power_of_2(stop_ids.shape[1]),
        terminated is not None,
    )
