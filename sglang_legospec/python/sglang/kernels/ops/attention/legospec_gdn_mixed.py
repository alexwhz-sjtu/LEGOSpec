"""Bounded Qwen3.5/FLA mixed-precision events and read-only block readout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    K,
    V,
    G,
    B,
    SLOTS,
    CU,
    ECU,
    CARRY,
    LENS,
    CK,
    CV,
    CG,
    CB,
    OK,
    OV,
    OG,
    OB,
    N: tl.constexpr,
    BN: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    F: tl.constexpr,
):
    event, feature = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, BN)
    ends = tl.load(ECU + rows + 1, rows < N, 2147483647)
    row = tl.sum((event >= ends).to(tl.int32), 0)
    start = tl.load(CU + row)
    length = tl.load(CU + row + 1) - start
    if HAS_LENGTHS:
        length = tl.load(LENS + row)
    slot = tl.load(SLOTS + row).to(tl.int64)
    carry = tl.load(CARRY + row)
    time = event - tl.load(ECU + row)
    source = time - carry
    old = (time < carry) & (slot > 0)
    new = (source >= 0) & (source < length) & (slot > 0)
    col = feature * F + tl.arange(0, F)
    key = tl.load(K + (start + source) * 2048 + col, new & (col < 2048), 0)
    key_old = tl.load(CK + (slot * 64 + time) * 2048 + col, old & (col < 2048), 0)
    tl.store(OK + event * 2048 + col, tl.where(old, key_old, key), col < 2048)
    vc = col - 2048
    value = tl.load(V + (start + source) * 4096 + vc, new & (vc >= 0) & (vc < 4096), 0)
    value_old = tl.load(
        CV + (slot * 64 + time) * 4096 + vc, old & (vc >= 0) & (vc < 4096), 0
    )
    tl.store(
        OV + event * 4096 + vc, tl.where(old, value_old, value), (vc >= 0) & (vc < 4096)
    )
    if feature == 0:
        heads = tl.arange(0, 32)
        gate = tl.load(G + (start + source) * 32 + heads, new, 0)
        beta = tl.load(B + (start + source) * 32 + heads, new, 0)
        gate_old = tl.load(CG + (slot * 64 + time) * 32 + heads, old, 0)
        beta_old = tl.load(CB + (slot * 64 + time) * 32 + heads, old, 0)
        tl.store(OG + event * 32 + heads, tl.where(old, gate_old, gate))
        tl.store(OB + event * 32 + heads, tl.where(old, beta_old, beta))


def pack_gdn_events(
    k,
    v,
    g,
    beta,
    slots,
    cu,
    event_cu,
    carry,
    cache_k,
    cache_v,
    cache_g,
    cache_beta,
    *,
    total_events,
    lengths=None,
):
    out_k = torch.empty((1, total_events, 16, 128), device=k.device, dtype=k.dtype)
    out_v = torch.empty((1, total_events, 32, 128), device=v.device, dtype=v.dtype)
    out_g = torch.empty((1, total_events, 32), device=g.device, dtype=torch.float32)
    out_b = torch.empty((1, total_events, 32), device=beta.device, dtype=beta.dtype)
    if total_events:
        _pack[(total_events, 24)](
            k,
            v,
            g,
            beta,
            slots,
            cu,
            event_cu,
            carry,
            lengths,
            cache_k,
            cache_v,
            cache_g,
            cache_beta,
            out_k,
            out_v,
            out_g,
            out_b,
            slots.numel(),
            triton.next_power_of_2(slots.numel()),
            lengths is not None,
            256,
        )
    return out_k, out_v, out_g, out_b


@triton.jit
def _cache(
    K,
    V,
    G,
    B,
    GC,
    SLOTS,
    CU,
    ECU,
    CARRY,
    LENS,
    CK,
    CV,
    CG,
    CB,
    CGC,
    COUNTS,
    HAS_LENGTHS: tl.constexpr,
):
    row, tile, feature = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(SLOTS + row).to(tl.int64)
    length = tl.load(CU + row + 1) - tl.load(CU + row)
    if HAS_LENGTHS:
        length = tl.load(LENS + row)
    if slot <= 0 or length <= 0:
        return
    context = length + tl.load(CARRY + row)
    count = context % 64
    start = tl.load(ECU + row) + (context // 64) * 64
    time = tile * 16 + tl.arange(0, 16)
    col = feature * 256 + tl.arange(0, 256)
    key = tl.load(
        K + (start + time[:, None]) * 2048 + col[None, :],
        (time[:, None] < count) & (col[None, :] < 2048),
        0,
    )
    tl.store(
        CK + (slot * 64 + time[:, None]) * 2048 + col[None, :],
        key,
        (time[:, None] < count) & (col[None, :] < 2048),
    )
    vc = col - 2048
    value = tl.load(
        V + (start + time[:, None]) * 4096 + vc[None, :],
        (time[:, None] < count) & (vc[None, :] >= 0) & (vc[None, :] < 4096),
        0,
    )
    tl.store(
        CV + (slot * 64 + time[:, None]) * 4096 + vc[None, :],
        value,
        (time[:, None] < count) & (vc[None, :] >= 0) & (vc[None, :] < 4096),
    )
    if feature == 0:
        h = tl.arange(0, 32)
        real = time[:, None] < count
        g = tl.load(G + (start + time[:, None]) * 32 + h[None, :], real, 0)
        beta = tl.load(B + (start + time[:, None]) * 32 + h[None, :], real, 0)
        tl.store(CG + (slot * 64 + time[:, None]) * 32 + h[None, :], g, real)
        tl.store(CB + (slot * 64 + time[:, None]) * 32 + h[None, :], beta, real)
        read = time[:, None] < tl.minimum(64, count + 8)
        gc = tl.load(GC + (start + time[:, None]) * 32 + h[None, :], read, 0)
        tl.store(CGC + (slot * 64 + time[:, None]) * 32 + h[None, :], gc, read)
        if tile == 0:
            tl.store(COUNTS + slot, count)


def cache_gdn_chunk(
    k,
    v,
    g,
    beta,
    gc,
    slots,
    cu,
    event_cu,
    carry,
    cache_k,
    cache_v,
    cache_g,
    cache_beta,
    cache_gc,
    counts,
    *,
    lengths=None,
):
    _cache[(slots.numel(), 4, 24)](
        k,
        v,
        g,
        beta,
        gc,
        slots,
        cu,
        event_cu,
        carry,
        lengths,
        cache_k,
        cache_v,
        cache_g,
        cache_beta,
        cache_gc,
        counts,
        lengths is not None,
        num_warps=4,
    )


@triton.jit
def _rewind(STATE, ENTRY, SLOTS, CARRY, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + row).to(tl.int64)
    if slot <= 0 or tl.load(CARRY + row) == 0:
        return
    col = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ENTRY + slot * 524288 + col, col < 524288, 0)
    tl.store(STATE + slot * 524288 + col, x, col < 524288)


def rewind_gdn_partial_chunk(state, entry, slots, carry):
    _rewind[(slots.numel(), 128)](state, entry, slots, carry, 4096, num_warps=4)


@triton.jit
def _read(Q, STATE, ENTRY, K, V, GC, COUNTS, SLOTS, OUT, BV: tl.constexpr):
    b, h, part_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    part, tile = part_tile % 2, part_tile // 2
    slot = tl.load(SLOTS + b).to(tl.int64)
    count = tl.load(COUNTS + slot, slot > 0, 0)
    if part == 1 and count <= 56:
        return
    time = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    v = tile * BV + tl.arange(0, BV)
    qi = time + part * 64 - count
    active = (qi >= 0) & (qi < 8) & (slot > 0)
    bo = tl.zeros((64, BV), tl.float32)
    A = tl.zeros((64, 64), tl.float32)
    for kk in range(2):
        k = kk * 64 + tl.arange(0, 64)
        q = tl.load(
            Q + ((b * 8 + qi[:, None]) * 16 + h // 2) * 128 + k[None, :],
            active[:, None],
            0,
        )
        if part == 0:
            hs = tl.load(
                ENTRY + ((slot * 32 + h) * 128 + v[None, :]) * 128 + k[:, None],
                (slot > 0) & (v[None, :] < 128),
                0,
            ).to(tl.bfloat16)
        else:
            hs = tl.load(
                STATE + ((slot * 32 + h) * 128 + v[None, :]) * 128 + k[:, None],
                (slot > 0) & (v[None, :] < 128),
                0,
            ).to(tl.bfloat16)
        bo += tl.dot(q, hs)
        if part == 0:
            key = tl.load(
                K + ((slot * 64 + cols[None, :]) * 16 + h // 2) * 128 + k[:, None],
                (slot > 0) & (cols[None, :] < count),
                0,
            )
            A += tl.dot(q, key)
    if part == 0:
        gq = tl.load(GC + (slot * 64 + time) * 32 + h, active, 0)
        gk = tl.load(GC + (slot * 64 + cols) * 32 + h, (slot > 0) & (cols < count), 0)
        bo = bo * tl.exp2(gq)[:, None]
        A = tl.where(
            active[:, None] & (cols[None, :] < count),
            A * tl.exp2(gq[:, None] - gk[None, :]),
            0,
        )
        value = tl.load(
            V + ((slot * 64 + cols[:, None]) * 32 + h) * 128 + v[None, :],
            (slot > 0) & (cols[:, None] < count) & (v[None, :] < 128),
            0,
        )
        bo = bo * (128**-0.5) + tl.dot(A.to(tl.bfloat16), value) * (128**-0.5)
    else:
        bo = bo * (128**-0.5)
    tl.store(
        OUT + ((b * 8 + qi[:, None]) * 32 + h) * 128 + v[None, :],
        bo,
        ((qi >= 0) & (qi < 8))[:, None] & (v[None, :] < 128),
    )


def mixed_gdn_read(q, state, entry, key, values, gc, counts, slots):
    if (
        state.dtype != torch.float32
        or entry.dtype != torch.float32
        or q.dtype != torch.bfloat16
        or state.shape[1:] != (32, 128, 128)
        or entry.shape != state.shape
        or q.shape != (slots.numel(), 8, 16, 128)
    ):
        raise ValueError(
            "Mixed GDN requires FP32 persistent states and normalized BF16 queries"
        )
    out = torch.empty((slots.numel(), 8, 32, 128), device=q.device, dtype=q.dtype)
    _read[(slots.numel(), 32, 8)](
        q,
        state,
        entry,
        key,
        values,
        gc,
        counts,
        slots,
        out,
        32,
        num_warps=4,
        num_stages=3,
    )
    return out
