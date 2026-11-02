"""Read-only block queries over FP32 GDN state, and exact BF16 gated norm."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.ops.layernorm.torch29_rmsnorm import torch29_sum_squares_128


@triton.jit
def _torch29_strided_read_sum(products, BV: tl.constexpr):
    # The frozen [head,K,V].sum(K) maps four warps to K and uses four
    # independent accumulators in each thread. Retain that addition order while
    # reading the production [head,V,K] layout; no state transpose is needed.
    local = tl.full((BV, 16), 0.0, tl.float32)
    for step in range(8):
        indices = tl.broadcast_to((step * 16 + tl.arange(0, 16))[None, :], (BV, 16))
        local += tl.gather(products, indices, 1)
    local = local.reshape((BV, 4, 4))
    a0 = tl.gather(local, tl.full((BV, 1, 4), 0, tl.int32), 1).reshape((BV, 4))
    a1 = tl.gather(local, tl.full((BV, 1, 4), 1, tl.int32), 1).reshape((BV, 4))
    a2 = tl.gather(local, tl.full((BV, 1, 4), 2, tl.int32), 1).reshape((BV, 4))
    a3 = tl.gather(local, tl.full((BV, 1, 4), 3, tl.int32), 1).reshape((BV, 4))
    return tl.sum(((a0 + a1) + a2) + a3, 1)


@triton.jit
def _read(
    Q,
    STATE,
    SLOTS,
    OUT,
    STRIDE: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    BV: tl.constexpr,
    ROUND_NORMALIZED_Q: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(SLOTS + b).to(tl.int64)
    k = tl.arange(0, 128)
    v = tile * BV + tl.arange(0, BV)
    # Load each FP32 state tile once and reuse it for all eight queries.
    state = tl.load(
        STATE + slot * STRIDE + h * 128 * 128 + v[:, None] * 128 + k[None, :],
        (slot > 0) & (v[:, None] < 128),
        0,
    ).to(tl.float32)
    for position in tl.static_range(8):
        q = tl.load(Q + ((b * 8 + position) * HK + h // (HV // HK)) * 128 + k).to(
            tl.float32
        )
        q = q * tl.rsqrt(torch29_sum_squares_128(q) + 1e-6)
        if ROUND_NORMALIZED_Q:
            q = q.to(tl.bfloat16).to(tl.float32)
        q = q * (128**-0.5)
        result = _torch29_strided_read_sum(state * q[None, :], BV)
        tl.store(OUT + ((b * 8 + position) * HV + h) * 128 + v, result, v < 128)


def legospec_gdn_read(q, state, slots, *, output_dtype=None, round_normalized_q=False):
    if state.dtype != torch.float32 or state.shape[1:] != (32, 128, 128):
        raise ValueError("GDN state must be FP32 [request,32,V=128,K=128]")
    q = q.contiguous().view(-1, 8, 16, 128)
    out = torch.empty(
        q.shape[0],
        8,
        32,
        128,
        device=q.device,
        dtype=q.dtype if output_dtype is None else output_dtype,
    )
    _read[(q.shape[0], 32, 8)](
        q,
        state,
        slots,
        out,
        state.stride(0),
        16,
        32,
        16,
        round_normalized_q,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _gated_norm(X, Z, W, OUT, EPS: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 128)
    x = tl.load(X + row * 128 + i).to(tl.float32)
    z = tl.load(Z + row * 128 + i).to(tl.float32)
    w = tl.load(W + i).to(tl.float32)
    inv = tl.rsqrt(torch29_sum_squares_128(x) * (1.0 / 128) + EPS)
    normed = (x * inv).to(tl.bfloat16).to(tl.float32)
    weighted = (normed * w).to(tl.bfloat16).to(tl.float32)
    activated = tl.div_rn(z, 1.0 + libdevice.exp(-z))
    tl.store(OUT + row * 128 + i, weighted * activated)


def legospec_gdn_gated_norm(x, z, weight, eps):
    out = torch.empty_like(x)
    _gated_norm[(x.numel() // 128,)](
        x, z, weight, out, eps, num_warps=4, enable_fp_fusion=False
    )
    return out


@triton.jit
def _prepare(
    SLOTS,
    CU,
    POSITIONS,
    CURSORS,
    GENERATIONS,
    EXPECTED,
    LENGTHS,
    INDICES,
    VALID,
    HAS_LENGTHS: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(SLOTS + b).to(tl.int64)
    start, end = tl.load(CU + b), tl.load(CU + b + 1)
    n = tl.load(LENGTHS + b) if HAS_LENGTHS else end - start
    position = tl.load(POSITIONS + start, n > 0, 0)
    cursor = tl.load(CURSORS + slot, slot > 0, 0)
    generation = tl.load(GENERATIONS + slot, slot > 0, 0)
    expected = tl.load(EXPECTED + b)
    valid = (n == 0) | ((slot > 0) & (cursor == position) & (generation == expected))
    if not valid:
        column = tl.arange(0, 8)
        values = tl.where(column == 0, b, slot)
        values = tl.where(column == 2, n, values)
        values = tl.where(column == 3, position, values)
        values = tl.where(column == 4, cursor, values)
        values = tl.where(column == 5, generation, values)
        values = tl.where(column == 6, expected, values)
        tl.device_print(
            "LEGOSpec GDN invalid commit [row,slot,n,position,cursor,generation,expected,slot]",
            values,
        )
    tl.store(VALID + b, valid)
    tl.store(INDICES + b, tl.where((slot > 0) & valid & (n > 0), slot, -1))


def legospec_gdn_prepare(
    slots, cu_seqlens, positions, cursors, generations, expected, lengths=None
):
    indices = torch.empty_like(slots, dtype=torch.int32)
    valid = torch.empty_like(slots, dtype=torch.bool)
    _prepare[(slots.numel(),)](
        slots,
        cu_seqlens,
        positions,
        cursors,
        generations,
        expected,
        lengths,
        indices,
        valid,
        lengths is not None,
    )
    torch._assert_async(valid.all(), "GDN context commit cursor/generation mismatch")
    return indices


@triton.jit
def _advance(SLOTS, CU, LENGTHS, CURSORS, HAS_LENGTHS: tl.constexpr):
    b = tl.program_id(0)
    slot = tl.load(SLOTS + b).to(tl.int64)
    n = tl.load(LENGTHS + b) if HAS_LENGTHS else tl.load(CU + b + 1) - tl.load(CU + b)
    if slot > 0 and n > 0:
        tl.store(CURSORS + slot, tl.load(CURSORS + slot) + n)


def legospec_gdn_advance(slots, cu_seqlens, cursors, lengths=None):
    _advance[(slots.numel(),)](slots, cu_seqlens, lengths, cursors, lengths is not None)
