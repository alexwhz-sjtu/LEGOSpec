"""Fused BF16 RMSNorm retaining PyTorch 2.9 CUDA Reduce.cuh addition order.

2.9 combines lanes with shfl-down offsets 1,2,4,8,16; newer Torch/Triton
use descending offsets. Reversing the five lane bits restores the former tree.
Inter-warp sums precede that tree, as in the original block_x_reduce.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reverse_lane(lane):
    return (
        ((lane & 1) << 4)
        | ((lane & 2) << 2)
        | (lane & 4)
        | ((lane & 8) >> 2)
        | ((lane & 16) >> 4)
    )


@triton.jit
def torch29_sum_squares_128(value):
    lane = _reverse_lane(tl.arange(0, 32))
    a = tl.gather(value, lane, 0)
    b = tl.gather(value, lane + 32, 0)
    c = tl.gather(value, lane + 64, 0)
    d = tl.gather(value, lane + 96, 0)
    return tl.sum(((a * a + b * b) + c * c) + d * d, 0)


@triton.jit
def _norm(
    X,
    R,
    W,
    Y,
    S,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    VECTORIZED: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    thread = tl.arange(0, WIDTH)
    logical = thread // 32 * 32 + _reverse_lane(thread % 32)
    a0 = tl.full((WIDTH,), 0, tl.float32)
    a1 = tl.full((WIDTH,), 0, tl.float32)
    a2 = tl.full((WIDTH,), 0, tl.float32)
    a3 = tl.full((WIDTH,), 0, tl.float32)
    for step in range(tl.cdiv(H, WIDTH * 4)):
        if VECTORIZED:
            p0 = (logical + step * WIDTH) * 4
            p1, p2, p3 = p0 + 1, p0 + 2, p0 + 3
        else:
            p0 = logical + step * WIDTH * 4
            p1, p2, p3 = p0 + WIDTH, p0 + 2 * WIDTH, p0 + 3 * WIDTH
        x0 = tl.load(X + row * H + p0, p0 < H, 0).to(tl.float32)
        x1 = tl.load(X + row * H + p1, p1 < H, 0).to(tl.float32)
        x2 = tl.load(X + row * H + p2, p2 < H, 0).to(tl.float32)
        x3 = tl.load(X + row * H + p3, p3 < H, 0).to(tl.float32)
        if HAS_RESIDUAL:
            x0 = (
                (x0 + tl.load(R + row * H + p0, p0 < H, 0).to(tl.float32))
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )
            x1 = (
                (x1 + tl.load(R + row * H + p1, p1 < H, 0).to(tl.float32))
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )
            x2 = (
                (x2 + tl.load(R + row * H + p2, p2 < H, 0).to(tl.float32))
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )
            x3 = (
                (x3 + tl.load(R + row * H + p3, p3 < H, 0).to(tl.float32))
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )
        a0, a1, a2, a3 = a0 + x0 * x0, a1 + x1 * x1, a2 + x2 * x2, a3 + x3 * x3
    local = ((a0 + a1) + a2) + a3
    across_warps = tl.sum(local.reshape((WIDTH // 32, 32)), 0)
    total = tl.sum(across_warps, 0)
    inv = tl.rsqrt(total * (1.0 / H) + EPS)
    i = tl.arange(0, OUT_BLOCK)
    x = tl.load(X + row * H + i, i < H, 0).to(tl.float32)
    if HAS_RESIDUAL:
        x = (
            (x + tl.load(R + row * H + i, i < H, 0).to(tl.float32))
            .to(X.dtype.element_ty)
            .to(tl.float32)
        )
        tl.store(S + row * H + i, x, i < H)
    normed = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + i, i < H, 0).to(tl.float32)
    tl.store(Y + row * H + i, normed * weight, i < H)


def torch29_rmsnorm(x, weight, eps, residual=None):
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("LEGOSpec reference RMSNorm requires BF16 activations/weights")
    x = x.contiguous()
    if residual is not None:
        residual = residual.contiguous()
    out = torch.empty_like(x)
    summed = torch.empty_like(x) if residual is not None else None
    h = x.shape[-1]
    n = x.numel() // h
    if n:
        if h not in (128, 2560):
            raise ValueError(f"Unaudited LEGOSpec RMSNorm width {h}")
        height = min(16, 1 << (n.bit_length() - 1))
        width = min(512 // height, 512 if h > 128 else 128)
        _norm[(n,)](
            x,
            residual,
            weight,
            out,
            summed,
            h,
            width,
            eps,
            residual is not None,
            h > 128,
            triton.next_power_of_2(h),
            num_warps=width // 32,
            enable_fp_fusion=False,
        )
    return out if residual is None else (out, summed)
