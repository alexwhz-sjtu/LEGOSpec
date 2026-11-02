"""Residual add and RMSNorm with both HF BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hf_add_norm(
    X, R, W, Y, S, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + i, i < H, 0).to(tl.float32)
    r = tl.load(R + row * H + i, i < H, 0).to(tl.float32)
    s = (x + r).to(X.dtype.element_ty).to(tl.float32)
    inv = tl.rsqrt(tl.sum(s * s, 0) / H + EPS)
    normalized = (s * inv).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + i, i < H, 0).to(tl.float32)
    tl.store(Y + row * H + i, normalized * weight, i < H)
    tl.store(S + row * H + i, s, i < H)


def hf_add_rmsnorm(x, residual, weight, eps):
    if (
        x.shape != residual.shape
        or not x.is_contiguous()
        or not residual.is_contiguous()
    ):
        raise ValueError("HF residual norm requires equal contiguous activation shapes")
    out, summed = torch.empty_like(x), torch.empty_like(residual)
    if x.numel() == 0:
        return out, summed
    h = x.shape[-1]
    _hf_add_norm[(x.numel() // h,)](
        x, residual, weight, out, summed, h, eps, triton.next_power_of_2(h)
    )
    return out, summed
