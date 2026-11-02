"""SwiGLU retaining HF's activation rounding before multiplication."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _hf_silu_mul(X, OUT, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // D, offsets % D
    gate = tl.load(X + row * (2 * D) + col, offsets < N, 0).to(tl.float32)
    up = tl.load(X + row * (2 * D) + D + col, offsets < N, 0).to(tl.float32)
    # PyTorch SiLU uses x / (1 + exp(-x)); an approximate sigmoid followed
    # by multiplication crosses BF16 halfway values for some gate inputs.
    activated = (
        tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        .to(X.dtype.element_ty)
        .to(tl.float32)
    )
    tl.store(OUT + offsets, activated * up, offsets < N)


def hf_silu_and_mul(x):
    if not x.is_contiguous() or x.shape[-1] % 2:
        raise ValueError("Expected contiguous [...,2*intermediate] activations")
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out
    _hf_silu_mul[(triton.cdiv(out.numel(), 1024),)](
        x, out, out.numel(), out.shape[-1], 1024
    )
    return out
