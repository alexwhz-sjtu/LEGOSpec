"""Fused DSpark gate, recurrent update and output activation for BF16 heads.

Generated for the LEGOSpec integration and checked against Torch 2.9/2.14.
The explicit BF16 conversions preserve the unfused head's rounding points.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _dspark_rnn_update(
    S,
    RAW,
    NEW,
    LATENT,
    N: tl.constexpr,
    R: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = x // R, x % R
    valid = x < N
    state = tl.load(S + x, valid, 0).to(tl.float32)
    g = tl.load(RAW + row * STRIDE + col, valid, 0).to(tl.float32)
    c = tl.load(RAW + row * STRIDE + R + col, valid, 0).to(tl.float32)
    o = tl.load(RAW + row * STRIDE + 2 * R + col, valid, 0).to(tl.float32)
    dtype = NEW.dtype.element_ty
    gate = (1.0 / (1.0 + libdevice.exp(-g))).to(dtype).to(tl.float32)
    # libdevice tanh flushes these FP32 subnormals. Torch keeps the BF16
    # input, which is also its correctly rounded tanh in this range.
    candidate = tl.where(tl.abs(c) < 1.1754943508222875e-38, c, libdevice.tanh(c))
    candidate = candidate.to(dtype).to(tl.float32)
    complement = (1.0 - gate).to(dtype).to(tl.float32)
    retained = (gate * state).to(dtype).to(tl.float32)
    update = (complement * candidate).to(dtype).to(tl.float32)
    tl.store(NEW + x, retained + update, valid)
    latent = tl.where(tl.abs(o) < 1.1754943508222875e-38, o, libdevice.tanh(o))
    tl.store(LATENT + x, latent, valid)


def dspark_rnn_update(state, raw):
    """Return BF16 ``(new_state, tanh(output_raw))`` for contiguous CUDA rows.

    ``state`` is [batch, rank], ``raw`` is the unchanged joint projection's
    [batch, 3*rank] gate/candidate/output result. Neither input is modified.
    """
    if (
        state.ndim != 2
        or raw.ndim != 2
        or state.shape[1] <= 0
        or raw.shape != (state.shape[0], state.shape[1] * 3)
        or state.dtype != torch.bfloat16
        or raw.dtype != torch.bfloat16
        or not state.is_cuda
        or state.device != raw.device
        or not state.is_contiguous()
        or not raw.is_contiguous()
    ):
        raise ValueError(
            "DSpark RNN update requires contiguous BF16 CUDA [B,R] and [B,3R]"
        )
    new, latent = torch.empty_like(state), torch.empty_like(state)
    if state.numel():
        _dspark_rnn_update[(triton.cdiv(state.numel(), 256),)](
            state,
            raw,
            new,
            latent,
            state.numel(),
            state.shape[1],
            raw.stride(0),
            256,
            enable_fp_fusion=False,
        )
    return new, latent
