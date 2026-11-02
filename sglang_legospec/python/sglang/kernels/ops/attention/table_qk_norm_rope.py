import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.torch29_rmsnorm import torch29_sum_squares_128


@triton.jit
def _table_qk_norm_rope_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    pos_ptr,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    HF_SEMANTICS: tl.constexpr,
    TORCH29: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)

    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * D + HALF + half_ar).to(tl.float32)
    if HF_SEMANTICS:
        cos = cos.to(tl.bfloat16).to(tl.float32)
        sin = sin.to(tl.bfloat16).to(tl.float32)

    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    w_ptr = tl.where(is_q, q_weight_ptr.to(tl.int64), k_weight_ptr.to(tl.int64)).to(
        tl.pointer_type(tl.bfloat16)
    )

    row = qkv_ptr + t * row_stride + col0
    x = tl.load(row + d_ar).to(tl.float32)
    ms = tl.sum(x * x, 0) / D
    if TORCH29 and D == 128:
        ms = torch29_sum_squares_128(x) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    if HF_SEMANTICS:
        inv = tl.rsqrt(ms + EPS)
    w1 = tl.load(w_ptr + half_ar).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
    x1 = tl.load(row + half_ar).to(tl.float32) * inv
    x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv
    if HF_SEMANTICS:
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
    x1 *= w1
    x2 *= w2
    x1 = x1.to(tl.bfloat16).to(tl.float32)
    x2 = x2.to(tl.bfloat16).to(tl.float32)
    if HF_SEMANTICS:
        o1 = (x1 * cos).to(tl.bfloat16).to(tl.float32) - (x2 * sin).to(tl.bfloat16).to(
            tl.float32
        )
        o2 = (x2 * cos).to(tl.bfloat16).to(tl.float32) + (x1 * sin).to(tl.bfloat16).to(
            tl.float32
        )
    else:
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
    tl.store(row + half_ar, o1.to(tl.bfloat16))
    tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))


def table_qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    eps: float,
    hf_semantics: bool = False,
    torch29_semantics: bool = False,
) -> None:
    """In-place QK RMSNorm + table-lookup neox RoPE on the fused QKV tensor.

    Reads cos/sin from the SAME rotary table as the unfused path, so there is
    no large-position angle drift (unlike theta-recompute kernels). V columns
    are untouched.
    """
    T = qkv.shape[0]
    if T == 0:
        return
    grid = (T, num_q_heads + num_k_heads)
    _table_qk_norm_rope_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        qkv.stride(0),
        num_q_heads * head_dim,
        NHQ=num_q_heads,
        D=head_dim,
        EPS=eps,
        HF_SEMANTICS=hf_semantics,
        TORCH29=torch29_semantics,
        enable_fp_fusion=not hf_semantics,
    )
