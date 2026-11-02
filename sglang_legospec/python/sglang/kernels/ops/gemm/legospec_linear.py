"""Frozen BF16 cuBLAS projections independent of target GEMM dispatch."""

from typing import Optional

import torch

from sglang.srt.utils.custom_op import register_custom_op


def _fake(x, weight, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@register_custom_op(op_name="legospec_cublas_linear", mutates_args=[], fake_impl=_fake)
def legospec_cublas_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("LEGOSpec cuBLAS projections require BF16 inputs and weights")
    matrix = x.reshape(-1, x.shape[-1])
    result = x.new_empty((matrix.shape[0], weight.shape[0]))
    # Deterministic target execution replaces aten::mm/addmm globally. The out
    # overload retains the original cuBLAS kernel and its frozen rounding. An
    # opaque op prevents Inductor from decomposing this back into overridden mm.
    # beta=0 never reads the uninitialized result used as the dummy addend.
    torch.addmm(
        result if bias is None else bias,
        matrix,
        weight.T,
        beta=0 if bias is None else 1,
        out=result,
    )
    return result.reshape(*x.shape[:-1], weight.shape[0])
