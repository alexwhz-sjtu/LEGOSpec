"""Frozen-protocol nucleus probabilities, using capture-safe GPU primitives."""

import torch


def stable_nucleus_probs(logits, top_ps, *, truncate=True):
    """FP32 logits [rows,vocab] -> normalized final p/q.

    Stable CUDA radix sort, native softmax and native cumsum deliberately keep
    the frozen harness's reduction order. A probability-threshold repair is not
    equivalent in FP32 at a 151936-token boundary. Every op is graph capturable;
    there is no host read or Python loop over vocabulary/request rows.
    """
    if logits.ndim != 2 or top_ps.numel() != logits.shape[0]:
        raise ValueError("Expected [rows,vocab] logits and [rows] top-p")
    if not truncate:
        return logits.float().softmax(-1)
    ordered, indices = logits.float().sort(dim=-1, descending=True, stable=True)
    values = ordered.softmax(-1)
    keep = values.cumsum(-1) - values < top_ps.reshape(-1, 1)
    values = values * keep
    values = values / values.sum(-1, keepdim=True)
    nucleus = torch.zeros_like(values).scatter_(-1, indices, values)
    # Mixed top-p batches retain a genuinely untruncated distribution on p=1
    # rows, rather than applying a rounded cumulative-probability boundary.
    return torch.where(top_ps.reshape(-1, 1) >= 1, logits.float().softmax(-1), nucleus)
