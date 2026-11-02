"""DSA selects history; ordinary within-block visibility is retained."""

from .base import build_visibility


def dsa_visibility(inputs, positions, valid, documents, query_length, **kwargs):
    visibility = build_visibility(inputs, positions, valid, documents, query_length)
    selection = inputs.selection
    if selection is None:
        return visibility
    import torch
    n, a, _ = selection.indices.shape
    counts = torch.zeros(n, a, positions.size(1), dtype=torch.long, device=positions.device)
    counts.scatter_add_(-1, selection.indices, selection.selected_valid.long())
    keep = (counts > 0).repeat_interleave(query_length // a, 1)
    visibility[..., :positions.size(1)] &= keep[:, None]
    return visibility
