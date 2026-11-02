"""Integer event topology; all trainable tensors stay inside model forward.

Write events follow real context order. Read events precede the write at the
same absolute position and have an identity state transition. This permits a
single differentiable chunk scan without replicating any anchor's prefix.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNEventLayout:
    context_indices: torch.Tensor
    query_indices: torch.Tensor
    write_events: torch.Tensor
    read_events: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    context_lengths: tuple[int, ...]
    samples: tuple[int, ...]
    num_events: int
    dense_shape: tuple[int, int] | None = None
    context_keep: torch.Tensor | None = None
    query_keep: torch.Tensor | None = None


def build_dense_event_layout(inputs, block_size):
    """Single-document batches: fixed-capacity device sorting, no D2H reads.

    Inactive context/slots become identity transitions after all useful events.
    Shape-only CPU boundaries serve the FP32 oracle; CUDA uses a dense scan and
    does not pass host boundaries to FLA.
    """
    device = inputs.draft_embeddings.device
    if (inputs.draft_state is not None or inputs.document_ids is not None
            or inputs.anchor_document_ids is not None or inputs.context_position_ids is not None):
        raise ValueError("Dense events require uncached single-document inputs")
    n, a = inputs.anchor_positions.shape
    s = inputs.target_hidden_states.size(1)
    if inputs.draft_embeddings.shape[:2] != (n, a * block_size):
        raise ValueError("draft_embeddings must have shape [N,A*block_size,D]")
    anchors = inputs.anchor_positions.detach().to(device)
    keep = inputs.block_keep_mask.detach().to(device).bool()
    if anchors.is_floating_point():
        raise ValueError("context and anchor positions must be integers")
    if keep.shape != (n, a):
        raise ValueError("anchor positions and validity must have shape [N,A]")
    nonnegative = ((anchors >= 0) | ~keep).all()
    if device.type == "cuda":
        torch._assert_async(nonnegative, "anchor positions must be nonnegative")
    elif not nonnegative:
        raise ValueError("anchor positions must be nonnegative")
    valid = (torch.ones(n, s, device=device, dtype=torch.bool)
             if inputs.context_valid_mask is None
             else inputs.context_valid_mask.detach().to(device).bool())
    if valid.shape != (n, s):
        raise ValueError("context positions, validity and documents must be [N,S]")
    last = (anchors.masked_fill(~keep, -1).amax(-1, keepdim=True)
            if a else anchors.new_full((n, 1), -1))
    positions = torch.arange(s, device=device).expand(n, -1)
    active = valid & (positions < last)
    packed = (~active).to(torch.int32).argsort(dim=-1, stable=True)
    packed_keep = active.gather(1, packed)
    sentinel = torch.iinfo(torch.long).max
    write_keys = (2 * packed + 1).masked_fill(~packed_keep, sentinel)
    read_keys = (2 * anchors).repeat_interleave(block_size, -1)
    read_keys = read_keys.masked_fill(~keep.repeat_interleave(block_size, -1), sentinel)
    width = s + a * block_size
    order = torch.cat((write_keys, read_keys), -1).argsort(dim=-1, stable=True)
    inverse = torch.empty_like(order)
    inverse.scatter_(1, order, torch.arange(width, device=device).expand(n, -1))
    offsets = torch.arange(n, device=device)[:, None] * width
    contexts = (torch.arange(n, device=device)[:, None] * s + packed).flatten()
    return GDNEventLayout(
        contexts, torch.arange(n * a * block_size, device=device),
        (offsets + inverse[:, :s]).flatten(),
        (offsets + inverse[:, s:]).flatten(),
        torch.arange(n + 1, device=device) * width,
        torch.arange(n + 1) * width,
        (s,) * n, tuple(range(n)), n * width,
        (n, width), packed_keep.flatten(), keep.repeat_interleave(block_size, -1).flatten(),
    )


def build_event_layout(inputs, block_size):
    """Keep event indices on the input device; cached/packed documents stay isolated."""
    device = inputs.draft_embeddings.device
    if (device.type == "cuda" and inputs.draft_state is None
            and inputs.document_ids is None and inputs.anchor_document_ids is None
            and inputs.context_position_ids is None):
        return build_dense_event_layout(inputs, block_size)
    n, a = inputs.anchor_positions.shape
    if inputs.draft_embeddings.shape[:2] != (n, a * block_size):
        raise ValueError("draft_embeddings must have shape [N,A*block_size,D]")
    s = inputs.target_hidden_states.size(1)
    positions = (
        torch.arange(s, device=device).expand(n, -1)
        if inputs.context_position_ids is None
        else inputs.context_position_ids.detach().to(device)
    )
    valid = (
        torch.ones(n, s, dtype=torch.bool, device=device)
        if inputs.context_valid_mask is None
        else inputs.context_valid_mask.detach().to(device).bool()
    )
    docs = (
        torch.zeros(n, s, dtype=torch.long, device=device)
        if inputs.document_ids is None else inputs.document_ids.detach().to(device)
    )
    anchor_docs = (
        torch.zeros(n, a, dtype=torch.long, device=device)
        if inputs.anchor_document_ids is None else inputs.anchor_document_ids.detach().to(device)
    )
    anchors = inputs.anchor_positions.detach().to(device)
    keep = inputs.block_keep_mask.detach().to(device).bool()
    if any(x.shape != (n, s) for x in (positions, valid, docs)):
        raise ValueError("context positions, validity and documents must be [N,S]")
    if anchor_docs.shape != (n, a) or keep.shape != (n, a):
        raise ValueError("anchor documents and validity must be [N,A]")
    if positions.is_floating_point() or anchors.is_floating_point():
        raise ValueError("context and anchor positions must be integers")
    if (anchors[keep] < 0).any():
        raise ValueError("anchor positions must be nonnegative")
    contexts, queries, writes, reads = [], [], [], []
    lengths, samples, boundaries = [], [], [0]
    slots = torch.arange(block_size, device=device)
    for sample in range(n):
        sequence_docs = docs[sample, valid[sample]]
        if sequence_docs.unique_consecutive().numel() != sequence_docs.unique().numel():
            raise ValueError("A document must occupy one contiguous context segment")
        for document in anchor_docs[sample, keep[sample]].unique().unbind():
            ai = torch.where(keep[sample] & (anchor_docs[sample] == document))[0]
            ap = anchors[sample, ai]
            ci = torch.where(valid[sample] & (docs[sample] == document))[0]
            cp = positions[sample, ci]
            if cp.numel() and ((cp < 0).any() or (cp[1:] <= cp[:-1]).any()):
                raise ValueError("Context positions must increase within each document")
            ci = ci[cp < ap.max()]
            cp = positions[sample, ci]
            qi = (sample * a * block_size + ai[:, None] * block_size + slots).flatten()
            keys = torch.cat((2 * cp + 1, (2 * ap[:, None]).expand(-1, block_size).flatten()))
            order = keys.argsort(stable=True)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel(), device=device)
            offset = boundaries[-1]
            contexts.append(sample * s + ci)
            queries.append(qi)
            writes.append(offset + inverse[:ci.numel()])
            reads.append(offset + inverse[ci.numel():])
            lengths.append(ci.numel())
            samples.append(sample)
            boundaries.append(offset + order.numel())

    def indices(parts):
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device)

    cpu_boundaries = torch.tensor(boundaries, dtype=torch.long)
    return GDNEventLayout(
        indices(contexts), indices(queries), indices(writes), indices(reads),
        cpu_boundaries.to(device), cpu_boundaries, tuple(lengths), tuple(samples),
        boundaries[-1],
    )
