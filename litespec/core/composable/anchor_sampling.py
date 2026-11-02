"""Objective-independent anchors, keyed by sample identity and epoch.

All random draws use a private CPU generator for each unpadded sample. This
keeps positions invariant to CUDA RNG use, padding, rank and batch slicing.
The layout is collected before model execution and reused during replay.
"""

from dataclasses import fields
import hashlib
import json

import torch

from .batch import BlockLayout, prepare_layout


ANCHOR_SAMPLING_VERSION = "sample_epoch_v1"


def prepare_sample_epoch_layout(input_ids, loss_mask, attention_mask, *,
                                block_size, num_anchors, seed, epoch,
                                sample_indices=None, supervision_mode="sampled", minimum_anchor=0):
    ids = input_ids.detach().cpu()
    mask = loss_mask.detach().cpu().float()
    valid = attention_mask.detach().cpu().bool()
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape != valid.shape:
        raise ValueError("Anchor inputs must have matching [batch, sequence] shapes")
    if int(epoch) < 0:
        raise ValueError("Anchor epoch must be nonnegative")
    indices = None if sample_indices is None else torch.as_tensor(sample_indices).reshape(-1).tolist()
    if indices is not None and len(indices) != len(ids):
        raise ValueError("One stable sample_index is required per anchor sample")
    layouts = []
    if supervision_mode == "all":
        num_anchors = max(1, ids.size(1) - 1)
    for row in range(len(ids)):
        positions = valid[row].nonzero().reshape(-1)
        length = int(positions[-1]) + 1 if positions.numel() else 1
        tokens = ids[row:row + 1, :length]
        labels = mask[row:row + 1, :length] * valid[row:row + 1, :length]
        if indices is None:
            # Token-only online datasets do not necessarily carry sample_index.
            # Include masks and exclude batch padding from this stable identity.
            identity = ["content", hashlib.sha256(json.dumps([
                tokens.tolist(), labels.tolist(), valid[row, :length].tolist()
            ], separators=(",", ":")).encode()).hexdigest()]
        else:
            identity = ["sample_index", int(indices[row])]
        key = json.dumps([ANCHOR_SAMPLING_VERSION, int(seed), int(epoch), identity],
                         separators=(",", ":")).encode()
        row_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        generator = torch.Generator(device="cpu").manual_seed(row_seed)
        layouts.append(prepare_layout(tokens, labels, block_size, num_anchors,
                                      complete_blocks=True, generator=generator,
                                      supervision_mode=supervision_mode, minimum_anchor=minimum_anchor))
    if not layouts:
        raise ValueError("Cannot prepare anchors for an empty batch")
    return BlockLayout(**{field.name: torch.cat([
        getattr(layout, field.name) for layout in layouts], dim=0)
        for field in fields(BlockLayout)}).to(input_ids.device)


def anchor_layout_digest(layout):
    """Small audit fingerprint; includes inactive-slot masks and prediction masks."""
    digest = hashlib.sha256()
    for name in ("anchor_positions", "block_keep_mask", "prediction_mask"):
        value = getattr(layout, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
