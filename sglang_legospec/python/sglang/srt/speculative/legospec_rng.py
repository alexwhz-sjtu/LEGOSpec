"""Stable request seeds, independent of batching, request slots and retractions."""

import hashlib


def legospec_request_seed(req, server_seed):
    explicit = req.sampling_params.sampling_seed
    if explicit is not None:
        return int(explicit)
    value = f"{int(server_seed)}|{req.rid}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def legospec_verify_uniforms(seeds, positions, width):
    """Request-indexed coins for the native SGLang chain verifier."""
    import torch
    from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

    columns = torch.arange(width, dtype=torch.int32, device=seeds.device)
    bits = murmur_hash32(seeds ^ 0x15691A2E35D44C71, positions, columns)
    uniforms = (bits.to(torch.int64) & 0xFFFFFF).float() * (2.0**-24)
    return uniforms[:, :-1].contiguous(), uniforms[:, -1].contiguous()
