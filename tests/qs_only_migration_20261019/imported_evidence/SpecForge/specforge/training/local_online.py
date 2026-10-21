"""Single-node colocated online target capture.

Every process owns one target TP shard and one draft/FSDP shard. TP peers run
the same packed target batch, then keep disjoint rows for draft training. The
captured tensors never leave the accelerator and never enter a FeatureStore.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from specforge.runtime.contracts import TrainBatch


def distributed_prompt_indices(
    size: int,
    *,
    dp_size: int,
    dp_rank: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> list[int]:
    """Build one deterministic DP shard with DistributedSampler semantics."""

    if size <= 0:
        return []
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank={dp_rank} must be in [0, {dp_size})")
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(epoch))
        indices = torch.randperm(size, generator=generator).tolist()
    else:
        indices = list(range(size))
    if drop_last:
        total_size = (size // dp_size) * dp_size
        indices = indices[:total_size]
    else:
        num_samples = (size + dp_size - 1) // dp_size
        total_size = num_samples * dp_size
        padding_size = total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            repeats = (padding_size + len(indices) - 1) // len(indices)
            indices += (indices * repeats)[:padding_size]
    return indices[dp_rank:total_size:dp_size]


class ColocatedOnlineDataLoader:
    """Synchronously capture target states and yield rank-local draft batches."""

    def __init__(
        self,
        *,
        prompts: Sequence[dict[str, Any]],
        capture,
        capture_layout,
        collate_fn,
        strategy: str,
        batch_size: int,
        tp_size: int,
        tp_rank: int,
        dp_size: int,
        dp_rank: int,
        seed: int,
        sampler_drop_last: bool = True,
    ) -> None:
        if batch_size < 1 or tp_size < 1 or dp_size < 1:
            raise ValueError("batch_size, tp_size, and dp_size must be positive")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank={tp_rank} must be in [0, {tp_size})")
        self.prompts = prompts
        self.capture = capture
        self.capture_layout = capture_layout
        self.collate_fn = collate_fn
        self.strategy = strategy
        self.batch_size = int(batch_size)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.dp_size = int(dp_size)
        self.dp_rank = int(dp_rank)
        self.seed = int(seed)
        self.sampler_drop_last = bool(sampler_drop_last)
        self.target_batch_size = self.batch_size * self.tp_size
        self._epoch = 0
        self._seek_batches = 0
        self._perf = {
            "wait_producer_s": 0.0,
            "wait_fetch_s": 0.0,
            "fetch_s": 0.0,
            "fetch_bytes": 0.0,
            "fetch_batches": 0.0,
            "fetch_samples": 0.0,
        }

    @property
    def dataset_size(self) -> int:
        if self.sampler_drop_last:
            per_dp = len(self.prompts) // self.dp_size
        else:
            per_dp = (len(self.prompts) + self.dp_size - 1) // self.dp_size
        target_batches = per_dp // self.target_batch_size
        return target_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._seek_batches = 0

    def seek(self, num_batches: int) -> None:
        skip = max(0, int(num_batches))
        available = self.dataset_size // self.batch_size
        if skip > available:
            raise ValueError(
                f"seek({num_batches}) skips past the end of local online data: "
                f"only {available} batches are available"
            )
        self._seek_batches = skip

    @staticmethod
    def _payload(prompt: dict[str, Any]) -> tuple[list[int], list[int]]:
        payload = prompt.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("prepared prompt must contain a payload mapping")
        input_ids = payload.get("input_ids")
        loss_mask = payload.get("loss_mask")
        if not isinstance(input_ids, list) or not isinstance(loss_mask, list):
            raise TypeError(
                "prepared prompt payload must contain list input_ids/loss_mask"
            )
        if len(input_ids) != len(loss_mask) or not input_ids:
            raise ValueError("prepared prompt input_ids/loss_mask lengths must match")
        return input_ids, loss_mask

    def _feature_sample(
        self,
        input_ids: list[int],
        loss_mask: list[int],
        aux: torch.Tensor,
        last: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = aux.device
        sample = {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long, device=device
            ).unsqueeze(0),
            "loss_mask": torch.tensor(
                loss_mask, dtype=torch.long, device=device
            ).unsqueeze(0),
            self.capture_layout.aux_feature: aux.unsqueeze(0),
        }
        last_name = self.capture_layout.last_hidden_feature
        if last_name is not None:
            sample[last_name] = last.unsqueeze(0)
        return sample

    def __iter__(self):
        indices = distributed_prompt_indices(
            len(self.prompts),
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            seed=self.seed,
            epoch=self._epoch,
            drop_last=self.sampler_drop_last,
        )
        skip, self._seek_batches = self._seek_batches, 0
        start = skip * self.target_batch_size
        stop = len(indices) - (len(indices) % self.target_batch_size)
        for offset in range(start, stop, self.target_batch_size):
            batch_indices = indices[offset : offset + self.target_batch_size]
            rows = [self._payload(self.prompts[index]) for index in batch_indices]
            started = time.perf_counter()
            aux_rows, last_rows = self.capture.capture_rows(
                [input_ids for input_ids, _ in rows]
            )
            if (
                len(aux_rows) != self.target_batch_size
                or len(last_rows) != self.target_batch_size
            ):
                raise RuntimeError(
                    "local SGLang capture returned a different number of rows "
                    f"than requested: aux={len(aux_rows)}, last={len(last_rows)}, "
                    f"requested={self.target_batch_size}"
                )
            rank_start = self.tp_rank * self.batch_size
            rank_stop = rank_start + self.batch_size
            samples = [
                self._feature_sample(input_ids, loss_mask, aux, last)
                for (input_ids, loss_mask), aux, last in zip(
                    rows[rank_start:rank_stop],
                    aux_rows[rank_start:rank_stop],
                    last_rows[rank_start:rank_stop],
                )
            ]
            tensors = self.collate_fn(samples)
            non_tensors = [
                name
                for name, value in tensors.items()
                if not isinstance(value, torch.Tensor)
            ]
            if non_tensors:
                raise TypeError(
                    f"local collator returned non-tensors for {non_tensors}"
                )
            elapsed = time.perf_counter() - started
            self._perf["fetch_s"] += elapsed
            self._perf["fetch_batches"] += 1.0
            self._perf["fetch_samples"] += float(self.batch_size)
            self._perf["fetch_bytes"] += float(
                sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors.values()
                )
            )
            selected = batch_indices[rank_start:rank_stop]
            yield TrainBatch(
                sample_ids=[f"local:{index}" for index in selected],
                strategy=self.strategy,
                tensors=tensors,
                metadata={
                    "target_repr": (
                        "hidden_state"
                        if self.capture_layout.last_hidden_feature is not None
                        else None
                    ),
                    "ttt_length": None,
                },
            )

    def perf_counters_snapshot(self, reset: bool = False) -> dict[str, float]:
        snapshot = dict(self._perf)
        if reset:
            for name in self._perf:
                self._perf[name] = 0.0
        return snapshot

    def close(self) -> None:
        close = getattr(self.capture, "close", None)
        if callable(close):
            close()


def build_local_capture(cfg, *, capture_layers, capture_method, target_batch_size):
    """Build the in-process SGLang runner from the canonical model config."""

    from specforge.offline_capture import load_local_capture

    dtype = getattr(torch, cfg.model.torch_dtype)
    capture = load_local_capture(
        cfg.model.target_model_path,
        torch_dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        attention_backend=cfg.model.sglang_attention_backend,
        mem_fraction_static=cfg.model.sglang_mem_fraction_static,
        context_length=(cfg.model.sglang_context_length or cfg.data.max_length + 7),
        enable_nccl_nvls=cfg.model.sglang_enable_nccl_nvls,
        enable_symm_mem=cfg.model.sglang_enable_symm_mem,
        enable_torch_compile=cfg.model.sglang_enable_torch_compile,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        ep_size=cfg.model.sglang_ep_size,
        disable_radix_cache=cfg.model.sglang_disable_radix_cache,
        max_running_requests=(
            cfg.model.sglang_max_running_requests or target_batch_size
        ),
        max_total_tokens=(
            cfg.model.sglang_max_total_tokens or target_batch_size * cfg.data.max_length
        ),
        download_dir=cfg.model.cache_dir,
    )
    capture.set_capture_layers(capture_layers, capture_method=capture_method)
    return capture


__all__ = [
    "ColocatedOnlineDataLoader",
    "build_local_capture",
    "distributed_prompt_indices",
]
