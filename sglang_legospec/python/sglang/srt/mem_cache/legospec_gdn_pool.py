"""Fixed-capacity, generation-scoped confirmed GDN memory."""

from dataclasses import dataclass

import torch


@dataclass
class GDNCommitLayout:
    slots: torch.Tensor
    cu_seqlens: torch.Tensor
    generations: torch.Tensor
    lengths_cpu: list[int] | None = None
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    reset_slots: torch.Tensor | None = None
    reset_generations: torch.Tensor | None = None
    mixed: object | None = None


@dataclass
class GDNMixedLayout:
    cu_seqlens: torch.Tensor
    carry: torch.Tensor
    read_chunks: torch.Tensor
    total_events: int
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    valid_event_lens: torch.Tensor | None = None


class GDNReadCache:
    """At most one 64-event chunk; no context-length-sized state snapshots."""

    def __init__(self, num_slots, device):
        def empty(*shape, dtype=torch.bfloat16):
            return torch.empty(num_slots, *shape, dtype=dtype, device=device)

        self.key = empty(64, 16, 128)
        self.value = empty(64, 32, 128)
        self.gate = empty(64, 32, dtype=torch.float32)
        self.beta = empty(64, 32)
        self.cumulative_gate = empty(64, 32, dtype=torch.float32).zero_()
        self.transformed_value = empty(64, 32, 128)
        self.entry_state = empty(32, 128, 128, dtype=torch.float32).zero_()
        self.count = torch.zeros(num_slots, dtype=torch.int32, device=device)

    def tensors(self):
        return tuple(vars(self).values())

    def reset(self, slots):
        for tensor in (self.entry_state, self.count, self.cumulative_gate):
            tensor.index_fill_(0, slots, 0)


class LEGOSpecGDNPool:
    def __init__(self, layer_ids, num_slots, device):
        self.layer_ids = list(layer_ids)
        self.indices = {layer: index for index, layer in enumerate(self.layer_ids)}
        self.states = torch.zeros(
            len(self.layer_ids),
            num_slots,
            32,
            128,
            128,
            dtype=torch.float32,
            device=device,
        )
        self.tails = torch.zeros(
            len(self.layer_ids), num_slots, 3, 6144, dtype=torch.bfloat16, device=device
        ).transpose(2, 3)
        self.cursors = torch.zeros(num_slots, dtype=torch.int64, device=device)
        self.generations = torch.zeros_like(self.cursors)
        self.owners = [None] * num_slots
        self.host_generations = [0] * num_slots
        self.device = device
        self.read_caches = [GDNReadCache(num_slots, device) for _ in self.layer_ids]

    def read_cache(self, layer_id):
        return self.read_caches[self.indices[layer_id]]

    def state(self, layer_id):
        return self.states[self.indices[layer_id]]

    def tail(self, layer_id):
        return self.tails[self.indices[layer_id]]

    def get_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.states,
                self.tails,
                self.cursors,
                self.generations,
                *(t for cache in self.read_caches for t in cache.tensors()),
            )
        )

    def prepare_prefill(self, reqs, prefixes, lengths, slots):
        reset, reset_generations, expected = [], [], []
        for req, prefix in zip(reqs, prefixes):
            slot = req.kv.req_pool_idx
            if not slot or slot >= len(self.owners):
                raise ValueError("GDN prefill has an invalid request slot")
            if prefix == 0:
                self.owners[slot] = req.rid
                self.host_generations[slot] += 1
                reset.append(slot)
                reset_generations.append(self.host_generations[slot])
            elif self.owners[slot] != req.rid:
                raise ValueError(
                    "GDN requires full confirmed-prefix prefill after slot reuse"
                )
            expected.append(self.host_generations[slot])

        def gpu(values, dtype=torch.int64):
            return torch.tensor(values, dtype=dtype, pin_memory=True).to(
                self.device, non_blocking=True
            )

        ends, offsets, chunks = [0], [0], []
        for row, length in enumerate(lengths):
            ends.append(ends[-1] + length)
            n = (length + 63) // 64
            offsets.append(offsets[-1] + n)
            chunks.extend((row, i) for i in range(n))
        carries = [prefix % 64 for prefix in prefixes]
        event_ends, event_offsets, event_chunks = [0], [0], []
        for row, (length, carry) in enumerate(zip(lengths, carries)):
            count = length + carry + 8 if length else 0
            event_ends.append(event_ends[-1] + count)
            n = (count + 63) // 64
            event_offsets.append(event_offsets[-1] + n)
            event_chunks.extend((row, i) for i in range(n))
        return GDNCommitLayout(
            slots=slots,
            cu_seqlens=gpu(ends, torch.int32),
            generations=gpu(expected),
            lengths_cpu=list(lengths),
            chunk_indices=gpu(chunks, torch.int32).reshape(-1, 2),
            chunk_offsets=gpu(offsets, torch.int32),
            reset_slots=gpu(reset),
            reset_generations=gpu(reset_generations),
            mixed=GDNMixedLayout(
                cu_seqlens=gpu(event_ends, torch.int32),
                carry=gpu(carries, torch.int32),
                read_chunks=gpu(
                    [(n + c) // 64 for n, c in zip(lengths, carries)], torch.int32
                ),
                total_events=event_ends[-1],
                chunk_indices=gpu(event_chunks, torch.int32).reshape(-1, 2),
                chunk_offsets=gpu(event_offsets, torch.int32),
            ),
        )

    def reset(self, layout):
        if layout.reset_slots is None or not layout.reset_slots.numel():
            return
        self.states.index_fill_(1, layout.reset_slots, 0)
        self.tails.index_fill_(1, layout.reset_slots, 0)
        self.cursors.index_fill_(0, layout.reset_slots, 0)
        self.generations.index_copy_(0, layout.reset_slots, layout.reset_generations)
        for cache in self.read_caches:
            cache.reset(layout.reset_slots)

    def generation_rows(self, reqs):
        values = []
        for req in reqs:
            slot = req.kv.req_pool_idx
            if self.owners[slot] != req.rid:
                raise ValueError("GDN decode request generation has not been prefilled")
            values.append(self.host_generations[slot])
        return values
