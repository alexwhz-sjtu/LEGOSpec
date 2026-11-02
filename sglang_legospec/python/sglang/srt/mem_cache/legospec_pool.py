"""Independent dense KV, bounded SWA rings and request-scoped GDN state."""

import torch
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter
from sglang.srt.mem_cache.legospec_gdn_pool import LEGOSpecGDNPool
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sglang.srt.runtime_context import get_exec


class LEGOSpecKVPool(KVCache):
    post_capture_active = False

    def __init__(self, config, size, num_requests, device, enable_memory_saver=False):
        layers = config.composable_config["backbone"]["layers"]
        self.full_layers = [
            i
            for i, s in enumerate(layers)
            if s["attention"] == "gqa" and s["access"] == "dense"
        ]
        self.swa_layers = [
            i
            for i, s in enumerate(layers)
            if s["attention"] == "gqa" and s["access"] == "sliding_window"
        ]
        self.gdn_layers = [i for i, s in enumerate(layers) if s["attention"] == "gdn"]
        super().__init__(
            size, 1, torch.bfloat16, len(layers), device, enable_memory_saver
        )
        self.ring_history, self.ring_stride = 512, 520
        self.swa_req_ring_size = self.ring_history if self.swa_layers else None
        self.num_request_slots = num_requests
        self.commit_stopped = torch.zeros(
            num_requests, dtype=torch.int32, device=device
        )
        kwargs = dict(
            page_size=1,
            dtype=torch.bfloat16,
            head_num=config.num_key_value_heads,
            head_dim=config.head_dim,
            device=device,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=True,
        )
        self.full_kv_pool = MHATokenToKVPool(
            size=size, layer_num=len(self.full_layers), **kwargs
        )
        self.swa_kv_pool = (
            MHATokenToKVPool(
                size=num_requests * self.ring_stride,
                layer_num=len(self.swa_layers),
                **kwargs,
            )
            if self.swa_layers
            else None
        )
        self._full_indices = {i: j for j, i in enumerate(self.full_layers)}
        self._swa_indices = {i: j for j, i in enumerate(self.swa_layers)}
        if self.swa_kv_pool is not None:
            for i in range(len(self.swa_layers)):
                self.swa_kv_pool.get_key_buffer(i)[: self.ring_stride + 1].zero_()
                self.swa_kv_pool.get_value_buffer(i)[: self.ring_stride + 1].zero_()
        self.gdn = (
            LEGOSpecGDNPool(self.gdn_layers, num_requests, device)
            if self.gdn_layers
            else None
        )
        self.mem_usage = sum(self.get_kv_size_bytes()) / (1 << 30)

    def _build_layer_router(self):
        maps = [(IdSpaceKind.FULL, self.full_kv_pool, self._full_indices)]
        if self.swa_kv_pool is not None:
            maps.append(
                (IdSpaceKind.SLIDING_WINDOW, self.swa_kv_pool, self._swa_indices)
            )
        router = KVLayerRouter.from_layer_maps(self, maps)
        if self.gdn is not None:
            router.bind_part(
                sub_pool="legospec_gdn",
                part=STATE_PART,
                view=self.gdn,
                indices=self.gdn.indices,
                view_layer_ids={i: i for i in self.gdn_layers},
            )
        return router

    def get_key_buffer(self, layer_id):
        return self.layer_router.read(layer_id).key()

    def get_value_buffer(self, layer_id):
        return self.layer_router.read(layer_id).value()

    def get_kv_buffer(self, layer_id):
        return self.layer_router.read(layer_id).kv()

    def get_kv_buffer_shape(self):
        k, v = self.get_kv_buffer(self.full_layers[0])
        return k.shape, v.shape

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        self.layer_router.layer(layer.layer_id).store(
            layer, loc, cache_k, cache_v, *args, **kwargs
        )

    def set_kv_buffer_prefix_valid(
        self, layer, loc_2d, commit_lens, cache_k, cache_v, *args
    ):
        if layer.layer_id not in self._full_indices:
            raise ValueError(
                "Window commits require explicit request/absolute-position addressing"
            )
        self.full_kv_pool.set_kv_buffer_prefix_valid(
            layer,
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
            *args,
            layer_id_override=self._full_indices[layer.layer_id],
        )

    def get_kv_size_bytes(self):
        sizes = [self.full_kv_pool.get_kv_size_bytes()]
        if self.swa_kv_pool is not None:
            sizes.append(self.swa_kv_pool.get_kv_size_bytes())
        total = [sum(s[i] for s in sizes) for i in range(2)]
        total[0] += self.commit_stopped.numel() * self.commit_stopped.element_size()
        if self.gdn is not None:
            total[0] += self.gdn.get_bytes()
        return tuple(total)

    def get_contiguous_buf_infos(self):
        infos = [self.full_kv_pool.get_contiguous_buf_infos()]
        if self.swa_kv_pool is not None:
            infos.append(self.swa_kv_pool.get_contiguous_buf_infos())
        return tuple([x for info in infos for x in info[i]] for i in range(3))

    def move_kv_cache(self, tgt_loc, src_loc):
        # Draft state is rebuilt from confirmed target hiddens; it must never
        # copy speculative SWA/GDN state through the target's token relocation.
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)


def create_legospec_kv_pool(configurator, sizes, req_to_token_pool):
    return LEGOSpecKVPool(
        configurator.model_config.hf_config,
        sizes.max_total_num_tokens,
        req_to_token_pool.req_to_token.shape[0],
        configurator.device,
        get_exec().features.enable_memory_saver,
    )
