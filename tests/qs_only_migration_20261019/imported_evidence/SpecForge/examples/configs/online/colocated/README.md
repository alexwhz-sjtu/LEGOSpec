# Online colocated recipes

These recipes run synchronous in-process SGLang capture and draft training on
one node. Every rank owns the matching target TP shard and draft/FSDP shard;
captured tensors stay on the rank's accelerator. No Mooncake service, external
capture server, FeatureStore, or serialization path is created.

`training.batch_size` is the per-draft-rank micro-batch. A target TP group
captures `batch_size * tp_size` prompts and each TP rank keeps its contiguous
`batch_size` slice. Sequence parallelism and nonzero DataLoader workers are not
supported in this topology.
