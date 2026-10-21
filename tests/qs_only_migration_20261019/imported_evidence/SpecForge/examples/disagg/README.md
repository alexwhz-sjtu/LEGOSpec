# Disaggregated examples

This directory contains optional launch helpers for producer/consumer runs; the
YAML recipes themselves live under `examples/configs`. Disaggregated training
uses the same typed config and public command as every other run:

```bash
specforge train -c examples/configs/online/disaggregated/external/qwen3-8b-dflash-disaggregated.yaml
```

Online producer and consumer roles run on disjoint physical node pools. Launch
the producer explicitly on a capture node and the consumer on the trainer
node(s). The consumer topology comes from `deployment.trainer` and uses torch
distributed when `nproc_per_node > 1`.

Choose the YAML from the category that matches the data flow:

| Category | Recipe directory | What the launcher owns |
| --- | --- | --- |
| Offline disaggregated | `examples/configs/offline/disaggregated/` | Producer and consumer; feature storage is `shared_dir` or Mooncake |
| Online disaggregated, external | `examples/configs/online/disaggregated/external/` | Producer and consumer only; Mooncake and SGLang already exist |

`external` is a lifecycle boundary: SpecForge never starts Mooncake or SGLang
for this topology. Single-node online work belongs under `online/colocated`
and does not use this producer/consumer data plane.

The scripts in this directory are optional thin examples. The single-node
wrappers only add the config path and forward arguments to `specforge train`;
they contain no second trainer, torchrun construction, or transport validation:

```bash
CONFIG=examples/configs/online/disaggregated/external/qwen3-8b-dflash-disaggregated.yaml \
  examples/disagg/run_online.sh --role producer

CONFIG=examples/configs/offline/disaggregated/qwen3-8b-eagle3-offline-disaggregated.yaml \
  examples/disagg/run_offline.sh
```

For offline ingestion and training on separate physical nodes, invoke the same
rank dispatcher on both nodes. The cluster launcher supplies
`RCLI_NODE_RANK=0` for the producer and `RCLI_NODE_RANK=1` for the consumer:

```bash
rcli exec --per-node <job> \
  'CONFIG=examples/configs/offline/disaggregated/qwen2.5-7b-eagle3-offline-disaggregated.yaml bash examples/disagg/run_offline_2node.sh'
```

This wrapper only maps rank to `specforge train --role`; the YAML still owns
the consumer process count and the shared `control_dir`/`store_root`. Both
nodes must resolve those paths to the same storage and start from a fresh
attempt directory. `NODE_RANK`/`NUM_NODES` are accepted when the scheduler does
not expose the corresponding `RCLI_*` variables.

## Explicit roles

Use the same YAML when producer and consumer belong to different pools:

```bash
specforge train -c examples/configs/online/disaggregated/external/qwen3-8b-dflash-disaggregated.yaml \
  --role producer

specforge train -c examples/configs/online/disaggregated/external/qwen3-8b-dflash-disaggregated.yaml \
  --role consumer
```

For a multi-node consumer, `deployment.trainer.nnodes`,
`nproc_per_node`, `master_addr`, and `master_port` remain identical on every
node. Supply only the node-local rank at invocation time:

```bash
# Trainer node 0
specforge train -c run.yaml --role consumer --node-rank 0

# Trainer node 1
specforge train -c run.yaml --role consumer --node-rank 1
```

Online disaggregated runs reject `--role both`; launch producer and consumer on
their respective node pools. The CLI does not SSH into other machines or start
scheduler jobs.

For the common two-physical-node demonstration (inference services + producer
on rank 0, one DP trainer pool on rank 1), run the checked-in infrastructure
wrapper once per node through the cluster launcher:

```bash
export DISAGG_STORE_ID=qwen3-8b-two-node-attempt-001
export DISAGG_RUN_ROOT=/shared/specforge/$DISAGG_STORE_ID

# The launcher supplies RCLI_NODE_RANK, RCLI_NUM_NODES, and RCLI_HEAD_IP.
rcli exec --per-node <job> \
  'bash examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh'
```

The wrapper owns Mooncake/SGLang readiness and cross-node lifecycle only. Both
roles still execute `specforge train -c ...`; it contains no legacy Python
trainer and constructs no `torchrun` command. Set `SERVER_GPUS`, `TRAINER_GPUS`,
`TRAINER_NPROC`, and `CONFIG` when the allocation differs. `SERVER_COUNT`
starts that many capture servers on rank 0, each on the next `SERVER_TP`
devices of `SERVER_GPUS` and on consecutive ports from `SERVER_PORT`; all of
them are handed to the producer as `deployment.disaggregated.server_urls`.
`MOONCAKE_DEFAULT_KV_LEASE_TTL` (milliseconds) sets the master's read lease
for large feature payloads, and both nodes bind Mooncake's transfer engine to
their published hostname (`MC_TCP_BIND_ADDRESS`). Both nodes must see
the fresh `DISAGG_RUN_ROOT` path. References stay under its shared `control`
directory, while the wrapper's lifecycle markers stay at the shared run root.
The consumer's SQLite/WAL and rank inboxes default to the trainer-node-local
`/tmp/specforge/$DISAGG_STORE_ID/consumer-state`; override
`DISAGG_CONSUMER_STATE_DIR` or `LOCAL_SCRATCH` when `/tmp` is unsuitable.
For a multi-node trainer, set `deployment.disaggregated.inbox_server_url` to a
private HTTP origin on trainer node 0. Rank 0 owns the SQLite/WAL and relays
only tensor-free `SampleRef` metadata to the other trainer nodes; feature
tensors continue to move directly through the selected feature store.

## Checkpoints without shared storage

Each distributed rank writes its own `training_state_rankN.pt`. If every
trainer node resolves `output_dir` to the same shared filesystem, no extra
step is required. If `output_dir` is node-local, run the checked-in relay on
both trainer nodes before training. It exchanges rank-local archives over the
private trainer network, verifies SHA-256, and atomically assembles a complete
checkpoint directory on each node:

```bash
# Trainer node 0 (ranks 0-7)
python examples/disagg/sync_distributed_checkpoints.py \
  --run-root /workspace/runs/$RUN_ID \
  --run-id "$RUN_ID" \
  --local-ranks 0-7 --peer-ranks 8-15 \
  --serve-host 10.0.0.3 --serve-port 35914 \
  --peer-url http://10.0.0.4:35915 \
  --max-archives 3

# Trainer node 1 (ranks 8-15)
python examples/disagg/sync_distributed_checkpoints.py \
  --run-root /workspace/runs/$RUN_ID \
  --run-id "$RUN_ID" \
  --local-ranks 8-15 --peer-ranks 0-7 \
  --serve-host 10.0.0.4 --serve-port 35915 \
  --peer-url http://10.0.0.3:35914 \
  --max-archives 3
```

The relay has no authentication or TLS; bind it only to a trusted private
interface. Set `--max-archives` to the same retention window as
`training.max_checkpoints` so relay archives cannot grow without bound.

The Inkling DSpark variant uses the same two-node lifecycle with the target
settings validated against SGLang
[#31847](https://github.com/sgl-project/sglang/pull/31847):

```bash
export DISAGG_STORE_ID=inkling-two-node-attempt-001
export DISAGG_RUN_ROOT=/shared/specforge/$DISAGG_STORE_ID

rcli exec --per-node <job> \
  'bash examples/disagg/run_inkling_dspark_disagg_2node.sh'
```

Rank 0 uses four GPUs for TP4 ModelOpt-FP4 capture; rank 1 defaults to four
FSDP trainer ranks. Override `TARGET_MODEL_PATH`, `SERVER_GPUS`,
`TRAINER_GPUS`, or `TRAINER_NPROC` for another allocation. The launcher keeps
the unified radix tree enabled and does not pass `--disable-radix-cache`.
SGLang v0.5.18 includes the Inkling support from #31847. Install v0.5.18 in
both nodes' environment; the wrapper applies SpecForge's checked-in v0.5.18
capture patch before starting the server.

The Qwen3.8-27B DFlash2 variant runs eight TP1 capture servers on rank 0 and
an eight-rank trainer on rank 1, with the Mooncake lease, segment sizes and
in-flight watermarks measured for its 277-500 MB feature payloads:

```bash
export DISAGG_STORE_ID=qwen3.8-27b-dflash2-attempt-001
export DISAGG_RUN_ROOT=/shared/specforge/$DISAGG_STORE_ID

rcli exec --per-node <job> \
  'bash examples/disagg/run_qwen3.8_27b_dflash2_disagg_2node.sh'
```

Its [runbook](../../docs/recipes/qwen3.8-27b-dflash2-disaggregated.md)
explains the server/trainer split per GPU generation, the RDMA variant, and
the measured throughput on B300 and H200 nodes.

## External services

Recipes under `online/disaggregated/external` require an already-running
Mooncake deployment and patched SGLang capture server. Those services are not
started or stopped by `specforge train`. Put stable, non-secret topology in the
typed `deployment.disaggregated` section; inject authentication tokens, each
node's Mooncake hostname, and device visibility through the deployment
environment.
The checked-in recipes contain representative endpoints; replace those fields
or override them with environment values. Set the positive
`capture_nnodes` to the number of physical capture/producer nodes. Those nodes
must not overlap the `trainer.nnodes` pool.

Online configs use Mooncake. Offline configs may use either a typed
`shared_dir` store or Mooncake. `deployment.disaggregated.control_dir` is the
one attempt root from which the launcher derives the reference channel or
manifest and lifecycle markers. An online deployment places the consumer's
rank-0 SQLite/WAL in the node-local
`deployment.disaggregated.consumer_state_dir`. Multi-node trainers require this
field and keep their rank inboxes under the shared `control_dir`. Always choose
fresh control and consumer-state directories for a new attempt. An offline
Mooncake producer must also set a positive
`deployment.disaggregated.producer_segment_size`; online roles and the offline
consumer use zero because they do not own feature allocations.

Use `specforge train -c run.yaml --plan` to inspect the resolved role and
process commands without starting workers. Secret-looking override values and
URL userinfo are redacted.

See the [disaggregated training guide](../../docs/sections/basic_usage/disaggregated_training.md)
for service prerequisites, recovery rules, and the online/offline data-plane
contracts.
