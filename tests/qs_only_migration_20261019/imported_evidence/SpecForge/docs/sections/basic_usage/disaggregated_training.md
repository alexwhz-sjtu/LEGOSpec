# Disaggregated training

Disaggregation is a producer/consumer launch topology of the canonical training
command:

```bash
specforge train -c run.yaml
```

The producer captures or ingests features and the consumer runs the canonical
trainer.

| Category | Producer responsibility | Feature store | Service ownership |
| --- | --- | --- | --- |
| Offline disaggregated | Ingest existing feature files and publish a static manifest | `shared_dir` or Mooncake | No SGLang lifecycle; configure the storage backend directly |
| Online disaggregated, external | Send prompts to existing capture endpoints and publish streaming refs | Mooncake | User or scheduler owns Mooncake and SGLang |

Online disaggregation always uses disjoint capture/producer and trainer node
pools. For single-node online training, use `deployment.mode:
local_colocated`; that path performs in-process target capture and does not use
Mooncake.

Choose checked-in recipes by directory:

| Workflow | Catalog | Representative config |
| --- | --- | --- |
| Offline disaggregated | [`offline/disaggregated/`](https://github.com/sgl-project/SpecForge/tree/main/examples/configs/offline/disaggregated) | [`qwen3-8b-eagle3-offline-disaggregated.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/offline/disaggregated/qwen3-8b-eagle3-offline-disaggregated.yaml) |
| Online, external services | [`online/disaggregated/external/`](https://github.com/sgl-project/SpecForge/tree/main/examples/configs/online/disaggregated/external) | [`qwen3-8b-dflash-disaggregated.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3-8b-dflash-disaggregated.yaml) |

The full model and strategy index is in the
[recipe catalog](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/README.md). Filename suffixes are
historical identifiers; the directory and typed YAML fields define the runtime
semantics.

Mooncake training uses **pinned receive pools by default**, with a lazily allocated
8 GiB retained-pool budget per trainer rank. On CUDA with loader prefetch enabled,
the loader copies received features to the GPU before yielding the batch. Returned
tensors, prefetched batches, and overflow allocations consume additional memory.
Set `deployment.disaggregated.receive_buffers: pageable` to use fresh CPU receives.

Patched CUDA capture servers automatically publish device tensors when
`MOONCAKE_PROTOCOL=rdma`. TCP and non-CUDA workers keep host publication.
Set `SGLANG_SPEC_CAPTURE_GPU_PUT=0` on an external server to disable GPU
publication explicitly.
GPU publication needs RDMA-registerable allocations; custom expandable CUDA
allocations may require `PYTORCH_ALLOC_CONF=expandable_segments:False` on the
capture server. The trainer's allocator is independent.

## One config owns the launch topology

Process topology and attempt paths live in the same typed run document as the
model and training settings. An online deployment has this shape:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 4
  disaggregated:
    control_dir: outputs/qwen3-8b-dflash-disaggregated/control
    capture_nnodes: 1
    consumer_state_dir: outputs/qwen3-8b-dflash-disaggregated/consumer-state
    backend: mooncake
    server_urls:
      - http://127.0.0.1:30000
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
    mooncake_protocol: tcp
```

`data.train_data_path` or `data.prompts_path` selects online mode;
`data.hidden_states_path` selects offline mode. There is no second scenario
flag to keep synchronized. The consumer world size is
`nnodes * nproc_per_node`; it must satisfy the TP/USP topology in `training`.

The control directory is attempt-scoped. The launcher deterministically derives
the online reference channel and lifecycle markers, or the offline manifest,
beneath that root. Online consumers put the rank-0 SQLite/WAL under
`consumer_state_dir`; that path should be node-local and is required for
multi-node trainers. By default rank inboxes stay under the shared
`control_dir`. When a cluster cannot mount that path on every trainer node, set
`inbox_server_url` to a private rank-0 HTTP origin: rank 0 keeps the inbox files
locally and relays only tensor-free references and consumed counters to remote
ranks. The producer and consumer rank 0 must still resolve `control_dir` to the
same path. For online training, its filesystem must support atomic hard links
so the consumer can publish a complete optimizer-window contract without
overwriting another consumer's claim. A fresh attempt requires fresh control
and consumer-state directories. `store_id` defaults to `run_id` and can be set
explicitly when the Mooncake deployment requires another namespace.

Long producer/consumer waits are unbounded unless `idle_timeout_s`,
`peer_wait_timeout_s`, or `producer_hold_s` is explicitly configured. This
avoids terminating healthy training merely because it runs longer than a fixed
wall-clock default. An explicitly configured timeout is terminal: expiration
fails the attempt and runs its feature cleanup.

The file loader translates the former `training.deployment_mode`,
`training.server_urls`, and `training.metadata_db_path` fields at its migration
boundary. New recipes use `deployment`; `training.role` remains the canonical
persisted role selection and normally stays `auto` for a shared config.

## Explicit online roles

Online capture/producer and trainer pools must be disjoint. Use the same
resolved config on both sides:

```bash
# Capture node
specforge train -c examples/configs/online/disaggregated/external/qwen3-8b-dflash-disaggregated.yaml --role producer

# Trainer node
specforge train -c examples/configs/online/disaggregated/external/qwen3-8b-dflash-disaggregated.yaml --role consumer
```

Use `--plan` together with the explicit role to inspect commands without
starting them. Plan output redacts secret-shaped overrides and URL credentials.

## Multi-server capture

`deployment.disaggregated.server_urls` is an ordered worker topology, not merely
a set of endpoints. The producer creates one capture adapter and one rollout
worker for every list entry. All workers lease disjoint prompts from one
controller, write into the same Mooncake namespace, and publish references to
the same channel concurrently:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 2
  disaggregated:
    control_dir: outputs/qwen36-two-capture-servers/control
    backend: mooncake
    server_urls:
      - http://capture-0.example:30000
      - http://capture-1.example:30000
```

Every server must use the same target model, revision, capture method, and
auxiliary layer ids. A server failure returns its leased prompts to the shared
pool; surviving workers can lease and finish them. A worker is removed after
repeated consecutive failures, and the producer fails loudly rather than
publishing a successful partial stream if every worker is lost or a prompt
exhausts its bounded retries.

Repeating one URL is also intentional: each occurrence creates another worker
that can issue a blocking capture request to the same server. This can improve
prefill occupancy without allocating another target replica, subject to that
server's request capacity. For example, the following temporary override uses
four producer workers against one server while keeping the checked-in YAML as
the source of every other setting:

```bash
specforge train -c examples/configs/online/disaggregated/external/qwen3-8b-domino-disaggregated.yaml \
  'deployment.disaggregated.server_urls=["http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000"]'
```

In the default external-services mode, listing one or more URLs never starts
those servers. They must already be healthy and attached to the configured
Mooncake deployment. Online disaggregated launch requires explicit roles on disjoint pools:

```bash
# Capture/producer node
specforge train -c run.yaml --role producer

# Trainer node (or each trainer node with --node-rank)
specforge train -c run.yaml --role consumer
```

`deployment.disaggregated.capture_nnodes` records the number of physical
capture/producer nodes. `deployment.trainer.nnodes` counts only trainer nodes.
SGLang target TP belongs to the external capture servers; the disaggregated
consumer keeps `training.tp_size: 1`.

## Split pools and multi-node consumers

The same YAML launches either role explicitly:

```bash
# Inference/ingestion pool
specforge train -c run.yaml --role producer

# Trainer pool
specforge train -c run.yaml --role consumer
```

For offline features split across two physical nodes, the generic wrapper maps
the launcher-provided rank to those same two commands:

```bash
rcli exec --per-node <job> \
  'CONFIG=examples/configs/offline/disaggregated/qwen2.5-7b-eagle3-offline-disaggregated.yaml bash examples/disagg/run_offline_2node.sh'
```

`RCLI_NODE_RANK=0` selects the producer and rank 1 selects the consumer. The
wrapper contains no trainer or transport setup; the shared `control_dir` and
`store_root` remain in YAML and must resolve to the same fresh attempt storage
on both nodes. `NODE_RANK`/`NUM_NODES` are accepted on other schedulers.

For multiple consumer nodes, record the shared topology once:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 2
    nproc_per_node: 8
    master_addr: trainer-0.example
    master_port: 29500
  disaggregated:
    # With inbox_server_url, only the producer and consumer rank 0 need this
    # path; remote ranks receive their tensor-free reference stream over HTTP.
    control_dir: /local/rank0/control/attempt-001
    consumer_state_dir: /local/rank0/consumer-state/attempt-001
    inbox_server_url: http://trainer-0.example:35900
    backend: mooncake
    server_urls: [http://capture-server:30000]
```

Then provide only the node-local identity on each trainer host:

```bash
# trainer-0
specforge train -c run.yaml --role consumer --node-rank 0

# trainer-1
specforge train -c run.yaml --role consumer --node-rank 1
```

Online disaggregation rejects `--role both`: SpecForge does not SSH to remote
hosts or impersonate a cluster scheduler. An existing torchrun environment is
detected and used as the worker environment rather than nesting another
torchrun. A producer is rejected inside a multi-rank torchrun to prevent
duplicate capture/ingestion roles.

`inbox_server_url` is optional; omit it when every trainer rank shares
`control_dir`. When set, rank 0 binds the configured port on all interfaces and
remote ranks pull only their private metadata stream. The endpoint has no
built-in authentication or TLS, so expose it only on a trusted cluster network.
The producer must run on the rank-0 host (or otherwise share rank 0's
`control_dir`); payload tensors never traverse this HTTP service.

The separate-inference-node Qwen3-8B example keeps that scheduler boundary but
restores full development-stack orchestration:

```bash
export DISAGG_STORE_ID=qwen3-8b-two-node-attempt-001
export DISAGG_RUN_ROOT=/shared/specforge/$DISAGG_STORE_ID
rcli exec --per-node <job> \
  'bash examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh'
```

Rank 0 starts and health-checks Mooncake and patched SGLang, then invokes the
canonical producer role. Rank 1 invokes the canonical consumer role, whose
process count still comes from the YAML. The wrapper coordinates only
node-local services and shared lifecycle markers; it does not contain another
trainer or construct `torchrun`. `NODE_RANK`/`NUM_NODES`/`HEAD_IP` may be used
instead of the corresponding `RCLI_*` variables on another scheduler. Both
nodes must see a fresh `DISAGG_RUN_ROOT`. References stay under its shared
`control` directory, while the wrapper's lifecycle markers stay at the shared
run root. SQLite/WAL and rank inboxes default to the trainer-local
`/tmp/specforge/$DISAGG_STORE_ID/consumer-state`. Set
`DISAGG_CONSUMER_STATE_DIR` or `LOCAL_SCRATCH` to select another node-local
path. This split-state form currently supports one trainer node only.

The same wrapper starts several capture servers on rank 0 when `SERVER_COUNT`
is set: server `i` owns the `i`-th group of `SERVER_TP` devices from
`SERVER_GPUS`, listens on `SERVER_PORT + i`, and every server is handed to the
producer as `deployment.disaggregated.server_urls`.
`examples/disagg/run_qwen3.8_27b_dflash2_disagg_2node.sh` uses it for eight
TP1 Qwen3.8-27B capture servers feeding an eight-rank DFlash2 trainer; see its
[recipe page](/recipes/qwen3.8-27b-dflash2-disaggregated).

## External services

The training CLI owns only the explicitly requested SpecForge role. Mooncake
and patched SGLang remain external, usually long-lived services managed by
Kubernetes, Slurm, systemd, or the deployment environment. Capture services
and the producer live outside the trainer node pool.

Install a Mooncake client compatible with the store's `put_from`/`get_into`
wire contract on the producer and consumer images:

```bash
# CUDA earlier than 13
pip install mooncake-transfer-engine

# CUDA 13 or later
pip install mooncake-transfer-engine-cuda13
```

Online configs require Mooncake. Stable endpoints may be supplied by the typed
deployment section or the environment:

```bash
export MOONCAKE_METADATA_SERVER=http://metadata-server:8080/metadata
export MOONCAKE_MASTER_SERVER_ADDR=mooncake-master:50051
export MOONCAKE_LOCAL_HOSTNAME=this-node-routable-address
```

Keep `DISAGG_AUTH_TOKEN`, node-local hostnames, and device visibility out of
checked-in YAML. `MOONCAKE_PROTOCOL` and `MOONCAKE_RDMA_DEVICES` may also be
node-local deployment values.

The online producer sends prompts to the URLs in
`deployment.disaggregated.server_urls`. Start a patched SGLang server separately
with the model, capture method, and auxiliary layer ids matching the draft
config. DFlash, DFlash2, and Domino use the DFlash capture contract, DSpark
uses its dedicated K3 capture contract, and EAGLE3 and P-EAGLE use the EAGLE3
capture contract. Capture rejects chunked prefill and
gives every request attempt a unique radix-cache namespace so cached prefixes
cannot truncate the captured sequence. Online capture is text-only: VLM
training, including Qwen2.5-VL, is not supported. Online evaluation is also not
supported.

Offline disaggregation has no capture-server lifecycle and is distinguished by its `shared_dir` or Mooncake feature-store backend.

For the strict e2e one-sample overfit and serving validation procedure, follow
[`scripts/gates/README.md`](https://github.com/sgl-project/SpecForge/blob/main/scripts/gates/README.md).

It starts and health-checks Mooncake and SGLang, invokes the canonical training
entry, verifies overfit and serving behavior, and cleans up every process it
owns. This preserves complete automated validation without turning the training
CLI into a production service manager.

## Online runtime contract

Consumer ranks always use one path:

```text
RefDistributor (rank 0)
  -> one InboxChannel per rank
  -> one StreamingRefQueue per rank
  -> FeatureDataLoader
  -> DPAckController at the optimizer boundary
```

Rank 0 publishes a global dispatch quantum before capture:

```text
quantum = consumer_world_size * batch_size * accumulation_steps
```

When both `training.total_steps` and `training.max_steps` are omitted, the
producer publishes the schedule horizon derived from the prepared prompt count,
`training.num_epochs`, and this quantum. The consumer validates that contract,
uses it for optimizer and Domino loss schedules, and still trains until EOF.

The producer waits for that handshake. Configure bounded capture under
`runtime`:

```yaml
runtime:
  producer_lease: 8
  in_flight_high_watermark: 256
  in_flight_low_watermark: 192
  resident_high_watermark_bytes: null
  resident_low_watermark_bytes: null
  feature_store_max_resident_bytes: null
```

High/low thresholds implement hysteresis. The optional hard resident cap rejects
and cleans a new capture before publication. A partial quantum at EOF is settled
without dispatch rather than training an incomplete optimizer step.

Consumer rank 0 is the only source-channel reader and SQLite writer. At an
optimizer boundary, `DPAckController` commits sample ids and the durable marker,
removes their feature objects, then advances the channel counters. A crash
before the transaction replays refs; a crash after it skips them.

## Offline shared-directory and Mooncake stores

The checked-in offline recipe uses a shared directory:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 1
  disaggregated:
    control_dir: outputs/qwen3-8b-eagle3-offline-disaggregated/control
    backend: shared_dir
    store_root: outputs/qwen3-8b-eagle3-offline-disaggregated/features
```

Both paths must be visible at the same location from producer and consumer
nodes. The producer ingests existing EAGLE3, DFlash/DFlash2, Domino, or DSpark
features and publishes a fixed manifest. Offline epochs remain re-iterable.

Set `backend: mooncake` instead, omit `store_root`, and provide the Mooncake
endpoints to use the remote store. Also set a positive
`deployment.disaggregated.producer_segment_size`: unlike online server-owned
captures, an offline producer owns the objects it ingests until the consumer
acknowledges them. `MOONCAKE_GLOBAL_SEGMENT_SIZE` remains a compatibility
fallback. The consumer uses a zero-sized client segment. A Mooncake offline
producer remains alive until the consumer reports completion; the single-node
supervisor handles that lifecycle.

## Resume and freshness

For every new online attempt, the consumer database and its WAL/SHM sidecars
must not exist. The launcher checks that rule on global rank 0. Resume requires
the retained database and a matching checkpoint:

```bash
specforge train -c run.yaml --role consumer \
  training.resume_from=outputs/run/run-latest
```

With the default `--role auto`, a disaggregated config containing
`training.resume_from` resolves to consumer-only. Reuse the same control directory,
store id, run id, output directory, consumer topology, and Mooncake objects.
The durable optimizer-step acknowledgement must equal the checkpoint step.
Because acknowledgements may advance between periodic checkpoints, a crash in
that interval is deliberately rejected instead of replaying already-consumed
samples or silently skipping optimizer state; use a more frequent save interval
when that recovery window matters.

Offline disaggregated resume also launches only the consumer against the
retained manifest/store. Changing trainer world size during exact optimizer
state resume is outside the current contract.

The optional scripts `examples/disagg/run_online.sh` and
`examples/disagg/run_offline.sh` are thin delegates to the same command. Their
arguments use the CLI form, for example `run_online.sh --role producer`; they do
not parse `NPROC_PER_NODE`, construct torchrun, or manage transport state.
