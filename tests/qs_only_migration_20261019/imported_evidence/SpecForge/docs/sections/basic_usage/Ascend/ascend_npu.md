# Ascend NPU Tutorial

This is an end-to-end tutorial for running SpecForge on Ascend NPU hosts. It
walks through **installation → data preparation → online disaggregated training
with external services and split multi-node
roles**, using Qwen3.5-4B DFlash as the running example. Validated on a 16-card
A3 (64GB) host.

---

## 1. Installation

You need an Ascend host with the driver, CANN, and a `torch_npu`-enabled
PyTorch already installed, plus SGLang `0.5.18` with NPU support. Then install
SpecForge without touching that stack:

```bash
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
python -m pip install -e . --no-deps
```

`--no-deps` keeps pip from pulling CUDA wheels over the working NPU
torch/sglang. If a later step reports a missing lightweight dependency, install
just that package, also with `--no-deps`.

### Apply the SGLang capture patches (online runs only)

Online capture needs two patches on top of the installed SGLang, applied **in
this order**:

```bash
# Base capture patch: --enable-spec-capture server flags + the capture sink
bash scripts/apply_sglang_spec_capture_patch.sh

# Ascend companion patch: skip the wildcard segment mount that Ascend
# Mooncake rejects, and mount the feature segment with location="cpu"
SGLANG_DIR=$(python -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
cd "$SGLANG_DIR" && git apply /path/to/SpecForge/patches/sglang/v0.5.18/spec-capture-ascend-mount.patch
```

Skip both for offline training, which reads features from disk. The companion
patch is a no-op on non-Ascend hosts (it keys off `ASCEND_RT_VISIBLE_DEVICES`),
so a shared installation stays CUDA-safe.

### Device visibility on Ascend

Ascend selects devices through `ASCEND_RT_VISIBLE_DEVICES`, and the driver
rejects an *empty* value — hiding devices from a process means **unsetting**
the variable, not setting it to `""`. SpecForge handles this internally: device
ordinals from the config are injected through `CUDA_VISIBLE_DEVICES` on CUDA
hosts and `ASCEND_RT_VISIBLE_DEVICES` on Ascend hosts, and a "hide all devices"
role unsets the variable. You only need to export the visible set once for the
supervisor:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

Two more environment settings are recommended for long online runs:

```bash
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

### Attention backends on Ascend

Use `sdpa` for the **trainer**. Sequence parallelism (`sp_ulysses_size` /
`sp_ring_size` > 1, the `usp` backend) requires `yunchang`, whose import probes
the CUDA device and crashes NPU-only torch builds, so USP is CUDA-only for now.
SpecForge imports `yunchang` lazily only when SP sizes exceed 1, so the default
SP=1 path never touches it.

For the **capture server**, the recipes set `model.sglang_attention_backend:
ascend`. If a config leaves it at the flashinfer default, the launcher falls
back to `ascend` automatically on Ascend hosts (flashinfer does not exist
there).

---

## 2. Data preparation

Data preparation is platform independent. From the repository root:

```bash
python scripts/prepare_data.py --dataset sharegpt
```

This writes `./cache/dataset/sharegpt_train.jsonl`. For custom datasets and
target-model regeneration, see the [Data Preparation](../data_preparation.md)
guide.

---

## 3. Online training (external capture server)

This section uses **disaggregated** online training: a producer on the capture
node drives prompts through patched SGLang, features stream through Mooncake,
and a consumer runs on a disjoint trainer node. The checked-in
[`qwen3.5-4b-dflash-online-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml)
recipe targets an externally started capture server.

### Step 1: Start Mooncake and the capture server

```bash
mooncake_master --enable_http_metadata_server=true \
  --rpc_port=35551 --http_metadata_server_port=35880 \
  --metrics_port=35903 --enable_metric_reporting=false &
```

Start the patched capture server on NPU 0. The `--spec-capture-aux-layer-ids`
must match the draft's `target_layer_ids` — for DFlash these come from
`configs/qwen3.5-4b-dflash.json`: `1 8 15 22 29`. A mismatch produces zero
features with no error:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30)) \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --skip-tokenizer-init \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --attention-backend ascend \
  --enable-spec-capture --spec-capture-method dflash \
  --spec-capture-aux-layer-ids 1 8 15 22 29 \
  --host 127.0.0.1 --port 30000 &
```

Wait for `curl --fail http://127.0.0.1:30000/health` to return 200.

### Step 2: Launch producer and trainer

On the capture node, start the CPU producer after SGLang is healthy:

```bash
env -u ASCEND_RT_VISIBLE_DEVICES \
  specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml --role producer
```

On the trainer node:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml --role consumer
```

Before rerunning, clear stale control state:
`rm -rf outputs/qwen3.5-4b-dflash-npu-online`.

### Success criteria

- Producer log ends with `prompts_failed=0`.
- Consumer prints `step N: {...loss..., acc...}` lines and no
  `could not drain` error at teardown.
- If the capture producer dies with `ACL_ERROR_RT_CONTEXT_NULL` (107002), the
  installed SpecForge predates the NPU transport bind — upgrade past #722.

### Training DFlash 2 drafts

DFlash 2 reuses the DFlash strategy and capture path end to end, so the
external-server setup above carries over unchanged — including the
`--spec-capture-aux-layer-ids 1 8 15 22 29` flags, which match
`configs/qwen3.5-4b-dflash2.json`. Use the checked-in
[`qwen3.5-4b-dflash2-online-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash2-online-npu.yaml)
recipe:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash2-online-npu.yaml --role consumer
```

Notes specific to DFlash 2 on Ascend:

- **`attention_backend: sdpa` is mandatory** (already set in the recipe). The
  schema default `flex_attention` is rejected at the first training forward
  with `flex_attention is not available on this device; use sdpa/eager`.
- **`sdpa` and `eager` share the explicit boolean-mask path**; both implement
  the same block-local draft visibility, and the grouped convolutions operate
  inside each proposal block independently of the attention mask. The checked-in
  4B draft uses `full_attention` layers only, so no sliding-window mask is
  involved.
- **Selector knobs keep their defaults** (`dflash2_selector_loss_alpha: 1.0`,
  zero warmup/ramp). The selector and convolutions are stock torch ops and need
  no NPU-specific settings.
- **Serving the export** still requires an SGLang build with DFlash 2 support
  (SGLang PR #35371); training-side capture is unchanged.

---

## 4. Split producer and consumer roles across nodes

The same configs split across nodes with an explicit `--role`:

```bash
specforge train -c run.yaml --role producer   # inference / capture pool
specforge train -c run.yaml --role consumer --node-rank 0   # trainer-0
specforge train -c run.yaml --role consumer --node-rank 1   # trainer-1
```

Every capture server must use the same target model, capture method, and aux
layer ids. For external-service prerequisites, freshness rules, and resume, see
the [Disaggregated training](../disaggregated_training.md) guide.

---

## 5. NPU notes and troubleshooting

- **Empty `ASCEND_RT_VISIBLE_DEVICES` is invalid** — the Ascend driver rejects
  it. SpecForge unsets the variable instead of emptying it; do not export
  `ASCEND_RT_VISIBLE_DEVICES=` by hand.
- **USP is CUDA-only for now** — `yunchang` probes CUDA at import. Keep
  `sp_ulysses_size` / `sp_ring_size` at 1 and use the `sdpa` trainer backend.
- **Ascend Mooncake rejects wildcard buffer registration** — the trainer side
  forces `local_buffer_size=0` automatically (SpecForge roles are pure
  zero-copy clients), and the capture side needs the companion patch from
  Section 1.
- **`ACL_ERROR_RT_CONTEXT_NULL` in the capture producer** — the Mooncake
  transfer engine needs a bound device context; SpecForge binds the local NPU
  before `setup()`. Seeing this error means the installation predates #722.

---
