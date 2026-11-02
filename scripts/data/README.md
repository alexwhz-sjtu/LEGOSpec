# FlashMTP data pipeline

The data preparation flow has one entry point and one schema boundary:

```text
arbitrary dataset
  -> convert_dataset.py
  -> standard prompt JSONL
  -> regenerate_train_data.py
  -> regen_token_only (generated-answer JSONL)
  -> prepare_hidden_states.py
  -> regen_full (token/mask/hidden-state tar shards)
```

## 1. Convert a source dataset

Inspect a few records, create a dataset adapter under `adapters/`, then convert
the entire dataset:

```bash
python scripts/data/convert_dataset.py inspect \
  --input /path/to/source.parquet --rows 5

python scripts/data/convert_dataset.py convert \
  --input /path/to/source.parquet \
  --adapter scripts/data/adapters/my_dataset.json \
  --output /path/to/prompts.jsonl \
  --turn-mode multi
```

Hugging Face Parquet inputs can be filtered while they are streamed. Repeating
`--hf-filter` combines predicates with AND and lets the Parquet reader skip
files or row groups when their metadata permits it:

```bash
.venv/bin/python scripts/data/convert_dataset.py convert \
  --input allenai/WildChat-4.8M \
  --input-format hf \
  --split train \
  --hf-filter language '==' Chinese \
  --hf-filter model '==' gpt-4-0314 \
  --adapter scripts/data/adapters/wildchat_4_8m.json \
  --output cache/data/prompts/wildchat_chinese_gpt-4-0314.jsonl \
  --turn-mode first \
  --skip-invalid
```

This does not materialize the complete Hub dataset locally. Filtering is pushed
into the Hugging Face Parquet loader; row groups that cannot be excluded from
Parquet statistics may still be transferred and scanned. The WildChat adapter
stores the conversation-level language in `category`. `--turn-mode first` keeps
the first user request; use `multi` only when all user turns should be retained
after the original assistant turns are removed.
Empty messages inside a conversation are ignored. `--skip-invalid` skips the
rare conversation that has no usable user message at all.

The converter emits exactly this record contract:

```json
{"id": 0, "conversations": [{"role": "user", "content": "..."}], "source": "my-dataset", "category": null}
```

Only `system` and `user` messages are retained at this stage. The converter is
the only component that knows the source dataset's original fields.

## 2. Generate assistant responses

```bash
python scripts/data/regenerate_train_data.py \
  --model /data/wanghanzhen/models/Qwen/Qwen3-4B \
  --input-file-path /path/to/prompts.jsonl \
  --server-address localhost:30000
```

Successful records keep the same four-field contract and gain generated
`assistant` messages. Failed records are written to a separate error JSONL and
are not included in the successful output. Use `--resume` to continue by input
`id`. This is the `regen_token_only` save mode: generated answer tokens are
stored as decoded text without target-model hidden states. Its default location
is `./cache/data/regen_token_only/`; `--output-file-path` overrides it.

For regeneration input, `category` is optional. When it is absent,
`regenerate_train_data.py` normalizes it to `null` in successful and error
outputs so the downstream four-field contract remains stable.

## 3. Save tokens and hidden states

```bash
cd /data/wanghanzhen/projects/SpecDecoding/LiteSpec
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --nproc-per-node=8 \
scripts/data/prepare_hidden_states.py \
  --target-model-path /data/wanghanzhen/models/Qwen3-4B \
  --target-model-backend sglang \
  --data-path /data/wanghanzhen/projects/SpecDecoding/LiteSpec/cache/data/regen_token_only/wildchat_chinese_turns_le2_qwen3_4b_temp0.7_topk15_max4096.jsonl \
  --output-path /data/wanghanzhen/projects/SpecDecoding/LiteSpec/cache/data/regen_full/wildchat_chinese_qwen3_4b \
  --target-layer-ids 0,8,16,24,31 \
  --max-length 4096 \
  --tp-size 1 \
  --sglang-ep-size 1 \
  --batch-size 1 \
  --shard-size 512

```

Each saved sample contains `input_ids`, `attention_mask`, `loss_mask`, and the
selected hidden-state layers. A full shard contains 512 successful samples;
failed samples are logged and later successful samples fill their places. Use
`--resume` after an interrupted run. This is the `regen_full` save mode, whose
default root is `./cache/data/regen_full/`; `--output-path` overrides it.

`regen_full` schema v2 always stores the target model's final-norm hidden state
as `final_hidden_state`, even when `--target-layer-ids` selects only intermediate
layers. If the selected IDs already contain the final layer, it is not stored a
second time in `hidden_states`.

The two names describe save granularity for the same regenerated dataset:

- `regen_token_only`: regenerated answers in JSONL, without hidden states.
- `regen_full`: token IDs, masks, and hidden states in training shards.



## 4. Train from regen_full

All DLite training entrypoints accept either online JSONL or an offline cache.
Use exactly one of `--train-data-path` and `--train-hidden-states-path`:

```bash
torchrun --nproc-per-node=8 -m scripts.dlite.train_sft \
  --target-model-path /path/to/model \
  --train-hidden-states-path ./cache/data/regen_full/dataset_model \
  --tp-size 1 \
  --output-dir ./cache/models/run \
  ...
```

Offline training loads only the frozen target embedding and LM head. Target
logits are computed from cached `final_hidden_state` at the sampled anchor
positions. Offline mode requires `--tp-size 1` and does not support
`--shard-draft-by-tp`; multiple torchrun workers operate as data-parallel ranks.

The shell launchers use `TRAIN_DATA_PATH` for online mode or
`TRAIN_HIDDEN_STATES_PATH` for offline mode. These environment variables are
mutually exclusive.


## Optimized offline capture and raw-tar training

The optimized entry point is independent of local experiment directories. Its
defaults enable the full optimized path: SGLang, asynchronous capture,
`tar-raw-v1`, packed requests, streaming SHA256, and asynchronous tar commits.
The Qwen3-4B configuration captures BF16 layers 1, 9, 17, 25, 33 and the final
normalized layer 35, retaining original responses and max_length=4096:

```bash
python -m torch.distributed.run --nproc-per-node=2 --master-port=29679 \
  -m scripts.data.capture_hidden_states \
  --target-model-path /path/to/Qwen3-4B --target-model-backend sglang \
  --sglang-attention-backend fa3 --tp-size 1 --batch-size 4 \
  --data-path /path/to/data/training/answers.jsonl \
  --output-path /path/to/data/hidden/full80k/cache \
  --cache-dir /path/to/data/preprocessed/capture --max-length 4096 \
  --tar-shard-bytes 2147483648 \
  --pipeline-depth 2 --writer-workers 2 --max-pending-bytes 8589934592 \
  --fail-on-error
```

The five optimization flags are enabled by default and need not be repeated.
Use `--capture-mode sync`, `--no-packed-requests`, `--no-tar-stream-sha`, or
`--no-tar-async-commit` to override individual choices. To use legacy synchronous
output, pass `--capture-mode sync --cache-format tar-v2 --no-packed-requests
--no-tar-stream-sha --no-tar-async-commit` on one command line. Packed capture
requires SGLang; explicit `--target-model-backend hf` must use
`--no-packed-requests` and `--max-batch-tokens 0`. The separate legacy
`prepare_hidden_states` entry point retains its original behavior.

Raw members store tensor descriptors and exact tensor bytes inside ordinary
uncompressed tar shards. They avoid per-sample hidden clones and `torch.save`
packing. Each rank publishes its own final shards plus a unified sample index;
training ranks use the distributed sampler rather than owning capture-rank files.
The writer retains buffers until writes finish and propagates asynchronous errors.
Use `--resume` only with the same input/model/capture identity; already committed
shards are verified on resume, and only uncommitted samples are recaptured.

SHA256 is computed while writing; fsync and atomic data/index publication remain.
This entry point does not perform an additional full-dataset readback after
capture. Optional `--profiling-output /path/to/data/profiling/run1` writes CPU/CUDA
timing outside the cache. Do not put datasets, checkpoints or profiling artifacts
in test/tests directories.

The training reader supports `tar-v2`, `tar-indexed-v2`, and `tar-raw-v1` with the
same tokens, masks and hidden fields. Start from
`scripts/config/offline_dflare_parallel_ce.yaml` or
`scripts/config/offline_dflare_parallel_ce_tv.yaml`, replace `/path/to` paths,
then run:

```bash
python -m scripts.config.launch_composable \
  --config scripts/config/offline_dflare_parallel_ce.yaml
```

Both templates use two GPUs, batch4 per GPU, accumulation1 (global batch8),
512 total anchors, objective_chunk_blocks256, max_length4096 and six epochs.
The CE+TV template keeps the original 0.1 CE / 0.9 TV weighting. Training validates
manifest/index structure and sample tensors, but full shard checksum readback is
off by default. Opt in with `verify_hidden_cache_checksums: true` or
`--verify-hidden-cache-checksums` when explicitly needed.
