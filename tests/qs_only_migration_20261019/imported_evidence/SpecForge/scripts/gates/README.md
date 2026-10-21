# One-sample overfit validation

This guide contains three stages:

1. regenerate a small subset of datasets by target model;
2. overfit one sample on Qwen3-4B DSpark training with `specforge train`;
3. export the checkpoint, serve it with SGLang, and verify that one complete
   16-token draft block is accepted.

The training command below colocates target capture and draft training on one
GPU. We use Qwen3-4B DSpark as an example; the same topology is available for
DFlash, DFlash2, and FlashMTP.

## Validate the ports

Run this check from the SpecForge repository before preparing data. It checks
only whether the regeneration and serving ports are available.

```bash
source scripts/gates/_e2e_common.sh

gate_report_tcp_ports python 127.0.0.1 \
  regeneration_port 30000 \
  serving_port 32001
```

The check always returns to the shell. If one or more ports are occupied, it
prints every conflict without stopping or killing the owning processes.

## Prepare one sample
### Stage 1: Regen the datasets
Launch the sglang server:
```shell
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-4B \
  --mem-fraction-static 0.7 \
  --tp 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --reasoning-parser qwen3
```
Regen the datasets:
```
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3-4B \
  --concurrency 128 \
  --max-tokens 4096 \
  --server-address localhost:30000 \
  --temperature 0 \
  --input-file-path ./cache/dataset/sharegpt_train.jsonl \
  --output-file-path ./cache/dataset/sharegpt_train_regen_10.jsonl \
  --resume \
  --reasoning save \
  --num-samples 10
```


### Stage 2: train with SpecForge

```bash
export TRAIN_DATA_PATH=./cache/dataset/sharegpt_train_regen_10.jsonl
export MODEL_NAME=Qwen3-4B
export SPEC_METHOD=DSpark
export DRAFT_MODEL_CONFIG=configs/qwen3-4b-dspark.json
export MODEL=/data/wanghanzhen/models/Qwen3-4B
export TRAINING_CONFIG=examples/configs/online/colocated/qwen3-4b-dspark-online.yaml

specforge train \
  --config ${TRAINING_CONFIG} \
  model.target_model_path=${MODEL} \
  model.draft_model_config=${DRAFT_MODEL_CONFIG} \
  model.sglang_context_length=4200 \
  data.train_data_path=${TRAIN_DATA_PATH} \
  data.max_length=4096 \
  data.chat_template=qwen \
  training.num_epochs=600 \
  training.max_steps=6000 \
  training.total_steps=6000 \
  training.batch_size=1 \
  training.accumulation_steps=1 \
  training.eval_interval=0 \
  training.save_interval=1000 \
  training.log_interval=1 \
  deployment.trainer.nproc_per_node=1 \
  tracking.report_to=none \
  run_id=${MODEL_NAME}-${SPEC_METHOD}-overfit \
  output_dir=outputs/${MODEL_NAME}-${SPEC_METHOD}-overfit
```
<span style="color: red;"><strong>Note:</strong></span> The training data must be inferenced by the target model use temperature=0 and corresponding thinking mode.</span>

This command creates the in-process SGLang target and draft trainer on the same
rank. It does not start Mooncake, an external capture server, or a FeatureStore.

After training, the checkpoint will be stored in:

```text
outputs/${MODEL_NAME}-${SPEC_METHOD}-overfit/${MODEL_NAME}-${SPEC_METHOD}-overfit
```

To repeat the experiment independently, use a new suffix consistently in
`run_id`, `output_dir`, and the sample paths.

### Stage 3: serve with SGLang and check accept length

First export the trained draft:

```bash
# Export the draft model to HF format
specforge export \
  --to hf \
  --checkpoint outputs/${MODEL_NAME}-${SPEC_METHOD}-overfit/${MODEL_NAME}-${SPEC_METHOD}-overfit-latest \
  --draft-config ${DRAFT_MODEL_CONFIG} \
  --output-dir outputs/${MODEL_NAME}-${SPEC_METHOD}-overfit/draft_hf \
  --embedding-source ${MODEL} \
  --embedding-key model.language_model.embed_tokens.weight

# Normalize the config
python scripts/gates/normalize_dflash_export.py \
  --config outputs/${MODEL_NAME}-${SPEC_METHOD}-overfit/draft_hf/config.json \
  --block-size 16
```

The normalizer preserves DFlash/Domino exports as ``DFlashDraftModel``. For a
DSpark export it selects SGLang's ``Qwen3DSparkModel`` architecture and promotes
the Markov/confidence settings required by the SGLang loader to the top level.

Start SGLang in the first terminal:

```bash
python -m sglang.launch_server \
  --model-path ${MODEL} \
  --tp-size 1 \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --context-length 4096 \
  --max-running-requests 1 \
  --max-total-tokens 4096 \
  --chunked-prefill-size -1 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --trust-remote-code \
  --port 32001 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path outputs/${MODEL_NAME}-${SPEC_METHOD}-overfit/draft_hf \
  --reasoning-parser qwen3
```

After SGLang is ready, split all the data in TRAIN_DATA_PATH and run the validation for each data item in a second terminal:

```bash
python scripts/gates/run_dflash_chat_serving_gate.py \
  --server-url http://127.0.0.1:32001 \
  --model-path ${MODEL} \
  --served-model ${MODEL_NAME}-${SPEC_METHOD}-overfit \
  --data-path ${TRAIN_DATA_PATH} \
  --output-path outputs/${MODEL_NAME}-${SPEC_METHOD}-overfit/serving-gate.json \
  --enable-thinking \
  --system-prompt "" \
  --block-size 16 \
  --max-tokens 16
```

The terminal prints a concise summary. The complete request, SGLang response,
and server information remain in `serving-gate.json`; add `--verbose` only when
the full result is needed in the terminal.

The validation passes only when the result contains:

```json
{
  "passed": true,
  "input_format": "training_jsonl",
  "spec_accept_length": 16.0,
  "target_prefix_match_tokens": 16,
  "generated_tokens": 16,
  "target_tokens": 59,
  "clean_block_tokens": 16,
  "errors": [],
  "result_path": ""
}
```

`spec_accept_length >= 16` and `target_prefix_match_tokens >= 16` mean that the
complete DSpark block was accepted and agrees with the target continuation.
Stop the SGLang server with `Ctrl-C` after validation.
