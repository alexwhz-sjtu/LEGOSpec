# Training launchers

Run the single-node online/colocated default recipe for each draft algorithm from any working
directory:

```bash
./train/run_dflash.sh
./train/run_dflash2.sh
./train/run_dspark.sh
./train/run_flashmtp.sh
```

All launchers delegate to the canonical `specforge train` entry point. Model,
data, training, and deployment parameters live only in YAML. Shell launchers
own run metadata (`RUN_ID`, `OUTPUT_DIR`) and W&B settings, including a common
`ENABLE_WANDB=true|false` switch. Set `CONFIG` to select another YAML file:

```bash
CONFIG=/path/to/run.yaml ./train/run_dflash2.sh
```

The colocated recipes read PerfectBlend JSONL directly, run the target
in-process on every trainer rank, and do not require Mooncake or a separate
hidden-state preparation step. `training.tp_size` controls target tensor
parallelism; `training.batch_size` remains the per-rank draft micro-batch.

## FlashMTP

The bundled FlashMTP profile supports `model.flashmtp_markov_head_type=markov`
with additive logits and `rnn_easy` with direct logits. Configure the selected
profile and every other run parameter in the YAML recipe. The v2swa grouped
dynamic convolution is controlled by `model.flashmtp_backbone_conv_enabled`;
its two hyperparameters are `model.flashmtp_conv_kernel_size` and
`model.flashmtp_conv_group_size`. The checked-in FlashMTP recipes enable it
with kernel size 2 and group size 16.

Prepare its offline features with:

```bash
torchrun --nproc_per_node=1 scripts/prepare_hidden_states.py \
  --strategy flashmtp \
  --target-model-path Qwen/Qwen3-8B \
  --data-path ./cache/dataset/sharegpt.jsonl \
  --output-path ./cache/hidden_states/qwen3-8b-flashmtp-sharegpt \
  --chat-template qwen \
  --max-length 4096
```

Launch the default Qwen3-8B RNN-easy/direct profile:

```bash
./train/run_flashmtp.sh
```

Unless overridden, the launcher derives both `run_id` and the output model
directory name from the selected YAML using this pattern:

```text
sf_flashmtp_draft_num_hidden_layers-<depth>_flashmtp_markov_rank-<rank>_flashmtp_chs_num_layers-<count>_flashmtp_local_position-<bool>_flashmtp_backbone_conv_enabled-<bool>_flashmtp_conv_kernel_size-<size>_flashmtp_conv_group_size-<size>
```

Run target-only and FlashMTP inference benchmarking with:

```bash
TARGET_MODEL=Qwen/Qwen3-8B \
DRAFT_NAME_OR_PATH=./outputs/flashmtp/checkpoint \
DATASET=gsm8k \
  bash evaluation/run_benchmark_flashmtp.sh
```



### train FlashMTP

```bash
cd /data/wanghanzhen/projects/SpecDecoding/SpecForge
source .venv/bin/activate

RUN_ID=qwen3-8b-flashmtp-rnn512-chs12 \
OUTPUT_DIR=/data/wanghanzhen/outputs/qwen3-8b-flashmtp-rnn512-chs12 \
ENABLE_WANDB=true \
WANDB_PROJECT=flashmtp_v3 \
WANDB_NAME=qwen3-8b-flashmtp-rnn512-chs12 \
WANDB_RUN_ID=qwen3-8b-flashmtp-rnn512-chs12 \
  bash ./train/run_flashmtp.sh
```

To select another complete recipe without introducing duplicated settings:

```bash
cd /data/wanghanzhen/projects/SpecDecoding/SpecForge
source .venv/bin/activate
CONFIG=/data/wanghanzhen/projects/SpecDecoding/SpecForge/examples/configs/online/colocated/qwen3-8b-flashmtp-online.yaml \
bash /data/wanghanzhen/projects/SpecDecoding/SpecForge/train/run_flashmtp.sh
```
