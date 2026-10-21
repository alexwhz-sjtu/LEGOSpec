#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"

TARGET_MODEL="${TARGET_MODEL:?set TARGET_MODEL to the target checkpoint}"
DRAFT_NAME_OR_PATH="${DRAFT_NAME_OR_PATH:-${DRAFT_MODEL:-}}"
[[ -n "$DRAFT_NAME_OR_PATH" ]] || {
  printf 'error: set DRAFT_NAME_OR_PATH or DRAFT_MODEL\n' >&2
  exit 2
}

BLOCK_SIZE="${BLOCK_SIZE:-16}"
HISTORY_MODE="${HISTORY_MODE:-token}"
SLIDING_WINDOW_SIZE="${SLIDING_WINDOW_SIZE:-1}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  "$ROOT_DIR/evaluation/benchmark.py" \
  --model-name-or-path "$TARGET_MODEL" \
  --draft-name-or-path "$DRAFT_NAME_OR_PATH" \
  --block-size "$BLOCK_SIZE" \
  --history-mode "$HISTORY_MODE" \
  --sliding-window-size "$SLIDING_WINDOW_SIZE" \
  --dataset "$DATASET" \
  --max-samples "$MAX_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  "$@"
