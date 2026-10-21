#!/usr/bin/env bash
# Standalone DSpark single-node online-training launcher.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/online/colocated/qwen3-4b-dspark-online.yaml}"

RUN_ID="${RUN_ID:-qwen3-4b-dspark-online-colocated}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_ID}"
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-dspark-training}"
WANDB_NAME="${WANDB_NAME:-$RUN_ID}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/wandb}"

clip_wandb_value() {
  local value="$1"
  local max_len=128
  if (( ${#value} <= max_len )); then
    printf '%s' "$value"
    return
  fi
  local digest
  digest="$(printf '%s' "$value" | sha1sum | cut -c1-8)"
  printf '%s_%s' "${value:0:$((max_len - 9))}" "$digest"
}
WANDB_NAME="$(clip_wandb_value "$WANDB_NAME")"
WANDB_RUN_ID="$(clip_wandb_value "$WANDB_RUN_ID")"

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: run_dspark.sh [SPECFORGE_TRAIN_ARG ...]

Environment defaults:
  CONFIG, RUN_ID=qwen3-4b-dspark-online-colocated,
  OUTPUT_DIR=./outputs/$RUN_ID, ENABLE_WANDB=true,
  WANDB_PROJECT=dspark-training, WANDB_NAME=$RUN_ID,
  WANDB_RUN_ID=$RUN_ID, WANDB_OFFLINE=false, WANDB_DIR=./wandb

Model, data, training, and deployment parameters come only from the YAML.
EOF
  exit 0
fi

case "$ENABLE_WANDB" in
  true) REPORT_TO=wandb ;;
  false) REPORT_TO=none ;;
  *)
    printf 'error: ENABLE_WANDB must be true or false, got %s\n' \
      "$ENABLE_WANDB" >&2
    exit 2
    ;;
esac
case "$WANDB_OFFLINE" in
  true|false) ;;
  *)
    printf 'error: WANDB_OFFLINE must be true or false, got %s\n' \
      "$WANDB_OFFLINE" >&2
    exit 2
    ;;
esac
[[ -f "$CONFIG" ]] || {
  printf 'error: CONFIG does not exist: %s\n' "$CONFIG" >&2
  exit 2
}
CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd)/$(basename -- "$CONFIG")"
specforge_bin="$(command -v specforge)" || {
  printf 'error: specforge is not on PATH\n' >&2
  exit 2
}

cd "$ROOT_DIR"
exec "$specforge_bin" train --config "$CONFIG" \
  "run_id=$RUN_ID" \
  "output_dir=$OUTPUT_DIR" \
  "tracking.report_to=$REPORT_TO" \
  "tracking.wandb_project=$WANDB_PROJECT" \
  "tracking.wandb_name=$WANDB_NAME" \
  "tracking.wandb_run_id=$WANDB_RUN_ID" \
  "tracking.wandb_offline=$WANDB_OFFLINE" \
  "tracking.wandb_dir=$WANDB_DIR" \
  "$@"
