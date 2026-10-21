#!/usr/bin/env bash
# Standalone FlashMTP single-node online-training launcher.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/online/colocated/qwen3-8b-flashmtp-online.yaml}"

ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-flashmtp_v3}"
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
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: run_flashmtp.sh [SPECFORGE_TRAIN_ARG ...]

Environment defaults:
  CONFIG, RUN_ID=sf_flashmtp_<model-parameter-tags>,
  OUTPUT_DIR=./outputs/$RUN_ID, ENABLE_WANDB=true,
  WANDB_PROJECT=flashmtp_v3, WANDB_NAME=$RUN_ID,
  WANDB_RUN_ID=$RUN_ID, WANDB_OFFLINE=false, WANDB_DIR=./wandb

The default RUN_ID is derived from the selected YAML and contains draft layer,
Markov rank, CHS layer, local-position, and backbone-convolution settings.
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
cd "$ROOT_DIR"
python_bin="$(command -v python)" || {
  printf 'error: python is not on PATH; activate the SpecForge environment\n' >&2
  exit 2
}

default_run_id="$("$python_bin" - "$CONFIG" <<'PY'
import sys

from specforge.config import load_config

config = load_config(sys.argv[1])
model = config.model
if model.draft_num_hidden_layers is None:
    raise SystemExit(
        "model.draft_num_hidden_layers must be explicit in the FlashMTP YAML "
        "so the generated run name records the actual draft depth"
    )

def boolean_tag(value: bool) -> str:
    return "true" if value else "false"

print(
    "sf_flashmtp_"
    f"draft_num_hidden_layers-{model.draft_num_hidden_layers}_"
    f"flashmtp_markov_rank-{model.flashmtp_markov_rank}_"
    f"flashmtp_chs_num_layers-{model.flashmtp_chs_num_layers}_"
    f"flashmtp_local_position-{boolean_tag(model.flashmtp_local_position)}_"
    "flashmtp_backbone_conv_enabled-"
    f"{boolean_tag(model.flashmtp_backbone_conv_enabled)}_"
    f"flashmtp_conv_kernel_size-{model.flashmtp_conv_kernel_size}_"
    f"flashmtp_conv_group_size-{model.flashmtp_conv_group_size}"
)
PY
)"

RUN_ID="${RUN_ID:-$default_run_id}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_ID}"
WANDB_NAME="${WANDB_NAME:-$RUN_ID}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
WANDB_NAME="$(clip_wandb_value "$WANDB_NAME")"
WANDB_RUN_ID="$(clip_wandb_value "$WANDB_RUN_ID")"

specforge_bin="$(command -v specforge)" || {
  printf 'error: specforge is not on PATH\n' >&2
  exit 2
}

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
