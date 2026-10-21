#!/usr/bin/env bash
# Wait until all local GPUs have no compute process, then launch FlashMTP.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/online/colocated/qwen3-8b-flashmtp-online.yaml}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
IDLE_CONFIRM_INTERVAL="${IDLE_CONFIRM_INTERVAL:-15}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/train/logs}"
STATE_PATH="${STATE_PATH:-$LOG_DIR/wait_flashmtp.state}"
WATCH_LOG="${WATCH_LOG:-$LOG_DIR/wait_flashmtp.log}"
LOCK_PATH="${LOCK_PATH:-$LOG_DIR/.wait_flashmtp.lock}"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  printf 'error: another FlashMTP GPU waiter already owns %s\n' "$LOCK_PATH" >&2
  exit 3
fi

write_state() {
  local status="$1"
  shift
  {
    printf 'status=%s\n' "$status"
    printf 'waiter_pid=%s\n' "$$"
    printf 'updated_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '%s\n' "$@"
  } >"$STATE_PATH"
}

write_state waiting "config=$CONFIG"
printf '[%s] waiter started pid=%s config=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "$CONFIG" >>"$WATCH_LOG"

idle_checks=0
while (( idle_checks < 2 )); do
  process_count="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | wc -l
  )"
  if (( process_count == 0 )); then
    idle_checks=$((idle_checks + 1))
    printf '[%s] GPUs idle (%d/2 confirmation)\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$idle_checks" >>"$WATCH_LOG"
    if (( idle_checks < 2 )); then
      sleep "$IDLE_CONFIRM_INTERVAL"
    fi
  else
    idle_checks=0
    gpu_summary="$(
      nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
        | tr '\n' ';'
    )"
    printf '[%s] waiting: %s compute processes; %s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      "$process_count" \
      "$gpu_summary" >>"$WATCH_LOG"
    sleep "$POLL_INTERVAL"
  fi
done

train_log="$LOG_DIR/flashmtp_$(date -u '+%Y%m%dT%H%M%SZ').log"
source "$ROOT_DIR/.venv/bin/activate"
write_state \
  launched \
  "config=$CONFIG" \
  "train_log=$train_log" \
  "run_id=derived_by_run_flashmtp"
printf '[%s] GPUs confirmed idle; launching training log=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$train_log" >>"$WATCH_LOG"

set +e
CONFIG="$CONFIG" bash "$SCRIPT_DIR/run_flashmtp.sh" >>"$train_log" 2>&1
exit_code=$?
set -e
if (( exit_code == 0 )); then
  final_status=completed
else
  final_status=failed
fi
write_state \
  "$final_status" \
  "config=$CONFIG" \
  "train_log=$train_log" \
  "exit_code=$exit_code"
printf '[%s] training %s exit_code=%s log=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  "$final_status" \
  "$exit_code" \
  "$train_log" >>"$WATCH_LOG"
exit "$exit_code"
