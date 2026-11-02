#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MASTER_ADDR:?Set node0 address}"
: "${NODE_RANK:?Set this node rank}"
export PET_MASTER_ADDR="$MASTER_ADDR" PET_NODE_RANK="$NODE_RANK"
export PET_NNODES="${NNODES:-2}" PET_NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PET_MASTER_PORT="${MASTER_PORT:-29501}"
cd "$ROOT"
exec "${PYTHON_BIN:-python3}" -m scripts.config.launch_composable --config "${1:?Supply a YAML path}" "${@:2}"
