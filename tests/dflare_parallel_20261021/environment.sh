#!/usr/bin/env bash
# Every writable runtime/cache/temp location is on shared storage.
export DAC27_ROOT=/mnt/dai-sys/wanghanzhen/projects/DAC27
export TMPDIR="$DAC27_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$DAC27_ROOT/cache/xdg"
export XDG_DATA_HOME="$DAC27_ROOT/runtime_data"
export HF_HOME="$DAC27_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export MODELSCOPE_CACHE="$DAC27_ROOT/cache/modelscope"
export PIP_CACHE_DIR="$DAC27_ROOT/cache/pip"
export UV_CACHE_DIR=/mnt/dai-sys/wanghanzhen/.cache/uv
export UV_PYTHON_INSTALL_DIR=/mnt/dai-sys/wanghanzhen/.local/share/uv/python
export UV_PROJECT_ENVIRONMENT="$DAC27_ROOT/envs/litespec-py311-cu130"
export TORCH_HOME="$DAC27_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$DAC27_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$DAC27_ROOT/cache/triton"
export CUDA_CACHE_PATH="$DAC27_ROOT/cache/cuda"
export NUMBA_CACHE_DIR="$DAC27_ROOT/cache/numba"
export MPLCONFIGDIR="$DAC27_ROOT/cache/matplotlib"
export WANDB_CACHE_DIR="$DAC27_ROOT/cache/wandb"
export WANDB_DATA_DIR="$DAC27_ROOT/cache/wandb/data"
export WANDB_CONFIG_DIR="$DAC27_ROOT/cache/wandb/config"
export WANDB_DIR="$DAC27_ROOT/logs/wandb"
export SGLANG_CACHE_DIR="$DAC27_ROOT/cache/sglang"
export FLASHINFER_WORKSPACE_BASE="$DAC27_ROOT/cache/flashinfer"
export PYTHONPYCACHEPREFIX="$DAC27_ROOT/cache/pycache"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export UV_HTTP_TIMEOUT=300
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" \
    "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$MODELSCOPE_CACHE" \
    "$PIP_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
    "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" \
    "$WANDB_DATA_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DIR" "$SGLANG_CACHE_DIR" \
    "$FLASHINFER_WORKSPACE_BASE" "$PYTHONPYCACHEPREFIX"

export UV_PROJECT_ENVIRONMENT="/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/litespec-py311-cu130-h8002"
export NETRC="/mnt/dai-sys/wanghanzhen/projects/DAC27/private/credentials/H800_1.netrc"
export PATH="/mnt/dai-sys/wanghanzhen/projects/DAC27/envs/litespec-py311-cu130-h8002/bin:$PATH"
