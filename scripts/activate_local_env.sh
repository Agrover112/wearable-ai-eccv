#!/usr/bin/env bash
# Source this file to activate the project-local environment and cache policy.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run: source scripts/activate_local_env.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="/CT/RelightableGaussianCodecAvatarsHead/work/miniforge3"
ENV_NAME="${WEARABLE_AI_ENV_NAME:-wearable-ai-eccv}"
ENV_PREFIX="$REPO_ROOT/.conda/envs/$ENV_NAME"

export HOME="$REPO_ROOT/.cache/home"
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
export CONDA_PKGS_DIRS="$REPO_ROOT/.cache/conda/pkgs"
export CONDA_ENVS_PATH="$REPO_ROOT/.conda/envs"
export UV_CACHE_DIR="$REPO_ROOT/.cache/uv"
export PIP_CACHE_DIR="$REPO_ROOT/.cache/pip"
export HF_HOME="$REPO_ROOT/.cache/hf"
export TORCH_HOME="$REPO_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
export VLLM_CACHE_ROOT="$REPO_ROOT/.cache/vllm"
export CUDA_CACHE_PATH="$REPO_ROOT/.cache/cuda"
export MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$REPO_ROOT/.cache/numba"
export TMPDIR="$REPO_ROOT/.cache/tmp"

mkdir -p \
  "$HOME" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$UV_CACHE_DIR" \
  "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" \
  "$NUMBA_CACHE_DIR" "$TMPDIR"

# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
unset REPO_ROOT CONDA_ROOT ENV_NAME ENV_PREFIX
