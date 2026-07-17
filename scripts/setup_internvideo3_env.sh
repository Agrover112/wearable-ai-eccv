#!/usr/bin/env bash
# Create the project-local InternVideo3 inference environment with uv.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-/CT/RelightableGaussianCodecAvatarsHead/work/miniforge3/bin/conda}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
ENV_PREFIX="$PROJECT_ROOT/.conda/envs/internvideo3"

mkdir -p \
  "$PROJECT_ROOT/.conda/envs" \
  "$PROJECT_ROOT/.cache/home" \
  "$PROJECT_ROOT/.cache/xdg" \
  "$PROJECT_ROOT/.cache/conda/pkgs" \
  "$PROJECT_ROOT/.cache/uv" \
  "$PROJECT_ROOT/.cache/pip" \
  "$PROJECT_ROOT/.cache/tmp"

export HOME="$PROJECT_ROOT/.cache/home"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export CONDA_PKGS_DIRS="$PROJECT_ROOT/.cache/conda/pkgs"
export CONDA_ENVS_PATH="$PROJECT_ROOT/.conda/envs"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" python=3.10
fi

"$UV_BIN" pip install \
  --python "$ENV_PREFIX/bin/python" \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

"$UV_BIN" pip install \
  --python "$ENV_PREFIX/bin/python" \
  -r "$PROJECT_ROOT/baselines/longqa/requirements-internvideo3.txt"

# Transformers uses TorchCodec only for CPU video extraction in this pipeline.
"$UV_BIN" pip install \
  --python "$ENV_PREFIX/bin/python" \
  --no-deps \
  torchcodec==0.10.0 \
  --index-url https://download.pytorch.org/whl/cpu

# FlashAttention has no wheel for this Torch release, so build its H100 kernels locally.
"$CONDA_BIN" install --yes --prefix "$ENV_PREFIX" \
  --channel nvidia \
  cuda-nvcc=12.8
"$UV_BIN" pip install \
  --python "$ENV_PREFIX/bin/python" \
  ninja packaging psutil
CUDA_HOME="$ENV_PREFIX" \
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}" \
MAX_JOBS="${MAX_JOBS:-8}" \
"$UV_BIN" pip install \
  --python "$ENV_PREFIX/bin/python" \
  --no-build-isolation \
  flash-attn==2.8.3.post1

echo "InternVideo3 environment ready."
echo "Activate it with: source scripts/activate_internvideo3_env.sh"
