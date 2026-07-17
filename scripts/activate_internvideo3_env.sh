#!/usr/bin/env bash
# Source this file to activate the InternVideo3-compatible inference environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run: source scripts/activate_internvideo3_env.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEARABLE_AI_ENV_NAME=internvideo3
# Conda's cuda-nvcc activation hook expands these variables under `set -u`.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
# shellcheck source=activate_local_env.sh
source "$SCRIPT_DIR/activate_local_env.sh"
unset WEARABLE_AI_ENV_NAME SCRIPT_DIR
