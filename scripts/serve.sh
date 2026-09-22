#!/usr/bin/env bash
set -euo pipefail

# Run on an allocated GPU inside the project environment. Settings match the
# reported runs; only Qwen3.5 used the reasoning parser and Triton GDN prefill
model="${1:-Qwen/Qwen3.5-27B}"
extra=()
if [[ "${model,,}" == *qwen3.5* ]]; then
    extra=(--reasoning-parser qwen3 --gdn-prefill-backend triton)
fi
export VLLM_FLASH_ATTN_VERSION=3
exec vllm serve "$model" \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 49152 \
    --max-logprobs 100 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":64}' \
    --mm-processor-kwargs '{"min_pixels":784,"max_pixels":451584}' \
    "${extra[@]}"
