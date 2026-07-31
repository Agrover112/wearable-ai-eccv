#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=/CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/qwen3_5_0_8b_hf_images64_temporal_pivot_c128_smoke2_local_sdpa_2026-07-30"
FEATURE_CACHE="$PROJECT_ROOT/.cache/longqa/siglip2_so400m_c128"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export QWEN35_ATTN_IMPLEMENTATION=sdpa
export QWEN35_ENABLE_THINKING=0
export QWEN35_IMAGE_MIN_PIXELS=784
export QWEN35_IMAGE_MAX_PIXELS=451584
export QWEN35_SEED=0

mkdir -p "$RUN_DIR" "$FEATURE_CACHE"

python baselines/longqa/run_generate_longqa_proofpack.py \
  --input /CT/RGCAHead3/nobackup/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  --max-samples 2 \
  --strategy temporal_pivot \
  --candidate-frames 128 \
  --pivot-centers 2 \
  --target-centers 8 \
  --eventlet-radius 1 \
  --anchor-k 24 \
  --bridge-k 8 \
  --final-max-frames 64 \
  --temporal-nms-seconds 10 \
  --fill-mode semantic_boundary \
  --grounder-model google/siglip2-so400m-patch14-384 \
  --grounder-device cuda \
  --grounder-batch-size 16 \
  --grounder-dtype bfloat16 \
  --grounder-cache-dir "$FEATURE_CACHE" \
  --model-type qwen3_5_images \
  --llm-model Qwen/Qwen3.5-0.8B \
  --backend hf \
  --tp 1 \
  --concurrency 1 \
  --prompt-variant qwen3_5 \
  --longqa-max-new-tokens 1024 \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json" \
  --grounding-output "$RUN_DIR/proofpack.jsonl" \
  --no-resume-grounding \
  --no-resume-predictions
