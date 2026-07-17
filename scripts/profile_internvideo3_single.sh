#!/usr/bin/env bash
# Profile a representative timestamp-sensitive EgoLongQA query on one H100.

set -euo pipefail

PROJECT_ROOT=/CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/internvideo3_timestamp_profile_2026-07-14"

cd "$PROJECT_ROOT"
source scripts/activate_internvideo3_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python baselines/longqa/profile_internvideo3.py \
  --input /CT/RGCAHead3/nobackup/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --video-name 31cdcd6a7135a92b.mp4 \
  --frames 512 1024 2048 \
  --min-pixels 65536 \
  --max-pixels 131072 \
  --max-new-tokens 256 \
  --attn-implementation flash_attention_2 \
  --output "$RUN_DIR/profile.json"
