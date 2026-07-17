#!/usr/bin/env bash
# Evaluate timestamp-grounded InternVideo3 on the complete reduced dev140 split.

#SBATCH --job-name=iv3-ts512-d140
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/internvideo3_timestamp512_fa2_dev140_2026-07-14"

cd "$PROJECT_ROOT"
source scripts/activate_internvideo3_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_DIR" slurm_logs

python baselines/longqa/run_internvideo3_timestamp_pilot.py \
  --input ../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file configs/egolongqa_dev140_seed20260709.json \
  --output "$RUN_DIR/predictions.jsonl" \
  --attn-implementation flash_attention_2 \
  --frames 512 \
  --min-pixels 65536 \
  --max-pixels 131072 \
  --max-new-tokens 192
