#!/usr/bin/env bash
#SBATCH --job-name=qwen35-img64-dev140
#SBATCH --account=gpu24_default
#SBATCH --partition=gpu24
#SBATCH --qos=gpu24_default
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/qwen3_5_0_8b_hf_images64_px451584_dev140_2026-07-30"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export QWEN35_ATTN_IMPLEMENTATION=flash_attention_2
export QWEN35_ENABLE_THINKING=0
export QWEN35_IMAGE_MIN_PIXELS=784
export QWEN35_IMAGE_MAX_PIXELS=451584
export QWEN35_SEED=0

mkdir -p "$RUN_DIR" slurm_logs

python baselines/longqa/run_generate_longqa.py \
  --input /CT/RGCAHead3/nobackup/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  --model-type qwen3_5_images \
  --backend hf \
  --num-gpus 1 \
  --max-frames 64 \
  --frames-per-interval 64 \
  --batch-size 1 \
  --prompt-variant qwen3_5 \
  --longqa-max-new-tokens 1024 \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json" \
  --no-resume-predictions
