#!/usr/bin/env bash
#SBATCH --job-name=lfm25vl-dev140
#SBATCH --account=gpu22_ct-prio
#SBATCH --partition=gpu22
#SBATCH --qos=gpu22_default
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/lfm2_5_vl_1_6b_hf_uniform64_dev140_2026-07-30"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export LFM_MIN_IMAGE_TOKENS=64
export LFM_MAX_IMAGE_TOKENS=256

mkdir -p "$RUN_DIR" slurm_logs

python baselines/longqa/run_generate_longqa.py \
  --input /CT/RGCAHead3/nobackup/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  --model-type lfm2_5_vl \
  --backend hf \
  --num-gpus 1 \
  --max-frames 64 \
  --frames-per-interval 64 \
  --batch-size 1 \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json"
