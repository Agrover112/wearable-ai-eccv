#!/usr/bin/env bash
#SBATCH --job-name=iv3-dev140
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13"

cd "$PROJECT_ROOT"
source scripts/activate_internvideo3_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export INTERNVIDEO3_REVISION=c4602918b65225650d152db2850fe34e01d21fcd
export VISION_MIN_PIXELS=262144
export VISION_MAX_PIXELS=451584

mkdir -p "$RUN_DIR"

python baselines/longqa/run_generate_longqa.py \
  --input /CT/RGCAHead3/nobackup/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  --model-type internvideo3 \
  --backend hf \
  --num-gpus 1 \
  --max-frames 64 \
  --frames-per-interval 64 \
  --batch-size 1 \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json"
