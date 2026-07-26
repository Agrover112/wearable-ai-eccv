#!/usr/bin/env bash
#SBATCH --job-name=q35-native-proxies
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=slurm_outputs/qwen35_native_video_dev140/proxy_%A_%a_%j.out
#SBATCH --error=slurm_outputs/qwen35_native_video_dev140/proxy_%A_%a_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VIDEO_FPS_VALUES=(0.25 0.5)
PROXY_WIDTHS=(480 352)
PROXY_HEIGHTS=(256 192)
VIDEO_FPS="${VIDEO_FPS_VALUES[$SLURM_ARRAY_TASK_ID]}"
PROXY_WIDTH="${PROXY_WIDTHS[$SLURM_ARRAY_TASK_ID]}"
PROXY_HEIGHT="${PROXY_HEIGHTS[$SLURM_ARRAY_TASK_ID]}"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python baselines/longqa/run_generate_longqa_native_video.py \
  --input "$PROJECT_ROOT/data/wearable_ai_2026_egolongqa_val_700.jsonl" \
  --video-folder "$PROJECT_ROOT/data/videos" \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  --video-fps "$VIDEO_FPS" \
  --proxy-cache-dir "$PROJECT_ROOT/.cache/egolongqa/native_video_proxies" \
  --proxy-width "$PROXY_WIDTH" \
  --proxy-height "$PROXY_HEIGHT" \
  --proxy-workers "$SLURM_CPUS_PER_TASK" \
  --prepare-proxies-only
