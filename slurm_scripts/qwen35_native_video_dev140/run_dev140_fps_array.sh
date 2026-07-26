#!/usr/bin/env bash
#SBATCH --job-name=q35-native-video
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=0-1%1
#SBATCH --output=slurm_outputs/qwen35_native_video_dev140/%A_%a_%j.out
#SBATCH --error=slurm_outputs/qwen35_native_video_dev140/%A_%a_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VIDEO_FPS_VALUES=(0.25 0.5)
PROXY_WIDTHS=(480 352)
PROXY_HEIGHTS=(256 192)
VIDEO_FPS="${VIDEO_FPS_VALUES[$SLURM_ARRAY_TASK_ID]}"
PROXY_WIDTH="${PROXY_WIDTHS[$SLURM_ARRAY_TASK_ID]}"
PROXY_HEIGHT="${PROXY_HEIGHTS[$SLURM_ARRAY_TASK_ID]}"
FPS_TAG="${VIDEO_FPS/./p}"
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/qwen35_native_video_fps${FPS_TAG}_dev140"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export VLLM_RPC_TIMEOUT=900000
export VLLM_LOG_DIR="$RUN_DIR/vllm_logs"

mkdir -p "$RUN_DIR" "$VLLM_LOG_DIR" \
  "$PROJECT_ROOT/slurm_outputs/qwen35_native_video_dev140"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Video FPS: $VIDEO_FPS"
echo "Proxy resolution: ${PROXY_WIDTH}x${PROXY_HEIGHT}"
echo "Started at: $(date)"

python baselines/longqa/run_generate_longqa_native_video.py \
  --input "$PROJECT_ROOT/data/wearable_ai_2026_egolongqa_val_700.jsonl" \
  --video-folder "$PROJECT_ROOT/data/videos" \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  --llm-model Qwen/Qwen3.5-9B \
  --video-fps "$VIDEO_FPS" \
  --proxy-cache-dir "$PROJECT_ROOT/.cache/egolongqa/native_video_proxies" \
  --proxy-width "$PROXY_WIDTH" \
  --proxy-height "$PROXY_HEIGHT" \
  --require-proxies \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --tp 2 \
  --concurrency 1 \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json"

echo "Finished at: $(date)"
