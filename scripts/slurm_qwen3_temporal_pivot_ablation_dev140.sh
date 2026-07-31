#!/usr/bin/env bash
#SBATCH --job-name=q3-pivot-ablate
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
ANNOTATIONS=/CT/RGCAHead3/nobackup/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
VIDEO_FOLDER=/scratch/inf0/user/kkumar/val
FEATURE_CACHE="$PROJECT_ROOT/.cache/longqa/siglip2_so400m_c128"
RETRIEVAL_QUERY_MODE="${1:-${RETRIEVAL_QUERY_MODE:-question}}"
PIVOT_MASK_MODE="${2:-${PIVOT_MASK_MODE:-primary}}"
MAX_SAMPLES="${3:-${MAX_SAMPLES:-}}"
RUN_NAME="${4:-${RUN_NAME:-qwen3_vl_8b_pivot_${RETRIEVAL_QUERY_MODE}_${PIVOT_MASK_MODE}_dev140_2026-07-31}}"
EXCLUDE_PIVOT_TARGET_OVERLAP="${5:-${EXCLUDE_PIVOT_TARGET_OVERLAP:-0}}"
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/$RUN_NAME"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export QWEN_MIN_PIXELS=784
export QWEN_MAX_PIXELS=451584
export VLLM_QWEN_MAX_MODEL_LEN=49152
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_LOG_DIR="$RUN_DIR"

mkdir -p "$RUN_DIR" "$FEATURE_CACHE" slurm_logs

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$EXCLUDE_PIVOT_TARGET_OVERLAP" == "1" ]]; then
  EXTRA_ARGS+=(--exclude-pivot-target-overlap)
fi

python baselines/longqa/run_generate_longqa_proofpack.py \
  --input "$ANNOTATIONS" \
  --video-folder "$VIDEO_FOLDER" \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  "${EXTRA_ARGS[@]}" \
  --strategy temporal_pivot \
  --retrieval-query-mode "$RETRIEVAL_QUERY_MODE" \
  --candidate-frames 128 \
  --pivot-centers 2 \
  --pivot-mask-mode "$PIVOT_MASK_MODE" \
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
  --model-type qwen \
  --llm-model Qwen/Qwen3-VL-8B-Instruct \
  --backend vllm \
  --tp 1 \
  --concurrency 8 \
  --prompt-variant baseline \
  --longqa-max-new-tokens 16 \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json" \
  --grounding-output "$RUN_DIR/proofpack.jsonl"
