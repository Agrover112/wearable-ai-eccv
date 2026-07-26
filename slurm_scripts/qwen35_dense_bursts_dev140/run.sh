#!/usr/bin/env bash
#SBATCH --job-name=q35-bursts-dev140
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_outputs/qwen35_dense_bursts_dev140/%x_%j.out
#SBATCH --error=slurm_outputs/qwen35_dense_bursts_dev140/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_NAME="${RUN_NAME:-qwen35_dense_bursts_dev140}"
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/$RUN_NAME"

PROOFPACK="${PROOFPACK:-$PROJECT_ROOT/runs/egolongqa/qwen35_parent_temporal_pivot_dev140/proofpack.jsonl}"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export QWEN_MAX_PIXELS=451584
export VLLM_QWEN_MAX_MODEL_LEN=65536
export VLLM_MAX_NUM_BATCHED_TOKENS=65536
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_GPU_MEMORY_UTILIZATION=0.92
export VLLM_LOG_DIR="$RUN_DIR/vllm_logs"

mkdir -p "$RUN_DIR" "$PROJECT_ROOT/slurm_outputs/qwen35_dense_bursts_dev140"

python baselines/longqa/run_generate_longqa_dense_bursts.py \
  --input "$PROJECT_ROOT/data/wearable_ai_2026_egolongqa_val_700.jsonl" \
  --video-folder "$PROJECT_ROOT/data/videos" \
  --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
  --parent-proofpack "$PROOFPACK" \
  --frame-metadata-output "$RUN_DIR/frame_metadata.jsonl" \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json" \
  --model-type qwen \
  --llm-model Qwen/Qwen3.5-9B \
  --backend vllm \
  --tp 1 \
  --concurrency 1 \
  --max-frames 64
