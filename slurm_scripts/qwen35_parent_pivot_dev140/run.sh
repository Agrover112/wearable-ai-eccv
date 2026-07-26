#!/usr/bin/env bash
#SBATCH --job-name=q35-parent-pivot
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/qwen35_parent_pivot_dev140/%j.out
#SBATCH --error=slurm_outputs/qwen35_parent_pivot_dev140/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_NAME="${RUN_NAME:-qwen35_parent_temporal_pivot_dev140}"
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/$RUN_NAME"
GROUNDING_CACHE_DIR="${GROUNDING_CACHE_DIR:-$PROJECT_ROOT/.cache/egolongqa/qwen3vl_embedding_c128}"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-784}"
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-451584}"
export VLLM_QWEN_MAX_MODEL_LEN="${VLLM_QWEN_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
export VLLM_LOG_DIR="$RUN_DIR/vllm_logs"

mkdir -p "$RUN_DIR" "$VLLM_LOG_DIR" "$GROUNDING_CACHE_DIR" \
    "$PROJECT_ROOT/slurm_outputs/qwen35_parent_pivot_dev140"

echo "Job ID: $SLURM_JOB_ID"
echo "Run directory: $RUN_DIR"
echo "Started at: $(date)"

python baselines/longqa/run_generate_longqa_proofpack.py \
    --input "$PROJECT_ROOT/data/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --video-folder "$PROJECT_ROOT/data/videos" \
    --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json" \
    --strategy temporal_pivot \
    --candidate-frames 128 \
    --final-max-frames 64 \
    --grounding-output "$RUN_DIR/proofpack.jsonl" \
    --grounder-cache-dir "$GROUNDING_CACHE_DIR" \
    --grounder-model Qwen/Qwen3-VL-Embedding-8B \
    --grounder-device cuda \
    --grounder-batch-size 4 \
    --grounder-dtype bfloat16 \
    --model-type qwen \
    --llm-model Qwen/Qwen3.5-9B \
    --backend vllm \
    --tp 1 \
    --concurrency 1 \
    --output "$RUN_DIR/predictions.jsonl" \
    --eval-output "$RUN_DIR/results.json"

echo "Finished at: $(date)"
