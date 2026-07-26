#!/usr/bin/env bash
#SBATCH --job-name=qwen35-fixed-pack-dev140
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/inf0/user/ssureshn/wearable-ai-eccv/slurm_outputs/qwen35_fixed_pack_dev140/%x_%j.out
#SBATCH --error=/scratch/inf0/user/ssureshn/wearable-ai-eccv/slurm_outputs/qwen35_fixed_pack_dev140/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$PROJECT_ROOT/scripts/activate_local_env.sh" || true
source "$PROJECT_ROOT/.venv/bin/activate"

FRAME_SOURCE="${FRAME_SOURCE:-uniform}"
PROOFPACK="${PROOFPACK:-}"
RUN_NAME="${RUN_NAME:-qwen35_fixed_uniform64_dev140}"
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/$RUN_NAME"
SUBSET_FILE="${SUBSET_FILE:-$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

if [[ "$FRAME_SOURCE" == "proofpack" && -z "$PROOFPACK" ]]; then
    echo "PROOFPACK must point to a prior proofpack.jsonl when FRAME_SOURCE=proofpack" >&2
    exit 2
fi
if [[ "$FRAME_SOURCE" != "uniform" && "$FRAME_SOURCE" != "proofpack" ]]; then
    echo "FRAME_SOURCE must be uniform or proofpack, got: $FRAME_SOURCE" >&2
    exit 2
fi
if [[ "$FRAME_SOURCE" == "proofpack" && ! -f "$PROOFPACK" ]]; then
    echo "PROOFPACK does not exist: $PROOFPACK" >&2
    exit 2
fi

mkdir -p "$RUN_DIR"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-784}"
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-451584}"
export VLLM_QWEN_MAX_MODEL_LEN="${VLLM_QWEN_MAX_MODEL_LEN:-65536}"
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
export VLLM_LOG_DIR="$RUN_DIR"

args=(
    --input "$PROJECT_ROOT/data/wearable_ai_2026_egolongqa_val_700.jsonl"
    --subset-file "$SUBSET_FILE"
    --video-folder "$PROJECT_ROOT/data/videos"
    --output "$RUN_DIR/predictions.jsonl"
    --frame-pack-output "$RUN_DIR/frame_packs.jsonl"
    --eval-output "$RUN_DIR/results.json"
    --frame-source "$FRAME_SOURCE"
    --frame-count 64
    --model-type qwen
    --llm-model Qwen/Qwen3.5-9B
    --backend vllm
    --tp 1
    --concurrency 1
)
if [[ "$FRAME_SOURCE" == "proofpack" ]]; then
    args+=(--proofpack "$PROOFPACK")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
    args+=(--max-samples "$MAX_SAMPLES")
fi

python "$PROJECT_ROOT/baselines/longqa/run_generate_longqa_fixed_pack.py" "${args[@]}"
