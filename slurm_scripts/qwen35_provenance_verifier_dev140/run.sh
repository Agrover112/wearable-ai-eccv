#!/usr/bin/env bash
#SBATCH --job-name=q35-provenance-verifier
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/inf0/user/ssureshn/wearable-ai-eccv/slurm_outputs/qwen35_provenance_verifier_dev140/%x_%j.out
#SBATCH --error=/scratch/inf0/user/ssureshn/wearable-ai-eccv/slurm_outputs/qwen35_provenance_verifier_dev140/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_NAME="${RUN_NAME:-qwen35_provenance_verifier_dev140}"
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/$RUN_NAME"

PRIMARY_PREDICTIONS="${PRIMARY_PREDICTIONS:-$PROJECT_ROOT/runs/egolongqa/qwen35_parent_temporal_pivot_dev140/predictions.jsonl}"
SECONDARY_PREDICTIONS="${SECONDARY_PREDICTIONS:-$PROJECT_ROOT/runs/egolongqa/qwen35_fixed_uniform64_dev140/predictions.jsonl}"
PRIMARY_PROOFPACK="${PRIMARY_PROOFPACK:-$PROJECT_ROOT/runs/egolongqa/qwen35_parent_temporal_pivot_dev140/proofpack.jsonl}"

if [[ ! -f "$PRIMARY_PREDICTIONS" ]]; then
    echo "PRIMARY_PREDICTIONS does not exist: $PRIMARY_PREDICTIONS" >&2
    exit 2
fi
if [[ ! -f "$SECONDARY_PREDICTIONS" ]]; then
    echo "SECONDARY_PREDICTIONS does not exist: $SECONDARY_PREDICTIONS" >&2
    exit 2
fi
if [[ ! -f "$PRIMARY_PROOFPACK" ]]; then
    echo "PRIMARY_PROOFPACK does not exist: $PRIMARY_PROOFPACK" >&2
    exit 2
fi

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

mkdir -p "$RUN_DIR" "$VLLM_LOG_DIR" \
    "$PROJECT_ROOT/slurm_outputs/qwen35_provenance_verifier_dev140"

args=(
    --input "$PROJECT_ROOT/data/wearable_ai_2026_egolongqa_val_700.jsonl"
    --video-folder "$PROJECT_ROOT/data/videos"
    --subset-file "$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json"
    --primary-predictions "$PRIMARY_PREDICTIONS"
    --secondary-predictions "$SECONDARY_PREDICTIONS"
    --primary-proofpack "$PRIMARY_PROOFPACK"
    --output "$RUN_DIR/predictions.jsonl"
    --evidence-output "$RUN_DIR/verifier_evidence.jsonl"
    --eval-output "$RUN_DIR/results.json"
    --llm-model Qwen/Qwen3.5-9B
    --tp 1
    --concurrency 1
)
if [[ "${THINKING:-0}" == "1" ]]; then
    args+=(--thinking)
fi

python baselines/longqa/run_generate_longqa_provenance_verifier.py "${args[@]}"
