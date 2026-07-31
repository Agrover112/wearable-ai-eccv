#!/usr/bin/env bash
#SBATCH --job-name=q35-verifier-full
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
PIVOT_RUN="$PROJECT_ROOT/runs/egolongqa/qwen3_5_0_8b_hf_images64_temporal_pivot_c128_full700_local_sdpa_2026-07-31"
UNIFORM_RUN="$PROJECT_ROOT/runs/egolongqa/qwen3_5_0_8b_hf_images64_px451584_full700_local_sdpa_2026-07-31"
RUN_DIR="$PROJECT_ROOT/runs/egolongqa/qwen3_5_0_8b_hf_pivot_uniform_verifier_full700_h100_fa2_2026-07-31"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export QWEN35_ATTN_IMPLEMENTATION=flash_attention_2
export QWEN35_ENABLE_THINKING=0
export QWEN35_IMAGE_MIN_PIXELS=784
export QWEN35_IMAGE_MAX_PIXELS=451584
export QWEN35_SEED=0

mkdir -p "$RUN_DIR"

# Exercise the H100/FA2 path on two real disagreements before the full run
python baselines/longqa/run_generate_longqa_verifier.py \
  --input "$ANNOTATIONS" \
  --video-folder /scratch/inf0/user/kkumar/val \
  --primary-predictions "$PIVOT_RUN/predictions.jsonl" \
  --secondary-predictions "$UNIFORM_RUN/predictions.jsonl" \
  --primary-proofpack "$PIVOT_RUN/proofpack.jsonl" \
  --max-frames 64 \
  --proofpack-quota 32 \
  --max-disagreements 2 \
  --verifier-prompt baseline \
  --model-type qwen3_5_images \
  --llm-model Qwen/Qwen3.5-0.8B \
  --backend hf \
  --tp 1 \
  --concurrency 1 \
  --prompt-variant qwen3_5 \
  --longqa-max-new-tokens 1024 \
  --output "$RUN_DIR/smoke_predictions.jsonl" \
  --eval-output "$RUN_DIR/smoke_results.json" \
  --no-resume-predictions

python baselines/longqa/run_generate_longqa_verifier.py \
  --input "$ANNOTATIONS" \
  --video-folder /scratch/inf0/user/kkumar/val \
  --primary-predictions "$PIVOT_RUN/predictions.jsonl" \
  --secondary-predictions "$UNIFORM_RUN/predictions.jsonl" \
  --primary-proofpack "$PIVOT_RUN/proofpack.jsonl" \
  --max-frames 64 \
  --proofpack-quota 32 \
  --verifier-prompt baseline \
  --model-type qwen3_5_images \
  --llm-model Qwen/Qwen3.5-0.8B \
  --backend hf \
  --tp 1 \
  --concurrency 1 \
  --prompt-variant qwen3_5 \
  --longqa-max-new-tokens 1024 \
  --output "$RUN_DIR/predictions.jsonl" \
  --eval-output "$RUN_DIR/results.json"
