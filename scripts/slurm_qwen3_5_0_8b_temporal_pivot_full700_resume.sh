#!/usr/bin/env bash
#SBATCH --job-name=q35-pivot-full
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

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export QWEN35_ATTN_IMPLEMENTATION=flash_attention_2

# The 560 proof packs are complete and prediction generation resumes from the local prefix
bash scripts/run_qwen3_5_0_8b_images64_temporal_pivot_full700_local.sh
