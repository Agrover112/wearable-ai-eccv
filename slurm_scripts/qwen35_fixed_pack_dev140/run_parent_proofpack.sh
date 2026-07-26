#!/usr/bin/env bash
#SBATCH --job-name=q35-fixed-parent
#SBATCH --partition=gpu24
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/inf0/user/ssureshn/wearable-ai-eccv/slurm_outputs/qwen35_fixed_pack_dev140/%x_%j.out
#SBATCH --error=/scratch/inf0/user/ssureshn/wearable-ai-eccv/slurm_outputs/qwen35_fixed_pack_dev140/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export PROJECT_ROOT
export FRAME_SOURCE=proofpack
export PROOFPACK="$PROJECT_ROOT/runs/egolongqa/qwen35_parent_temporal_pivot_dev140/proofpack.jsonl"
export RUN_NAME=qwen35_fixed_parent_pivot_dev140

bash "$PROJECT_ROOT/slurm_scripts/qwen35_fixed_pack_dev140/run.sh"
