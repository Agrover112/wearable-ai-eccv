#!/usr/bin/env bash
#SBATCH --job-name=longqa-evidence-audit
#SBATCH --partition=cpu20
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/longqa_evidence_recall_audit_dev140/%j.out
#SBATCH --error=slurm_outputs/longqa_evidence_recall_audit_dev140/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_NAME="${RUN_NAME:-longqa_evidence_recall_audit_qwen35_dev140}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJECT_ROOT/runs/egolongqa/$RUN_NAME}"
INPUT="${INPUT:-$PROJECT_ROOT/data/wearable_ai_2026_egolongqa_val_700.jsonl}"
SUBSET_FILE="${SUBSET_FILE:-$PROJECT_ROOT/configs/egolongqa_dev140_seed20260709.json}"
VIDEO_FOLDER="${VIDEO_FOLDER:-$PROJECT_ROOT/data/videos}"
FINAL_PACK_METADATA="${FINAL_PACK_METADATA:-}"
PROOFPACK="${PROOFPACK:-$PROJECT_ROOT/runs/egolongqa/qwen35_parent_temporal_pivot_dev140/proofpack.jsonl}"
PRIMARY_PREDICTIONS="${PRIMARY_PREDICTIONS:-$PROJECT_ROOT/runs/egolongqa/qwen35_parent_temporal_pivot_dev140/predictions.jsonl}"
SECONDARY_PREDICTIONS="${SECONDARY_PREDICTIONS:-$PROJECT_ROOT/runs/egolongqa/qwen35_fixed_uniform64_dev140/predictions.jsonl}"
ANNOTATIONS="${ANNOTATIONS:-}"
SELECTION_MODE="${SELECTION_MODE:-disagreements_or_errors}"
PREVIEW_MODE="${PREVIEW_MODE:-contact_sheets}"

cd "$PROJECT_ROOT"
source scripts/activate_local_env.sh || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mkdir -p "$ARTIFACT_DIR" "$PROJECT_ROOT/slurm_outputs/longqa_evidence_recall_audit_dev140"

args=(
  --input "$INPUT"
  --subset-file "$SUBSET_FILE"
  --candidate-metadata "$PROOFPACK"
  --video-folder "$VIDEO_FOLDER"
  --artifact-dir "$ARTIFACT_DIR"
  --selection-mode "$SELECTION_MODE"
  --preview-mode "$PREVIEW_MODE"
  --tolerance-seconds "${TOLERANCE_SECONDS:-1.0}"
)
if [[ -n "$FINAL_PACK_METADATA" ]]; then
  args+=(--final-pack-metadata "$FINAL_PACK_METADATA")
fi
if [[ -n "$PRIMARY_PREDICTIONS" ]]; then
  args+=(--primary-predictions "$PRIMARY_PREDICTIONS")
fi
if [[ -n "$SECONDARY_PREDICTIONS" ]]; then
  args+=(--secondary-predictions "$SECONDARY_PREDICTIONS")
fi
if [[ -n "$ANNOTATIONS" ]]; then
  args+=(--annotations "$ANNOTATIONS")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${PREVIEW_MAX_FRAMES:-}" ]]; then
  args+=(--preview-max-frames "$PREVIEW_MAX_FRAMES")
fi

python baselines/longqa/run_longqa_evidence_recall_audit.py "${args[@]}"
