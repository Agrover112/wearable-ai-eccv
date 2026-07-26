#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PARENT_PROOFPACK:?Set PARENT_PROOFPACK to the parent proofpack.jsonl before submitting.}"

cd "$PROJECT_ROOT"
mkdir -p slurm_outputs/qwen35_operator_adaptive_dev140
sbatch --noinfo --export=ALL,PARENT_PROOFPACK="$PARENT_PROOFPACK" \
    slurm_scripts/qwen35_operator_adaptive_dev140/run.sh
