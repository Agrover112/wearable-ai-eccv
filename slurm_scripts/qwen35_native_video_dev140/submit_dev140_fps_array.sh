#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p slurm_outputs/qwen35_native_video_dev140
sbatch --noinfo slurm_scripts/qwen35_native_video_dev140/run_dev140_fps_array.sh
