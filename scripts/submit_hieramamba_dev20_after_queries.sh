#!/bin/bash

# Submit the remaining HieraMamba stages with strict success dependencies.
set -euo pipefail

ROOT=/CT/NDF/work/waw-26
cd "${ROOT}"

bootstrap_submit=$(sbatch --parsable slurm_hieramamba_bootstrap_h100.sh)
bootstrap_job=${bootstrap_submit%%;*}
extract_submit=$(sbatch --parsable --dependency="afterok:${bootstrap_job}" \
    slurm_hieramamba_extract_dev20.sh)
extract_job=${extract_submit%%;*}
infer_submit=$(sbatch --parsable --dependency="afterok:${extract_job}" \
    slurm_hieramamba_infer_dev20.sh)
infer_job=${infer_submit%%;*}

printf 'Bootstrap: %s\nExtraction: %s (afterok:%s)\nInference: %s (afterok:%s)\n' \
    "${bootstrap_job}" \
    "${extract_job}" "${bootstrap_job}" \
    "${infer_job}" "${extract_job}"
