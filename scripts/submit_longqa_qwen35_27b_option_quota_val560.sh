#!/bin/bash
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
cd "${ROOT}"

selection_job=$(sbatch --parsable slurm_longqa_qwen35_option_quota_globalfix_val560.sh)
answer_job=$(sbatch --parsable --dependency="afterok:${selection_job}" \
    slurm_longqa_qwen35_27b_option_quota_globalfix_val560.sh)
merge_job=$(sbatch --parsable --dependency="afterok:${answer_job}" \
    slurm_longqa_qwen35_27b_option_quota_merge.sh)

echo "Corrected option-quota selection job: ${selection_job}"
echo "Dependent Qwen3.5-27B answer job: ${answer_job}"
echo "Dependent merge/analysis job: ${merge_job}"
