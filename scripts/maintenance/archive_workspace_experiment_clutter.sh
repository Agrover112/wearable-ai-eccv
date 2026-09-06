#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARCHIVE_ROOT=${ROOT}/archives/workspace_cleanup_2026-09-01
MODE=${1:---dry-run}

if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--apply" ]]; then
    echo "Usage: $0 [--dry-run|--apply]" >&2
    exit 2
fi

if command -v squeue >/dev/null 2>&1 && squeue -h -u "$(id -un)" | grep -q .; then
    echo "Refusing archival while Slurm jobs are queued or running" >&2
    exit 1
fi

mkdir -p "${ARCHIVE_ROOT}/manifests"

SLURM_SCRIPTS_LIST=${ARCHIVE_ROOT}/manifests/root_slurm_scripts.txt
SLURM_LOGS_LIST=${ARCHIVE_ROOT}/manifests/slurm_logs.txt
RUNTIME_LOGS_LIST=${ARCHIVE_ROOT}/manifests/nested_runtime_logs.txt

find "${ROOT}" -maxdepth 1 -type f -name 'slurm*.sh' -printf '%P\n' \
    | LC_ALL=C sort > "${SLURM_SCRIPTS_LIST}"
find "${ROOT}/slurm_logs" -type f ! -name 'README.md' -printf '%P\n' \
    | LC_ALL=C sort > "${SLURM_LOGS_LIST}"
{
    find "${ROOT}/runs" -type f \
        \( -name 'vllm_server*.log' -o -name 'slurm.out' -o -name 'slurm.err' \) \
        ! -path '*/qwen35_27b_dual_view_fusion_v3_container_smoke5/*' \
        -printf 'runs/%P\n'
    find "${ROOT}/data/wearable-ai/starter_kit/output" -type f \
        \( -name 'vllm_server*.log' -o -name 'slurm.out' -o -name 'slurm.err' \) \
        -printf 'data/wearable-ai/starter_kit/output/%P\n'
    find "${ROOT}" -maxdepth 1 -type f \
        \( -name 'vllm_server*.log' -o -name '*.out' -o -name '*.err' \) \
        -printf '%f\n'
} | LC_ALL=C sort -u > "${RUNTIME_LOGS_LIST}"

count_and_bytes() {
    local kind=$1
    local base=$2
    local list=$3
    local count=0
    local bytes=0
    while IFS= read -r relative; do
        [[ -n "${relative}" ]] || continue
        local path
        path=${base}/${relative}
        [[ -f "${path}" ]] || continue
        count=$((count + 1))
        bytes=$((bytes + $(stat -c %s "${path}")))
    done < "${list}"
    printf '%s\t%s\t%s\n' "${kind}" "${count}" "${bytes}"
}

{
    printf 'category\tfiles\tbytes\n'
    count_and_bytes root_slurm_scripts "${ROOT}" "${SLURM_SCRIPTS_LIST}"
    count_and_bytes slurm_logs "${ROOT}/slurm_logs" "${SLURM_LOGS_LIST}"
    count_and_bytes nested_runtime_logs "${ROOT}" "${RUNTIME_LOGS_LIST}"
} | tee "${ARCHIVE_ROOT}/inventory.tsv"

if [[ "${MODE}" == "--dry-run" ]]; then
    echo "Dry run only. Archives would be written under ${ARCHIVE_ROOT}"
    exit 0
fi

archive_and_verify() {
    local base=$1
    local list=$2
    local archive=$3
    local expected
    expected=$(grep -c . "${list}" || true)
    if [[ "${expected}" == 0 ]]; then
        return
    fi
    tar -C "${base}" -czf "${archive}" --files-from "${list}"
    local archived
    archived=$(tar -tzf "${archive}" | grep -c . || true)
    [[ "${archived}" == "${expected}" ]] || {
        echo "Archive verification failed: ${archive}" >&2
        exit 1
    }
}

archive_and_verify "${ROOT}" "${SLURM_SCRIPTS_LIST}" \
    "${ARCHIVE_ROOT}/root_slurm_scripts.tar.gz"
archive_and_verify "${ROOT}/slurm_logs" "${SLURM_LOGS_LIST}" \
    "${ARCHIVE_ROOT}/slurm_logs.tar.gz"

cp "${RUNTIME_LOGS_LIST}" \
    "${ARCHIVE_ROOT}/manifests/nested_runtime_logs_archived.txt"
archive_and_verify "${ROOT}" \
    "${ARCHIVE_ROOT}/manifests/nested_runtime_logs_archived.txt" \
    "${ARCHIVE_ROOT}/nested_runtime_logs.tar.gz"

sha256sum "${ARCHIVE_ROOT}"/*.tar.gz > "${ARCHIVE_ROOT}/archive_sha256.txt"

while IFS= read -r relative; do
    [[ -n "${relative}" ]] && rm -f -- "${ROOT}/${relative}"
done < "${SLURM_SCRIPTS_LIST}"
while IFS= read -r relative; do
    [[ -n "${relative}" ]] && rm -f -- "${ROOT}/slurm_logs/${relative}"
done < "${SLURM_LOGS_LIST}"
while IFS= read -r relative; do
    [[ -n "${relative}" ]] || continue
    rm -f -- "${ROOT}/${relative}"
done < "${RUNTIME_LOGS_LIST}"

find "${ROOT}/slurm_logs" -depth -type d -empty -delete
mkdir -p "${ROOT}/slurm_logs"
cat > "${ROOT}/slurm_logs/README.md" <<'EOF'
# Slurm logs

Historical Slurm output was archived after the competition. New jobs may write
logs here. See `archives/workspace_cleanup_2026-09-01/slurm_logs.tar.gz` and its
manifest for the preserved historical files.
EOF

rm -f -- "${ROOT}/documentation/cleanup_records/cleanup_20260901T121943Z_before.tsv"

cat > "${ARCHIVE_ROOT}/README.md" <<'EOF'
# Historical experiment launchers and logs

This directory replaces hundreds of completed root-level launchers and runtime
logs with three verified compressed archives. Predictions, metrics, diagnostic
JSON, implementation source, the final container smoke run, and the final image
were not removed.

- `root_slurm_scripts.tar.gz`: historical root-level Slurm launchers.
- `slurm_logs.tar.gz`: historical scheduler stdout and stderr.
- `nested_runtime_logs.tar.gz`: vLLM server logs and copied Slurm logs from run
  directories, excluding the final v3 container smoke run.
- `manifests/`: original relative paths.
- `archive_sha256.txt`: archive checksums.

Restore an archive with `tar -xzf <archive> -C <target>`. Read its manifest first
because the three archives use different path roots.
EOF

echo "Workspace archival complete: ${ARCHIVE_ROOT}"
