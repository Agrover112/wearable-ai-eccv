#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${OUT:-${ROOT}/documentation/reproducibility_manifests}
CONDA_ROOT=${CONDA_ROOT:-/CT/NDF/work/miniforge3}
ENV_PREFIX=${ENV_PREFIX:-${CONDA_ROOT}/envs/wearable-ai}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/inf0/user/agaur/wai-26}
HF_HUB=${HF_HUB:-${SCRATCH_ROOT}/cache/huggingface/hub}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/wearable-ai/egolongqa}
STARTER=${STARTER:-${ROOT}/data/wearable-ai/starter_kit}
SUBMISSION=${SUBMISSION:-${ROOT}/test_submission/qwen35_27b_dual_view_fusion_v2}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${SCRATCH_ROOT}/test_submission/qwen35_27b_dual_view_fusion_v2/artifacts}
RECOVERY_ROOT=${RECOVERY_ROOT:-${ROOT}/archives/submission_recovery_2026-09-01}
FINAL_SMOKE=${FINAL_SMOKE:-${ROOT}/runs/egolongqa/qwen35_27b_dual_view_fusion_v3_container_smoke5}
NVILA_ENV=${NVILA_ENV:-${SCRATCH_ROOT}/envs/nvila-autogaze}

mkdir -p "${OUT}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "Wearable AI environment is missing: ${ENV_PREFIX}" >&2
    exit 1
fi

{
    printf 'captured_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -srmo)"
    printf 'workspace=%s\n' "${ROOT}"
    printf 'environment=%s\n' "${ENV_PREFIX}"
    printf 'python=%s\n' "$("${ENV_PREFIX}/bin/python" --version 2>&1)"
} > "${OUT}/host.txt"

"${CONDA_ROOT}/bin/conda" list -p "${ENV_PREFIX}" --explicit \
    > "${OUT}/wearable-ai-conda-explicit.txt"
"${CONDA_ROOT}/bin/conda" list -p "${ENV_PREFIX}" --json \
    > "${OUT}/wearable-ai-conda-list.json"
"${ENV_PREFIX}/bin/python" -m pip freeze --all \
    > "${OUT}/wearable-ai-pip-freeze.txt"
cp "${ENV_PREFIX}/conda-meta/history" "${OUT}/wearable-ai-conda-history.txt"

if [[ -x "${NVILA_ENV}/bin/python" ]]; then
    "${NVILA_ENV}/bin/python" -m pip freeze --all \
        > "${OUT}/nvila-autogaze-pip-freeze.txt"
    {
        printf 'environment=%s\n' "${NVILA_ENV}"
        printf 'python=%s\n' "$("${NVILA_ENV}/bin/python" --version 2>&1)"
        printf 'bytes=%s\n' "$(du -s -B1 "${NVILA_ENV}" | cut -f1)"
    } > "${OUT}/nvila-autogaze-environment.txt"
fi

if [[ -d "${HF_HUB}" ]]; then
python - "${HF_HUB}" "${OUT}/huggingface-model-snapshots.tsv" <<'PY'
import pathlib
import sys

hub = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
rows = ["model\trevision\tfiles\tlogical_bytes"]
for model in sorted(hub.glob("models--*")):
    snapshots = model / "snapshots"
    for snapshot in sorted(snapshots.iterdir()) if snapshots.is_dir() else []:
        if not snapshot.is_dir():
            continue
        files = 0
        size = 0
        for path in snapshot.rglob("*"):
            if not path.is_file():
                continue
            files += 1
            size += path.stat().st_size
        name = model.name.removeprefix("models--").replace("--", "/")
        rows.append(f"{name}\t{snapshot.name}\t{files}\t{size}")
output.write_text("\n".join(rows) + "\n")
PY
elif [[ -s "${OUT}/huggingface-model-snapshots.tsv" ]]; then
    echo "Hugging Face cache is absent; preserving the existing snapshot manifest"
else
    echo "Hugging Face cache is absent and no prior snapshot manifest exists" >&2
    exit 1
fi

if [[ -d "${DATA_ROOT}/val" ]]; then
    VIDEO_COUNT=$(find "${DATA_ROOT}/val" -maxdepth 1 -type f -name '*.mp4' | wc -l)
else
    VIDEO_COUNT=0
fi
if [[ "${VIDEO_COUNT}" == 700 ]]; then
python - "${DATA_ROOT}" "${OUT}/dataset-inventory.tsv" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
rows = ["path\tfiles\tbytes\tsha256_if_metadata"]
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(root)
    digest = ""
    if path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".txt"}:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(f"{relative}\t1\t{path.stat().st_size}\t{digest}")
output.write_text("\n".join(rows) + "\n")
PY
elif [[ -s "${OUT}/dataset-inventory.tsv" ]]; then
    echo "EgoLongQA videos are absent; preserving the existing dataset manifest"
else
    echo "EgoLongQA videos are absent and no prior dataset manifest exists" >&2
    exit 1
fi

find "${STARTER}" "${SUBMISSION}" -type f \
    \( -name '*.py' -o -name '*.sh' -o -name 'Containerfile*' -o -name '*.json' -o -name '*.md' \) \
    -print0 | sort -z | xargs -0 sha256sum \
    > "${OUT}/source-sha256.txt"

if [[ -f "${ARTIFACT_ROOT}/kth-saar-qwen35-dual-view-v3.tar" ]]; then
python - "${ARTIFACT_ROOT}" "${RECOVERY_ROOT}" "${OUT}/final-image-receipt.txt" <<'PY'
import pathlib
import sys

artifacts = pathlib.Path(sys.argv[1])
recovery = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
archive = artifacts / "kth-saar-qwen35-dual-view-v3.tar"
sidecar = artifacts / "kth-saar-qwen35-dual-view-v3.tar.sha256"
marker = recovery / "validation" / "CONTAINER_SMOKE_PASSED_V3"
reference = recovery / "artifacts" / "ecr_immutable_reference_v3.txt"
lines = [
    f"archive={archive}",
    f"archive_bytes={archive.stat().st_size if archive.exists() else 'MISSING'}",
    f"archive_sha256_record={sidecar.read_text().strip() if sidecar.exists() else 'MISSING'}",
    f"smoke_marker={marker.read_text().strip() if marker.exists() else 'MISSING'}",
    f"immutable_ecr_reference={reference.read_text().strip() if reference.exists() else 'MISSING'}",
]
output.write_text("\n".join(lines) + "\n")
PY
elif [[ -s "${OUT}/final-image-receipt.txt" ]]; then
    echo "Final OCI archive is absent; preserving the existing image receipt"
else
    echo "Final OCI archive is absent and no prior receipt exists" >&2
    exit 1
fi

python - "${FINAL_SMOKE}" "${OUT}/final-container-runtime.json" <<'PY'
import json
import pathlib
import sys

smoke = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
reports = list((smoke / "inference_diagnostics").glob("vllm_startup_*.json"))
if len(reports) != 1:
    raise SystemExit(f"expected one final vLLM startup report, found {len(reports)}")
report = json.loads(reports[0].read_text())
component = json.loads(
    (smoke / "inference_diagnostics" / "dual_view_startup.json").read_text()
)
result = {
    "python": report.get("python"),
    "platform": report.get("platform"),
    "packages": report.get("packages"),
    "command": report.get("command"),
    "environment": report.get("environment"),
    "gpus": report.get("gpus"),
    "startup_status": report.get("status"),
    "component_startup": component,
}
output.write_text(json.dumps(result, indent=2) + "\n")
PY

echo "Reproducibility manifests written to ${OUT}"
