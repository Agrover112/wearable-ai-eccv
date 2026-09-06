# Scratch asset restoration

This document records the external assets removed from
`/scratch/inf0/user/agaur/wai-26` on 2026-09-01. The workspace retains source,
predictions, result summaries, diagnostics, paper material, environment
manifests, model revisions, and container provenance. Scratch can be rebuilt
from the commands below.

## Inventory at deletion

| Scratch component | Bytes | Purpose |
|---|---:|---|
| `data/` | 218,568,813,056 | EgoLongQA annotations and 700 validation videos |
| `cache/` | 106,133,596,672 | Hugging Face model snapshots and UV package cache |
| `test_submission/` | 67,720,557,568 | Final v3 OCI archive and checksum sidecar |
| `latency_smoke/` | 13,513,728 | Regenerable SigLIP2 and uncertainty caches |
| `src/` | 1,416,704 | AutoGaze Git checkout |
| `envs/` | 0 | Empty after the earlier NVILA environment cleanup |
| **Total** | **392,437,898,240** | **Scratch tree before deletion** |

The before-deletion inventory is also stored in
`documentation/cleanup_records/scratch_deletion_20260901_before.tsv`.

## EgoLongQA dataset

Source repository: `facebook/wearable-ai` on Hugging Face.

The deleted local subset contained:

- `egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl`;
- 700 MP4 files under `egolongqa/val/`;
- 218,568,813,056 bytes in the scratch data tree;
- annotation SHA-256
  `e42d1d3de86dd15de73ff7e303652000b156f384c0f28700b49116043803436f`.

The original download command was not recorded. The starter kit documents a
full Git LFS clone. For this project, the smaller equivalent reconstruction is:

```bash
export HF_HOME=/scratch/inf0/user/agaur/wai-26/cache/huggingface
hf download facebook/wearable-ai \
  --repo-type dataset \
  --include 'egolongqa/**' 'README.md' 'LICENSE' \
  --local-dir /scratch/inf0/user/agaur/wai-26/data/wearable-ai
```

The restoration script verifies the annotation checksum and 700-video count,
then links `data/wearable-ai/egolongqa` to the restored scratch directory:

```bash
bash scripts/maintenance/restore_wearable_ai_assets.sh --dataset
```

The upstream dataset revision was not captured. The checksum check prevents a
silently changed annotation file from being treated as the original validation
set. `documentation/reproducibility_manifests/dataset-inventory.tsv` preserves
all 700 video names and byte sizes.

## Models

The final cache contained five exact snapshots:

| Model and revision | Use in this project |
|---|---|
| `Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654` | Final answer model and 27B baselines |
| `google/siglip2-so400m-patch14-384@e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | Final question-conditioned frame retrieval |
| `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Earlier Qwen3-VL baselines and temporal experiments |
| `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5` | Initial baseline and comparison experiments |
| `IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71` | Targeted object and crop experiments |

Restore the final two checkpoints with:

```bash
bash scripts/maintenance/restore_wearable_ai_assets.sh --core-models
```

Restore the three retained baselines with:

```bash
bash scripts/maintenance/restore_wearable_ai_assets.sh --baselines
```

Revisions for checkpoints removed during the earlier exploratory cleanup remain
in `documentation/reproducibility_manifests/huggingface-model-snapshots.tsv`.

## Software environments and caches

The main Conda environment remains under `/CT/NDF/work/miniforge3/envs/wearable-ai`;
it was not part of scratch. Its exact Conda and pip manifests are in
`documentation/reproducibility_manifests/`.

The 11.13 GB UV cache, Hugging Face cache, latency features, Triton products,
and other compiler/runtime caches were regenerable and were not archived.

The removed AutoGaze checkout was at revision
`ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab`. Restore it with:

```bash
bash scripts/maintenance/restore_wearable_ai_assets.sh --autogaze
```

The removed NVILA/AutoGaze environment is recorded in
`nvila-autogaze-pip-freeze.txt` and `nvila-autogaze-environment.txt` under the
reproducibility manifests.

## Final submission image

The deleted archive was:

```text
kth-saar-qwen35-dual-view-v3.tar
67,720,555,008 bytes
SHA-256 d467137d5e3bfcd41dba817aa2e5f8643d72ff54fe39b5a3b764e63735f8db0d
```

Its checksum sidecar, smoke marker, clean recovery logs, immutable ECR
reference, and build receipts remain in
`archives/submission_recovery_2026-09-01/`. The immutable registry reference was:

```text
510788268643.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/kth-saar@sha256:1408a9dd1e17af9017a9349e2f77e7998564295d8c21da06552f8dd86eeff5d0
```

The creation sequence was:

1. `test_submission/qwen35_27b_dual_view_fusion/prepare_build_context.sh`
   staged the exact Qwen3.5-27B and SigLIP2 revisions with starter-kit source.
2. `build_oci_archive.sh` built the original OCI image from the pinned CUDA
   12.8.1 Ubuntu 22.04 base.
3. `test_submission/qwen35_27b_dual_view_fusion_v2/create_patched_image.sh`
   added the audited dual-view source and runtime configuration to the immutable
   v1 ECR image.
4. `test_phase/Containerfile.compiler_repair` added `build-essential` and
   Python headers required by Triton.
5. `slurm_repair_and_container_smoke_2xh100.sh` built and tested v3 on two H100s.
6. `push_validated_image.sh` validated the local archive and pushed the exact
   digest recorded above.

The complete commands, failures, runtime settings, and validation gates are in
`REPRODUCIBILITY_GUIDE.md` and
`test_submission/qwen35_27b_dual_view_fusion_v2/RECOVERY_RUNBOOK.md`.

Deleting the local OCI archive means byte-for-byte recovery now depends on the
registry retaining the immutable digest. The source and pinned model revisions
are sufficient to rebuild an equivalent image, but a fresh build is not
guaranteed to produce the same archive SHA-256 because layer metadata and the
base registry state can differ.
