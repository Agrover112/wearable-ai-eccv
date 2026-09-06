# Storage cleanup audit

**Audit date:** 2026-09-01  
**Scope:** `/CT/NDF/work/waw-26`, `/scratch/inf0/user/agaur/wai-26`, the former `/CT/HyperAvatar/work/wearable-ai-submission-recovery`, and accessible user-owned temporary paths.

## Safety status

The initial inventory was read-only. The organizers subsequently evaluated the
corrected image and the team finished fourth. The following compact final record
must remain intact:

- The v3 checksum sidecar and image receipt under
  `archives/submission_recovery_2026-09-01/` and
  `documentation/reproducibility_manifests/`
- `archives/submission_recovery_2026-09-01/artifacts/ecr_immutable_reference_v3.txt`
- `archives/submission_recovery_2026-09-01/validation/CONTAINER_SMOKE_PASSED_V3`
- Clean validation, build, and push logs in the same workspace archive
- `test_submission/qwen35_27b_dual_view_fusion_v2/`
- `test_submission/qwen35_27b_dual_view_fusion/SUBMISSION_RECEIPT.md` and the complete v1 submission material

The v3 archive has local SHA-256:

```text
d467137d5e3bfcd41dba817aa2e5f8643d72ff54fe39b5a3b764e63735f8db0d
```

The recorded immutable ECR reference is:

```text
510788268643.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/kth-saar@sha256:1408a9dd1e17af9017a9349e2f77e7998564295d8c21da06552f8dd86eeff5d0
```

## Measured overview

Sizes are apparent bytes reported by `du -sb` or `stat`, converted approximately to decimal GB where useful. A blank or `unmeasured` value means that the audit host could list the path but could not traverse every child; it is not evidence that the path is empty.

| Path/pattern | Measured size | Classification | Rationale | Exact cleanup command candidate |
|---|---:|---|---|---|
| `runs/` | 1,101,784,646 bytes (about 1.10 GB) | REVIEW | 285 experiment directories and 1,724 files. Contains the primary historical results, audits, partial runs, and logs used for the paper. | `find /CT/NDF/work/waw-26/runs -mindepth 1 -maxdepth 1 -type d -print` then remove only an explicitly approved run directory with `rm -rf --one-file-system -- <dir>` |
| `data/wearable-ai/starter_kit/output/` | 343,276,286 bytes (about 343 MB for `data/`) | REVIEW | Many outputs mirror files under `runs/`; some are the only copy of a result. Verify each pair before removing one side. | `cmp -s <runs-file> <output-file> && rm -- <output-file>` only after a manifest and result review |
| `runs/**` and `data/**/output/**` mirrored result files | Not summed separately; numerous same-size pairs observed | REVIEW | Same experiment names and files occur in both locations. Same size alone does not prove identical content. | `sha256sum <a> <b>` followed by removal of the approved duplicate |
| `slurm_logs/` | 1,376,082 bytes (about 1.4 MB) | COMPRESS | 152 `.out` and 152 `.err` files; historical logs are small but useful for reproducibility and failure analysis. | `tar -C /CT/NDF/work/waw-26 -czf /approved/archive/slurm_logs-<date>.tar.gz slurm_logs` |
| `slurm_logs/archive/` | 288 KB | COMPRESS | Already separated historical logs from July/August. | `tar -C /CT/NDF/work/waw-26 -czf /approved/archive/slurm_logs-archive.tar.gz slurm_logs/archive` |
| `team_share/egolongqa_ensemble_predictions_2026-07-31/` | 5,953,140 bytes (about 6.0 MB) | PROTECT | Deliberately curated handoff predictions and analysis for collaborators. | No command until the team confirms the handoff is no longer needed |
| `documentation/` | 279,787 bytes (about 280 KB) | PROTECT | Research notes, meeting briefs, implementation history, and this audit. | None |
| `wearable-ai-eccv/` working tree | 227,660,313 bytes (about 228 MB) | PROTECT | Separate project repository containing the maintained implementation, documentation, features, and history. | `git -C /CT/NDF/work/waw-26/wearable-ai-eccv status --short`; clean only through intentional Git/repository decisions |
| `wearable-ai-eccv/src/mamba/wheels/` | At least 80,000,000 bytes (about 80 MB) | REVIEW | HieraMamba-related wheels are present. They may be unused now, but are part of the separate repository and should not be removed automatically. | `find wearable-ai-eccv/src/mamba/wheels -type f -print`; remove only after confirming no environment or reproduction workflow needs them |
| Python `__pycache__/` and `*.pyc` | About 2.4 MB in the principal trees; exact total varies | DELETE-SAFE | Regenerable bytecode only. It is not part of the source, results, or submission image. | `find /CT/NDF/work/waw-26 -type d -name __pycache__ -prune -exec rm -rf -- {} +` |
| `*.partial*` and partial Slurm outputs | About 2 MB observed among the largest partial files | REVIEW | Most are interrupted experiments, but a partial file can be useful for recovery or diagnosing a timeout. | `find /CT/NDF/work/waw-26 -type f -name '*.partial*' -print`; delete only selected stale files with `rm -- <file>` |
| Former `/scratch/.../test_submission/qwen35_27b_dual_view_fusion_v2/artifacts/kth-saar-qwen35-dual-view-v3.tar` | 67,720,555,008 bytes (67.72 GB) | REMOVED | Full SHA-256, sidecar, source, smoke proof, build history, and immutable ECR digest were retained before scratch deletion. | Removed with scratch on 2026-09-01 |
| `/scratch/.../v2/artifacts/kth-saar-qwen35-dual-view-v2.tar` | 67,601,324,032 bytes (67.60 GB) | DELETE-SAFE | Superseded by the preserved and smoke-tested v3 archive. | Removed by `scripts/maintenance/cleanup_wearable_ai_storage.sh --apply` |
| `/scratch/.../v2/artifacts/crane-cache/sha256:3686bf...` | 67,601,268,460 bytes (67.60 GB) | DELETE-SAFE | Intermediate layer cache; v3 archive, digest, source, and smoke proof are retained. | Removed by the cleanup script |
| `/scratch/.../test_submission/qwen35_27b_dual_view_fusion_v1/` | 60,165,635,550 bytes (60.17 GB apparent) | DELETE-SAFE | Unpacked build context; source and digest history remain in the workspace. Model files share storage with the retained cache. | Removed by the cleanup script |
| `/scratch/.../v2/starter_kit/my_weights/` | About 60 GB apparent | DELETE-SAFE | Unpacked build context; exact core model snapshots remain in the Hugging Face cache and v3 archive. | Removed by the cleanup script |
| `/scratch/.../cache/huggingface/` | 326,376,578,557 bytes (326.38 GB) | REVIEW | Largest cache. It may contain duplicate model snapshots, blobs, and snapshots already copied into the submission tree. | `du -sb .../cache/huggingface; find .../cache/huggingface -maxdepth 4 -type f -printf '%s\\t%p\\n'`; prune only verified duplicate snapshots with the Hugging Face cache tooling |
| `/scratch/.../cache/triton/` | Not measurable from this audit host | REGENERABLE | Triton compilation cache is regenerable, but may reduce startup time for future tests. | `find .../cache/triton -type f -print`; remove only when no job is running, using `rm -rf -- <approved cache>` |
| `/scratch/.../cache/cuda/` | 3,988 bytes measured | REGENERABLE | Small CUDA cache/configuration material. | `rm -rf -- /scratch/inf0/user/agaur/wai-26/cache/cuda` after checking no active job uses it |
| `/scratch/.../cache/flashinfer_workspace/` | 76,518 bytes measured | REGENERABLE | Compiled workspace cache; regenerable. | `rm -rf -- /scratch/inf0/user/agaur/wai-26/cache/flashinfer_workspace` |
| `/scratch/.../envs/nvila-autogaze/` | 14,840,866,304 bytes (14.84 GB) | DELETE-SAFE | Exploratory environment; Python version and full pip freeze are preserved in the reproducibility manifests. | Removed by the cleanup script |
| Former `/CT/HyperAvatar/work/wearable-ai-submission-recovery/` | 115,529,728 bytes at final measurement | REMOVED | Safe references, smoke proof, and clean logs were copied into the workspace; BuildKit binaries, stale process files, and one credential-bearing failed log were discarded. | Removed on 2026-09-01 |
| `/tmp` accessible from the audit host | 480 MB total | REVIEW | Includes small MCP/system directories, ECR tooling, build remnants, and temporary test directories. | `find /tmp -maxdepth 2 -user "$USER" -printf '%s\\t%p\\n'`; remove only identified stale paths, never shared/system directories |
| User-owned `/tmp` paths | Not fully measurable here because the audit shell could not resolve the user name for `ps`/ownership checks | REVIEW | Ownership must be checked from the user’s normal login shell before cleanup. | `find /tmp -xdev -user "$(id -un)" -printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' | sort -nr` |

## Important duplicate and retention notes

1. The same experiment outputs appear under both `runs/egolongqa/` and `data/wearable-ai/starter_kit/output/egolongqa/`. Treat `runs/` as the human-facing canonical record unless a result exists only under `data/`; verify content with SHA-256 before removing a mirror.
2. The v2 scratch directory contains two complete archives, a base layer cache, and the unpacked model/source tree. These are related, but they are not interchangeable until the v3 digest, local checksum, and a rebuild/recovery test have been recorded elsewhere.
3. Model snapshots in the Hugging Face cache may be hard-linked or content-addressed. Do not use a raw `rm` against cache blobs while any environment or submission tree may refer to them. First run the cache’s own information/prune tooling and inspect link counts.
4. The Slurm logs are not a storage problem at present: the entire repository log area is about 1.4 MB. Compression is useful for tidiness, not for meaningful capacity recovery.
5. No running Slurm job was observed during this audit. Cleanup should still begin with a fresh `squeue -u "$USER"` check because jobs can start after this document was written.

## Estimated recoverable space

These are planning estimates, not deletion approvals. They exclude all protected submission material.

| Confidence | Candidate space | Basis | Conditions |
|---|---:|---|---|
| High | About 2-5 MB in repository bytecode/partial files | `__pycache__`, `.pyc`, and clearly identified incomplete files | Confirm no process is using a partial output and retain any needed failure evidence |
| High | About 135 GB | Previous v2 archive and crane layer cache | v3 archive, checksum, smoke proof, immutable ECR reference, and source are retained |
| High | About 216 GB | Exploratory Hugging Face models and the legacy Transformers cache | Exact model revisions were captured before pruning; core Qwen/SigLIP2 baselines remain cached |
| High | About 14.8 GB | NVILA/AutoGaze environment | Full pip freeze and source revision were captured before removal |
| Medium | Up to about 80 MB | HieraMamba wheel files and about 111 MB BuildKit tools, if no longer needed | Confirm with the team and preserve an archive if the paper/reproduction plan needs them |
| Low/variable | Up to about 480 MB | User-owned temporary files and stale build tooling under `/tmp` | Ownership and active-process checks must be performed from the normal user shell |

The recoverable space was dominated by the scratch model and image caches, not
the repository or Slurm logs. Cleanup therefore targeted only explicit cache,
environment, and image-build paths after preserving their manifests.

## Cleanup result

The cleanup ran on 2026-09-01 after the final result was known. It removed
501,925,743,616 apparent bytes across the explicit target paths. This is not a
physical-space measurement because the two unpacked image trees used hard
links. The large independent removals were the superseded v2 archive and crane
layer, exploratory Hugging Face checkpoints and legacy cache, and the 14.84 GB
NVILA environment.

At the intermediate point after selective pruning, the Hugging Face hub occupied
94,996,862,464 bytes and contained the core Qwen3.5-27B, Qwen3-VL/Qwen2.5,
SigLIP2, and Grounding DINO assets. The submission artifact directory contained
only the 67,720,555,008-byte v3 archive and its checksum sidecar. The complete
project scratch tree was subsequently removed as recorded below. Exact before,
after, and deletion records are under `documentation/cleanup_records/`.

### Workspace archival

A second pass addressed file-count clutter without removing experiment results.
It compressed 234 historical root Slurm launchers, 304 scheduler logs, and 704
nested vLLM/Slurm text logs into three archives under
`archives/workspace_cleanup_2026-09-01/`. The archives retain original relative
paths, have SHA-256 checksums, and were listed successfully before the originals
were removed. Predictions, result summaries, evidence packs, structured
diagnostics, source code, and the final v3 container smoke logs remain unpacked.
Forty-five empty run/output directories were also removed.

### HyperAvatar cleanup

The Wearable AI recovery directory under `/CT/HyperAvatar/work` was removed
after its safe records were copied to
`archives/submission_recovery_2026-09-01/` and verified with SHA-256. The first
failed build log was deliberately excluded because it contained expired cloud
credentials and signed URLs. No other HyperAvatar project directory was
modified.

### Complete scratch deletion

The complete `/scratch/inf0/user/agaur/wai-26` tree was removed on 2026-09-01.
It occupied 392,437,898,240 bytes immediately before deletion: 218.57 GB of
EgoLongQA data, 106.13 GB of model and package caches, the 67.72 GB final OCI
archive, and about 15 MB of exploratory source and latency caches. The exact
before inventory and deletion receipt are under `documentation/cleanup_records/`.
Dataset restoration, exact model revisions, the AutoGaze source revision, and
the image creation history are documented in `SCRATCH_ASSET_RESTORATION.md`.

## Applied cleanup sequence

1. Preserve the v3 ECR reference, local checksum, smoke marker, diagnostics, build source, and receipts.
2. Capture the development and NVILA environments, dataset inventory, source hashes, model revisions, and final container runtime.
3. Retain `runs/`, `analysis/`, starter-kit outputs, Slurm logs, and the separate `wearable-ai-eccv` repository for paper analysis.
4. Remove only the explicit superseded image layers, unpacked build contexts, exploratory environments/models, bytecode, and known credential caches listed in the cleanup script.
5. Write before/after manifests and a cleanup receipt under `documentation/cleanup_records/`.
