# Wearable AI LongQA reproducibility guide

This guide records the implementation actually present in this workspace, including
the final test-phase container. It is intended to make a future rebuild possible
without relying on chat history. Facts come from source files, run metadata,
`RUN_LOG.md`, generated manifests, or preserved recovery artifacts. Unknown
facts are labelled rather than guessed.

## 1. Scope and results

The project targets the Wearable AI Workshop EgoLongQA task: answer a multiple-choice
question about a long egocentric video with one of A, B, C, or D.

The main final validation results are:

- Qwen3.5-27B, endpoint-inclusive uniform 64 frames: 608/700 = 86.86%.
- Its held-out 560-row partition: 478/560 = 85.36%.
- Exploratory label-free probability fusion: 618/700 = 88.29%.
- Later exploratory candidate-agreement analysis: 620/700 = 88.57%.

The last two are analyses over multiple prediction and evidence artifacts. They
are not the direct single-model behavior of the submitted container. The final
test container used the dual-view method described in Section 5. The earlier
endpoint-only submission is documented in
`documentation/LONGQA_SUBMISSION_QWEN35_ENDPOINT_2026-08-07.md`.

## 2. Layout

The repository root is /CT/NDF/work/waw-26.

| Path | Purpose |
|---|---|
| data/wearable-ai/starter_kit/ | Python implementation, requirements, and evaluator |
| data/wearable-ai/egolongqa/ | Dataset placeholder and restoration instructions |
| configs/ | Development subsets and split definitions |
| scripts/ | Experiment, evaluation, merge, audit, and export helpers |
| archives/workspace_cleanup_2026-09-01/root_slurm_scripts.tar.gz | Historical Slurm experiment launchers |
| archives/workspace_cleanup_2026-09-01/slurm_logs.tar.gz | Historical scheduler stdout/stderr |
| slurm_logs/ | Destination for logs from any new jobs |
| runs/egolongqa/ | Archived predictions, evidence, diagnostics, and summaries |
| analysis/ | Derived comparisons and error analyses |
| documentation/ | Reports and experiment notes |
| test_submission/qwen35_27b_dual_view_fusion/ | Original v1 test submission |
| test_submission/qwen35_27b_dual_view_fusion_v2/ | Recovery and v2/v3 build scripts |
| documentation/SCRATCH_ASSET_RESTORATION.md | Recreate deleted scratch datasets, models, and image inputs |
| archives/submission_recovery_2026-09-01/ | Final smoke marker, clean recovery logs, and ECR references |

The starter kit expects `starter_kit/` beside `egolongqa/`. The dataset is no
longer stored locally. The restoration script recreates the scratch data and
links it at the expected workspace path.

`documentation/reproducibility_manifests/dataset-inventory.tsv` records every
local dataset file, its byte size, and SHA-256 for annotation and metadata
files. It contains 702 entries: 700 validation videos and two metadata files.
The upstream dataset revision and original download command remain unknown.
`SCRATCH_ASSET_RESTORATION.md` provides a checked reconstruction command and
the annotation checksum required to detect a changed upstream copy.

## 3. Software

Slurm launchers activate the environment with:

~~~
source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
~~~

data/wearable-ai/starter_kit/requirements.txt states Linux, Python 3.10, and
CUDA 12.8. Its pinned packages are:

~~~
torch==2.10.0
torchvision==0.25.0
transformers==5.8.1
tokenizers==0.22.2
accelerate==1.13.0
huggingface_hub==1.15.0
opencv-python-headless==4.13.0.92
Pillow==12.2.0
qwen-vl-utils==0.0.14
numpy==2.2.6
scipy==1.15.3
nltk==3.9.4
sacrebleu==2.6.0
portalocker==3.2.0
vllm==0.19.1
~~~

The requirements comments install CUDA PyTorch first:

~~~
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r data/wearable-ai/starter_kit/requirements.txt
~~~

wearable-ai-eccv/requirements.txt contains minimum versions and is not an
exact replacement for the pinned starter-kit file.

When assets are restored, large caches are placed under:

~~~
/scratch/inf0/user/agaur/wai-26/cache/huggingface
/scratch/inf0/user/agaur/wai-26/cache/torch
/scratch/inf0/user/agaur/wai-26/cache/xdg
~~~

The generated manifests under `documentation/reproducibility_manifests/`
contain the Conda explicit specification, package list, installation history,
pip freeze, host kernel, and Python version. The development environment uses
Python 3.10.20. The final container is separate: Python 3.10.12, Torch
2.10.0+cu128, Transformers 5.8.1, vLLM 0.19.1, FlashInfer 0.6.6, and
`nvidia-cutlass-dsl` 4.7.0. See `final-container-runtime.json`. The development
environment has `nvidia-cutlass-dsl` 4.6.0, so the two package lists must not be
combined.

## 4. Checkpoints

The final build context stages these exact Hugging Face snapshots:

| Model | Revision | Image path |
|---|---|---|
| Qwen/Qwen3.5-27B | fc05daec18b0a78c049392ed2e771dde82bdf654 | /models/qwen35-27b |
| google/siglip2-so400m-patch14-384 | e8e487298228002f3d8a82e0cd5c8ea9c567f57f | /models/siglip2-so400m-patch14-384 |

The restoration script recreates these default local snapshot paths:

~~~
/scratch/inf0/user/agaur/wai-26/cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654
/scratch/inf0/user/agaur/wai-26/cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f
~~~

Other checkpoints appeared in exploratory work, including Qwen3-VL-8B,
Qwen3.5-9B, Qwen3.8-27B, Qwen3-VL-Embedding-2B, Qwen3-VL-Reranker-2B,
SigLIP, VideoJudge-7B, Molmo2, NVILA, InternVL, and HieraMamba components.
They are not in the final image. Hugging Face checkpoints that remained in the
cache at capture time have revisions, file counts, and logical sizes in
`documentation/reproducibility_manifests/huggingface-model-snapshots.tsv`.
The separate NVILA/AutoGaze environment is recorded in
`nvila-autogaze-environment.txt` and `nvila-autogaze-pip-freeze.txt`.

## 5. Pipeline

run_evaluation.py is the unified entry point. It loads JSONL annotations,
extracts frames, invokes the chosen model, writes predictions, and optionally
evaluates them. LongQA evaluation matches rows by stable sample keys in the
merge/evaluation utilities. LongQA predictions use video_path and mcq_answer;
other tracks can use different fields.

The standard prompt contains:

~~~
Question: <question>

Options:
<the four options>

Answer with ONLY ...
~~~

The default prompt variant is baseline.

### Final dual-view model

model.py registers qwen35_dual_view_fusion, implemented in
longqa_dual_view_fusion.py.

For each question:

1. The evaluator supplies 128 endpoint-inclusive candidate frames followed by
   the exact 64-frame endpoint-inclusive global grid.
2. SigLIP2 encodes the 128 candidates in batches of eight on GPU 1.
3. Question/option retrieval queries score the candidate image embeddings.
4. The temporal compiler interprets explicit after, before, first, last, and
   state-change wording. The selector uses two pivot centers and eight target
   centers, local/bridge context, temporal suppression, budget filling, and
   chronological ordering to produce 64 frames.
5. Qwen3.5-27B answers with the same question, options, and baseline prompt on
   both the global and option-conditioned frame views.
6. If the letters agree, that letter is returned.
7. If they disagree, Qwen scores A-D by vLLM choice-letter log probabilities on
   each view. Each view is normalized across the four letters; the two
   distributions are averaged and the highest letter is returned. An exact tie
   uses the direct answer fallback.

SigLIP2 selects evidence; it does not vote on the final answer. The final
runtime uses Qwen TP=1 on GPU 0 and SigLIP2 on GPU 1.

### Entry points

| File | Role |
|---|---|
| data/wearable-ai/starter_kit/run_evaluation.py | Unified CLI |
| data/wearable-ai/starter_kit/run_generate_longqa.py | Standard LongQA generation |
| data/wearable-ai/starter_kit/model.py | Registry, HF/vLLM, frame extraction, diagnostics |
| data/wearable-ai/starter_kit/longqa_dual_view_fusion.py | Final dual-view model |
| data/wearable-ai/starter_kit/run_generate_longqa_grounded.py | Standalone grounding |
| data/wearable-ai/starter_kit/run_generate_longqa_proofpack.py | Cached temporal evidence |
| data/wearable-ai/starter_kit/run_generate_longqa_uncertainty.py | Uncertainty selection |
| data/wearable-ai/starter_kit/run_generate_longqa_tcot.py | Temporal CoT selection |
| scripts/merge_longqa_*.sh | Label-free merges and analyses |

## 6. Baselines and experiments

The archived dated launchers document this progression:

1. Qwen2.5-VL and Qwen3-VL with 32 uniform frames.
2. Frame-count and resolution ablations.
3. SigLIP/SigLIP2 retrieval and temporal pivoting.
4. Uncertainty, AdaQ, FOCUS, QCA, eventlet, timeline, and proof-pack trials.
5. Temporal CoT, object/OCR/crop, occurrence, reranker, and VideoJudge trials.
6. Qwen3.5-9B/27B direct candidates, endpoint sampling, option-quota views,
   likelihood/probability fusion, and agreement checks.
7. Qwen3.8-27B and other open-model smoke tests.

Representative historical launchers:

~~~
slurm_longqa_qwen3_32frames.sh
slurm_longqa_qwen3_uniform64_px451584_dev.sh
slurm_longqa_qwen35_27b_uniform64_endpoint_dev.sh
slurm_longqa_qwen35_27b_uniform64_endpoint_val560.sh
slurm_longqa_qwen35_27b_evidence_rank_fusion_val560.sh
~~~

The root launchers were compressed after the competition. Inspect one without
restoring the full set:

~~~
tar -xOf archives/workspace_cleanup_2026-09-01/root_slurm_scripts.tar.gz \
  slurm_longqa_qwen35_27b_uniform64_endpoint_val560.sh | less
~~~

To rerun it from the workspace root, pipe the same content to `sbatch` after
checking its paths and environment assumptions.

The exact dated launcher and archived run must be checked before claiming that
an experiment completed. Many historical jobs produced partial or invalid
artifacts.

## 7. Slurm conventions

The archived launchers generally use partition gpu24, request H100s with
#SBATCH --gres=gpu:<n>, write slurm_logs/<name>_%j.out/.err, activate
wearable-ai, place caches in scratch, and archive under runs/egolongqa/.
The generic launcher requests one GPU, 8 CPUs, 32 GB, and 9 hours; individual
launchers override these values.

The final source smoke requested two H100s, 32 CPUs, 256 GB, and two hours. The
repair/container smoke requested two H100s, 32 CPUs, 256 GB, and four hours.

Check the batch step, not only Slurm's extern step:

~~~
sacct -j <job-id> \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList,Reason%60
scontrol show job -dd <job-id>
~~~

A COMPLETED extern step does not prove that the batch script succeeded.

## 8. Reproduce the final research run

Install the environment as above, then from data/wearable-ai/starter_kit/:

~~~
python run_evaluation.py \
  --task longqa \
  --model-type qwen \
  --backend vllm \
  --video-folder ../egolongqa/val \
  --max-frames 64
~~~

The final endpoint launcher additionally uses:

~~~
LLM_MODEL=Qwen/Qwen3.5-27B
QWEN_ENABLE_THINKING=0
VLLM_REASONING_PARSER=qwen3
MAX_FRAMES=64
FRAMES_PER_INTERVAL=64
QWEN_MAX_PIXELS=451584
CONCURRENCY=1
BATCH_SIZE=1
uniform-sampling endpoint_inclusive
~~~

The final container runtime uses:

~~~
WAI_QWEN_TP_SIZE=1
WAI_SIGLIP_GPU_INDEX=1
VLLM_QWEN_MAX_MODEL_LEN=32768
VLLM_GPU_MEMORY_UTILIZATION=0.90
QWEN_ENABLE_THINKING=0
QWEN_MIN_PIXELS=784
QWEN_MAX_PIXELS=451584
VLLM_MAX_LOGPROBS=100
VLLM_GDN_PREFILL_BACKEND=triton
~~~

Source smoke:

~~~
sbatch test_submission/qwen35_27b_dual_view_fusion_v2/slurm_source_smoke_test_2xh100.sh
~~~

The final container smoke produced 5 predictions, 4 correct, and per-question
generation times of 17.45--28.44 seconds. The source smoke also produced 5
valid predictions and 4 correct answers, but it is not the source of those
container timing numbers.

## 9. Container build and recovery

### v1

The v1 Containerfile uses the pinned base:

~~~
docker.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04
sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9
~~~

It installs Python 3.10, pip, certificates, CUDA 12.8 PyTorch, and pinned
requirements; copies source under /app and weights under /models; and has no
ENTRYPOINT because the evaluator invokes the script explicitly.

Original preparation/build commands:

~~~
bash test_submission/qwen35_27b_dual_view_fusion/prepare_build_context.sh
bash test_submission/qwen35_27b_dual_view_fusion/build_oci_archive.sh
~~~

The original v1 submission digest was:

~~~
sha256:5f8fad0c6994534b117613127d2f11b14f36551bc10866f30a909a32738fbe73
~~~

The v1 digest remains part of the build record because v2 used it as an
immutable base. The local unpacked v1 tree is no longer required after the
final test result and may be removed under the cleanup plan.

### v1 failure and v2 recovery

The old image reached vLLM initialization but did not finish successfully.
The earlier source smoke was not an image test and used a different GPU/memory
situation.

v2 kept the old root filesystem and added only the audited /app files:
model.py, run_evaluation.py, longqa_dual_view_fusion.py, and
SUBMISSION_BUILD_PROVENANCE.json. Runtime changes were Qwen TP=1 on GPU 0,
SigLIP2 on GPU 1, a 32,768-token cap, 0.90 GPU utilization, and retained
startup diagnostics.

The first container validator issue was a read-only rootless /tmp; Torch
passed with an ephemeral writable /tmp. The next attempt reached vLLM but
failed with RuntimeError: Failed to find C compiler during Triton helper
compilation.

### v3 repair

Containerfile.compiler_repair installs:

~~~
build-essential
python3.10-dev
~~~

and sets CC=/usr/bin/gcc. The repair launcher was:

~~~
sbatch test_submission/qwen35_27b_dual_view_fusion_v2/slurm_repair_and_container_smoke_2xh100.sh
~~~

Job 49948777 passed compiler/Triton checks, Torch CUDA import, two-GPU
visibility, vLLM startup, five predictions, timing, and archive validation.

The locally tested archive was:

~~~
/scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v2/artifacts/kth-saar-qwen35-dual-view-v3.tar
~~~

Recorded archive SHA-256:

~~~
d467137d5e3bfcd41dba817aa2e5f8643d72ff54fe39b5a3b764e63735f8db0d
~~~

Recorded size: 67720555008 bytes. The local archive was deleted with scratch
after its full SHA-256, checksum sidecar, smoke marker, build history, and ECR
digest were preserved in the workspace.

The archived smoke marker is:

~~~
archives/submission_recovery_2026-09-01/validation/CONTAINER_SMOKE_PASSED_V3
~~~

A new build should use the pinned source and model revisions. It may be
functionally equivalent without reproducing the same archive hash because OCI
layer metadata and the registry base state can change.

## 10. Gates and ECR

Before pushing, the recovery runbook requires:

1. validate_local_tarball.py confirms v1 filesystem integrity and the
   correction-layer file allowlist.
2. crane validate --tarball accepts the archive.
3. The exact image runs in a two-H100 container smoke test.
4. Five valid A-D predictions are written.
5. Every generation is below 300 seconds.
6. Dual-view startup reports Qwen TP=1, SigLIP GPU 1, and ready.
7. vLLM startup reports ready and its log is retained.
8. The uploaded immutable digest matches the remote tag digest.

The final ECR reference is preserved at:

~~~
archives/submission_recovery_2026-09-01/artifacts/ecr_immutable_reference_v3.txt
~~~

It records:

~~~
510788268643.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/kth-saar@sha256:1408a9dd1e17af9017a9349e2f77e7998564295d8c21da06552f8dd86eeff5d0
~~~

The push command was:

~~~
bash test_submission/qwen35_27b_dual_view_fusion_v2/push_validated_image.sh
~~~

The push worker ran in tmux using run_v3_push_worker.sh, returned exit status
0, and remote validation passed. Credentials and signed URLs are deliberately
not recorded here.

## 11. Outputs

A normal run writes:

~~~
<run>/predictions.jsonl
<run>/results.json
<run>/results_summary.json
~~~

The dual-view run additionally writes:

~~~
<run>/diagnostics.json
<run>/inference_diagnostics/dual_view_startup.json
<run>/inference_diagnostics/vllm_startup_*.json
<run>/inference_diagnostics/vllm_server_*.log
~~~

Diagnostics record prompt-token statistics where vLLM reports them: count,
minimum, mean, p50, p95, maximum, context window, and fill percentages. The
recovery smoke observed a maximum of approximately 28,250 tokens against the
32,768 cap, approximately 85.43% fill.

A validation prediction file contains 700 unique rows in annotation order and
one valid answer letter per row. The local generation format uses `video_path`
and `mcq_answer`; the public validation upload required an `answers` field in
its submission wrapper. Test-phase evaluation invoked the container directly.

## 12. Troubleshooting

- Time-zero Slurm failure: inspect the batch step and its .err; an unavailable
  runtime, environment retrieval hold, or missing prerequisite means inference
  did not run.
- Torch failure only in a validator: reproduce with writable ephemeral /tmp;
  do not ignore unrelated validator failures.
- Failed to find C compiler: use the compiler-repair Containerfile and verify
  gcc, Python headers, and Triton compilation inside the image.
- Memory/context failure: check visible GPUs, TP size, 32,768-token cap, 0.90
  utilization, frame limit, and 451584 maximum pixels.
- Invalid/incomplete predictions: check row count, raw/parsed answers, and
  strict retry behavior. Earlier short TCoT caps omitted final markers and made
  raw scores invalid.
- Merge disagreement: merge by stable sample keys, verify exact coverage,
  preserve annotation order, and label the artifact as direct, fused, or
  exploratory.
- Hugging Face failure: the final image is offline and has both checkpoints
  under /models; local setup must use the exact snapshot revisions.

## 13. Retained records and cleanup

The organizers evaluated the corrected image and the team finished fourth.
The compact final record is:

- v3 checksum sidecar, smoke marker, and immutable ECR reference under
  `archives/submission_recovery_2026-09-01/`;
- v2 recovery source/build scripts;
- v1 and v3 digest records and final submission receipts;
- pinned requirements, source, and checkpoint revision metadata.

Large `runs/` and analysis files are not needed to pull the uploaded image, but
remain available for the paper and audit trail. Historical launchers and logs
were compressed under `archives/workspace_cleanup_2026-09-01/`; their path
manifests and archive checksums are stored beside them. The dated storage record
is in `documentation/STORAGE_CLEANUP_AUDIT.md`.

## 14. Remaining gaps

These facts remain unknown or unavailable locally:

- exact upstream dataset revision and the original command used for the local
  download; a reconstruction command and annotation checksum are retained;
- complete organizer-side OS, NVIDIA, Slurm, and invocation details;
- organizer final test score and evaluator logs;
- final organizer invocation parameters;
- byte-for-byte recovery of the deleted OCI archive if the registry no longer
  retains its immutable digest;
- whether every historical run still has all raw inputs after cleanup.

These gaps do not invalidate the recorded v3 smoke result or fourth-place
outcome, but they prevent byte-for-byte reproduction of every historical
experiment.
