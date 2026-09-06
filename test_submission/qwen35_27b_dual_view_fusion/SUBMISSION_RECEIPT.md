# EgoLongQA test image submission receipt

## Uploaded image

- Team: `kth_saar`
- Registration ID: `WAI-6B70F7C8`
- Track/division: EgoLongQA / Large
- Repository: `510788268643.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/kth-saar`
- Tag: `qwen35-27b-dual-view-fusion-v1`
- Digest: `sha256:5f8fad0c6994534b117613127d2f11b14f36551bc10866f30a909a32738fbe73`
- Immutable reference: `510788268643.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/kth-saar@sha256:5f8fad0c6994534b117613127d2f11b14f36551bc10866f30a909a32738fbe73`

The digest was returned by `podman push` and independently verified using ECR
`batch-get-image`.

## Submission form fields

- Model name: `Qwen3.5-27B Dual-View Fusion`
- Model license: `Qwen License` (open weights)
- Total parameters (billions): `28.91743645`
- Active parameters (billions): `28.91743645`
- Image digest: `sha256:5f8fad0c6994534b117613127d2f11b14f36551bc10866f30a909a32738fbe73`

The count is the exact sum of packaged safetensor parameters:

- Qwen3.5-27B: `27.781427952B`
- SigLIP2 SO400M: `1.136008498B`

Both components run for every query, so the declared active and total counts
are equal.

## Validation performed

- Final local image size: `74,724,558,750` bytes (74.7 GB decimal).
- `/app` contains the evaluation runner, model registry, and custom wrapper.
- `/models` contains all 11 Qwen shards and the complete SigLIP2 checkpoint.
- No duplicate weights exist under `/app`.
- CUDA-enabled PyTorch, Transformers, vLLM, OpenCV, the custom registry key,
  and `run_evaluation.py --help` passed in an offline sandbox.
- Both packaged model configurations and the SigLIP2 processor load offline.
- OpenCV decoded a real 1080p EgoLongQA validation MP4 frame.

The official validator could not be run to completion with the login node's
temporary rootless VFS Podman installation because each `podman run` attempted
to duplicate the entire 74.7 GB filesystem. Equivalent checks passed through a
read-only bubblewrap sandbox.

The source-equivalent two-GPU smoke test subsequently completed successfully as
Slurm job `49668091`:

- 5/5 prediction rows and evaluation outputs were written.
- Smoke-subset accuracy: `4/5` (`80%`).
- Per-query generation time: 17.535--31.195 seconds (mean 22.827 seconds).
- One-time vLLM initialization: 246 seconds.
- Mean/max context fill: 56.88%/56.95% of the 49,152-token window.

The smoke test used the same pinned Qwen3.5-27B and SigLIP2 snapshots, custom
model wrapper, sampling mode, and inference settings packaged in the uploaded
image. The uploaded image itself could not be launched on this login node due
to its rootless VFS behavior, not due to a model or image validation failure.
