# Recovery runbook

The failed submission was:

`sha256:5f8fad0c6994534b117613127d2f11b14f36551bc10866f30a909a32738fbe73`

Do not delete it. The corrected image uses it as an immutable base so the
installed dependencies and model weights remain byte-for-byte identical.

## What failed in v1

The old image reached vLLM engine startup but did not finish initializing on
the organizer's H100 80 GB node. Our earlier smoke test ran the source tree,
not the submitted image, and placed Qwen on one H100 NVL with 96 GB of memory.
It therefore did not reproduce the submitted container or GPU memory limit.

## Runtime corrections in v2

- Qwen3.5-27B runs on GPU 0 without tensor parallelism.
- SigLIP2 runs independently on GPU 1.
- Qwen's context window is capped at 32,768 tokens. The maximum recorded input
  over all 700 validation questions was approximately 28,250 tokens.
- vLLM reserves 90 percent of GPU 0 instead of 95 percent.
- The full vLLM log, vLLM startup report, and dual-view component report are
  written under `inference_diagnostics/` beside `predictions.jsonl`.
- The answer policy and visual inputs are unchanged.

## Required gates

Do not push or send a digest until every gate passes.

1. `validate_local_tarball.py` confirms that the old root filesystem is intact
   and that the new layer contains only the four audited `/app` files.
2. `crane validate --tarball` reports a well-formed image.
3. `slurm_container_smoke_test_2xh100.sh` runs the finished image with the same
   two-GPU layout used during organizer evaluation.
4. The container writes five predictions, and every answer is one of `A-D`.
5. Every measured generation takes less than 300 seconds.
6. `dual_view_startup.json` reports Qwen TP=1, SigLIP on GPU 1, and status
   `ready`.
7. The vLLM startup report says `ready`, and the complete server log remains in
   the output directory.
8. The push script returns a new immutable digest, and `crane digest` returns
   the same value for the uploaded tag.

`create_patched_image.sh` stores the corrected manifest in ECR by digest but
does not assign it a tag. This avoids downloading the 67 GB base layer twice
and does not create a submission. The final tag is assigned only after the
finished archive passes the container smoke test.

The container smoke test writes
`/CT/HyperAvatar/work/wearable-ai-submission-recovery/validation/CONTAINER_SMOKE_PASSED`
only after gates 3 through 7 pass. `push_validated_image.sh` refuses to push
without that marker.

## Commands

After the local tarball is complete:

```bash
python test_submission/qwen35_27b_dual_view_fusion_v2/validate_local_tarball.py \
  /scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v2/artifacts/kth-saar-qwen35-dual-view-v2.tar

sbatch test_submission/qwen35_27b_dual_view_fusion_v2/slurm_container_smoke_test_2xh100.sh
```

Only after the Slurm job passes:

```bash
bash test_submission/qwen35_27b_dual_view_fusion_v2/push_validated_image.sh
```
