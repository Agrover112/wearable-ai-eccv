# Qwen3.5-27B dual-view fusion recovery image

This package replaces the failed `v1` image without modifying or deleting it.
The answering policy is unchanged. It combines a 64-frame endpoint-inclusive
view with a 64-frame SigLIP2-selected view and averages Qwen answer
probabilities when the two views disagree.

The runtime layout differs from `v1`:

- Qwen runs on GPU 0 without tensor parallelism.
- SigLIP2 runs independently on GPU 1.
- Qwen's context window is capped at 32,768 tokens, above the approximately
  28,250-token maximum observed over all 700 validation questions.
- vLLM reserves 90 percent of GPU 0 instead of 95 percent.
- The complete vLLM log and a machine-readable startup report are written to
  `inference_diagnostics/` beside the predictions file.

The model uses two GPUs even when the organizer exposes eight. This avoids the
TP=2 sampling deadlock found by the recovery smoke test while keeping retrieval
off Qwen's GPU.

## Required validation sequence

1. Authenticate crane to the team ECR repository using temporary credentials.
2. Run `create_patched_image.sh`. It downloads and verifies the immutable base
   layer through crane's local cache, appends the audited source layer, and
   stores the corrected ECR manifest by digest without assigning a tag.
3. Run the structural validator.
4. Run the finished image on one node with 2 x H100 GPUs and five validation
   questions.
5. Check that five predictions exist, all answers are in `A` through `D`, the
   vLLM startup record says `ready`, and the complete server log is retained.
6. Assign the final ECR tag only after all five checks pass.

The test command is in `slurm_container_smoke_test_2xh100.sh`. It refuses to
report success unless the exact two-GPU runtime layout starts and all output
checks pass.
