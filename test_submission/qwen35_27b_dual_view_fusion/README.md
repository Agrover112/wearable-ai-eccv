# Qwen3.5-27B dual-view fusion test image

This submission packages the label-free `618/700` validation pipeline:

1. Extract 128 endpoint-inclusive candidate frames and an independent 64-frame
   endpoint-inclusive global view.
2. Use pinned SigLIP2 SO400M embeddings to create a 64-frame question- and
   option-conditioned temporal view.
3. Ask the same pinned Qwen3.5-27B model for one answer under each view.
4. Keep an agreed answer. On disagreement, score A-D under both views,
   normalize each score vector, average the two probability vectors, and choose
   the highest-scoring letter.

No validation labels, cached predictions, or dataset paths are included.

## Prepare

```bash
bash test_submission/qwen35_27b_dual_view_fusion/prepare_build_context.sh \
  /scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v1
```

The default context is staged under scratch. It dereferences the pinned Qwen
and SigLIP2 snapshots so the image is independent of the Hugging Face cache.

## Build and validate

```bash
cd /scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v1/starter_kit/test_phase
./build_image.sh --tag kth-saar-qwen35-dual-view:v1
./validate_image.sh kth-saar-qwen35-dual-view:v1
```

Do not use `--push` until the local validator and five-row smoke test pass.

The first image was uploaded on 2026-08-12. Its immutable ECR reference and
the exact test-submission fields are recorded in `SUBMISSION_RECEIPT.md`.

The model registry key is `qwen35_dual_view_fusion`. It requests two visible
GPUs: Qwen runs on GPU 0 and SigLIP2 on GPU 1. The evaluation runner
automatically enforces the required 192-frame extraction bundle.
