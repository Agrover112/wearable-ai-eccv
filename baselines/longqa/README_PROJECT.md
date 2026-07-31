# LongQA baseline implementation

This directory contains the workshop starter-kit runtime plus the EgoLongQA
extensions used in our experiments:

- configurable uniform frame counts and Qwen image budgets;
- vLLM/Hugging Face generation with prompt-context accounting;
- resumable prediction generation;
- SigLIP/CLIP frame retrieval, anchors, temporal NMS, per-option union, and
  coarse-to-fine selection;
- reusable SigLIP/SigLIP2 frame-embedding caches with model/revision and video
  identity validation;
- eventlet, option-contrastive, and temporal-pivot proof-pack selection with
  resumable grounding and prediction fingerprints;
- schema-constrained visual event ledgers with optional open-vocabulary object
  detections;
- full-answer semantic likelihood scoring across separate retrieved and
  uniform visual contexts;
- disagreement verification through support/contradiction prompts, pairwise
  order swaps, option permutations, and calibrated candidate scores;
- matched video-blind and open-QA-to-option baselines;
- question-type and image-resolution analysis utilities;
- grouped cross-validation utilities for disagreement-routing diagnostics;
- prompt variants that preserve question-plus-options input and emit only an
  option letter;
- grounding-frame audits and rolling partial-run analysis.

See [`../../docs/longqa/LONGQA_EXPERIMENT_FLAGS.md`](../../docs/longqa/LONGQA_EXPERIMENT_FLAGS.md)
for the experiment controls and [`../../docs/longqa/RUN_LOG.md`](../../docs/longqa/RUN_LOG.md)
for validated results.

Proof-pack strategies and promotion criteria are documented in
[`../../docs/longqa/LONGQA_PROOFPACK_EXPERIMENTS.md`](../../docs/longqa/LONGQA_PROOFPACK_EXPERIMENTS.md).
Cluster-specific Slurm launchers are intentionally maintained outside this
repository; the Python entry points here remain scheduler-independent.

## InternVideo3 inference

InternVideo3 uses the same frame sampling, prompts, prediction format, resume logic, and
evaluation code as Qwen. Its checkpoint requires Transformers 4.57.x, so it has a separate
project-local environment instead of changing the Qwen/vLLM environment:

```bash
bash scripts/setup_internvideo3_env.sh
source scripts/activate_internvideo3_env.sh
```

Run a two-sample smoke test of the strongest uniform dev140 configuration from the repository
root with:

```bash
VISION_MAX_PIXELS=451584 python baselines/longqa/run_generate_longqa.py \
  --input ../../../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file ../../configs/egolongqa_dev140_seed20260709.json \
  --max-samples 2 \
  --model-type internvideo3 \
  --backend hf \
  --max-frames 64 \
  --frames-per-interval 64 \
  --batch-size 1 \
  --output ../../runs/egolongqa/internvideo3_uniform64_px451584_smoke/predictions.jsonl \
  --eval-output ../../runs/egolongqa/internvideo3_uniform64_px451584_smoke/results.json
```

Remove `--max-samples 2` and choose a new output directory for the later 140-sample run.
InternVideo3 is not supported by the installed vLLM release, so its backend is `hf`.

## Qwen3.5-0.8B inference

The `qwen3_5` adapter represents uniformly sampled frames as one native video. It passes the
original frame indices, source FPS, and total frame count to the processor, so Qwen3.5 receives
timestamped video tokens rather than the separate untimestamped images used by the older `qwen`
adapter. The default checkpoint is `Qwen/Qwen3.5-0.8B`.

Run a two-sample, non-thinking smoke test from the repository root with:

```bash
source scripts/activate_local_env.sh

python baselines/longqa/run_generate_longqa.py \
  --input /CT/RGCAHead3/nobackup/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file configs/egolongqa_dev140_seed20260709.json \
  --max-samples 2 \
  --model-type qwen3_5 \
  --backend hf \
  --num-gpus 1 \
  --max-frames 64 \
  --frames-per-interval 64 \
  --batch-size 1 \
  --prompt-variant qwen3_5 \
  --longqa-max-new-tokens 1024 \
  --output runs/egolongqa/qwen3_5_0_8b_uniform64_smoke/predictions.jsonl \
  --eval-output runs/egolongqa/qwen3_5_0_8b_uniform64_smoke/results.json \
  --no-resume-predictions
```

The adapter uses FlashAttention 2 by default and disables Qwen thinking so the response is a
single `{"answer":"X"}` object. Set `QWEN35_ENABLE_THINKING=1` for the thinking ablation, or
`QWEN35_ATTN_IMPLEMENTATION=sdpa` on a GPU not supported by the installed FlashAttention wheel.
`QWEN35_VIDEO_TOTAL_PIXELS` overrides the processor's default total video-pixel budget of
25,165,824, and `QWEN35_SEED` controls sampling with a default of 0. The current vLLM request
path sends separate images and is therefore not used for this native-video adapter.
