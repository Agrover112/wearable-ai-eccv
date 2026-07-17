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
- matched video-blind and open-QA-to-option baselines;
- question-type and image-resolution analysis utilities;
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
The Hugging Face backend uses the FlashAttention-2 extension installed by the setup script by
default. Override it only when an experiment specifically requires another implementation:

```bash
INTERNVIDEO3_ATTN_IMPLEMENTATION=sdpa \
  python baselines/longqa/run_generate_longqa.py ...
```

The setup script compiles FlashAttention for H100 (`SM90`) by default. Set
`FLASH_ATTN_CUDA_ARCHS=80` while running the setup script if the environment instead needs to
run on A100 GPUs.

To reproduce the native-video timestamp and challenge-latency profile on one H100, request an
interactive allocation and run:

```bash
scripts/profile_internvideo3_single.sh
```

The profiler samples 512, 1,024, and 2,048 frames from one representative 600-second video,
separately records processor and model-inference time, verifies the generated timestamp span,
and writes `runs/egolongqa/internvideo3_timestamp_profile_2026-07-14/profile.json`.

The deterministic 30-question promotion gate uses native video decoding, 512 frames, timestamped
evidence prompting, and resumable JSONL output:

```bash
source scripts/activate_internvideo3_env.sh
python baselines/longqa/run_internvideo3_timestamp_pilot.py \
  --input ../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --video-folder /scratch/inf0/user/kkumar/val \
  --subset-file configs/internvideo3_temporal_pilot30_seed20260714.json \
  --output runs/egolongqa/internvideo3_timestamp512_pilot30_fa2/predictions.jsonl \
  --attn-implementation flash_attention_2 \
  --frames 512 \
  --min-pixels 65536 \
  --max-pixels 131072 \
  --max-new-tokens 192
```

The subset balances prior 64-frame successes and failures across first/last, before/after, and
other temporal comparisons, plus six controls. It is a paired promotion test rather than an
estimate of dev140 accuracy.

To evaluate the complete reduced dev140 split with the promoted FA2 configuration, submit:

```bash
ssh slurm
cd /CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
sbatch scripts/slurm_internvideo3_timestamp512_dev140.sh
```

This runs all 140 questions, writes one resumable prediction row per question, and produces
`runs/egolongqa/internvideo3_timestamp512_fa2_dev140_2026-07-14/summary.json` when complete.
