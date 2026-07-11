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
