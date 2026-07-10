# LongQA baseline implementation

This directory contains the workshop starter-kit runtime plus the EgoLongQA
extensions used in our experiments:

- configurable uniform frame counts and Qwen image budgets;
- vLLM/Hugging Face generation with prompt-context accounting;
- resumable prediction generation;
- SigLIP/CLIP frame retrieval, anchors, temporal NMS, per-option union, and
  coarse-to-fine selection;
- prompt variants that preserve question-plus-options input and emit only an
  option letter;
- grounding-frame audits and rolling partial-run analysis.

See [`../../docs/longqa/LONGQA_EXPERIMENT_FLAGS.md`](../../docs/longqa/LONGQA_EXPERIMENT_FLAGS.md)
for the experiment controls and [`../../docs/longqa/RUN_LOG.md`](../../docs/longqa/RUN_LOG.md)
for validated results.
