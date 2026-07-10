# EgoLongQA Experiment Flags

## Subsets

Create stable dev subsets:

```bash
python scripts/make_longqa_dev_subset.py \
  --n 140 \
  --seed 20260709 \
  --output configs/egolongqa_dev140_seed20260709.json
```

Generation scripts accept:

```bash
--subset-file configs/egolongqa_dev140_seed20260709.json
```

## Prompt Variants

LongQA generation accepts:

```bash
--prompt-variant baseline
--prompt-variant evidence_first
--prompt-variant option_verify
--prompt-variant temporal_anchor
--prompt-variant anti_shortcut
--prompt-variant combined
```

`baseline` preserves the original prompt text.

## Grounding Retrieval

Grounded generation accepts:

```bash
--retrieval-query-mode question_options
--retrieval-query-mode question
--retrieval-query-mode per_option_union
--retrieval-query-mode question_temporal_boost
```

`question_options` is the default and preserves the previous question+options
retrieval behavior.

Temporal NMS:

```bash
--temporal-nms-seconds 10
--temporal-nms-candidates 64
```

Coarse-to-fine:

```bash
--coarse-to-fine \
--coarse-candidate-frames 128 \
--num-windows 4 \
--window-radius-candidates 2 \
--frames-per-window 8 \
--global-anchor-k 32 \
--final-max-frames 64
```

## Diagnostics

```bash
python scripts/eval_longqa_diagnostics.py \
  --predictions runs/egolongqa/<run>/predictions.jsonl \
  --annotations data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --output runs/egolongqa/<run>/diagnostics.json
```

## Flexible Dev SLURM Launchers

Prompt-only uniform64:

```bash
PROMPT_VARIANT=combined sbatch slurm_longqa_qwen3_uniform64_dev.sh
```

Grounded retrieval/NMS:

```bash
RETRIEVAL_QUERY_MODE=per_option_union \
TEMPORAL_NMS_SECONDS=10 \
PROMPT_VARIANT=combined \
sbatch slurm_longqa_qwen3_grounded_dev.sh
```

Coarse-to-fine:

```bash
RETRIEVAL_QUERY_MODE=question_options \
TEMPORAL_NMS_SECONDS=10 \
NUM_WINDOWS=4 \
FRAMES_PER_WINDOW=8 \
GLOBAL_ANCHOR_K=32 \
PROMPT_VARIANT=combined \
sbatch slurm_longqa_qwen3_coarse_to_fine_dev.sh
```
