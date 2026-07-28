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
--subset-file /CT/NDF/work/waw-26/configs/egolongqa_dev140_seed20260709.json
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

## Temporal Chain Of Thought

`run_generate_longqa_tcot.py` uses Qwen itself to select frame IDs directly
from chronological low-resolution candidates. The answer call receives the
selected temporal neighborhoods at the normal high-resolution image budget
and uses the established answer-only MCQ prompt, isolating frame selection from
answer-prompt changes. The uniform answer-CoT control uses a larger generation
budget and a strict answer-only retry if no explicit `Final Answer:` marker is
produced.
Raw selections are cached per sample under
`TCOT_SELECTION_CACHE_DIR` and are independent of final selected/uniform quotas.

Run the matched dev140 experiments with:

```bash
sbatch slurm_longqa_qwen3_uniform64_answer_cot_px451584_dev.sh
sbatch slurm_longqa_qwen3_tcot_single_c128_px451584_dev.sh
sbatch slurm_longqa_qwen3_tcot_dynamic_c256_sel64_px451584_dev.sh
sbatch slurm_longqa_qwen3_tcot_dynamic_c256_sel48_u16_px451584_dev.sh
```

The two dynamic launchers intentionally share the same selector configuration
and cache fingerprint. Their only difference is the final frame pack:

- `sel64`: up to 64 frames from Qwen-selected temporal neighborhoods;
- `sel48_u16`: up to 48 selected frames plus 16 globally uniform frames.

Run `sel64` first when possible so `sel48_u16` can reuse all selection calls.
Each archive records `selections.jsonl`, including segment justifications,
selected seed indices, expanded neighborhoods, and final frame indices.

## Object-Aware Visual Hints

`run_generate_longqa_object_hints.py` evaluates Minerva-Ego-inspired object
hints without changing the established baselines. All modes reuse the completed
64-frame, 672px temporal-pivot proof pack and cache Grounding DINO detections
under:

```text
/scratch/inf0/user/agaur/wai-26/data/wearable-ai/object_hint_detection_cache
```

Run the dev140 experiments in this order:

```bash
# Builds the shared detection cache and labels detected objects in place.
sbatch slurm_longqa_qwen3_object_labels_dev.sh

# These reuse the cache and may be submitted together after object_labels ends.
sbatch slurm_longqa_qwen3_object_crop_pairs_dev.sh
sbatch slurm_longqa_qwen3_object_evidence_panels_dev.sh
sbatch slurm_longqa_qwen3_object_ledger_dev.sh

# Run after checking detector quality on the first four experiments.
sbatch slurm_longqa_qwen3_object_tracks_dev.sh
sbatch slurm_longqa_qwen3_object_question_router_dev.sh
```

The controlled modes are:

- `object_labels`: the same 64 proof-pack moments, with automatic labels placed
  at detected object locations;
- `crop_pairs`: 56 full frames plus up to eight enlarged details paired with
  their source frames;
- `evidence_panels`: 56 full frames plus up to eight panels containing the
  source view and one or two enlarged details;
- `object_ledger`: the original proof pack plus a chronological text record of
  detected object identities, locations, and apparent sizes;
- `object_tracks`: lightweight first/peak/last occurrence tracks derived from
  detections, with one-second temporal neighbours and proof-pack coverage fill;
- `question_router`: deterministic routing among uniform coverage, temporal
  pivoting, crop pairs, and object tracks according to the requested evidence.

`object_tracks` is the inexpensive detector-track test, not dense SAM2 mask
propagation. It should establish whether object-conditioned temporal evidence
helps before paying for a separate SAM2 pass over dense video clips. Detection
JSON is archived with each run, while detector model caches remain in scratch.

## Qwen3-VL Thinking Control

Run the matched uniform64/672px reasoning ablation with:

```bash
sbatch slurm_longqa_qwen3_thinking_uniform64_px451584_dev.sh
```

This changes the checkpoint from `Qwen3-VL-8B-Instruct` to
`Qwen3-VL-8B-Thinking` while retaining the established dev140 subset, uniform
64-frame sampling, 672px image cap, and evaluator. It provides 2,048 generation
tokens for reasoning and requires a final `Final Answer: X` marker. If the
initial response lacks that marker, a short continuation receives the original
reasoning as assistant context and emits only the final answer.
