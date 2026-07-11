# Approved EgoLongQA Experiments

These experiments are separate from the established direct-MCQ baselines.

## Open QA plus option similarity

The VLM sees video frames and the question, but no options. Its concise answer
is cached in `open_answers.jsonl`, encoded with
`sentence-transformers/all-MiniLM-L6-v2`, and matched to the four option
embeddings by cosine similarity. Predictions include all four scores and the
top-1 margin.

```bash
sbatch slurm_longqa_qwen3_openqa_dev.sh
```

Use `SUBSET_FILE=configs/egolongqa_dev20_seed20260709.json` for a first smoke
run. The archived artifacts include raw open answers, predictions, evaluation,
shortcut-aware diagnostics, and the vLLM server log.

## Video-blind ablation

The same question-plus-options prompt is sent to Qwen3-VL with an empty frame
list. This isolates the language branch and dataset shortcuts.

```bash
sbatch slurm_longqa_qwen3_video_blind.sh
```

This run also writes shortcut-aware diagnostics so its macro-letter and non-C
accuracy can be compared with the always-C floor.

## SigLIP2 grounding

The old SigLIP default is unchanged. This launcher explicitly selects
`google/siglip2-so400m-patch14-384`, BF16, c128/top24/anc8/final32, and caches
normalized frame features in scratch. Grounding JSONL records carry a complete
selection fingerprint, so incompatible caches are rejected.

Set `GROUNDER_REVISION` to a Hugging Face commit SHA to pin both loading and
feature-cache identity. If omitted, the cache is scoped to the model's current
default revision.

```bash
sbatch slurm_longqa_qwen3_siglip2_grounded_dev.sh
```

The same `SUBSET_FILE` override can select dev20 before dev140. Reusing the
same `GROUNDING_CACHE_DIR` avoids re-encoding frames in later retrieval modes.

## Resolution grid

The default job runs 32 frames at 672px-equivalent (`451584` pixels). Other
points use environment overrides:

```bash
QWEN_MAX_PIXELS=50176 sbatch slurm_longqa_qwen3_resolution_dev.sh
QWEN_MAX_PIXELS=200704 sbatch slurm_longqa_qwen3_resolution_dev.sh
QWEN_MAX_PIXELS=451584 sbatch slurm_longqa_qwen3_resolution_dev.sh
MAX_FRAMES=64 QWEN_MAX_PIXELS=50176 sbatch slurm_longqa_qwen3_resolution_dev.sh
```

Analyze completed prediction files with:

```bash
python scripts/analyze_longqa_resolution.py \
  --run px224=path/to/224/predictions.jsonl \
  --run px448=path/to/448/predictions.jsonl \
  --run px672=path/to/672/predictions.jsonl \
  --output-json runs/egolongqa/resolution_analysis.json
```

The analyzer aligns stable sample keys and reports overall, overlapping
question-type, exclusive question-type, category, and pairwise win/loss results.
