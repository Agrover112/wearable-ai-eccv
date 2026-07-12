# LongQA Proof-Pack Experiments

These six dev140 experiments preserve the direct question-and-options task and
do not modify previous baseline outputs. The four grounding experiments reuse
cached c128 SigLIP2 image embeddings from:

```text
/scratch/inf0/user/agaur/wai-26/data/wearable-ai/grounder_features
```

## Launch Order

### 1. Uniform64 at 672px

```bash
sbatch slurm_longqa_qwen3_uniform64_px451584_dev.sh
```

- 64 uniform frames
- `max_pixels=451584` (~672x672)
- 49K context window
- Existing baseline MCQ prompt

### 2. Uniform48 at 560px

```bash
sbatch slurm_longqa_qwen3_uniform48_px313600_dev.sh
```

- 48 uniform frames
- `max_pixels=313600` (~560x560)
- 32K context window
- Existing baseline MCQ prompt

### 3. Eventlet hybrid

```bash
sbatch slurm_longqa_qwen3_eventlet_hybrid_dev.sh
```

- 128 SigLIP2 candidates
- 32 global anchors
- 8 relevant event centers expanded to center +/-1 candidate
- 8 requested semantic-boundary frames, then boundary fill to 64
- 64-frame cap at 448px

### 4. Option-contrastive eventlets

```bash
sbatch slurm_longqa_qwen3_option_contrastive_eventlets_dev.sh
```

- One visual hypothesis per option
- Per-option score is normalized and contrasted against the strongest competing
  option at each candidate frame
- 4 centers per option, expanded to center +/-1 candidate
- 16 global anchors and semantic-boundary fill
- 64-frame cap at 448px

### 5. Temporal-pivot proof pack

```bash
sbatch slurm_longqa_qwen3_temporal_pivot_dev.sh
```

- Deterministic operators: `AFTER`, `BEFORE`, `FIRST`, `LAST`, `STATE_CHANGE`
- 2 pivot centers and 8 target centers
- Forward/backward filtering for explicit `AFTER`/`BEFORE` questions
- 24 global anchors, local eventlets, and up to 8 bridge frames
- Questions without a supported operator fall back to eventlet-hybrid selection

### 6. Structured temporal-pivot proof pack

```bash
sbatch slurm_longqa_qwen3_temporal_pivot_structured_dev.sh
```

This uses the same temporal-pivot selection policy as experiment 5. The prompt
additionally lists image numbers, timestamps, evidence roles, pivot, and temporal
direction. Qwen still receives the original frames and must return one letter;
no prose timeline or visual-to-text compression is introduced.

## Smoke Tests

Use the stable dev20 subset before dev140 when validating a new environment:

```bash
SUBSET_FILE=configs/egolongqa_dev20_seed20260709.json \
MAX_SAMPLES=20 \
sbatch slurm_longqa_qwen3_eventlet_hybrid_dev.sh
```

The same `SUBSET_FILE` and `MAX_SAMPLES` overrides work for the other proof-pack
launchers. Set a unique `RUN_NAME` when retaining both smoke and full outputs.

## Artifacts

Each proof-pack archive contains:

```text
predictions.jsonl
proofpack.jsonl
results.json
results_summary.json
diagnostics.json
vllm_server.log
slurm.out
slurm.err
```

`proofpack.jsonl` records every selected frame's source index, timestamp,
similarity score, evidence role, selection fingerprint, and whether the cached
image embedding was reused. Grounding and prediction files are resumable, but
configuration fingerprints prevent reuse after a selector or prompt change.

## Promotion Criteria

Compare every run against uniform32/672 (`106/140`) and uniform64/448
(`104/140`). Promote to full validation when a run either:

- improves dev140 by at least 3-4 correct answers, or
- has enough unique correct answers to justify a later disagreement verifier.

Always report overall accuracy, non-C accuracy, temporal accuracy, paired
only-correct counts, context fill, and runtime.

## Next-Stage Launchers

The first proof-pack batch established uniform64/672 (`110/140`) and temporal
pivot/448 (`109/140`) as the two strongest dev candidates. The next launchers
are:

```bash
# Full-validation confirmation of the strongest simple baseline.
sbatch slurm_longqa_qwen3_uniform64_px451584_full.sh

# Combine the original pivot policy with 672px frames.
sbatch slurm_longqa_qwen3_temporal_pivot_px451584_dev.sh

# Test whether 96 high-resolution frames remain within the useful frontier.
sbatch slurm_longqa_qwen3_uniform96_px451584_dev.sh

# Replace generic semantic-boundary filler with temporal coverage at 448px.
sbatch slurm_longqa_qwen3_temporal_pivot_coverage_px200704_dev.sh

# Same revised selector at 672px.
sbatch slurm_longqa_qwen3_temporal_pivot_coverage_px451584_dev.sh
```

The revised pack requests 32 anchors, two pivot triplets, six directional target
triplets, up to eight bridges, and fills remaining slots by repeatedly sampling
the largest uncovered temporal gap. It does not use generic semantic-boundary
filler.

The full revised-pivot launcher is deliberately gated:

```bash
sbatch slurm_longqa_qwen3_temporal_pivot_coverage_px451584_full.sh
```

Run it only if the revised 672px dev experiment remains competitive with or
improves upon uniform64/672. It requests 24 hours because approximately 560
full-validation videos do not yet have cached SigLIP2 c128 image features. The
grounding and prediction outputs are resumable if the cluster time limit is
shorter.

## Routing And Verification Stage

The full revised-pivot run completed the SigLIP2 c128 cache and reached
`529/700`; uniform64/672 reached `514/700`. Run the next stage in this order.

### 1. Full original pivot/672

```bash
sbatch slurm_longqa_qwen3_temporal_pivot_px451584_full.sh
```

This is the direct promotion of the best dev run (`111/140`). All 700 image
feature caches now exist, so it should skip cold SigLIP2 image encoding.

### 2. Exact operator router

```bash
sbatch slurm_longqa_qwen3_operator_router_px451584_dev.sh
```

The router uses the baseline's exact uniform64 index formula for `GLOBAL`
questions and the original temporal-pivot policy for explicit operators. If the
dev result is competitive, promote it with:

```bash
sbatch slurm_longqa_qwen3_operator_router_px451584_full.sh
```

### 3. Disagreement audit

```bash
python scripts/analyze_longqa_disagreements.py \
  --annotations data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --first runs/egolongqa/<pivot-run>/predictions.jsonl \
  --second runs/egolongqa/<uniform-run>/predictions.jsonl \
  --first-label pivot \
  --second-label uniform \
  --output-json runs/egolongqa/pivot_uniform_disagreements.json \
  --output-jsonl runs/egolongqa/pivot_uniform_disagreements.jsonl
```

The JSONL contains sample keys, operator/category, both answers, gold answer,
and paired correctness for frame audits.

### 4. Gated verifier

The dev verifier can run immediately from the completed dev candidates:

```bash
sbatch slurm_longqa_qwen3_disagreement_verifier_dev.sh
```

Agreement rows copy the primary prediction without a model call. Disagreement
rows receive a 64-frame chronological union containing prioritized pivot/target
evidence and exact uniform coverage.

After the full original-pivot archive exists, submit:

```bash
CANDIDATE_DATE=YYYY-MM-DD \
sbatch slurm_longqa_qwen3_disagreement_verifier_full.sh
```

Override `PRIMARY_RUN` or `SECONDARY_RUN` when using non-default archive names.
The launcher fails before model startup if predictions or proof-pack metadata are
missing.
