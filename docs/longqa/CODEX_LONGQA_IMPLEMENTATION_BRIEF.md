# Codex Implementation Brief: Wearable AI Challenge 3 / EgoLongQA

**Date:** 2026-07-09
**Track:** Wearable AI Workshop Challenge 3 — Long Video Q&A
**Goal:** Improve multiple-choice LongQA validation accuracy while reducing reliance on answer-option shortcuts.

---

## 0. Current State and Key Facts

### Best runs so far

| System | Accuracy | Correct | Notes |
|---|---:|---:|---|
| Qwen3-VL + vLLM, 4 frames, 224px | 0.5300 | 371/700 | Initial starter-kit behavior; `--max-frames 32` did not imply 32 sampled frames. |
| Qwen3-VL + vLLM, 32 frames, 224px | 0.6429 | 450/700 | First true 32-frame Qwen3 baseline. |
| Qwen3-VL + vLLM, 32 frames, 448px | 0.6629 | 464/700 | Larger per-frame image budget helps. |
| Qwen3-VL + SigLIP grounding, c128 top24 anc8 final32, 448px | 0.7000 | 490/700 | Question-conditioned frame retrieval helps. |
| Qwen3-VL + uniform64, 448px | 0.7071 | 495/700 | Strongest simple baseline. |
| Qwen3-VL + hybrid c128 top16 anc64 final64, 448px | **0.7114** | **498/700** | Current best full run. |

### Important warnings from dataset analysis

The validation split has strong multiple-choice shortcuts:

- Always choosing **C** scores **63.4%**.
- Always choosing the **shortest option** scores about **40.3%**.
- A video-blind “most typical option” heuristic scores about **45.4%**.
- About **79.1%** of questions contain explicit temporal markers such as `after`, `before`, `first`, `then`, or `during`.
- Videos are typically around **10 minutes**, 15 fps, about **9,000 frames** each.

Therefore raw accuracy alone is not enough. Every new run should report both leaderboard-style raw accuracy and shortcut-robust diagnostics.

A note from my side: Keep in mind that the final test set (hidden) might not have a similar bias and hence we need to be careful in that regard.

---

## 1. Near-Term Strategy

Do **not** pivot immediately to a new temporal architecture such as HieraMamba. It is relevant as a future grounding model, but integration risk is too high for the July 9 → August 8 window.

Prioritize cheap, reversible improvements around the existing strong pipeline:

1. Prompt-level evidence forcing.
2. Option-aware retrieval.
3. Temporal non-maximum suppression for retrieved frames.
4. Coarse-to-fine event windows using existing SigLIP/Qwen infrastructure.
5. Shortcut-robust evaluation and subset gating.
6. Lightweight ensembling/routing using existing full-run predictions.

---

## 2. Implementation Principles for Codex

When editing the repo:

1. Preserve existing script behavior by default.
2. Add new flags rather than changing default behavior.
3. Keep output JSONL schema backward compatible.
4. All new experiments must support resume/caching where possible.
5. All new runs must write:
   - predictions JSONL,
   - eval results JSON,
   - config metadata JSON,
   - optional grounding/debug JSONL.
6. Avoid launching full 700-video jobs for untested variants. Add subset support first.
7. Do not re-run expensive c256 grounding unless cached embedding reuse exists.

---

## 3. Task A — Add Prompt Variants

### Objective

Make prompt changes configurable so we can test whether Qwen3-VL attends better to object semantics, visual attributes, temporal anchors, and option-by-option evidence.

### Files to inspect

Start by searching for prompt construction in:

- `run_evaluation.py`
- `run_generate_longqa.py`
- `run_generate_longqa_grounded.py`
- model adapter files for Qwen/Qwen3-VL
- any dataset/task-specific LongQA prompt helper

Search patterns:

```bash
rg "multiple-choice|option|LongQA|question|A\)|B\)|C\)|D\)|prompt|messages" .
```

### Required CLI additions

Add a flag to all LongQA generation paths:

```bash
--prompt-variant {baseline,evidence_first,option_verify,temporal_anchor,anti_shortcut,combined}
```

Default must be `baseline` and should reproduce current behavior.

### Prompt variants to implement

#### 3.1 `evidence_first`

Use this system/user instruction around the existing question/options:

```text
You are answering a multiple-choice question about a long first-person video.

First identify the relevant time period(s) in the video. Pay special attention to:
- objects being handled or viewed,
- object attributes such as color, text, shape, quantity, and location,
- actions before/after the event mentioned in the question,
- whether the answer requires counting, ordering, or comparing options.

Evaluate the options against the visual evidence. Then output only the final option letter: A, B, C, or D.
```
Or maybe modify it based on your current understanding of the codebase since I got the above snippet from GPT5.5 standalone.

#### 3.2 `option_verify`

```text
You are answering a multiple-choice question about a long first-person video.

For each option A, B, C, and D, compare the option against the visual evidence in the video. Check object identity, attributes, quantities, spatial relationships, and actions. Reject options contradicted by the video. Choose the single option best supported by the video.

Output only the final option letter: A, B, C, or D.
```

#### 3.3 `temporal_anchor`

```text
The question may depend on temporal order. First find the anchor event mentioned in the question. Then inspect what happened immediately before or after it as required by the wording. Prefer evidence from the relevant time period over global impressions.

Output only the final option letter: A, B, C, or D.
```

#### 3.4 `anti_shortcut`

```text
Do not choose based on option letter frequency, option length, or which option sounds most typical. The correct answer must be supported by visual evidence from the video.

Output only the final option letter: A, B, C, or D.
```

#### 3.5 `combined`

Combine all ideas without requiring long chain-of-thought output:

```text
You are answering a multiple-choice question about a long first-person video.

Use the video evidence, not option priors. Do not choose based on option letter frequency, option length, or which option sounds most typical.

First locate the relevant time period(s). If the question contains words like after, before, first, then, earlier, later, or during, identify the anchor event and reason from the correct temporal neighborhood.

Evaluate each option against visible evidence. Pay attention to object identity, color, text, shape, quantity, location, spatial relationships, and actions.

Return only the final option letter: A, B, C, or D.
```

### Output parsing requirement

The model may still produce extra text. Make answer parsing robust:

1. Prefer an explicit final standalone letter.
2. Then parse `Answer: X`, `Final answer: X`, `(X)`, or `Option X`.
3. If multiple letters appear, prefer the final occurrence after the words `answer` or `final`.
4. Log raw output and parsed answer.

Again check the current code context and update this requirement accordingly.

### Acceptance criteria

- Existing baseline results are reproducible when `--prompt-variant baseline`.
- Every LongQA generation script accepts the prompt flag.
- Predictions include metadata: `prompt_variant`.
- A 20-sample smoke run completes for every prompt variant.

---

## 4. Task B — Add Shortcut-Robust Evaluation Metrics

### Objective

Add metrics that distinguish actual video reasoning from validation-set answer priors.

### New script

Create:

```bash
scripts/eval_longqa_diagnostics.py
```

### Inputs

```bash
python scripts/eval_longqa_diagnostics.py \
  --predictions path/to/predictions.jsonl \
  --annotations path/to/val_annotations.jsonl_or_dataset \
  --output path/to/diagnostics.json
```

Adjust the annotation loader to match the actual dataset format in the repo.

### Required metrics

Report:

1. `accuracy_raw`
2. `correct`
3. `total`
4. `margin_over_always_c` = `accuracy_raw - 0.6342857` if using the known val split prior, or compute from annotations dynamically.
5. `always_c_accuracy`
6. `always_shortest_accuracy`
7. `macro_letter_accuracy`
8. `per_letter_accuracy`
9. `answer_distribution_predicted`
10. `answer_distribution_gold`
11. `non_c_accuracy`
12. `c_gold_accuracy`
13. `non_shortest_gold_accuracy`
14. `temporal_question_accuracy`
15. `non_temporal_question_accuracy`
16. `category_accuracy`
17. `category_count`
18. `prompt_variant` if available
19. `run_id` if available

### Temporal cue lexicon

Use at least:

```python
TEMPORAL_CUES = {
    "after", "before", "first", "then", "earlier", "later", "during",
    "while", "when", "next", "previous", "prior", "following", "last",
    "start", "end", "beginning", "finally", "eventually"
}
```

### Acceptance criteria

- The script can evaluate any existing predictions JSONL.
- It writes a JSON file and also prints a concise table.
- It does not require video decoding.
- It handles missing category fields gracefully.

---

## 5. Task C — Add Stratified Dev Subset Support

### Objective

Prevent expensive full-run exploration. All new variants should first run on a fixed stratified 100–140 sample subset.

### New script

Create:

```bash
scripts/make_longqa_dev_subset.py
```

### Requirements

Inputs:

```bash
python scripts/make_longqa_dev_subset.py \
  --annotations path/to/val_annotations.jsonl_or_dataset \
  --n 140 \
  --seed 20260709 \
  --output configs/egolongqa_dev140_seed20260709.json
```

Stratify by available fields:

1. scene/category if available,
2. gold answer letter,
3. temporal vs non-temporal question.

If exact stratification is difficult, use greedy proportional sampling.

### Generation script integration

Add a shared flag to LongQA scripts:

```bash
--subset-file configs/egolongqa_dev140_seed20260709.json
```

The subset file should contain stable sample identifiers or video IDs. Generation should skip all other rows.

### Acceptance criteria

- Same subset file gives identical sample list across runs.
- Supports 20-sample smoke subset and 140-sample dev subset.
- Subset predictions can be passed to existing evaluator and new diagnostics evaluator.

---

## 6. Task D — Option-Aware SigLIP Retrieval

### Objective

Improve temporal grounding by using semantic clues from answer options, not only the question.

### Files to inspect

- `run_generate_longqa_grounded.py`
- any SigLIP retrieval helper
- frame sampling utilities
- grounding JSONL writer

### New CLI flag

```bash
--retrieval-query-mode {question,question_options,per_option_union,question_temporal_boost}
```

Default: `question`.

### Modes

#### 6.1 `question`

Current behavior. Use only the question text.

#### 6.2 `question_options`

Retrieve using a single text query:

```text
Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
```

#### 6.3 `per_option_union`

Run separate text queries:

```text
Question: {question}
Candidate answer: {option_a}
```

and similarly for B/C/D.

Then union top frames from each option-specific query. For example:

- `top_k_per_option = ceil(top_k / 4)`
- merge all retrieved frames,
- rerank by maximum score across queries,
- apply temporal NMS if enabled,
- add anchors,
- cap at `final_max_frames`.

Add optional flag:

```bash
--top-k-per-option INT
```

If absent, derive from `top_k`.

#### 6.4 `question_temporal_boost`

Append temporal cue instructions to retrieval query only:

```text
Find the frames relevant to answering this question, especially the event before or after the anchor event if the question asks about temporal order.
Question: {question}
Options: ...
```

### Grounding output metadata

Each `grounding.jsonl` row should include:

- `retrieval_query_mode`
- selected frame indices/timestamps
- source of each selected frame: `retrieved`, `anchor`, `window`, `option_A`, etc.
- score if available
- query text hash or short query label, not necessarily full long query

### Acceptance criteria

- Existing grounding result is reproducible with `--retrieval-query-mode question`.
- `question_options` and `per_option_union` run on a 20-sample smoke set.
- Selected frame counts and source labels are logged.

---

## 7. Task E — Temporal NMS for Retrieved Frames

### Objective

Avoid wasting retrieved-frame budget on many near-duplicate frames from the same short moment.

### New flags

```bash
--temporal-nms-seconds FLOAT
--temporal-nms-candidates INT
```

Defaults:

- `--temporal-nms-seconds 0` means disabled.
- `--temporal-nms-candidates` defaults to something larger than `top_k`, e.g. `4 * top_k`, so NMS has candidates to choose from.

### Algorithm

1. Score all candidate frames.
2. Sort by descending score.
3. Iterate candidates.
4. Accept a frame only if its timestamp is at least `temporal_nms_seconds` away from all already accepted retrieved frames.
5. Stop when `top_k` accepted, or when exhausted.
6. If not enough frames after NMS, backfill with next highest-scoring frames regardless of distance.
7. Add anchors and windows after NMS.
8. Sort final selected frames chronologically before passing to Qwen.

### Suggested experiment values

- `--temporal-nms-seconds 5`
- `--temporal-nms-seconds 10`
- `--temporal-nms-seconds 20`

Videos are ~10 minutes, so 10–20 seconds is reasonable for event diversity.

### Acceptance criteria

- NMS is optional and disabled by default.
- Frame selection remains deterministic.
- Grounding JSONL reports how many frames were removed by NMS.

---

## 8. Task F — Coarse-to-Fine Window Selection

### Objective

Approximate the benefit of hierarchical temporal grounding without integrating a new architecture.

### New flags

```bash
--coarse-to-fine
--coarse-candidate-frames 128
--num-windows 4
--window-radius-candidates 2
--frames-per-window 8
--global-anchor-k 32
--final-max-frames 64
```

### Algorithm

1. Uniformly sample `coarse_candidate_frames` from the full video.
2. Score candidates with SigLIP using chosen retrieval query mode.
3. Apply temporal NMS to select `num_windows` center frames.
4. Around each selected center, sample `frames_per_window` frames from the local temporal region.
   - If the existing candidate grid is used, choose ±`window_radius_candidates` candidate neighbors.
   - If frame extraction supports timestamps, sample a fixed time window around center, e.g. ±10 seconds.
5. Add `global_anchor_k` uniform anchors across the whole video.
6. Deduplicate frames/timestamps.
7. Sort chronologically.
8. Cap at `final_max_frames`.
9. Pass selected frames to Qwen.

### First experiments

Run on dev140 first:

```bash
# CFT-1
python run_generate_longqa_grounded.py \
  --video-folder ../egolongqa/val \
  --subset-file configs/egolongqa_dev140_seed20260709.json \
  --retrieval-query-mode question_options \
  --coarse-to-fine \
  --coarse-candidate-frames 128 \
  --num-windows 4 \
  --window-radius-candidates 2 \
  --frames-per-window 8 \
  --global-anchor-k 32 \
  --final-max-frames 64 \
  --prompt-variant combined \
  --model-type qwen \
  --llm-model Qwen/Qwen3-VL-8B-Instruct \
  --backend vllm \
  --tp 1 \
  --concurrency 1 \
  --batch-size 1
```

```bash
# CFT-2
python run_generate_longqa_grounded.py \
  --video-folder ../egolongqa/val \
  --subset-file configs/egolongqa_dev140_seed20260709.json \
  --retrieval-query-mode per_option_union \
  --coarse-to-fine \
  --coarse-candidate-frames 128 \
  --num-windows 5 \
  --window-radius-candidates 1 \
  --frames-per-window 6 \
  --global-anchor-k 32 \
  --final-max-frames 64 \
  --prompt-variant combined \
  --model-type qwen \
  --llm-model Qwen/Qwen3-VL-8B-Instruct \
  --backend vllm \
  --tp 1 \
  --concurrency 1 \
  --batch-size 1
```

You may create suitable SLURM scripts based on the starter kit and other shell scripts that we have created here.

### Acceptance criteria

- Works on 20-sample smoke set.
- Works on dev140.
- Does not require c256.
- Grounding output clearly identifies window centers and local frames.

---

## 9. Task G — Ensemble and Routing over Existing Runs

### Objective

Use complementarity between uniform64 and SigLIP/hybrid predictions. This can completely change in case we have some better models, so let's address the ensembling part later. But again nice to have that support soon.

### New script

Create:

```bash
scripts/ensemble_longqa_predictions.py
```

### Inputs

```bash
python scripts/ensemble_longqa_predictions.py \
  --annotations path/to/val_annotations.jsonl_or_dataset \
  --pred uniform64=path/to/uniform64/predictions.jsonl \
  --pred hybrid=path/to/hybrid/predictions.jsonl \
  --pred siglip32=path/to/siglip/predictions.jsonl \
  --mode agreement_category_route \
  --output output/egolongqa/ensemble_predictions.jsonl \
  --eval-output output/egolongqa/ensemble_results.json
```

### Modes

#### 9.1 `majority_vote`

Take majority over provided predictions. If tie, use priority order.

#### 9.2 `agreement_category_route`

Rule:

- If uniform64 and hybrid agree, use agreed answer.
- If they disagree:
  - prefer hybrid for categories where grounding helped:
    - `Travel-Sightseeing (Outdoors)`
    - `Gardening`
    - `Outdoor Activities and Sports`
    - `Daily Activities`
    - `Travel-Sightseeing (Indoors)`
    - `Hiking-Outdoors`
  - prefer uniform64 for categories where uniform coverage helped:
    - `Pets / social gatherings`
    - `Hobbies-Daily Activities`
    - `Sightseeing`
    - `Fashion Advice`
    - `Shopping`
  - fallback to hybrid if category missing.

#### 9.3 `learned_route_dev`

Optional later: learn a small routing table on dev subset only, but avoid overfitting full validation.

### Acceptance criteria

- The script aligns predictions by sample/video/question ID, not row index unless IDs are unavailable.
- It reports disagreement count.
- It reports per-category changes over each base run.
- It writes a prediction JSONL compatible with the existing evaluator.

---

## 10. Experiment Matrix to Run First

Run on `dev140` before full validation.

### Prompt-only experiments on uniform64

| ID | Base | Prompt | Expected value |
|---|---|---|---|
| P0 | uniform64 448px | baseline | Reference |
| P1 | uniform64 448px | evidence_first | Better object/action grounding |
| P2 | uniform64 448px | option_verify | Better option discrimination |
| P3 | uniform64 448px | temporal_anchor | Better before/after questions |
| P4 | uniform64 448px | anti_shortcut | Better non-C/non-shortest metrics |
| P5 | uniform64 448px | combined | Best candidate prompt |

### Retrieval experiments on hybrid64

| ID | Candidate frames | Retrieved | Anchors | Final | Query mode | NMS | Prompt |
|---|---:|---:|---:|---:|---|---:|---|
| R0 | 128 | 16 | 64 | 64 | question | 0s | baseline |
| R1 | 128 | 16 | 64 | 64 | question_options | 0s | combined |
| R2 | 128 | 16 | 64 | 64 | per_option_union | 0s | combined |
| R3 | 128 | 16 | 64 | 64 | question_options | 10s | combined |
| R4 | 128 | 16 | 64 | 64 | per_option_union | 10s | combined |
| R5 | 128 | 24 | 40 | 64 | question_options | 10s | combined |
| R6 | 128 | 12 | 48 | 64 | per_option_union + window1 | 10s | combined |

### Coarse-to-fine experiments

| ID | Query mode | Windows | Frames/window | Anchors | Final | Prompt |
|---|---|---:|---:|---:|---:|---|
| CFT1 | question_options | 4 | 8 | 32 | 64 | combined |
| CFT2 | per_option_union | 5 | 6 | 32 | 64 | combined |
| CFT3 | question_options | 4 | 10 | 24 | 64 | combined |

### Promotion rule to full 700

Promote an experiment to full validation only if it beats both current references on dev140 in at least 3 of the following 5 metrics:

1. raw accuracy,
2. margin over always-C,
3. non-C accuracy,
4. temporal-question accuracy,
5. category-average accuracy.

Current full-run references:

- uniform64 448px: 0.7071,
- hybrid c128 top16 anc64 final64 448px: 0.7114.

---

## 11. Suggested Full-Run Commands After Dev Selection

These are templates. Update output directories to include date and run ID. And also check current context and update these commands accordingly.

### Best prompt on uniform64

```bash
export QWEN_MAX_PIXELS=200704
export VLLM_QWEN_MAX_MODEL_LEN=32768

python run_evaluation.py \
  --task longqa \
  --model-type qwen \
  --llm-model Qwen/Qwen3-VL-8B-Instruct \
  --backend vllm \
  --tp 1 \
  --concurrency 1 \
  --video-folder ../egolongqa/val \
  --max-frames 64 \
  --frames-per-interval 64 \
  --batch-size 1 \
  --prompt-variant combined \
  --predictions output/egolongqa/qwen3_vl_8b_vllm_uniform64_px200704_prompt_combined/predictions.jsonl \
  --eval-output output/egolongqa/qwen3_vl_8b_vllm_uniform64_px200704_prompt_combined/results.json
```

### Option-aware hybrid64

```bash
export QWEN_MAX_PIXELS=200704
export VLLM_QWEN_MAX_MODEL_LEN=32768

python run_generate_longqa_grounded.py \
  --video-folder ../egolongqa/val \
  --candidate-frames 128 \
  --top-k 16 \
  --anchor-k 64 \
  --window-radius 0 \
  --final-max-frames 64 \
  --retrieval-query-mode question_options \
  --temporal-nms-seconds 10 \
  --prompt-variant combined \
  --grounder-model google/siglip-base-patch16-224 \
  --grounder-device cuda \
  --grounder-batch-size 32 \
  --model-type qwen \
  --llm-model Qwen/Qwen3-VL-8B-Instruct \
  --backend vllm \
  --tp 1 \
  --concurrency 1 \
  --batch-size 1 \
  --output output/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_qopts_nms10_prompt_combined/predictions.jsonl \
  --eval-output output/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_qopts_nms10_prompt_combined/results.json \
  --grounding-output output/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_qopts_nms10_prompt_combined/grounding.jsonl
```

---

## 12. What Not to Prioritize Right Now

Do not spend main engineering effort on:

1. Full HieraMamba integration.
2. c256/window grounding without precomputed/cached frame embeddings.
3. Naive natural-language timeline summaries.
4. Uniform128 at low resolution as a main path.
5. Qwen2.5/HF full sweeps unless used for an ensemble after small-subset evidence.

Rationale:

- HieraMamba-style methods are relevant but integration-heavy and not directly MCQ answerers.
- The current c256 jobs were killed during grounding and did not reach generation.
- The timeline prototype scored only 0.6643 despite two Qwen calls per sample.
- Uniform128 224px timed out and partial accuracy was not better than uniform64/hybrid64.

---

## 13. Deadline-Oriented Plan

### July 9–12: Prompt + diagnostics sprint

- Implement prompt variants.
- Implement diagnostics metrics.
- Implement dev subset support.
- Run prompt-only experiments on dev140.

### July 13–18: Option-aware retrieval sprint

- Implement retrieval query modes.
- Implement temporal NMS.
- Run R1–R6 on dev140.

### July 19–24: Coarse-to-fine + ensemble sprint

- Implement coarse-to-fine selection if R1–R6 are promising.
- Implement ensemble/routing script.
- Compare against uniform64 and hybrid64.

### July 25–31: Full-run candidates

- Promote only the top 2–3 variants to full 700 validation.
- Run diagnostics on all full candidates.
- Keep uniform64 and current hybrid64 as reference backups.

### August 1–6: Finalization

- Run final selected configuration.
- Prepare submission predictions.
- Prepare ablation table: prompt, retrieval, NMS, hybrid, ensemble.

### August 7–8: Freeze

- Freeze highest reliable raw-accuracy system.
- Also retain a robust fallback prediction file.

---

## 14. Final Expected Deliverables

Codex should produce or modify code so that the repo contains:

1. Prompt variants integrated into generation scripts.
2. `scripts/eval_longqa_diagnostics.py`.
3. `scripts/make_longqa_dev_subset.py`.
4. Option-aware retrieval modes in grounded generation.
5. Temporal NMS for frame selection.
6. Optional coarse-to-fine selection mode.
7. `scripts/ensemble_longqa_predictions.py`.
8. Updated README or experiment notes describing the new flags.
9. Smoke-test commands and expected output paths.

---

## 15. Minimal First Codex Prompt

Paste this into Codex first if doing the work incrementally:

```text
We are working on the Wearable AI Challenge 3 EgoLongQA codebase. Please implement prompt-variant support for LongQA generation without changing default behavior.

Tasks:
1. Search the repo for LongQA prompt construction in run_evaluation.py, run_generate_longqa.py, run_generate_longqa_grounded.py, and model adapters.
2. Add a CLI flag --prompt-variant with choices baseline,evidence_first,option_verify,temporal_anchor,anti_shortcut,combined. Default must be baseline.
3. Implement the prompt templates from CODEX_LONGQA_IMPLEMENTATION_BRIEF.md.
4. Ensure predictions JSONL includes prompt_variant metadata if the schema allows metadata; otherwise write a sidecar config JSON.
5. Improve final answer parsing for A/B/C/D while preserving raw model output.
6. Add a 20-sample smoke command to the README or a scripts/comments section.
7. Do not change unrelated behavior.

Acceptance criteria:
- Existing baseline path behaves the same with --prompt-variant baseline.
- All prompt variants run on a small subset.
- Raw output and parsed answer are both preserved.
```

After this is merged, proceed to diagnostics and subset support.
