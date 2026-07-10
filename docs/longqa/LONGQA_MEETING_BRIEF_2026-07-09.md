# EgoLongQA Meeting Brief - 2026-07-09

## Goal

We are working on Wearable AI Workshop Challenge 3 / EgoLongQA: answer long-horizon multiple-choice questions over egocentric videos. The practical goal has been to establish strong reproducible baselines, understand the cost/accuracy tradeoff of visual context, and add temporal grounding methods that can select the right evidence frames before calling the VLM.

## Data And I/O

Input data:

- Validation annotations: `data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl`
- Videos: `data/wearable-ai/egolongqa/val`
- Each sample contains a video, question, options, answer, and category.

Main outputs:

- Full run archives: `runs/egolongqa/<run_name>/`
- Starter-kit outputs: `data/wearable-ai/starter_kit/output/egolongqa/<run_name>/`
- `predictions.jsonl`: one row per sample, including raw answer, parsed MCQ letter, prompt variant, and metadata.
- `results.json` / `results_summary.json`: accuracy and summary metrics.
- `grounding.jsonl`: selected frame indices/timestamps/scores for grounded runs.
- Frame audits: saved outside the repo under `/scratch/inf0/user/agaur/wai-26/data/wearable-ai/frame_audits`.

## Model Stack

Primary VLM:

- `Qwen/Qwen3-VL-8B-Instruct` via vLLM on 1 H100 NVL.
- Main stable setting: `QWEN_MAX_PIXELS=200704`, approximately 448x448 input frames.
- `VLLM_QWEN_MAX_MODEL_LEN=32768`, `VLLM_GPU_MEMORY_UTILIZATION=0.92`.

Other baselines:

- `Qwen2.5-VL-7B` HF path for comparison.
- Llama 4 Scout 4xH100 SLURM script was created, but Qwen became the main experimental path because it ran reliably and gave strong baselines.

Grounder:

- `google/siglip-base-patch16-224`.
- Used only to score candidate frames against text queries; final answer still comes from Qwen3-VL.

## Key Implementation Changes

Frame sampling:

- Fixed the early ambiguity around `--max-frames`. The original "default32" runs only capped the max frames; they still used the default 4 frames per interval.
- Added explicit `--frames-per-interval` so runs like 32, 64, and 128 frames are truly sampled.

Context accounting:

- Added prompt-token/context-fill tracking so we know how close each setting gets to the VLM context budget.
- This made 64 frames at 448px look realistic: about 39% max fill of a 32K context.

Image size control:

- Added `QWEN_MIN_PIXELS` and `QWEN_MAX_PIXELS` support.
- Important settings:
  - `50176` ~= 224x224, about 64 visual tokens/frame.
  - `200704` ~= 448x448, about 256 visual tokens/frame.
  - `451584` ~= 672x672, about 576 visual tokens/frame.
  - `802816` ~= 896x896, about 1024 visual tokens/frame.

Resume/caching:

- Uniform and grounded generation can resume existing `predictions.jsonl`.
- Grounded runs also resume `grounding.jsonl`, which matters because SigLIP candidate scoring can become the bottleneck.

Prompt and parsing:

- Added robust answer parsing and preserved both raw and parsed answers.
- Added prompt variants: `baseline`, `evidence_first`, `option_verify`, `temporal_anchor`, `anti_shortcut`, and `combined`.
- We kept the input format as question + options for reproducibility with previous baselines.

Diagnostics:

- Added stratified dev subsets: `configs/egolongqa_dev20_seed20260709.json` and `configs/egolongqa_dev140_seed20260709.json`.
- Added shortcut-aware diagnostics: always-C, shortest-option, non-C accuracy, temporal vs non-temporal accuracy, category breakdowns, and prediction distribution.
- Added rolling/prefix analysis for partial runs.

## Methods Tried

Uniform frame scaling:

- Feed uniformly sampled frames from the whole video directly to Qwen.
- This is the simplest way to improve narrative coverage.
- It worked surprisingly well: 4 -> 32 -> 64 frames gave large gains.

SigLIP temporal grounding:

- Uniformly sample candidate frames, score them against the question/options using SigLIP, then pass only selected frames to Qwen.
- The baseline grounded setup used top retrieved frames plus uniform anchors.
- Why anchors matter: SigLIP can find local evidence, but EgoLongQA often asks for relations across time; anchors preserve global narrative context.

Hybrid grounding:

- Best current idea: combine many uniform anchors with a smaller number of SigLIP-selected frames.
- This preserves the "story" of the video while injecting high-similarity evidence frames.

Frame audits:

- Added a utility to export selected frames/contact sheets to scratch.
- This lets us inspect which frames helped or hurt, especially for cases where grounding improved over uniform sampling.

Timeline summary prototype:

- Tried a two-stage method: summarize temporal timeline, then answer with selected frames.
- It did not help enough and added a second VLM call per sample, so it is currently de-prioritized.

Offline routing/ensemble:

- Existing full predictions from uniform64 and hybrid are complementary.
- A quick category/agreement routing experiment over existing predictions reached 0.7214 on validation, but this should be treated as an analysis signal, not the final submission strategy yet.

## Validated Full-Run Results

| Run | Frames / Method | Resolution | Accuracy | Runtime | Notes |
|---|---:|---:|---:|---:|---|
| Qwen2.5-VL HF default | effectively 4 | default | 0.5300 | - | Early baseline |
| Qwen3-VL vLLM default | effectively 4 | default | 0.5300 | - | Early baseline |
| Qwen2.5-VL HF 32 frames | 32 uniform | default | 0.6543 | 5h25m | Strong jump from 4 frames |
| Qwen3-VL 32 frames | 32 uniform | 224px | 0.6429 | 3h40m | Faster than HF path |
| Qwen3-VL 32 frames | 32 uniform | 448px | 0.6629 | 3h42m | Higher res helped |
| Qwen3-VL SigLIP | c128 top24 + 8 anchors | 448px | 0.7000 | 9h29m | First strong grounding result |
| Qwen3-VL uniform64 | 64 uniform | 448px | 0.7071 | 7h42m | Strongest simple baseline |
| Qwen3-VL timeline | 64 summary + 32 answer | 448px | 0.6643 | 11h30m | Not worth continuing now |
| Qwen3-VL hybrid | c128 top16 + 64 anchors, final64 | 448px | 0.7114 | 12h02m | Best validated full run |
| Qwen3-VL uniform128 | 128 uniform | 224px | partial 0.6976 | timed out | 668/700 predictions only |

Main inference:

- More temporal coverage is the dominant improvement so far.
- 448px matters because many questions involve signs, labels, prices, object identity, and text in the scene.
- Uniform64 at 448px is a very strong baseline because it captures both temporal narrative and visual detail.
- Pure retrieval can miss narrative continuity; hybrid retrieval + anchors is safer.
- Naive timeline summarization is not currently competitive.

## Current Experiments

Three dev140 jobs were launched on 2026-07-09:

- `qwen3_vl_8b_vllm_uniform64_px200704_prompt_combined_dev_2026-07-09`
  - uniform64, 448px, `combined` prompt.
  - Current artifact observed: 94/140 predictions written.

- `qwen3_vl_8b_vllm_grounded_per_option_union_nms10_prompt_combined_dev_2026-07-09`
  - SigLIP c128, top16, 64 anchors, final64, per-option-union retrieval, temporal NMS=10s, `combined` prompt.
  - Current artifact observed: grounding cache started, 12 rows written.

- `qwen3_vl_8b_vllm_cft_question_options_nms10_prompt_combined_dev_2026-07-09`
  - Coarse-to-fine grounding, question+options retrieval, temporal NMS=10s, `combined` prompt.
  - Current artifact observed: grounding cache started, 13 rows written.

No final summaries for these three dev jobs were present at the time this brief was written.

## Bottlenecks

- Full validation runs take 8-12+ hours for strong settings.
- Candidate-heavy grounding, especially c256/windowed selection, is too slow without better embedding caching.
- Uniform128 at low resolution timed out and did not clearly beat uniform64/hybrid in partial results.
- Validation set has answer-option biases, so raw accuracy alone can overstate progress. We now track non-C, shortcut baselines, temporal subsets, and category metrics.
- Grounding is useful, but the model still needs narrative continuity across distant events; selecting isolated "relevant" frames is not enough.

## Recommended Next Steps

Short term:

- Finish the three dev140 jobs and run diagnostics on them.
- Compare each new method against two references: uniform64 and hybrid.
- Promote only methods that improve raw accuracy and at least one robustness metric: non-C accuracy, temporal-question accuracy, category macro accuracy, or margin over shortcut baselines.

Next experiment queue:

- Prompt ablations on dev140: `baseline`, `evidence_first`, `option_verify`, `temporal_anchor`, `anti_shortcut`, `combined`.
- Retrieval ablations on dev140: `question_options`, `per_option_union`, `question_temporal_boost`.
- Temporal NMS ablations: 0s, 5s, 10s, 20s.
- Coarse-to-fine variants with fewer windows or fewer final frames to reduce runtime.
- Full validation only for the top 2-3 dev winners.

Higher-risk ideas:

- More advanced temporal models like HieraMamba are interesting, but they are probably not the next best use of time until the lighter grounding/prompt/routing experiments saturate.
- If we revisit advanced temporal grounding, first build reusable video/frame embeddings so experimentation is not blocked by repeated candidate scoring.

Most promising current direction:

- Keep uniform64 at 448px as the reliability baseline.
- Improve hybrid grounding so it keeps enough global anchors for narrative continuity while using SigLIP to add evidence frames.
- Use frame audits to understand which selected frames are actually responsible for improvements and to catch retrieval failures early.
