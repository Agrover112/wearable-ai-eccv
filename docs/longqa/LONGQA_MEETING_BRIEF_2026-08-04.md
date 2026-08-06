# EgoLongQA Meeting Brief - 2026-08-04

## Current Best Pipeline

Our best system uses **Qwen3.5-27B** and scores **572/700 (81.71%)** after the
deterministic output repair described below. The raw 27B generations score
569/700 because three responses contain no final option letter. It runs on one
H100. On the 560-question held-out portion of our validation split, the final
27B answering stage averaged 25.21 seconds per question and never exceeded
59.57 seconds. The frame selections were read from cache during that run, so
the complete uncached selection-plus-answer pipeline still needs a separate
latency check against the workshop's 300-second limit.

The pipeline processes each question as follows.

### 1. Read The Video And Question

The input is one long first-person video, one multiple-choice question, and all
four answer options. Because the full video contains far more images than the
model can receive efficiently, we first choose a 64-image summary of the video.

### 2. Build Three Frame Pools

- **Global coverage:** sample images at regular intervals throughout the video.
  This preserves the overall sequence and protects against missing an event
  simply because it did not look relevant to a retrieval model.
- **Question-focused coverage:** sample 128 candidate images across the video
  and use SigLIP2 to compare them with the question and answer options. Keep
  images around the strongest matches, including nearby moments that show what
  happened immediately before or after the matched action.
- **Uncertainty-focused coverage:** a smaller Qwen3.5 model scores candidate
  moments by how uncertain their answer distribution is. Ambiguous moments and
  frames around the likely reference event receive higher priority.

These are frame-selection signals, not three final answer models. Their frame
lists can overlap. We take up to 24 question-focused images, about 24 globally
spaced images, and up to 16 uncertainty-focused images; remove duplicates; fill
any remaining positions from the same pools; and sort the final 64 images by
their original timestamps.

### 3. Answer Once With Qwen3.5-27B

All 64 chronological images are passed **together in one request** to a single
Qwen3.5-27B model, along with the complete question and four options. The model
is not shown answers or vote counts from the smaller models. Long-form reasoning
is disabled in the current best run, and the model returns one option letter.

There is therefore no disagreement verification or ensemble in this pipeline.
The smaller Qwen and SigLIP2 components help decide which images the 27B model
sees; Qwen3.5-27B alone makes the submitted answer.

### 4. Validate The Output

The 27B model produced a valid letter on 697 questions. On three questions it
started a longer explanation and reached the generation limit before returning
a letter. The intended label-free fallback is the fixed majority answer from
the five previously completed 9B runs. A Python membership edge case prevented
that fallback from being written in the original merged artifact; the exporter
now applies it correctly and records which rows were repaired. This correction
does not inspect the reference answers. The three fixed fallback answers are
correct, raising the submission-ready result from 569/700 to 572/700.

## Work For The Next Few Days

1. Run Qwen3.5-27B with a fixed 1,024-token reasoning budget on dev140, using
   exactly the same 64-image input. Require an explicit `Final Answer: X` and
   check both accuracy and the 300-second limit.
2. Run the reasoning variant on val560 only if it improves meaningfully over
   the current `120/140` development result.
3. Compare the reasoning and non-reasoning predictions to determine whether
   extra reasoning helps temporal order, repeated events, object identity, or
   reading text in the scene.
4. Freeze the final submission only after its 700 answers, ordering, parameter
   declaration, and runtime have all been checked.
