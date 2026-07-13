# EgoLongQA HieraMamba T4 Pilot Configuration

This document records the current Google Colab T4 prototype. It is a smoke-test
configuration, not the recommended final evaluation configuration.

## Hardware and models

- GPU: NVIDIA T4, approximately 15 GB usable VRAM.
- Grounder: official HieraMamba Ego4D-NLQ checkpoint.
- Video/text encoder: official EgoVLP checkpoint.
- Query converter: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Answer model: `Qwen/Qwen2.5-VL-3B-Instruct` in 4-bit NF4 with double
  quantization, FP16 compute, and SDPA attention.
- HieraMamba and Qwen run sequentially and are unloaded between stages.

## Current pipeline

1. Convert each original question into 1-4 answer-free declarative grounding
   queries using `wheels/HIERAMAMBA_QUERY_CONVERSION_PROMPT.md`.
2. Review and validate the generated JSON. The converter can misclassify
   multi-event questions, so its output must not yet be trusted blindly.
3. Extract EgoVLP video features and a separate token-level text feature array
   for every converted query.
4. Run HieraMamba independently for each query and retain up to five raw span
   proposals per query.
5. Uniformly sample a shared budget from the raw proposals, remove timestamps
   closer than 0.75 seconds, and sort the remaining frames chronologically.
6. Resize frames with a 200,704-pixel budget while preserving aspect ratio.
   The tested portrait videos became 504x364 pixels.
7. Pass at most 32 frames, the original question, and all MCQ options to Qwen.

The converted JSON preserves relations such as `after:q1`, but full
relation-aware proposal filtering is not implemented in the current Qwen
sampling prototype. It currently samples all raw proposals after timestamp
deduplication.

## Feature configuration

The pilot uses `--target-clips 64`, which uniformly places only 64 EgoVLP
windows across the complete video. Each window contains 16 sampled frames,
uses a 32/30-second duration, and produces one 256-dimensional feature.

This is intentionally coarse. The official Ego4D recipe uses an 8/30-second
stride, producing roughly 2,250 features for a ten-minute video. The 64-clip
pilot therefore has much poorer temporal resolution and has produced broad,
overlapping HieraMamba spans.

## Pilot results

| Video | Converted queries | Raw spans | Qwen | Gold | Result |
|---|---:|---:|---:|---:|---:|
| `0b38cc26c3cf4364` | 1 | 5 | C | C | correct |
| `1a5db01abab90677` | 2 | 10 | D | C | incorrect |
| `bbf6e152b6ee763a` | 3 | 15 | A | C | incorrect |

The current end-to-end pilot is 1/3. This is not enough data to estimate
accuracy, but it shows that unioning many coarse top-five spans can cover most
of the source video and defeat the purpose of localization.

## Next defensible experiment

- Use dense official-stride EgoVLP features on a small, easier subset.
- Compare equal 32-frame budgets: full-video uniform, HieraMamba top-1,
  relation-filtered multi-query spans, and localized frames plus global anchors.
- Add semantic validation/retry for converted query programs.
- Manually timestamp a small audit set before making localization-quality
  claims.

## Alternatives

Recommended order for a T4:

1. **Uniform anchors plus SigLIP2 retrieval.** Select about 16 global anchors
   and 16 question/option-relevant frames from a larger candidate pool. This is
   simpler and has already performed well in the feature-branch experiments.
2. **HieraMamba top-1 plus anchors.** Use only the strongest proposal from each
   converted query, then spend the remaining frame budget on global anchors.
   This avoids the near-full-video union created by top-five proposals.
3. **Qwen coarse-to-fine.** Give Qwen many low-resolution frames to identify
   candidate periods, then decode a smaller number of higher-resolution frames
   around those periods for the final answer.
4. **Dense HieraMamba features.** Retain the current architecture but replace
   the 64-clip smoke test with official-stride EgoVLP features and apply
   relation-aware filtering before frame sampling.
5. **Direct uniform Qwen baseline.** Use 32 uniformly sampled frames at the
   same resolution and model settings. Every grounding method must beat this
   equal-budget baseline to justify its complexity.
6. **Domain adaptation.** Manually annotate 100-300 declarative query spans and
   fine-tune the Ego4D-NLQ checkpoint. This is the highest-effort option and
   should follow a timestamped audit confirming that domain shift is the main
   failure source.

For the immediate pilot, option 1 is the strongest practical alternative.

## Relevant scripts

- `convert_hieramamba_queries.py`: prompt-based question conversion.
- `extract_egovlp_dev.py`: resumable video and per-query feature extraction.
- `prepare_hieramamba_smoke.py`: creates a HieraMamba experiment for all
  converted queries associated with one video.
