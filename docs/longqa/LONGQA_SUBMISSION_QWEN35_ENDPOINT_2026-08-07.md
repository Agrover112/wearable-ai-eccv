# EgoLongQA Submission: Qwen3.5-27B Endpoint-Uniform-64

## Result

This submission contains 700 EgoLongQA predictions and scores **608/700
(86.86%)** on the available validation annotations. It is a single-model,
label-free pipeline and does not use retrieval, ensembling, answer-prior
calibration, or reference answers during prediction.

## Method

For each video, 64 frames are sampled in chronological order. The first and
last video frames are always included, with 62 frames distributed uniformly
between them. Qwen3.5-27B receives these frames together with the complete
question and all four answer options, then directly returns one answer without
extended reasoning.

The held-out 560-row run scores `478/560` (`85.36%`). Combined with the
previously evaluated 140-row development partition, the strict key-aligned
full result is `608/700` (`86.86%`).

## Submission Fields

| Field | Value |
|---|---|
| Track | `longqa` |
| Division | `large` |
| Model name | `Qwen3.5-27B Endpoint-Uniform-64` |
| Model license | `Qwen` |
| Open weights | checked |
| Total params (billions) | `27.781427952` |
| Active params (billions) | `27.781427952` |

The checkpoint is dense, so all declared parameters are active during model
inference. OpenCV frame sampling introduces no additional learned parameters.

## Upload

Upload:

`submissions/egolongqa/qwen35_27b_endpoint_uniform64_2026-08-07/predictions.jsonl`

The file contains exactly 700 unique video IDs in annotation order and one
valid A/B/C/D answer per row. Select the **LongQA** track before validating;
the LongQA schema uses `video_path` and `mcq_answer`, whereas the other tracks
may request an `answers` field.

The held-out run took 14,454 seconds including model startup, approximately
25.81 seconds per question when amortized over 560 rows. This is below the
workshop's 300-second per-question limit.
