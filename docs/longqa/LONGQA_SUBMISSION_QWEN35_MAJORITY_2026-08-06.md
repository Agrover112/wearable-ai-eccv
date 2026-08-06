# EgoLongQA Submission: Qwen3.5 Evidence Majority

## Submission Summary

This submission contains 700 EgoLongQA predictions and obtains **582/700
(83.14%)** exact-match accuracy on the available validation annotations. It is
a label-free ensemble of three direct model answers:

1. Qwen3.5-27B answering without extended reasoning.
2. The same Qwen3.5-27B checkpoint answering with a bounded reasoning budget.
3. Qwen3.5-9B answering from 64 evenly spaced frames that include both ends of
   the video.

The final answer is selected by a simple majority vote. This submission does
not use the Bayesian answer-prior calibrator, does not inspect reference answers
when producing a prediction, and does not include a verifier-generated answer
as a candidate.

## Input

Each example consists of:

- one long egocentric video;
- one question about events in that video; and
- four answer options labelled A, B, C, and D.

The video is too long to pass every frame to a vision-language model. Each
answer branch therefore receives a chronological set of 64 selected frames.

## Branch 1: Direct Qwen3.5-27B

The first 27B branch receives one mixed 64-frame view. That view combines three
sources of visual evidence:

- **Global coverage:** 24 frames distributed across the full video preserve the
  overall sequence and reduce the chance of completely missing an event.
- **Question-focused coverage:** up to 24 frames come from a temporal-pivot
  retrieval. SigLIP2 first compares 128 uniformly spaced candidate frames with
  the question and options. It identifies likely occurrences of the referenced
  event and the event being asked about, then includes nearby frames to retain
  local before-and-after context.
- **Uncertainty-focused coverage:** up to 16 frames come from a Qwen3.5-9B
  uncertainty pass. Candidate moments are scored using the spread of the
  model's A/B/C/D probabilities. Moments where the model is less certain are
  treated as places where additional visual evidence may be useful.

Duplicate frames are removed, remaining positions are filled from the available
pools, and the final 64 frames are sorted by their original timestamps. The
complete question and all four options are then shown to Qwen3.5-27B in one
request. Extended reasoning is disabled, and the model returns one option.

This branch scores **572/700 (81.71%)** after three incomplete generations are
replaced by their predetermined, label-free fallback answers.

## Branch 2: Reasoning Qwen3.5-27B

The second branch uses the same Qwen3.5-27B weights and the same mixed 64-frame
evidence. The controlled difference is decoding: reasoning is enabled with a
maximum budget of 1,024 tokens, followed by an explicit final option.

If a response fails to produce a valid final option, the runner performs a
short answer-only retry and then uses its predetermined fallback if necessary.
This branch scores **565/700 (80.71%)**. Although it is weaker alone, some of
its answers differ usefully from the direct 27B branch.

## Branch 3: Endpoint-Uniform Qwen3.5-9B

The third branch provides a deliberately different visual view. It selects 64
frames at regular intervals over the complete video while explicitly including
the first and last frames. It does not use question-conditioned retrieval for
this view. Qwen3.5-9B receives those frames, the complete question, and all four
options, then returns one option without extended reasoning.

This branch scores **542/700 (77.43%)**. Its purpose in the ensemble is not to
beat the 27B model alone, but to preserve endpoint and global evidence that the
mixed retrieval view may underrepresent.

## Final Vote

For every question, the three option letters are compared:

- if at least two branches agree, their answer is selected;
- if all three answers differ, the direct 27B answer is retained because it is
  the strongest standalone branch.

The branches are unanimous on 509 questions. Exactly two agree on 182
questions, and all three differ on 9 questions. Relative to the direct 27B
branch, voting changes 25 predictions: 16 changes correct an error, 6 replace a
correct answer with an error, and 3 exchange one wrong answer for another. The
net result is **582/700 (83.14%)**.

## Parameter Declaration

Exact parameter counts were read from the cached checkpoint tensor shapes:

| Checkpoint | Parameters |
|---|---:|
| Qwen3.5-27B | 27,781,427,952 |
| Qwen3.5-9B | 9,653,104,368 |
| SigLIP2 SO400M | 1,136,008,498 |
| **Total unique parameters** | **38,570,540,818** |

The two 27B branches reuse the same learned checkpoint. Calling the checkpoint
twice increases inference time, but it does not create a second set of learned
parameters. The Qwen models and SigLIP2 checkpoint are dense for this purpose,
so every declared parameter is used somewhere in the pipeline. The leaderboard
entries should therefore be:

- **Total params (billions): `38.570540818`**
- **Active params (billions): `38.570540818`**

This is a conservative system-level count of all distinct learned checkpoints,
including the retrieval model. It is not a count of parameter executions, which
would vary with the number of candidate frames and repeated model calls and is
better represented by measured runtime.

## Leaderboard Fields

Use the following values on the validation submission form:

| Field | Value |
|---|---|
| Track | `longqa` |
| Division | `large` |
| Model name | `Qwen3.5-27B/9B Evidence Majority` |
| Model license | `Qwen` |
| Open weights | checked |
| Total params (billions) | `38.570540818` |
| Active params (billions) | `38.570540818` |

The required LongQA row schema is `video_path` plus `mcq_answer`. The current
leaderboard requires `answers` only for the ConvQA and Proactive tracks. If an
upload reports a missing `answers` key, verify that **LongQA** rather than
ConvQA is selected before validating the file.

## Files

- Upload file:
  `submissions/egolongqa/qwen35_27b_thinking_endpoint_majority_2026-08-06/predictions.jsonl`
- Machine-readable declaration:
  `submissions/egolongqa/qwen35_27b_thinking_endpoint_majority_2026-08-06/submission_metadata.json`
- Source ensemble:
  `runs/egolongqa/qwen35_q27_thinking_endpoint_majority_full_2026-08-05/`

## Validation And Limitation

The upload file contains exactly 700 unique video IDs in annotation order and
one valid A/B/C/D answer per row. Individual answer calls remain below 300
seconds, but the completed fresh dev20 audit measured **698.2 seconds per
question** for the full sequential pipeline. SigLIP2 and uncertainty selection
alone account for 430.7 seconds per question. The current uncached sequential
implementation must therefore not be reported as meeting the workshop limit.
It requires a reduced or parallelized execution path whose accounting is
confirmed with the organizers before hidden-test submission.
