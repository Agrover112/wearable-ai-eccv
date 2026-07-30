# EgoLongQA Model Disagreement Analysis - 2026-07-30

## Scope

This analysis compares the six strongest completed full-validation systems:

- Qwen3 with uniform frames;
- Qwen3 with temporal-pivot frames;
- Qwen3 pivot/uniform verifier;
- Qwen3.5 with uniform frames;
- Qwen3.5 with temporal-pivot frames; and
- Qwen3.5 pivot/uniform verifier.

The purpose is to determine whether the remaining gain is more likely to come
from another frame-selection method or from choosing correctly among existing
model answers.

## New Completed Results

The resumed Qwen3.5 uniform run completed all 700 questions:

| System | Correct | Accuracy |
| --- | ---: | ---: |
| Qwen3.5 uniform64, 672px | 538/700 | 76.86% |
| Qwen3.5 temporal pivot | 537/700 | 76.71% |
| Qwen3 verifier | 539/700 | 77.00% |
| Qwen3.5 verifier | 539/700 | 77.00% |

The fixed three-system majority specified before the uniform run completed uses
Qwen3.5 temporal pivot, Qwen3.5 uniform, and the Qwen3 verifier. It obtains:

- `552/700` (`78.86%`) overall;
- `436/560` (`77.86%`) outside the repeatedly inspected dev140 subset;
- `78.84%` on temporal questions; and
- `82.03%` when the correct answer is not option C.

This is the strongest completed full-validation result. Its held-out gain means
the improvement is not explained only by tuning on dev140.

## Where The Majority Gain Comes From

The three inputs agree on 487 questions. They are correct on 439 of these and
all wrong on 48. No selector limited to these three answers can repair those 48
questions.

They disagree on 213 questions:

| Disagreement type | Questions | Majority correct | At least one correct |
| --- | ---: | ---: | ---: |
| Two models versus one | 189 | 107 | 170 |
| Three different answers | 24 | 6 | 20 |
| Total | 213 | 113 | 190 |

The large gap between `113` majority-correct and `190` oracle-correct answers
shows that answer selection is now a high-value problem. The theoretical oracle
for this fixed triple is `629/700` (`89.86%`), although an implementable router
will recover only part of that gap.

The all-different cases deserve special treatment. The current deterministic
tie-break follows Qwen3.5 temporal pivot and gets only `6/24`; Qwen3.5 uniform
gets `8/24`, while at least one input is correct on `20/24`.

## What Not To Use As A Router

Simple rules trained on dev140 using question category, temporal operator, or
question prefix did not generalize:

| Router | Dev140 | Remaining 560 |
| --- | ---: | ---: |
| Fixed three-system majority | 116/140 | 436/560 |
| Best system per temporal operator | 121/140 | 424/560 |
| Best system per category | 120/140 | 421/560 |
| Best system per question prefix | 120/140 | 424/560 |

These rules fit the small development split but reduce held-out accuracy.
Future routing must use per-example evidence and confidence features, with
grouped or nested cross-validation.

## Uncertainty And Temporal Chain Of Thought

The broad uncertainty-plus-dynamic-TCoT experiment has already been tested on
dev20. It used uncertainty to shortlist 64 of 128 candidate frames, split them
into four chronological sections, let Qwen select evidence within each section,
and added 16 uniform frames. It scored `15/20`, matched the existing temporal
pivot answers exactly, and remained below uniform sampling at `17/20`.

This does not justify running that pipeline over all 700 questions. A more
focused test is to invoke uncertainty-guided TCoT only when the three strong
systems disagree. It should first be evaluated on the 24 all-different cases,
then on uncertain two-versus-one cases. Unanimous answers should remain
untouched.

## Uncertainty-Guided Crops

The broad crop experiment has also been completed. Grounding DINO proposed
object crops, Qwen uncertainty ranked the full-frame/crop pairs, and the eight
highest-ranked crops were combined with 56 temporal-pivot frames. It scored
`106/140`, below the original crop-pair result (`109/140`) and temporal pivot
(`111/140`).

A narrower crop test remains defensible. For each proposed crop, compare:

1. answer uncertainty from the complete source frame; and
2. answer uncertainty from the source frame plus the crop.

Use a crop only when it produces a meaningful uncertainty reduction. Apply
this only to object-detail or spatial questions on which strong systems
disagree. This tests whether the crop adds visible evidence instead of merely
being ranked above other crops.

## Recommended Next Evaluation

1. Keep the fixed three-system majority as the deployment baseline.
2. Leave all 487 unanimous cases unchanged.
3. Run a candidate-constrained arbiter on the 24 all-different cases first.
4. Add answer entropy, option likelihood margins, verifier outputs, blind-model
   confidence, and retrieval statistics to the 189 two-versus-one cases.
5. Evaluate a small calibrated router with grouped or nested cross-validation.
6. Test conditional uncertainty-guided TCoT and delta-gated crops only where
   the router is uncertain.

The row-level six-system audit is stored in
`runs/egolongqa/model_disagreement_audit_2026-07-30/`. It contains 258
questions on which at least two of the six systems disagree, together with all
candidate answers, correctness labels, temporal operators, categories, vote
counts, and oracle status.
