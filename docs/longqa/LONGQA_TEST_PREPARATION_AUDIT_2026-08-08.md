# EgoLongQA Test Preparation Audit - 2026-08-08

## Current Position

Our strongest validation artifact is the dual-view probability fusion pipeline:

- Qwen3.5-27B answers from 64 endpoint-inclusive uniform frames.
- SigLIP2 constructs a second 64-frame question-and-option-conditioned view.
- When the two direct answers disagree, Qwen scores all four answer letters under
  each view and the two normalized probability distributions are averaged.
- The result is `618/700` (`88.29%`). The policy changes 47 endpoint answers,
  producing 28 fixes, 18 regressions, and one wrong-to-wrong change.

The predeclared mean-log-probability variant scores `615/700` (`87.86%`). The
older frozen temporal route scores `613/700` (`87.57%`), and endpoint-only Qwen
scores `608/700` (`86.86%`).

## Replacement For The Old Dev140 Split

The July dev140 split was stratified by category, answer letter, and a binary
temporal cue. It nevertheless contains 130 endpoint-correct examples, while the
full-set expectation is about 122. It also contains only 14 endpoint/option-view
disagreements, versus an expectation of about 18 or 19. Its `92.86%` endpoint
score therefore gave an optimistic development signal.

Do not replace it with one more randomly chosen split. Use five disjoint folds:

```text
configs/egolongqa_balanced_fold0_of5_20260808.json
configs/egolongqa_balanced_fold1_of5_20260808.json
configs/egolongqa_balanced_fold2_of5_20260808.json
configs/egolongqa_balanced_fold3_of5_20260808.json
configs/egolongqa_balanced_fold4_of5_20260808.json
```

Each fold contains 140 examples. They jointly cover all 700 videos exactly once
and balance category, temporal operator, answer position, video-duration bin,
frozen endpoint correctness, and endpoint/option agreement. Endpoint scores
`122, 121, 122, 122, 121` across the five folds, instead of `130` on the old
dev set.

For expensive experiments, use fold 0 as the first gate and fold 1 as an
independent confirmation. For routing or threshold selection, use nested
five-fold evaluation: choose the rule on four folds and report it on the fifth,
then rotate the held-out fold. A rule should be promoted only if its combined
out-of-fold result improves endpoint and it does not depend on raw option-letter
identities.

Current fold scores are:

| Method | Fold correct counts | Full |
|---|---|---:|
| Endpoint Qwen | 122, 121, 122, 122, 121 | 608 |
| Frozen temporal route | 121, 122, 124, 123, 123 | 613 |
| Mean log-probability fusion | 123, 123, 123, 125, 121 | 615 |
| Mean probability fusion | 124, 121, 124, 127, 122 | 618 |

Probability fusion beats log-probability fusion on four of the five balanced
folds, although its gain over the temporal route is not uniform on every fold.

## What The Leaderboard Does And Does Not Tell Us

The validation board shows model name, total parameters, open-weight status, and
accuracy. It does not publish prompts, frame counts, retrieval algorithms, model
calls, or participant code. The private result records are not publicly readable.
Consequently, names such as `ambient-agent-v0.5` cannot be reverse-engineered
from the board.

Parameter count is not inference compute. Reusing Qwen3.5-27B several times with
different frame packs still contributes one copy of its weights. A system listed
near 27--29B can therefore perform retrieval, multiple Qwen passes, option-wise
scoring, and a final judge while remaining in the same parameter range. The
`EgoAssist-Qwen3.5-27B-OptRetrieval` name and its 28.66B declaration are
consistent with a 27B Qwen plus a retrieval model, but its exact implementation
is not disclosed.

Prompt engineering may matter, but our own controlled evidence does not support
a generic prompt as the missing several points. Timestamped option verification
fell to `116/140`, and several reasoning or summary prompts also underperformed.
The larger opportunity is to change what evidence is shown and to score complete
option claims, while keeping the final answer prompt short and deterministic.

## Error Anatomy Of The 618/700 Pipeline

The probability-fusion output has 82 errors:

| Diagnostic bucket | Errors | Meaning |
|---|---:|---|
| Endpoint and option view agree incorrectly | 35 | The disagreement gate never invokes probability scoring. Every direct candidate also misses these rows. |
| Fusion regression | 18 | Endpoint was correct, but averaging the two views changed it to a wrong answer. |
| A different direct candidate is correct, but both rank views miss | 19 | Useful evidence exists in another branch, especially uncertainty or pivot views. |
| One rank view places the correct option first, but averaging rejects it | 7 | The evidence is present; the aggregation rule suppresses it. |
| Every direct candidate and both rank views miss | 3 | A clear evidence or reasoning failure. |

The seven-member direct pool contains the correct answer for 44 of the 82 errors.
The remaining 38 are absent from the whole pool; 25 of those are unanimous wrong
answers. The direct-pool oracle is `662/700` (`94.57%`), which shows that stronger
arbitration could approach the current leaders, but only if it identifies the
right specialist rather than taking a majority vote.

Among the 82 fusion errors, the uncertainty branch is correct 25 times, the
rotation/pivot branch 22 times, and the original SigLIP and option branches 20
times each. This supports adding uncertainty as a third evidence view inside
the judge. It does not support replacing the primary model with uncertainty.

The highest error rates are Shopping (`14/72`), Gardening (`6/32`), and Daily
Activities (`22/127`). Common failures require:

- distinguishing repeated instances of similar objects;
- identifying exact first, last, before, or after occurrences;
- reading small prices, names, labels, years, and counts;
- satisfying both clauses of a compound answer rather than matching one clause;
- tracking where an object moved or how its state changed.

Many wrong options share one correct clause with the gold option and differ only
in the second object, time, price, or location. A single global answer score can
therefore be high even when one decisive clause is unsupported.

## Test-Phase Development Priorities

### 1. Cross-validated conservative probability gate

Preserve endpoint unless the alternative has strong support under label-invariant
features such as probability margin, view agreement, entropy, and rank stability.
Fit any threshold or small classifier with five-fold out-of-fold evaluation. Do
not use raw answer letters or select a rule from its score on the same fold.

### 2. Third uncertainty evidence view

For endpoint/option disagreements, score all four options under the uncertainty
frame pack as well. Compare robust three-view aggregation and a conservative
judge that sees identical option claims under all three evidence packs. This is
directly motivated by uncertainty being correct on 25 current fusion errors.

### 3. Atomic option verification for consensus-hard questions

For questions with multiple events or two-part options, split each option into
small factual claims. Retrieve evidence for each claim independently, preserve
chronological order, and ask Qwen whether every claim is visibly supported.
Only override an endpoint/option consensus when one option has complete support
and the current answer has a specific contradicted or unsupported clause.

This is the main route for the 38 errors missed by every current direct model.
It should first target repeated-object, first/last, OCR, price, count, and
state-change questions rather than all 700 examples.

## Three Test Candidates

Subject to container latency checks, reserve the three test submissions for:

1. The current 618/700 dual-view probability-fusion pipeline.
2. A cross-validated conservative fusion gate or three-view uncertainty fusion,
   whichever has the stronger out-of-fold result.
3. Atomic option verification on consensus-hard questions, with the current
   pipeline as fallback.

The endpoint-only or predeclared mean-log-probability pipeline remains the safe
fallback if either new method fails latency or cross-validation checks.

## Reproducible Artifacts

- Fold generator: `scripts/make_longqa_balanced_folds.py`
- Fold evaluator: `scripts/evaluate_longqa_balanced_folds.py`
- Fold summary: `analysis/egolongqa_balanced_folds_20260808.json`
- Fold scores: `analysis/longqa_balanced_fold_scores_20260808.json`
- Error analyzer: `scripts/analyze_longqa_fusion_errors.py`
- Error summary: `analysis/longqa_probability_fusion_error_summary_20260808.json`
- Row-level audit: `analysis/longqa_probability_fusion_errors_20260808.jsonl`
