# LongQA Final-Day Ensemble Audit

## Current Position

The strongest single direct answerer is Qwen3.5-27B with 64 endpoint-inclusive
frames. It scores `608/700` (`86.86%`). The current primary route uses that
answer by default and changes to the direct option-quota 27B answer only for
`BEFORE` and multi-time disagreements. The rule was fixed before the 560-row
complement completed and scores `613/700` (`87.57%`).

The direct-candidate audit deliberately excludes the Bayesian calibrator and
the historical candidate-blind judge outputs. Those are useful diagnostics but
are not independent visual answerers and should not be counted as ensemble
votes.

| Direct candidate | Dev140 | Held-out 560 | Full 700 |
| --- | ---: | ---: | ---: |
| Endpoint Qwen3.5-27B | 130 | 478 | 608 |
| Option-quota Qwen3.5-27B | 123 | 468 | 591 |
| Pivot with rotation averaging, 9B | 120 | 445 | 565 |
| Option-quota, 9B | 117 | 429 | 546 |
| Endpoint, 9B | 116 | 426 | 542 |
| SigLIP2 temporal pivot, 9B | 115 | 422 | 537 |
| Uncertainty pivot, 9B | 113 | 424 | 537 |

## What the Disagreements Show

The seven direct candidates have an oracle of `662/700` (`94.57%`): at least
one of them is correct on 662 questions. The endpoint and option-quota 27B pair
alone has an oracle of `643/700`. Thus the missing accuracy is largely an
answer-selection problem, but this does not mean that ordinary voting can
recover it.

Unweighted plurality across all seven candidates scores only `573/700`.
Weighting each vote by its smoothed dev140 accuracy reaches `582/700`. Both
fail because the six weaker systems are correlated: several retrieval-heavy
models can outvote endpoint while sharing the same missing event or distractor.
Model count is therefore not a trustworthy measure of evidence strength.

Agreement remains useful for identifying where arbitration is needed. When all
seven direct candidates agree with endpoint, endpoint is correct on `426/451`
questions (`94.46%`). When only one candidate supports endpoint, it is correct
on `15/31` (`48.39%`). A future verifier should operate only on low-agreement
rows, but it must inspect visual evidence rather than merely recount votes.

The current temporal route is the only completed rule with a clean held-out
gain. It adds five answers over endpoint on dev140 and held-out combined, and
adds three on the 560-row complement alone. Retrospective combinations can be
made to score higher, but they are not promoted here because repeatedly
choosing rules after reading the held-out answers would turn the validation set
into training data.

The machine-readable audit is stored at
`analysis/longqa_direct_vote_fusion_2026-08-07.json` and can be regenerated with
`scripts/analyze_longqa_direct_vote_fusion.py`.

## Classical Retrieval and Ranking

Reciprocal-rank fusion, BM25, and related information-retrieval methods require
ranked evidence. Our archived model outputs usually contain only one answer
letter, so applying reciprocal-rank fusion directly to those letters is not
well-defined. The dev-weighted hard-vote result above is the closest available
classical baseline and is clearly worse than endpoint.

Rank fusion is more appropriate before Qwen answers. The new RRF experiment
ranks 128 candidate frames separately for the question target and each of the
four answer options. It gives each frame reciprocal-rank credit, assigns half
of the total weight to the target and half equally across the options, then
uses MMR and a temporal separation rule to avoid selecting many near-identical
frames. Forty-eight endpoint anchors are retained and only 16 frames are
replaced by retrieval, so the experiment preserves the global coverage that
made endpoint sampling strong.

BM25 would become useful if we first created reliable OCR or caption text for
every candidate frame. Current object and caption caches do not cover all 700
videos consistently, so building that index now would be a larger and riskier
final-day project than rank fusion over the existing SigLIP2 features.

## Experiments Through the Deadline

All six dev140 jobs are independent and can run simultaneously when GPUs are
available:

```bash
sbatch slurm_longqa_qwen35_27b_uniform48_endpoint_dev.sh
sbatch slurm_longqa_qwen35_27b_uniform64_endpoint_thinking_dev.sh
sbatch slurm_longqa_qwen35_27b_endpoint_timestamps_verify_dev.sh
sbatch slurm_longqa_qwen35_27b_balanced_option_quota_dev.sh
sbatch slurm_longqa_qwen35_27b_endpoint_mmr_hybrid_dev.sh
sbatch slurm_longqa_qwen35_27b_endpoint_rrf_hybrid_dev.sh
```

Priority when only one H100 is available:

1. Run endpoint-48. It is the cleanest test of the reported distractor-density
   failure and should finish faster than endpoint-64.
2. Run direct endpoint reasoning. It creates a genuinely different direct
   candidate without using the historical candidate-blind judge.
3. Run timestamped option verification. It tests whether explicit temporal
   positions help without changing the proven frames.
4. Run balanced option-quota. It corrects the old one-center-per-option behavior
   and is the most relevant possible replacement for the temporal specialist.
5. Run MMR and RRF-MMR. Their difference isolates score averaging from robust
   rank fusion while keeping the same 48-plus-16 frame budget.

Promote a dev result to the 560-row complement only if it either exceeds
`130/140`, or recovers at least three endpoint errors while causing at most one
regression. For a weaker specialist, define the routing condition from dev140
before its held-out job completes. Do not promote a plain majority merely
because its candidate oracle is high.

The practical final sequence is: finish the six dev gates, immediately expand
at most the best one or two candidates, then test only endpoint-preserving
rules between direct answerers. Endpoint-only remains the runtime-safe
submission; the `613/700` temporal route remains the accuracy submission unless
a predeclared new candidate or route surpasses it.
