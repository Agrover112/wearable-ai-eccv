# LongQA Pre-Test Deep Push - 2026-08-08

## Current Safe Submissions

- **Accuracy submission:** endpoint Qwen3.5-27B by default, with option-quota
  Qwen3.5-27B used only for `BEFORE` and multi-time disagreements. This frozen
  route scores `613/700` (`87.57%`).
- **Simpler submission:** endpoint-inclusive Qwen3.5-27B scores `608/700`
  (`86.86%`) and avoids the second retrieval and answer branch.

These artifacts remain unchanged while the experiments below run.

## What the Final Audit Found

The current route has 87 errors. At least one existing direct answerer is
correct on 49 of them, while every direct answerer is wrong on the other 38.
Twenty-five errors are unanimous across all seven direct candidates. This
separates two remaining problems:

1. **Arbitration:** 49 answers already exist but ordinary voting cannot identify
   them reliably.
2. **Evidence or reasoning:** 38 answers require a genuinely better view or
   better inference, not a different vote over existing letters.

Unweighted voting scores `573/700`, dev-weighted hard voting scores `582/700`,
and the strongest conservative hard-consensus rule tested scores below the
`613/700` route. More weak candidates therefore do not provide a useful vote.

No gold answer is included in the Qwen prompt. Sample keys, chronological frame
order, proof-pack sizes, and endpoint indices were rechecked. The `answer`
field retained in archived rows is output metadata; prompt construction uses
only `question` and `mcq_options`.

## Literature Check

The [BlackSwanSuite paper](https://arxiv.org/abs/2412.05725) studies inference
about deliberately hidden unexpected events from pre-event and post-event
context. EgoLongQA is not the same task, but the boundary-state idea is relevant:
the exact beginning and ending states can disambiguate first, last, and state
change questions. In our controlled experiment, endpoint-inclusive sampling
beat legacy and midpoint grids by seven and nine dev answers.

[Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
combines complete ranked lists using `sum(1 / (60 + rank))`. It cannot be
properly applied to one-letter outputs. Our frame-level RRF experiment failed
because its evidence overlapped ordinary MMR by `62.85/64` frames, so the
rankings lacked diversity. The new experiment instead obtains complete A-D
rankings from two genuinely different evidence views before applying RRF.

[VideoJudge](https://arxiv.org/abs/2509.21451) supports multimodal rather than
text-only judging, but our out-of-the-box VideoJudge-7B runs reached only
`114/140` and `113/140`. [VidCtx](https://arxiv.org/abs/2412.17415) supports
segment-level decisions and max pooling, while
[HiMu](https://arxiv.org/abs/2603.18558) supports decomposing compositional
questions into atomic predicates. Both remain promising test-phase work, but
implementing either fully is riskier than the two controlled runs below.

## Experiment 1: Boundary-Guarded Midpoint

Use 62 midpoint-phase interior frames plus the exact first and last frames.
This preserves the two boundaries that endpoint sampling needs while testing a
different interior phase that recovered four endpoint dev errors on its own.
It changes only frame positions; model, resolution, prompt, frame budget, and
reasoning setting remain fixed.

Because the deadline is August 8 at 13:59 Berlin time, run dev140 and val560
together if two H100s are available:

```bash
sbatch slurm_longqa_qwen35_27b_endpoint_guarded_midpoint_dev.sh
sbatch slurm_longqa_qwen35_27b_endpoint_guarded_midpoint_val560.sh
```

Expected durations are approximately 2.2 hours and 4 hours respectively,
assuming queue and node behavior match the completed endpoint run.

After both complete:

```bash
bash scripts/merge_longqa_qwen35_27b_guarded_midpoint.sh
```

## Experiment 2: Evidence-View Option Rank Fusion

Endpoint and option-quota disagree on only 17 dev rows. For each disagreement,
Qwen scores all four option letters once using endpoint evidence and once using
option-quota evidence. The evaluator reports view consensus, mean probability,
mean log probability, candidate-restricted scoring, cross-view confirmation,
and genuine RRF. This requires 34 dev model calls rather than another 140-row
answer pass.

Run now:

```bash
sbatch slurm_longqa_qwen35_27b_evidence_rank_fusion_dev.sh
```

Run the following only if a label-free policy reaches at least `132/140` or
changes endpoint answers with no regressions:

```bash
sbatch slurm_longqa_qwen35_27b_evidence_rank_fusion_val560.sh
```

After val560 completes, replace `POLICY` with the promoted policy reported by
the dev summary:

```bash
bash scripts/merge_longqa_qwen35_27b_evidence_rank_fusion.sh POLICY
```

## Lower Priority

The repaired direct-thinking job is valid but likely needs roughly three hours
for dev140 and substantially longer for all 700 rows. It cannot realistically
produce a complete pre-deadline submission unless spare compute is available:

```bash
sbatch slurm_longqa_qwen35_27b_uniform64_endpoint_thinking_dev.sh
```

Do not spend the remaining window on another SigLIP/MMR variant, timestamped
prompt, hard-vote ensemble, VideoJudge rerun, or full HiMu implementation. The
completed evidence does not support those as short-path improvements.
