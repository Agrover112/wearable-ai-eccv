# Qwen3.5 Agentic EgoLongQA Evaluation

## Motivation

This experiment tests a training-free, Symphony-inspired agentic pipeline for
EgoLongQA. The goal is to determine whether role-conditioned reasoning over a
shared temporal proofpack can improve Qwen3.5 answers while keeping model
inference within the five-minute per-question limit.

The dev20 run is a gate before a full dev140 evaluation. Results below distinguish
completed evaluations from partial outputs so that the interrupted agentic run is
not presented as a final dev20 score.

## Protocol

- Evaluation split: the 20-question dev20 configuration used by the existing
  parent temporal-pivot experiment.
- Parent baseline: Qwen3.5-9B receives a 64-frame question-conditioned temporal
  proofpack.
- Agentic method: a hypothesis agent proposes an answer and evidence needs;
  temporal and visual agents inspect the shared proofpack in parallel; an
  organizer resolves their reports into the final multiple-choice answer.
- The four roles are calls to the same Qwen3.5 model with different prompts, not
  independently trained or independently initialized models.
- All inter-agent messages and saved traces use strict JSON schemas.
- Proofpack construction selects 64 final frames from 128 candidates.
- Qwen3.5 revision:
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Qwen3-VL-Embedding-8B revision:
  `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- Qwen context length: 65,536 tokens.
- Maximum image pixels: 451,584.
- Hardware: one H100 for each submitted job.
- Agent generation limits: 768 hypothesis tokens, 1,024 tokens for each
  specialist, and 768 organizer tokens.
- Official budget: at most 300 seconds per question for question-conditioned
  proofpack retrieval plus model calls. Frame extraction and data loading are
  tracked separately.

The intended dev140 follow-up uses
`configs/egolongqa_dev140_seed20260709.json`, but it did not begin because its
dependency chain was blocked by the dev20 failure.

## Metrics

- Accuracy is exact multiple-choice accuracy.
- The partial agentic accuracy is computed only over the 16 completed samples.
- The comparable parent score is recomputed over the same 16-sample prefix.
- A fix is a sample where the parent is wrong and the agentic method is correct.
- A regression is a sample where the parent is correct and the agentic method is
  wrong.
- Budgeted inference time is proofpack retrieval plus all four agent calls.
- Full pipeline time additionally includes evidence and frame preparation.

## Runs

| Job | Experiment | State | Elapsed | Exit |
| --- | --- | --- | ---: | ---: |
| 49276501 | dev20 parent and timed proofpack | Completed | 01:01:19 | 0 |
| 49276502 | dev20 agentic | Failed after 16/20 | 00:34:59 | 1 |
| 49277152 | dev140 timed proofpack refresh | Pending, dependency never satisfied | 00:00:00 | - |
| 49277154 | dev140 agentic | Pending behind 49277152 | 00:00:00 | - |

The table reflects the Slurm state checked on 2026-07-27. The two dev140 jobs
have produced no evaluation results.

## Results

### Accuracy

| Method | Evaluated | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Parent temporal-pivot, full dev20 | 20 | 12 | 60.00% |
| Parent temporal-pivot, matched completed prefix | 16 | 9 | 56.25% |
| Agentic, partial completed prefix | 16 | 12 | 75.00% |

On the 16 samples that completed, the agentic output differed from the parent on
four questions. It fixed three parent errors, introduced no regressions, and had
one disagreement where both methods were wrong.

The 75.00% value is not a final dev20 result. Four questions are missing, and
the interrupted sample ordering can make the completed prefix easier or harder
than the full set.

### Timing

| Component | Mean | P95 | Maximum |
| --- | ---: | ---: | ---: |
| Question-conditioned proofpack retrieval | 108.86 s | 175.52 s | 175.52 s |
| Evidence and frame preparation | 46.07 s | 76.21 s | 76.21 s |
| Four-agent orchestration | 67.08 s | 75.49 s | 75.49 s |
| Budgeted inference | 175.94 s | 238.26 s | 238.26 s |
| Full pipeline wall time | 222.01 s | 314.47 s | 314.47 s |

All 16 completed samples stayed below the official 300-second model-inference
budget. Two full pipeline wall times exceeded 300 seconds when frame preparation
was included. Retrieval, rather than agent orchestration, was the largest
budgeted cost.

The parent prompts used a mean of 27,988 tokens, or 42.71% of the 65,536-token
context. The maximum was 28,247 tokens, or 43.10%.

## Interpretation

The partial run is encouraging: on a matched prefix, the agentic organizer
preserved every parent-correct answer and recovered three parent mistakes.
However, 16 questions are too few to establish a reliable gain, and the failed
run prevents a valid full-dev20 comparison.

The timing result is stronger operational evidence. Every completed question met
the official inference budget, with about 124 seconds of mean headroom. The
orchestration itself averaged about 67 seconds; optimizing retrieval would offer
more latency reduction than removing an agent role.

No conclusion can yet be drawn for dev140 because neither queued dev140 job
started.

## Issues And Fixes

- The dev20 agentic job failed on sample 17,
  `bbf6e152b6ee763a.mp4`, while answering: "After I purchased the Freeze Time
  card, how did the market change, and what dice rolls by the opponent enabled
  that change?"
- The hypothesis response reached the exact 768-token generation cap. vLLM
  returned `finish_reason=length`, leaving an unterminated JSON string and
  causing `json.loads` to fail. This is an output-cap failure, not an evaluated
  wrong answer.
- The run can resume from its 16 saved predictions after increasing the
  hypothesis token cap or making the hypothesis schema more compact.
- Job 49277152 was submitted with an `afterok` dependency on the failed dev20
  job, so Slurm marks the dependency as never satisfied. Job 49277154 remains
  pending behind that blocked job. The dev140 chain must be resubmitted or
  released after dev20 completes.
- Before submission, the separate high-reasoning audit identified and prompted
  fixes for the exact dev20 proofpack prerequisite, inclusion of
  question-conditioned retrieval in the timing budget, model and input
  fingerprinting, output-directory tracking, and wording that could imply the
  role-conditioned reports were independent agents. The re-audit found no
  remaining blocking or high-confidence issues.

## Reproduction

The launchers are:

```bash
sbatch slurm_scripts/qwen35_parent_pivot_dev20/run.sh
sbatch slurm_scripts/qwen35_agentic_dev20/run.sh
sbatch slurm_scripts/qwen35_parent_pivot_dev140_agentic/run.sh
sbatch slurm_scripts/qwen35_agentic_dev140/run.sh
```

The current dev20 agentic launcher retains the 768-token hypothesis limit, so it
should be updated before resubmission. Existing JSONL outputs support resuming
the remaining four samples rather than recomputing the first 16.

## Artifact Index

- Parent predictions and evaluated results:
  `runs/egolongqa/qwen35_parent_temporal_pivot_dev20/`
- Parent summary:
  `runs/egolongqa/qwen35_parent_temporal_pivot_dev20/results_summary.json`
- Partial agentic predictions and traces:
  `runs/egolongqa/qwen35_agentic_dev20/`
- Parent Slurm output:
  `slurm_outputs/qwen35_parent_pivot_dev20/`
- Agentic failure log:
  `slurm_outputs/qwen35_agentic_dev20/q35-agentic-dev20_49276502.err`
- Agentic progress log:
  `slurm_outputs/qwen35_agentic_dev20/q35-agentic-dev20_49276502.out`
- Agentic implementation:
  `baselines/longqa/agents/`
- Main agentic runner:
  `baselines/longqa/run_generate_longqa_agentic.py`
- Agentic launcher:
  `slurm_scripts/qwen35_agentic_dev20/run.sh`
