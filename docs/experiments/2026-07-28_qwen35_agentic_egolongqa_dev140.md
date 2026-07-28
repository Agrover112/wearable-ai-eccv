# Qwen3.5 Agentic EgoLongQA Dev140

## Motivation

This run evaluates the training-free, role-conditioned Qwen3.5 agentic pipeline
on the canonical 140-question EgoLongQA development subset. It follows the
initial dev20 test and uses a larger hypothesis generation limit to avoid the
JSON truncation observed there.

## Protocol

- Split: `configs/egolongqa_dev140_seed20260709.json`.
- Parent method: Qwen3.5-9B over a 64-frame temporal-pivot proofpack.
- Agentic method: hypothesis compiler, parallel temporal and visual specialists,
  and an organizer. All roles use the same Qwen3.5-9B checkpoint.
- Qwen3.5 revision:
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Qwen3-VL-Embedding-8B revision:
  `2c4565515e0f265c6511776e7193b22c0968ddc7`.
- Generation limits: 1,536 hypothesis tokens, 1,024 specialist tokens, and 768
  organizer tokens.
- Hardware: one H100 per job.
- Official inference budget: 300 seconds per question, excluding frame
  extraction and data loading.

## Metrics

Accuracy is exact multiple-choice accuracy. The partial agentic result is
reported only for the 12 successfully completed questions and compared against
the parent predictions for those same questions. Budgeted time includes
question-conditioned proofpack retrieval and all agent calls. Full pipeline time
also includes evidence and frame preparation.

## Runs

| Job | Experiment | State | Elapsed | Exit |
| --- | --- | --- | ---: | ---: |
| 49279291 | dev140 proofpack and parent | Completed | 06:02:45 | 0 |
| 49279292 | dev140 agentic | Failed after 12/140 | 00:27:34 | 1 |

## Results

| Method | Evaluated | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Parent temporal-pivot, full dev140 | 140 | 112 | 80.00% |
| Parent temporal-pivot, matched prefix | 12 | 8 | 66.67% |
| Agentic, partial matched prefix | 12 | 9 | 75.00% |

Across the 12 completed questions, the agentic method and parent disagreed
twice. The agentic method fixed one parent error, introduced no regressions, and
changed one parent-wrong answer to a different wrong answer.

The 75.00% agentic number is not a dev140 result: 128 questions are missing.

| Timing component | Mean | P95 | Maximum |
| --- | ---: | ---: | ---: |
| Proofpack retrieval | 99.90 s | 195.39 s | 195.39 s |
| Evidence and frame preparation | 43.56 s | 96.87 s | 96.87 s |
| Agent orchestration | 63.70 s | 73.47 s | 73.47 s |
| Budgeted inference | 163.59 s | 256.11 s | 256.11 s |
| Full pipeline | 207.15 s | 352.98 s | 352.98 s |

All 12 completed questions met the official 300-second inference budget. One
full pipeline time exceeded 300 seconds when frame preparation was included.

## Interpretation

The completed parent run establishes an 80.00% result on the canonical dev140
subset. The agentic prefix remains mildly encouraging, with one net correction
and no regression, but 12 questions are far too few for a meaningful comparison.
The current evidence supports feasibility under the official time budget, not an
accuracy claim for the full agentic method.

Retrieval again costs more time than the four-agent orchestration, averaging
99.90 seconds versus 63.70 seconds.

## Issues And Fixes

The agentic run failed on question 13 for
`1c8033f0712bae4d.mp4`. The visual specialist produced exactly 1,024 completion
tokens and stopped with `finish_reason=length`, leaving an unterminated JSON
string. The response repeated long lists of frame IDs in its summary.

The increased 1,536-token hypothesis limit avoided the previous hypothesis
failure, but the unchanged specialist limit exposed the same failure mode in the
next role. A rerun should either compact the specialist prompt and schema output
or raise the specialist generation limit before resuming from the 12 saved
records.

## Reproduction

```bash
sbatch slurm_scripts/qwen35_parent_pivot_dev140_agentic/run.sh
sbatch --dependency=afterok:<parent-job-id> \
  slurm_scripts/qwen35_agentic_dev140/run.sh
```

The current agentic output contains 12 aligned prediction and trace records, so
the runner can resume rather than recompute them after the specialist truncation
issue is fixed.

## Artifact Index

- Parent results:
  `runs/egolongqa/qwen35_parent_temporal_pivot_dev140_schema3/results.json`
- Parent summary:
  `runs/egolongqa/qwen35_parent_temporal_pivot_dev140_schema3/results_summary.json`
- Parent proofpack:
  `runs/egolongqa/qwen35_parent_temporal_pivot_dev140_schema3/proofpack.jsonl`
- Partial agentic predictions:
  `runs/egolongqa/qwen35_agentic_dev140/predictions.jsonl`
- Partial agent traces:
  `runs/egolongqa/qwen35_agentic_dev140/agent_traces.jsonl`
- Agentic failure log:
  `slurm_outputs/qwen35_agentic_dev140/q35-agentic-dev140_49279292.err`
