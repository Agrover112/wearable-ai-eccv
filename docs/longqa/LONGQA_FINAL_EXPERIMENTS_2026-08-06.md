# EgoLongQA Final Experiments - 2026-08-06

## Frozen Baseline

The baseline for these experiments is the label-free majority of direct
Qwen3.5-27B, bounded-reasoning Qwen3.5-27B, and endpoint-uniform Qwen3.5-9B. It
scores **582/700 (83.14%)** overall and **122/140 (87.14%)** on dev140. No
Bayesian answer-prior calibration is used.

## Experiment 1: Full-Evidence Pairwise Verification

The majority differs from direct Qwen3.5-27B on only 25 of the 700 questions.
Those are the only questions processed by this experiment. Six occur in dev140.

For each disagreement, Qwen3.5-27B receives the exact 64 chronological frames
recorded by the direct 27B run. The prompt contains the question and only two
complete candidate answer texts: the current majority answer and the direct 27B
answer. It explicitly warns that repeated nearby frames are not independent
evidence.

The comparison is run twice with candidate order reversed. A third response,
"evidence is insufficient or ambiguous," permits abstention. The frozen
conservative policy changes the majority answer only when:

1. the direct 27B challenger is the highest-probability choice in both orders;
2. its probability is at least 0.55 in both orders;
3. insufficient evidence has probability at most 0.35 in both orders; and
4. its mean odds over the majority answer are at least 1.5 to 1.

All other cases retain the existing majority answer. A consensus-only policy is
also emitted for diagnosis, but it must not be promoted merely because it looks
better after inspecting validation labels.

Run first:

```bash
sbatch slurm_longqa_qwen35_27b_full_evidence_pairwise_dev.sh
```

The conservative policy must beat the existing `122/140` dev result, or make a
clearly correct high-confidence change without a regression, before running:

```bash
sbatch slurm_longqa_qwen35_27b_full_evidence_pairwise_full.sh
```

The full run makes only 50 model calls: two candidate orders for each of 25
questions. Its baseline is the already submitted 582-answer majority, so an
abstention leaves the answer unchanged.

## Experiment 2: Fresh End-To-End Latency Smoke

This job executes the complete ensemble path on dev20 using one H100 and fresh
selection caches:

1. SigLIP2 temporal-pivot selection and its Qwen3.5-9B answer pass;
2. Qwen3.5-9B uncertainty-guided selection;
3. endpoint-uniform Qwen3.5-9B answering;
4. direct Qwen3.5-27B answering;
5. bounded-reasoning Qwen3.5-27B answering; and
6. majority voting.

Model weights remain in the normal Hugging Face cache, but selected-frame,
uncertainty-score, and prediction caches are fresh. Model startup and shutdown
are included once per stage, making the amortized measurement conservative.
Each stage's wall time and the total seconds per question are written to
`latency_summary.json`.

Run independently:

```bash
sbatch slurm_longqa_qwen35_final_pipeline_latency_dev20.sh
```

This job has no data dependency on the pairwise experiment. It may run at the
same time if two H100s are available. With one H100, run the short pairwise dev
job first and then this latency job. The measured amortized total must remain
below 300 seconds per question.

## Pairwise Dev Result

Job `49499116` completed the six dev disagreements. The conservative policy
made no changes and retained **122/140**. The consensus-only policy made one
change, but it replaced a correct majority answer with an incorrect direct-27B
answer and fell to **121/140**. Neither of the two useful direct-27B minority
answers was recovered reliably across both candidate orders.

The full pairwise run is therefore rejected. Do not submit
`slurm_longqa_qwen35_27b_full_evidence_pairwise_full.sh`; proceed directly to
the fresh latency smoke.

## Reproducible Submission Builder

The following CPU-only command rebuilds the 582/700 majority from its three
completed direct branches, evaluates it, and exports the official two-key
LongQA upload format:

```bash
bash scripts/build_longqa_qwen35_final_majority.sh
```

The builder is a reproducibility check, not a new model experiment. Its output
must contain the same 700 answer letters as
`qwen35_q27_thinking_endpoint_majority_full_2026-08-05`.

## Fresh Latency Result

Job `49499389` completed the full dev20 audit successfully. All stages produced
20 valid outputs. The measured wall times were:

| Stage | Total seconds | Seconds per question |
|---|---:|---:|
| SigLIP2 temporal pivot | 3,328 | 166.4 |
| Uncertainty selection | 5,286 | 264.3 |
| Endpoint-uniform 9B | 1,178 | 58.9 |
| Direct 27B | 1,474 | 73.7 |
| Reasoning 27B | 2,698 | 134.9 |
| **Complete sequential pipeline** | **13,964** | **698.2** |

The complete uncached sequential implementation therefore exceeds the
workshop's 300-second limit. The two selection stages alone require 430.7
seconds per question. This audit includes checkpoint startup once per stage,
so it is conservative, but the margin is too large to claim compliance.

The fresh final vote scores 13/20; the cached 700-row vote scores 12/20 on the
same subset. Endpoint-uniform reproduces exactly, while fresh direct and
reasoning 27B generations differ on one and four rows. The resulting one-answer
gain is not treated as a model improvement because dev20 is small, strongly
answer-position-skewed, and intended only for timing.

## Decision Order

1. Keep the existing 83.14% output as the accuracy reference.
2. Do not run the rejected full pairwise verifier.
3. Do not claim the current uncached sequential ensemble meets the 300-second
   limit.
4. Before submission, define a compliant execution path by removing expensive
   retrieval branches, reusing evidence, or confirming that parallel branch
   wall time is the organizer's accepted accounting method.
