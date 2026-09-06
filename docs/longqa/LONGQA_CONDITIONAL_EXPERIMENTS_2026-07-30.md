# EgoLongQA Conditional Experiments - 2026-07-30

## Fixed Reference

All four experiments use the same three completed systems:

1. Qwen3.5 temporal pivot: `537/700`.
2. Qwen3.5 uniform64: `538/700`.
3. Qwen3 pivot/uniform verifier: `539/700`.

Their deterministic majority is the fixed reference at `552/700` (`78.86%`).
The systems disagree on 213 questions, including 24 questions for which all
three answers differ. Interventions copy the fixed-majority answer outside
their stated gate.

## Experiment 1: Candidate-Constrained Three-Way Arbiter

Script:

```bash
sbatch slurm_longqa_qwen35_all_different_arbiter_full.sh
```

The job touches only the 24 all-different questions. It builds a 64-frame
mixture of temporal-pivot evidence and uniform coverage, then asks Qwen3.5 to
choose only among the three proposed answer texts. The other 676 rows copy the
fixed majority.

Primary comparison: the fixed majority gets `6/24` of these cases correct,
while at least one input system is correct on `20/24`.

## Experiment 2: Confidence Router With Nested Grouped CV

Script:

```bash
sbatch slurm_longqa_qwen35_disagreement_confidence_router_full.sh
```

For each of the 213 disagreements, Qwen3.5 scores all four options using:

- temporal-pivot frames;
- uniform frames;
- the mixed verifier frame pack; and
- question/options without video.

The CPU stage constructs confidence features from option probabilities,
margins, entropy, and vote count. It evaluates a regularized logistic router
with five outer folds and three inner folds. Rows from the same video remain in
the same fold. Category and temporal-operator labels are intentionally excluded
because earlier routing rules based on them did not generalize.

The reported full score is an out-of-fold analysis estimate. It is not yet a
deployable router trained on independent labeled data.

## Experiment 3: Conditional Uncertainty-Guided Dynamic TCoT

Script:

```bash
sbatch slurm_longqa_qwen3_conditional_ug_dynamic_tcot_dev.sh
```

This dev140 job touches its 37 three-system disagreements. For each:

1. sample 128 candidates across the video;
2. use Qwen answer uncertainty to retain four candidates in each of 16
   chronological regions;
3. let Qwen select evidence independently within four larger chronological
   sections;
4. expand around selected frames and add 16 uniform frames; and
5. choose only among answers proposed by the three reference systems.

The remaining 103 dev rows copy the fixed majority. Promote this branch only
if it improves the fixed dev majority by at least three answers without merely
shifting errors between option letters.

## Experiment 4: Delta-Gated Object Crops

Script:

```bash
sbatch slurm_longqa_qwen3_conditional_delta_crops_dev.sh
```

This dev140 job also touches only the 37 disagreements. Grounding DINO
detections come from the existing cache. For every proposed crop, Qwen scores:

1. the complete source frame; and
2. the complete frame together with the crop.

A crop is admitted only when the second input reduces answer entropy by at
least `0.05` and its preferred answer is one of the ensemble candidates. At
most eight admitted crops are combined with temporal-pivot context. This tests
whether the crop adds useful evidence rather than merely looking salient.

## Experiment 5: Finer Conditional Dynamic TCoT

Script:

```bash
sbatch slurm_longqa_qwen3_conditional_ug_dynamic_tcot_s8x3_dev.sh
```

This is a controlled temporal-granularity comparison with Experiment 3. Both
start from the same 128 candidates, retain four uncertainty-selected frames in
each of 16 chronological regions, and preserve the same final frame budget.

- Experiment 3 divides the 64-frame shortlist into four sections and selects
  up to six seed frames per section: at most 24 seeds.
- Experiment 5 divides the shortlist into eight sections and selects up to
  three seed frames per section: also at most 24 seeds.

For a ten-minute video, the sections shrink from approximately 2.5 minutes to
1.25 minutes. The finer partition may better separate repeated actions and
local before/after transitions without increasing the selected or final frame
budgets. Uncertainty scores are reusable across the two experiments, while the
TCoT selection cache remains isolated by its configuration fingerprint.

## Recommended Order

The scripts have no result dependency and can technically run together.
For prioritization:

1. Run Experiments 1 and 2 first; they directly target the largest measured
   ensemble opportunity.
2. Run Experiments 3 and 4 together afterward, or concurrently if two more
   H100s are available.
3. Do not promote Experiments 3 or 4 to all 700 questions until the dev140
   result exceeds the fixed majority by at least three answers.

Every runner supports prediction resume, validates required inputs before model
startup, writes a row-level audit file, and archives completed outputs under
`runs/egolongqa/`.
