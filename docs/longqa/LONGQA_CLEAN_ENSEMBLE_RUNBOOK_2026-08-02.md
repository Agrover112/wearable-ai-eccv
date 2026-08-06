# Clean Qwen3.5 Ensemble Runbook - 2026-08-02

## Invariant

Every candidate is an independent Qwen3.5 answer generated from a distinct
visual input. A verifier output is never reused as a candidate for another
verifier. The old verifier-derived Candidate C is excluded.

The frozen candidate set is:

1. uncertainty-pivot;
2. option-quota temporal pivot;
3. endpoint-inclusive uniform sampling.

Their dev140 majority is `121/140` (`86.43%`) and their oracle is `124/140`.

## Dev140 Verification

The two jobs below may run simultaneously on separate GPUs:

```bash
sbatch slurm_longqa_qwen35_clean_trio_verifier_dev.sh
sbatch slurm_longqa_qwen35_clean_trio_specialist_verifier_dev.sh
```

Both preserve unanimous answers and produce three policies:

- `all_disagreements`: apply the verifier on every disagreement;
- `all_different_only`: change only the two rows where all candidates differ;
- `conservative`: always resolve all-different rows, but overturn a 2-1 vote
  only when the fused probability margin is at least `0.10` and at least two
  evidence views support the dissenting answer.

The specialist job adds existing OCR transcriptions and object observations as
optional judge evidence. Those outputs are not candidate answers. Freeze one
policy after this comparison; do not tune it on val560.

## Val560 Candidate Runs

These jobs are independent and may run concurrently:

```bash
sbatch slurm_longqa_qwen35_option_quota_pivot_val560.sh
sbatch slurm_longqa_qwen35_endpoint_uniform_val560.sh
sbatch slurm_longqa_qwen35_uncertainty_pivot_val560_array.sh
```

The uncertainty launcher is a four-task array with 140 rows per task. To use
only one GPU at a time, submit it as:

```bash
sbatch --array=0-3%1 slurm_longqa_qwen35_uncertainty_pivot_val560_array.sh
```

All candidate launchers have stable run names and resume their own output. A
timed-out shard can therefore be resubmitted without discarding completed rows.

## Merge And Validate

After all six candidate jobs have completed (option-quota, endpoint-uniform,
and four uncertainty shards), run:

```bash
bash scripts/merge_longqa_clean_trio_val560.sh
```

This validates exact shard coverage, creates val560 and full-700 versions of
all three candidates, and writes clean majority predictions for both scopes.

Then run the frozen terminal verifier on val560:

```bash
sbatch slurm_longqa_qwen35_clean_trio_verifier_val560.sh
```

Finally combine the same frozen policy across dev140 and val560. For example:

```bash
POLICY=conservative bash scripts/merge_longqa_clean_trio_verifier_full.sh
```

The val560 accuracy is the promotion criterion. The full-700 merge is only a
packaging step and must not be used to retune the policy.
