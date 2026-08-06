# EgoLongQA final-push runbook: 3 August 2026

## Decision from the clean-trio validation

The uncertainty-pivot, option-quota, and endpoint-uniform majority reached
`121/140` (`86.43%`) on dev140 but only `437/560` (`78.04%`) on the disjoint
val560 subset. Its combined result is `558/700` (`79.71%`). This ensemble is
rejected; the validated `565/700` rotation-pivot pipeline remains primary.

## Experiment 1: Qwen3.5-27B selective judge

The larger model is called only on disagreements among five independently
generated Qwen branches: pivot, uniform, uncertainty-pivot, option-quota, and
endpoint-uniform. The old verifier-derived Candidate C is excluded. Their
five-way majority is `559/700` and their candidate oracle is `614/700`.
Each call receives 64 chronological
frames: 24 temporal-pivot frames, 24 global frames, and 16 frames prioritized
by the uncertainty analysis. It sees all four complete answer options and may
select an option absent from the candidate set.

Run the dev140 pilot first:

```bash
sbatch slurm_longqa_qwen35_27b_primary_judge_dev.sh
```

The independent majority scores `119/140` on this subset and has 28 candidate
disagreements. The relevant policies are:

- `all_disagreements`: always accept the 27B answer on a disagreement;
- `candidate_supported`: accept it only when an earlier candidate proposed it;
- `plurality_ties_only`: accept it only when the five candidates have a tied
  highest vote count.

The completed pilot showed that the useful fixed rule is
`plurality_ties_only`: invoke Qwen3.5-27B only when two or more answers share
the highest vote count. It reaches `121/140` with two fixes and no regressions;
the unrestricted policy falls to `117/140`.

The original `122/140` promotion threshold was not met. Because the tie-only
rule was identified after inspecting the dev changes, it remains exploratory
until val560 confirms it. The val560 subset contains only four plurality ties,
so this check is inexpensive and must use the frozen rule without alternatives.

The held-out check subsequently rejected the rule: it made one fix and two
regressions on val560, producing `560/700` after merging with dev. Broad 27B
arbitration and tie-only arbitration are therefore not promoted.

Run the held-out evaluation with the selected policy:

```bash
JUDGE_POLICY=<selected-policy> sbatch slurm_longqa_qwen35_27b_primary_judge_val560.sh
```

After it finishes:

```bash
JUDGE_POLICY=<selected-policy> bash scripts/merge_longqa_qwen35_27b_primary_judge.sh
```

## Experiment 2: focused reasoning diagnosis

The fixed audit contains ten primary errors requiring cross-time ordering, ten
shopping or text-reading errors, and ten errors missed by all six strong runs.
Unlike the selective pilot, the judge is called on every audit row.

```bash
sbatch slurm_longqa_qwen35_27b_primary_judge_audit30.sh
```

Run this after the dev pilot has downloaded and validated the model. The audit
is diagnostic and must not be merged into a submission.

## Experiment 3: lightweight router analysis

After dev and val judge predictions have been merged, test a small grouped
router using only five-model vote support, model-source agreement, the frozen
majority answer, and the 27B answer. Option-letter identity is excluded:

```bash
JUDGE_POLICY=<selected-policy> bash scripts/run_longqa_qwen35_27b_router_cv.sh
```

This is grouped out-of-fold analysis. Promote routing only if its estimated
full accuracy exceeds both the fixed primary and the direct 27B policy by at
least two percentage points. A deployment fit/export step would still be
required for unseen test data.

## Runtime behavior

The 27B model is conditional: agreement rows retain the validated primary
answer without a model call. The pilot records model runtime and frame indices;
strict per-question timing still needs to be measured from the first run before
the val560 job is submitted.

## Experiment 4: candidate-blind 27B answerer

The hard-error audit showed three correct 27B answers that were absent from all
five independent 9B candidates. To test this capability without anchoring the
larger model to previous answers, run Qwen3.5-27B on every dev140 question with
the same 24 pivot + 24 global + 16 uncertainty evidence pack but no candidate
suggestions in the prompt:

```bash
sbatch slurm_longqa_qwen35_27b_candidate_blind_dev.sh
```

This is a standalone candidate experiment. Expand it to val560 only if it
reaches at least `122/140`, stays below 300 seconds per question, and adds
several correct answers absent from the independent five-model pool.

The completed dev run reached `120/140`, with six correct answers absent from
all five 9B candidates. This raises the candidate oracle from `124/140` to
`130/140`; all 140 rows remained below 300 seconds. Candidate diversity
therefore justifies a held-out run despite missing the standalone threshold:

```bash
sbatch slurm_longqa_qwen35_27b_candidate_blind_val560.sh
```

After completion:

```bash
bash scripts/merge_longqa_qwen35_27b_candidate_blind.sh
```

The completed held-out run scored `449/560`; merged accuracy is `569/700`
(`81.29%`). Adding this candidate raises the independent candidate oracle to
`647/700` (`92.43%`), but simple voting and a grouped vote/source router do not
capture that ceiling. Candidate-blind 27B is retained as the strongest
Candidate-C-free standalone model.
