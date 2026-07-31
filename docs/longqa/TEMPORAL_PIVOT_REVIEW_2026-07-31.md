# Temporal-pivot review

Date: 2026-07-31
Branch: `exp/molmo2-er-and-2b`
Prediction bundle: `egolongqa_ensemble_predictions_2026-07-31.zip`

## Bottom line

The temporal-pivot idea is useful as a complementary view, not as a universal
selector. It improves Qwen3-VL-8B from `514/700` to `528/700`, but changes
Qwen3.5 from `538/700` to `537/700`. Its main value is diversity: the Qwen3
pivot/uniform pair has a `575/700` candidate oracle and the Qwen3.5 pair has a
`580/700` oracle. The corresponding verifiers reach `539/700`.

The cleanest robust success case is a question beginning with `After ...`.
This is also the only syntax for which the compiler cleanly separates the
pivot clause from the requested target clause. Pivot selection gains `+5` for
Qwen3 and `+10` for Qwen3.5 on these 204 questions. Results are mixed or
negative for embedded `after`, `before`, and unparsed `GLOBAL` wording.

I found three concrete selector/input defects and implemented opt-in ablations
for each. A 20-example screen did not find a configuration worth promoting to
dev140. The important outcome is therefore to retain the current pivot as a
diverse verifier candidate, avoid the failed simplifications below, and focus
the next small experiment on the weak `GLOBAL` allocation rather than on a
larger prompt or reasoning system.

## Prediction audit

### Overall results

| System | Correct |
| --- | ---: |
| Qwen3 uniform | 514/700 |
| Qwen3 temporal pivot | 528/700 |
| Qwen3 pivot/uniform verifier | 539/700 |
| Qwen3.5 uniform | 538/700 |
| Qwen3.5 temporal pivot | 537/700 |
| Qwen3.5 pivot/uniform verifier | 539/700 |
| Fixed three-way majority | 552/700 |
| Candidate arbiter | 553/700 |

### Where pivoting helps

The following is a mutually exclusive syntax partition of all 700 questions.
Values are pivot minus uniform correct answers.

| Question structure | N | Qwen3 delta | Qwen3.5 delta |
| --- | ---: | ---: | ---: |
| Leading `After ...` | 204 | +5 | +10 |
| Embedded `after` | 60 | +1 | -3 |
| `before` | 54 | +2 | -2 |
| `first` | 128 | +4 | +1 |
| `last` | 7 | 0 | 0 |
| State/change wording | 40 | +3 | +1 |
| `GLOBAL` fallback | 207 | -1 | -8 |

The broad operator totals obscure this distinction: `AFTER` is `+6` for Qwen3
and `+7` for Qwen3.5 only because the leading form offsets the embedded form.

A Qwen3.5 rule that uses pivot only for leading `After` and uniform otherwise
scores `548/700`. It is a useful standalone routing observation, but it is not
a replacement for the pivot candidate: its pair with uniform has only 31
disagreements and a `554/700` oracle, versus 106 disagreements and a `580/700`
oracle for the original pivot/uniform pair.

### What the verifier is using

Qwen3 pivot and uniform disagree on 122 rows. Pivot alone is correct on 61,
uniform alone on 47, and neither on 14. The verifier recovers 48/61 pivot-only
answers, 23/47 uniform-only answers, and 1/14 rows where neither candidate is
correct. This confirms that making the pivot more uniform-like can raise its
standalone score while making the downstream verifier worse.

Across the six supplied base systems, 61 questions are missed by all of them.
Their primary heuristic types are dominated by cross-time ordering (23), OCR or
named detail (14), and spatial location (13). Qualitative inspection shows many
multi-event identity/state-tracking questions. These are not just failures to
sample one nearby frame; answer-stage relation tracking is also a bottleneck.

### Answer-letter behavior

The labels are highly imbalanced: A/B/C/D occur 8/205/444/43 times. Pivoting
changes Qwen3 accuracy by `0/-6/+18/+2` and Qwen3.5 by `+1/-8/+4/+2` across
those letters. There is no simple option-A prediction bias, so the retrieval
truncation described below should not be claimed as the cause of the score
pattern. It is nevertheless a silent loss of input semantics and makes option
order affect retrieval.

## Implementation audit

### 1. Most retrieval queries are truncated

SigLIP2's text tower accepts 64 positions. The current target query has token
length min/median/p95/max `50/104/200/363`; 642/700 rows exceed the text
window. After actual processor truncation, the visible option markers are:

| Visible option markers | Rows |
| --- | ---: |
| A only | 322 |
| A and B | 155 |
| A, B, and C | 64 |
| All four | 73 |
| None | 85 |
| A and D (tokenization edge case) | 1 |

The combined query therefore often means “question plus option A,” not
“question plus all options.” Standalone option queries largely avoid this:
only 10/2,800 compact option strings exceed 64 tokens.

The compiler already extracts `program.target`, but the original selector
never uses it for target retrieval. A compact target-plus-options query reduces
the number of overlength queries from 642 to 519; a target-only query exceeds
64 tokens on only 6 rows. Both forms were screened below.

### 2. Two pivots are selected but only one controls direction

The selector retrieves two pivot hypotheses and renders both eventlets, then
uses only `pivots[0]` to construct the forward/backward target mask. On 139/264
`AFTER` rows and 33/54 `BEFORE` rows, the second pivot lies on the side that
would reopen an interval discarded by the primary mask. The median reopened
timeline fraction is about 22% for `AFTER` and 27% for `BEFORE`.

An opt-in per-pivot mask now divides the unchanged target budget between the
two directional hypotheses and bridges each pivot to its best target. It does
not increase the final 64-frame budget.

### 3. Pivot and target quotas can select the same event

At least one exact pivot center is also a target center on 93/700 rows:

| Operator | Rows with exact center overlap |
| --- | ---: |
| AFTER | 25 |
| BEFORE | 8 |
| FIRST | 41 |
| LAST | 2 |
| STATE_CHANGE | 17 |

This is most common where pivot and target query text are identical or nearly
identical. Deduplication means the nominal target quota silently becomes more
semantic-boundary fill. An opt-in exclusion now prevents target centers from
reusing any pivot eventlet position.

### 4. `GLOBAL` is not actually non-temporal

The 207 `GLOBAL` rows include 138 with `later`, 77 with `earlier`, and 55 with
both. The fallback uses eight generic relevance eventlets rather than an
explicit temporal program. At the same time, directly replacing all of these
with uniform frames removes useful pivot-only answers and disagreement
diversity. The next sensible ablation is a near-uniform `GLOBAL` pack—roughly
48-56 uniform anchors plus 2-4 retrieved eventlets—rather than an exact uniform
route or another broad multi-event parser.

## Implemented ablations

The changes are opt-in and preserve the legacy defaults:

- `--retrieval-query-mode question`
- `--retrieval-query-mode question_option_mean`
- `--retrieval-query-mode temporal_target_options`
- `--retrieval-query-mode temporal_target_option_mean`
- `--pivot-mask-mode per_pivot`
- `--exclude-pivot-target-overlap`

`question_option_mean` and `temporal_target_option_mean` score the question and
four compact standalone options independently, normalize each score vector,
and give the question and option set equal influence. The option set is
averaged rather than given four equal frame quotas.

The selector records per-pivot target centers and the new modes in proof-pack
metadata. Non-default selector settings are included in the fingerprint. A
dedicated Slurm launcher is available at
`scripts/slurm_qwen3_temporal_pivot_ablation_dev140.sh`.

## Focused screen

The fixed screen is the first 20 rows of the canonical dev140 subset. Historical
scores on these exact keys are 14/20 for pivot and 15/20 for uniform. The
promotion bar was set at 16/20 before looking at results.

| Variant | Correct | Decision |
| --- | ---: | --- |
| Exact legacy control | 14/20 | Reproduces historical pivot |
| Question only, primary mask | 13/20 | Reject |
| Question + independent option mean, primary mask | 12/20 | Reject |
| Compact target + options, primary mask | 13/20 | Reject |
| Compact target + options, per-pivot mask | 12/20 | Reject |
| Compact target + independent option mean, primary mask | 12/20 | Reject |
| Legacy query + no pivot/target overlap, primary mask | 14/20 | Tie; do not scale |

Redundant factorial combinations were stopped once their clean prefixes were
already below both baselines. One pre-fix per-pivot run was excluded because a
bridge-bookkeeping compatibility bug changed non-directional rows; a regression
test now guarantees that the default primary path bridges its legacy first two
targets. No valid result justifies a dev140 run.

## Recommendations

1. Keep the original Qwen3 pivot as the disagreement verifier's temporal view.
   Its diversity is more valuable than a post-hoc router's higher standalone
   score.
2. Do not promote question-only or equal option-mean retrieval. Options matter
   to grounding, and removing or symmetrically averaging them lost answers in
   the screen.
3. Do not promote per-pivot masking or overlap exclusion unless a larger
   selector-only audit shows better evidence recall. Their implementation
   defects are real, but fixing an internal inconsistency is not automatically
   an accuracy gain.
4. Run one small operator-specific budget experiment next: keep the existing
   selector for explicit operators and make `GLOBAL` near-uniform while
   retaining a few retrieved eventlets. This directly targets the Qwen3.5
   `-8` `GLOBAL` gap without collapsing all 34 Qwen3.5 `GLOBAL` disagreements.
5. If interval syntax is revisited, support only explicit `after X ... before
   Y` / `between X and Y` masks first. Avoid another general clause-rewriting
   system; the existing multi-event and structured-prompt experiments did not
   improve the baseline.

All routing observations use labeled validation predictions. Treat them as
diagnostics and validate any promoted policy on grouped out-of-fold splits.
