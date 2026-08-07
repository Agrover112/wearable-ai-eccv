# EgoLongQA Final-Day Arbitration Experiments - 2026-08-07

## Current Reference

The safe label-free reference remains the majority of direct Qwen3.5-27B,
bounded-reasoning Qwen3.5-27B, and endpoint-uniform Qwen3.5-9B. It scores
`582/700` (`83.14%`) overall and `122/140` (`87.14%`) on dev140. The Bayesian
answer-position calibrator is excluded.

## Experiment A: Label-Invariant Router

The router considers direct 27B, reasoning 27B, endpoint-uniform 9B, and the
rotation-averaged temporal-pivot 9B model. On dev140 these candidates disagree
on 35 rows and have a `131/140` candidate oracle. The router predicts the
correctness of each proposed semantic answer rather than predicting A/B/C/D.
Its inputs contain candidate agreement, evidence-family support, temporal
operator, coarse question type, and cheap retrieval statistics. No option
letter or learned answer-position prior is included.

Grouped five-fold out-of-fold evaluation did not pass the promotion gate. The
core router made one intervention and regressed from `122/140` to `121/140`.
Adding the already-cached four-rotation semantic confidence made two
interventions, with no fixes, one regression, and one both-wrong change; it also
finished at `121/140`. Do not run the frozen val560 router or the 179-row val
rotation job from these results. The code and launchers are retained for audit.

## Experiment B: Qwen3.5-27B Endpoint-Uniform

This is the first GPU priority. It gives direct Qwen3.5-27B exactly 64
chronological frames selected by the tested endpoint-inclusive grid: first and
last frame plus 62 uniformly distributed interior frames. It uses the known
`451584` pixel limit, 49,152-token context, one H100, direct MCQ decoding, and
no retrieval model.

Run:

```bash
sbatch slurm_longqa_qwen35_27b_uniform64_endpoint_dev.sh
```

The job automatically compares the completed candidate against the current
primary and records unique fixes, regressions, oracle coverage, temporal
operator breakdown, and all disagreement rows. Promote to val560 only if it
improves dev or contributes several correct primary-error recoveries without
excessive regressions:

```bash
sbatch slurm_longqa_qwen35_27b_uniform64_endpoint_val560.sh
bash scripts/merge_longqa_qwen35_27b_endpoint.sh
```

## Experiment C: Rotation-Averaged Confidence

The required semantic remapping was already implemented and validated. Four
cyclic option placements are mapped back to the original answer meanings before
log probabilities are averaged. The completed dev140 run was itself weaker
than the primary, and its confidence features did not improve the new OOF
router. Therefore the prepared val-disagreement launcher is gated off rather
than being run speculatively.

## Experiment D: Qwen3.5-27B Corrected Option-Quota Evidence

This is the second GPU candidate. It reuses the corrected dev140 proof packs
from job `49431158`; SigLIP2 embeddings and frame retrieval are not recomputed.
Each semantic answer received an evidence quota, queries were token-safe, frames
were deduplicated, and the final 64-frame pack remained chronological.

Run after Experiment B, or concurrently only when a second H100 is genuinely
available:

```bash
sbatch slurm_longqa_qwen35_27b_option_quota_globalfix_dev.sh
```

Its standalone score is secondary to complementarity with the endpoint 27B and
current primary. A corrected val560 proof pack and 27B promotion launcher are
prepared, but both remain gated on the dev result:

```bash
sbatch slurm_longqa_qwen35_option_quota_globalfix_val560.sh
sbatch --dependency=afterok:<proofpack_job_id> \
  slurm_longqa_qwen35_27b_option_quota_globalfix_val560.sh
```

## Run Order

1. Run the 27B endpoint dev experiment.
2. Run the 27B corrected option-quota dev experiment.
3. Compare standalone accuracy, primary-only fixes, regressions, and combined
   candidate oracle.
4. Promote only candidates with convincing dev complementarity.
5. Do not run the CPU router val560 or rotation val179 jobs: both failed their
   dev gate.

## Completed Dev Results

Both new 27B jobs completed successfully.

| Candidate | Dev140 | Difference from primary | Two-model oracle |
|---|---:|---:|---:|
| Current primary | 122/140 | - | - |
| Endpoint-inclusive 27B | **130/140** | +8 | **134/140** |
| Corrected option-quota 27B | 123/140 | +1 | 131/140 |

Endpoint 27B is the clear promotion candidate. It fixes 12 primary errors and
regresses on four rows. The frozen secondary policy uses the stricter v2
temporal compiler: endpoint 27B handles its 27 `GLOBAL` rows and the prior
primary handles all explicitly temporal rows. It scores `129/140` (`92.14%`),
changing eight predictions for a net gain of seven. This fixed policy and
endpoint standalone must be evaluated on val560 without further tuning.

Option-quota 27B adds three correct answers beyond endpoint on dev140, but its
standalone score is seven answers lower. Its val560 pipeline remains available
as a secondary use of compute, not the immediate priority.

## Option-Quota Held-Out Run

The complete val560 path can now be queued with one command:

```bash
bash scripts/submit_longqa_qwen35_27b_option_quota_val560.sh
```

This submits three dependent jobs: corrected option-quota frame selection,
Qwen3.5-27B answering on those frames, and CPU merging. The selection job only
creates evidence packs; it no longer runs an unused 9B answer stage.

Once the endpoint full result also exists, the merge script compares the two
27B candidates and creates a frozen three-way majority with the previous
primary. Endpoint 27B is the deterministic fallback when all three candidates
disagree. This majority does not improve endpoint on dev140, so it is retained
as a held-out comparison rather than assumed to be better.

If endpoint merging completes after the option-quota chain, rebuild the
comparison with:

```bash
bash scripts/merge_longqa_qwen35_27b_option_quota.sh
```

## Frozen Temporal Route

The option-quota merge now also builds a simple label-free route. Endpoint 27B
remains the default. Corrected option-quota 27B is used only when it disagrees
with endpoint on a `BEFORE` question or a question comparing multiple times or
occurrences. This changes two dev140 predictions, both correctly, for
`132/140` (`94.29%`). The same rule improves the older complete 9B comparison
from `542/700` to `548/700`, and it is therefore frozen before the current
val560 jobs complete.

## Remaining Input Tests

Two focused model-input tests are prepared:

```bash
sbatch slurm_longqa_qwen35_27b_endpoint_timestamps_verify_dev.sh
sbatch slurm_longqa_qwen35_27b_endpoint_video_smoke5.sh
```

The first keeps the proven endpoint frames but lists their exact timestamps and
asks Qwen to verify every option. The second is only a five-row compatibility
test: it sends the same chronological frames through Qwen's native video input
instead of representing them as unrelated images. Run
`slurm_longqa_qwen35_27b_endpoint_video_dev.sh` only if that smoke test finishes
cleanly and produces sensible predictions.

## Code and Candidate Audit

The archived outputs were re-indexed by `(video_path, question)`. All 136
prediction artifacts contain dataset keys in annotation order, with no duplicate
or foreign rows. Re-scoring the endpoint dev output after strict key alignment
preserves `130/140`, so its gain is not caused by row order or accidental label
reuse. Evaluation, resume handling, and fixed-proof-pack decoding were hardened
to reject stale sample keys and incomplete frame extraction.

The option-quota audit found one configuration mismatch. The historical
`option_quota_pivot` run selected one SigLIP2 center for each answer option, then
used the rest of its eight-center target budget for the combined question query;
the configured `centers-per-option=4` value was not used by this strategy. The
completed result remains valid under that exact definition. A separate
`balanced_option_quota_pivot` strategy now performs round-robin allocation from
multiple ranked centers per option without changing or invalidating the running
experiment. Its gated dev launcher is:

```bash
sbatch slurm_longqa_qwen35_27b_balanced_option_quota_dev.sh
```

This is lower priority than the timestamped endpoint test because dev already
shows that endpoint coverage is substantially stronger than retrieval-heavy
evidence packs.

An exhaustive category diagnostic also rejects coarse category routing. Picking
the best candidate per category on dev scores `127/140`, but the frozen choices
score only `442/560` on the held-out complement, below direct 27B. Category
labels therefore do not provide a reliable arbitration rule.

## Endpoint 27B Full Result

The endpoint-inclusive held-out job `49531487` completed all 560 rows cleanly at
`478/560` (`85.36%`). Merging it with the untouched dev140 partition gives
**`608/700` (`86.86%`)**, the strongest label-free full-set result in the
project. The model uses 64 chronological frames comprising the first frame, the
last frame, and 62 evenly spaced interior frames; it performs no retrieval or
learned routing.

The gain is not driven by answer-position calibration. Macro accuracy across
the four answer positions is `88.34%`, non-C accuracy is `87.50%`, and the full
prediction distribution is `A/B/C/D = 46/200/393/61`. Compared with the prior
`582/700` primary, endpoint 27B makes 56 unique fixes and 30 regressions across
96 disagreements; their diagnostic two-model oracle is `638/700`.

Held-out execution took `14,454` seconds including model startup, or about
`25.81` seconds per question when amortized over 560 rows. Mean context fill was
`56.87%` and the maximum was `57.47%` of the 49,152-token context.

## Sampling-Phase and Conservative-Retrieval Suite

The endpoint result motivates a controlled test of temporal sampling phase.
Three grids use the identical Qwen3.5-27B checkpoint, 64-frame budget,
672-by-672 pixel limit, baseline prompt, and disabled reasoning:

- `endpoint`: positions `i/63`, including both ends; completed at `130/140`.
- `legacy`: positions `i/64`, including the start but not the end.
- `midpoint`: the center of each of 64 equal temporal bins, including neither
  exact endpoint.

Run the two missing candidates independently:

```bash
sbatch slurm_longqa_qwen35_27b_uniform64_legacy_dev.sh
sbatch slurm_longqa_qwen35_27b_uniform64_midpoint_dev.sh
```

After both finish, construct the endpoint-first three-phase majority and report
standalone, unique-correct, agreement, majority, and oracle statistics:

```bash
bash scripts/merge_longqa_qwen35_27b_sampling_phases_dev.sh
```

The conservative retrieval branch keeps 48 endpoint-inclusive anchors from a
128-frame candidate grid. It adds only 16 question-relevant frames selected by
SigLIP2 with maximal marginal relevance, which penalizes visually redundant
retrievals. It never allows retrieval to replace the majority of global
coverage:

```bash
sbatch slurm_longqa_qwen35_27b_endpoint_mmr_hybrid_dev.sh
```

Two previously prepared complementary tests remain available:

```bash
sbatch slurm_longqa_qwen35_27b_endpoint_timestamps_verify_dev.sh
sbatch slurm_longqa_qwen35_27b_balanced_option_quota_dev.sh
```

All experiments are dev140-gated. Val560 launchers should be added only for a
candidate or frozen majority that improves endpoint dev or materially expands
its candidate oracle.

## Completed Phase Tests and New Primary

The two controlled 64-frame phase ablations are complete. The old grid, which
includes the first frame but omits the exact final frame, scores `123/140`
(`87.86%`). Sampling the middle of every temporal bin scores `121/140`
(`86.43%`). Both are clearly below the endpoint-inclusive grid at `130/140`
(`92.86%`). A majority vote across all three grids also falls to `127/140`, so
the endpoint gain should be treated as a property of its sampling pattern, not
as a reason to ensemble every phase.

The corrected option-quota 27B run is also complete. It scores `468/560` on the
held-out complement and `591/700` (`84.43%`) after merging with dev140. Although
weaker than endpoint alone, it is correct on 35 full-set disagreements where
endpoint is wrong. The inference output is complete and valid; its SLURM job
encountered an obsolete wrapper quotation error only after evaluation, during
archival, so the run does not need to be repeated.

The route frozen before held-out completion now has a final result. It uses
endpoint 27B by default and changes to option-quota 27B only when the two models
disagree on a `BEFORE` question or a question that compares multiple times or
occurrences. It scores `132/140` on dev and `481/560` held out, giving
**`613/700` (`87.57%`)** overall. The 19 routed changes contain 12 fixes and
seven regressions. This is a real but modest held-out gain of three answers over
endpoint alone; it uses neither answer labels nor a fitted calibrator.

The current primary artifact is therefore:

```text
runs/egolongqa/qwen35_27b_endpoint_optionquota_temporal_route_full_2026-08-07/
```

Endpoint-only remains the simplest operational submission at `608/700`
(`86.86%`). The routed result is preferable for accuracy if the additional
SigLIP2 retrieval and second Qwen answer pass satisfy the final runtime and
parameter-reporting requirements.
