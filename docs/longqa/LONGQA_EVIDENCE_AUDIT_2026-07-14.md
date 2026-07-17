# EgoLongQA Evidence Audit - 2026-07-14

## Scope

This audit examines the frame evidence behind:

- the two QCA predictions uniquely correct beyond pivot and uniform;
- all five multi-event predictions that differ from pivot;
- all 24 full-verifier fixes relative to pivot;
- all 13 full-verifier regressions relative to pivot.

Samples are aligned by `video_path||question`, not row position. The machine-
readable manifest is `runs/egolongqa/evidence_audit_2026-07-14.json`. Contact
sheets and selected JPEGs are stored outside the repository at:

`/scratch/inf0/user/agaur/wai-26/data/wearable-ai/frame_audits/evidence_audit_2026-07-14`

Each sample directory contains metadata and up to 16 representative frames per
evidence source. Frames combine high-priority retrieval roles with temporal
coverage; they are an inspection view, not the complete 64-frame model input.

## QCA Findings

Standalone QCA scores `104/140`. It uniquely solves two questions missed by
both pivot and uniform, but regresses 13 pivot-correct answers.

### Dynamic allocation collapsed

For every one of the 140 samples, QCA produced the identical quota vector:

```text
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
```

The segment relevance and content-deviation values differ, but the current
softmax/power/floor pipeline compresses those differences so heavily that every
16-segment video receives exactly four frames per segment. Therefore this run
tests stratified relevance-diversity selection, not the intended dynamic QCA
budget allocation.

This explains why replacing the `GLOBAL` route with QCA changes only three
answers. Before another QCA run, contribution scores need calibrated scaling or
temperature and a test asserting non-uniform quotas on non-uniform synthetic
inputs. Quota entropy and min/max allocation should be logged for every run.

### Two unique wins

- `4ee6962b7a404ef9`: both QCA and pivot inspection sheets contain the globe and
  surrounding buildings. QCA has broader segment anchors, but the decisive
  building word is not clearly exclusive to its representative frames. This
  looks more like answer arbitration sensitivity than demonstrated selector
  recall.
- `f11fb6958943a777`: both packs broadly cover the first and second driving
  environments, but the representative sheets do not clearly expose the white
  SUV comparison. QCA and multi-event both answer this row correctly, suggesting
  that broader state coverage may help, but it needs a dense frame-level audit
  around the two SUV occurrences before claiming causal evidence improvement.

Conclusion: retain QCA as a complementary candidate source, but the current run
does not validate the paper's dynamic allocation mechanism.

## Multi-Event Findings

The multi-event router changes five of 31 routed answers: two fixes and three
regressions.

### Clause parsing is the first bottleneck

Only two changed questions yield three distinct clauses. Three fall back to one
generic full-question query. Some extracted clauses are syntactically weak, for
example:

```text
did I purchase at the first store I visited
and in what type of store was I
seen carrying it
```

The parser splits temporal language but does not recover semantic event
arguments such as `purchase(item)` and `carry(item, store_type)`. Consequently,
the selector may retrieve storefronts and carrying scenes without preserving
the shared object identity.

### Visual comparison

- Fix `e80db9b80583d44e`: the multi-event pack spreads evidence across early
  vendor/shop encounters, bridges, and the final Peaceful Stone Lights scene.
  This is the intended behavior for a first-versus-last question.
- Regression `9bab9b303c81a11d`: multi-event evidence is spread across a
  convenience store, streets, and clothing stores. Pivot contains clearer
  close-up product interactions and store interiors. The multi-event pack gains
  narrative breadth but dilutes object-identity evidence.

Conclusion: the method needs structured event arguments and entity linking,
not more event centers. A useful next representation would explicitly encode
`event_1`, `event_2`, and `shared_entity`, then require evidence for both events
before spending the remaining budget on bridges.

## Verifier Findings

The full verifier improves pivot from `528/700` to `539/700` by making 24 fixes
and 13 regressions.

### It is primarily choosing between candidates

| Outcome relative to pivot | Chooses uniform | Chooses a third option |
| --- | ---: | ---: |
| Fix | 23 | 1 |
| Regression | 11 | 2 |

Across all disagreements, the verifier emits an answer outside the pivot and
uniform candidates eight times. Only one of those eight third answers is
correct; pivot is correct for two and uniform for four. A retrospective fallback
to pivot on third answers would score `540/700`; fallback to uniform would score
`542/700`. These are diagnostics, not valid promoted results, but they strongly
motivate a candidate-constrained verifier experiment.

### Visual comparison

- Fix `09ea3872eb883ec1`: the verifier union provides broad chronological
  coverage and the verifier adopts the uniform candidate. The representative
  sheets do not clearly show the decisive tram illustration, illustrating that
  the final choice can depend on frames outside a 16-frame audit view.
- Regression `016c4ad8fda54a97`: representative evidence mostly shows a wooded
  path and transition into the street. It does not clearly establish the child,
  stairs, playground, or sports field. With weak decisive evidence, the verifier
  overturns the correct pivot answer in favor of the incorrect uniform answer.

The category-level verifier net gain is concentrated in Travel-Tourism (`+4`),
Gardening (`+3`), and Daily Activities (`+2`). Outdoor Activities and Sports is
net `-1`; several other categories are neutral because fixes and regressions
cancel.

Conclusion: the current verifier succeeds when one candidate has captured the
missing evidence, but it lacks a reliable abstention rule when the union remains
ambiguous. Third-option freedom is almost entirely harmful.

## Failure Attribution

| Component | Evidence from audit | Current diagnosis |
| --- | --- | --- |
| Candidate timeline | QCA and pivot often both span relevant periods | Not the dominant failure in inspected cases |
| QCA allocation | All 140 quota vectors are uniform | Score calibration/implementation bottleneck |
| Multi-event compiler | Generic fallback or malformed clauses in changed rows | Event representation bottleneck |
| Evidence packing | Multi-event regressions trade close detail for breadth | Quota/entity-linking bottleneck |
| Verifier | 34 of 37 changes follow uniform; third answers are 1/8 | Candidate arbitration bottleneck |
| Final VLM | Different answers can arise from similarly adequate sheets | Residual reasoning/calibration sensitivity |

## Recommended Experiments

1. Add a candidate-constrained verifier that must choose pivot or uniform, with
   an optional `KEEP_PRIMARY` abstention decision. Run dev140 first.
2. Fix QCA score scaling and add quota-distribution tests before rerunning it.
   Do not run current QCA on full validation.
3. Replace punctuation-based multi-event splitting with structured event and
   shared-entity extraction. Test selector recall on the five audited rows before
   another dev140 generation run.
4. Add verifier evidence sufficiency signals: maximum relevance, pivot/target
   coverage, and candidate-frame overlap. Preserve pivot when neither candidate
   has clearly stronger evidence.

The highest-confidence next experiment is candidate-constrained verification.
It directly addresses the strongest audit signal and requires model calls only
on existing disagreements.
