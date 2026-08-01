# Judge Approaches

The approaches are ordered from most to least promising for hidden-test
generalization.

## Approach 1: Disagreement-Triggered Adaptive Hypothesis Verification

This is the preferred first implementation. It uses disagreement to trigger
fresh evidence retrieval instead of relying on predictor profiles, learned
weights, or validation-derived rules. It permits exactly one conditional
evidence-refinement round; it never repeats an unchanged judgment call.

The approach is motivated by
[VideoHV-Agent (CVPR 2026)](https://arxiv.org/abs/2603.04977), which reformulates
answer candidates as testable hypotheses, derives a discriminative clue,
retrieves localized video evidence, and verifies the hypotheses before
answering. The implementation below simplifies that framework for the current
three-predictor ensemble.

This is not a reproduction of VideoHV-Agent. That method answers each question
from all options using summary, hypothesis, clue, verification, refinement, and
answer agents. Here, hypothesis verification is a post-hoc arbitrator activated
for every predictor disagreement. It compares the two unique candidates in a
two-versus-one case or all three candidates in an all-different case, allows
one evidence-conditioned follow-up, and uses a deterministic conservative
decision rule instead of a generative answer agent.

### Algorithm

```text
If all predictors agree:
    Return their shared answer.

If the predictors disagree:
    C = the 2 or 3 unique candidate answers

    1. Convert every candidate c in C into a testable visual hypothesis.

    2. Derive one neutral discriminative clue:
       "What directly visible fact would distinguish these candidates?"

    3. Retrieve fresh evidence from the full video:
       - search with SigLIP2 using the question, hypotheses, and clue;
       - use an equal retrieval budget for every unique candidate;
       - exclude or deduplicate already selected frames;
       - add temporal neighbours around the strongest retrieval hits;
       - merge with a capped set of predictor evidence; and
       - sort every frame chronologically.

    4. Run the first VLM verification call.
       Hide predictor identities, frame sources, and vote counts.

       For every hypothesis, require:
       - directly supporting timestamps;
       - directly contradicting timestamps;
       - required temporal-order or identity checks; and
       - SUPPORTED / CONTRADICTED / INSUFFICIENT.

       If the evidence is not decisive, also require:
       - the single visible fact that is still missing;
       - one or two concrete retrieval queries;
       - optional anchor frame IDs; and
       - BEFORE / AFTER / AROUND / FULL_VIDEO as the search region.

    5. If the first report is inconclusive and contains a valid request:
       - retrieve new, non-duplicate frames for the missing fact;
       - retain the most important evidence from the first round;
       - sort the revised evidence pack chronologically; and
       - verify once more from the combined evidence.

       Do not run an unchanged retry. Do not perform a third verification.

    6. If 2-vs-1:
           return the minority only when it is SUPPORTED
           and the majority is CONTRADICTED.
           Otherwise return the majority.

       If all-different:
           return a candidate only when it is uniquely SUPPORTED
           and both alternatives are CONTRADICTED.
           Otherwise return the frozen temporal-pivot fallback.
```

Missing evidence is always `INSUFFICIENT`; it is not evidence that a hypothesis
is false. A refinement round is an active request for new evidence, not a
second sample of the same judgment.

The all-different fallback is used only after judging is inconclusive or fails;
all-different cases no longer bypass the judge.

### Why This Should Generalize

- It uses the current question and video rather than validation-set profiles.
- Fresh retrieval can break correlated evidence-selection failures.
- Conditional refinement lets the verifier search for a specifically missing
  fact instead of guessing from an incomplete first evidence pack.
- A balanced per-candidate evidence budget prevents vote count from determining
  how much evidence an answer receives.
- Hiding votes and predictor identities reduces majority and source bias.
- The conservative selection rule protects the frozen case-specific fallback
  when evidence is ambiguous.

The evaluation target is:

```text
Net gain = all disagreement fixes - all disagreement regressions
```

### Implementation Contract: Version 1.2

This section is the executable specification for Codex. Values under
**Observed** come from completed repository runs. Values under **Proposed
default** are the first configuration to test; they are not claimed to be
optimal.

#### Observed Constraints

| Item | Observed value |
|---|---:|
| Validation questions and distinct videos | 700 |
| Questions with explicit temporal cues | 554/700 |
| Frozen three-predictor ensemble | 552/700 |
| Unanimous cases | 487 |
| Two-versus-one cases | 189 |
| Majority correct within two-versus-one | 107 |
| Minority correct within two-versus-one | 63 |
| Neither proposed candidate correct within two-versus-one | 19 |
| All-different cases | 24 |
| At least one correct candidate within all-different | 20 |
| No correct candidate within all-different | 4 |
| Existing SigLIP2 candidate grid | 128 frames |
| Existing VLM frame limit | 64 frames |
| Existing high-resolution frame budget | 451,584 pixels |

The fixed ensemble is:

| Predictor | Completed result | Evidence source |
|---|---:|---|
| Qwen3.5 temporal pivot | 537/700 | Saved temporal-pivot proof pack |
| Qwen3.5 uniform64 | 538/700 | Deterministic uniform64 indices |
| Qwen3 pivot/uniform verifier | 539/700 | Derived from pivot and uniform evidence |

The third predictor is not a third independent evidence selector. Its frames
are reconstructed from the same pivot and uniform sources and must not receive
a third evidence quota.

#### Target Code

Implement against the latest Qwen baseline branch. At the time this contract
was written, that is:

```text
origin/feat/egolongqa-qwen-baselines @ f91698c
```

Extend:

```text
baselines/longqa/run_generate_longqa_conditional.py
```

Reuse:

```text
majority_answer()
sample_key()
normalize_answer()
baseline_uniform_indices()
build_prediction_row()
VLLMModel.generate_json()
```

Add two conditional modes and one disagreement gate:

```text
hypothesis_existing   # Approach 2 control: no fresh retrieval
hypothesis_fresh      # Approach 1
all_disagreements     # intervene on 2-vs-1 and all-different cases
max_evidence_refinements = 0 or 1
```

The existing structured-JSON and citation-validation utilities on
`origin/feat/videochat3-baseline` may be copied or adapted. Keep the runner
resumable and fingerprint every prompt, model, and evidence-selection setting.

#### Exact Decision Flow

```text
INPUT:
    sanitized question, options, and video
    predictions from the fixed three systems
    saved pivot proof pack

query_deadline = monotonic_start + max_query_seconds - timeout_reserve_seconds

Normalize all predictor answers to A/B/C/D.

unique_candidates = DEDUPLICATE_ANSWERS(predictor_answers)

If len(unique_candidates) == 1:
    return the shared answer
    make no planner or verifier call

If len(unique_candidates) == 2:
    case_type = TWO_VS_ONE
    M = majority answer
    m = minority answer
    fallback = M

If len(unique_candidates) == 3:
    case_type = ALL_DIFFERENT
    fallback = Qwen3.5 temporal-pivot answer

For every disagreement:
    Map unique candidates to neutral IDs X, Y, and optionally Z.
    Permute the mapping deterministically using a hash of sample_key.

    planner_report = PLAN_QUERIES(
        question, neutral candidate texts,
        deadline=query_deadline
    )
    evidence = BUILD_EVIDENCE(
        planner_report,
        deadline=query_deadline
    )
    report_1 = VERIFY(
        question, neutral candidates, evidence,
        refinement_available=true,
        deadline=query_deadline
    )

    if IS_DECISIVE_SET(report_1, case_type):
        return APPLY_CONSERVATIVE_RULE(
            report_1, case_type, fallback
        )

    if CAN_REFINE(report_1, query_deadline, max_evidence_refinements):
        request = VALIDATE_REFINEMENT_REQUEST(report_1)
        new_evidence = RETRIEVE_MISSING_EVIDENCE(
            request,
            exclude=evidence,
            deadline=query_deadline
        )

        if new_evidence is not empty:
            refined_pack = BUILD_REFINED_PACK(
                previous=evidence,
                additional=new_evidence,
                cap=64
            )
            report_2 = VERIFY(
                question, neutral candidates, refined_pack,
                refinement_available=false,
                deadline=query_deadline
            )
            return APPLY_CONSERVATIVE_RULE(
                report_2, case_type, fallback
            )

    return fallback
```

For `TWO_VS_ONE`, `IS_DECISIVE_SET` requires one valid `SUPPORTED` candidate
and one valid `CONTRADICTED` candidate. The minority is returned only when it
is supported and the majority is contradicted; otherwise the majority is
returned.

For `ALL_DIFFERENT`, it requires exactly one valid `SUPPORTED` candidate and
two valid `CONTRADICTED` candidates. The uniquely supported candidate is
returned. Every other verdict combination returns the frozen temporal-pivot
fallback.

`CAN_REFINE` is true only for a valid, non-decisive first report containing a
valid request for missing visual evidence, when no refinement has yet run and
the monotonic deadline has not expired. Every optional retrieval and verifier
call receives the same hard deadline; timeout cancels refinement and returns
the case-specific fallback. Malformed output is a failure, not a reason to
retry.

All 213 disagreement rows may change in Version 1.2: 189 two-versus-one cases
and 24 all-different cases. The 487 unanimous rows remain unchanged.

#### Input Contract And Gold-Label Isolation

Join every artifact using:

```text
sample_key = id if present, otherwise video_path + "||" + question
```

Read a predictor answer from `mcq_answer_parsed`; if absent, normalize
`mcq_answer`.

Normalize evidence records to:

```json
{
  "frame_index": 1234,
  "timestamp_s": 82.267,
  "score": 0.318,
  "source": "pivot"
}
```

Saved proof-pack records already provide:

```json
{
  "sample_key": "...",
  "video_path": "...",
  "selected": [
    {
      "frame_index": 1234,
      "timestamp": 82.267,
      "score": 0.318,
      "source": "pivot"
    }
  ]
}
```

Prediction rows copy fields from the labeled annotation file. Therefore, never
pass an entire annotation or prediction row into the planner or verifier.
Whitelist only:

```text
sample_key
video_path
question
mcq_options
predictor answer
evidence metadata
```

The free-form gold `answer`, labeled `mcq_answer`, `category`, and every
correctness field must never enter a model prompt.

#### Neutral Candidate Mapping

Do not expose option letters, predictor names, evidence-source ownership, or
vote counts to the planner or verifier.

```text
candidate IDs = X, Y for two unique answers
candidate IDs = X, Y, Z for three unique answers
mapping = deterministic permutation seeded by SHA-1(sample_key)
```

Prompts contain only neutral candidate IDs and their answer texts. The audit
file retains the private mapping so that code can apply the final decision.
Do not use Python's built-in `hash()`, because it is not stable across
processes. Use separate strict two-candidate and three-candidate schemas.

#### Evidence Pack

Use a balanced allocation based on the number of unique candidates:

| Disagreement | Fresh retrieval | Prior predictor evidence | Uniform anchors | Total |
|---|---:|---:|---:|---:|
| Two candidates | 12 per candidate | 8 per candidate | 24 | 64 |
| Three candidates | 8 per candidate | 5 per candidate | 25 | 64 |

For the `hypothesis_existing` control, replace the 24 fresh frames with 24
additional uniform anchors and set `max_evidence_refinements=0`. Keep the
planner, first-pass verifier prompt, model, and decision rule identical. This
is the loop-disabled control for whether first-pass fresh retrieval causes a
gain.

##### Fresh Retrieval

| Parameter | Proposed default |
|---|---:|
| Grounder | `google/siglip2-so400m-patch14-384` |
| Candidate grid | 128 uniform frames over the full video |
| Queries per candidate | At most 2 |
| Retrieved centers per candidate | 4 total |
| Temporal NMS between centers | 10 seconds |
| Context around each center | Candidate-grid neighbor at `-1`, center, and `+1` |
| Maximum fresh frames per candidate | 12 with two candidates; 8 with three |

For a candidate query group \(Q_c\), rank a frame \(f\) using:

```text
score(c, f) = max cosine_similarity(frame_embedding(f), text_embedding(q))
              over q in Q_c
```

Do not compare raw SigLIP2 scores between candidates as if they were calibrated
probabilities. Rank each candidate independently and give all supplied
candidates identical budgets.

The 128 image embeddings must be computed only once per test video and reused
by the temporal-pivot predictor and this retrieval step. Validation caches may
be reused during development, but the hidden-test pipeline must not depend on
pre-existing video caches.

##### Prior Predictor Evidence

For each candidate, union the evidence from predictors that proposed it,
deduplicate exact frame indices, and retain at most eight temporally diverse
frames for two candidates or five for three. The verifier predictor's
reconstructed frames may participate in this union, but duplicates from the
pivot or uniform source count only once.

Use the existing source priority:

```text
pivot / directional_target
bridge / pivot_context / directional_target_context
event_center / event_center_context
uniform
anchor
coverage_fill
semantic_boundary
```

##### Finalization

1. Merge fresh, prior, and uniform evidence.
2. Deduplicate by exact `frame_index`.
3. Preserve intentional `-1/center/+1` temporal neighborhoods.
4. If fewer than 64 frames remain, round-robin additional retrieval hits across
   X, Y, and Z when present.
5. Fill any remaining capacity using the largest uncovered uniform time gaps.
6. Sort by `(timestamp_s, frame_index)`.
7. Assign displayed IDs `F001 ... F064` only after sorting.

The VLM cites displayed frame IDs. Code maps them back to trusted timestamps;
the model must not generate its own timestamps.

##### Conditional Evidence Refinement

The query planner runs once. Only the first verifier may request one targeted
follow-up retrieval. A valid request contains:

```text
one concrete missing visible fact
one or two observable retrieval queries
zero or more supplied anchor frame IDs
BEFORE | AFTER | AROUND | FULL_VIDEO
```

Build a genuinely new candidate pool only when refinement is triggered:

- for `AROUND`, decode up to 64 unseen candidates inside a 30-second window on
  each side of the supplied anchor frame IDs;
- for `BEFORE` or `AFTER`, decode up to 64 unseen candidates uniformly in the
  requested side of the anchor; and
- for `FULL_VIDEO`, use up to 128 midpoint-offset candidates between the
  original uniform-grid timestamps.

Reuse cached embeddings whenever timestamps coincide and compute SigLIP2
embeddings only for new candidates. Exclude every frame already shown, apply
the same temporal NMS, and retrieve at most four new centers with their
immediate candidate-pool neighbours. The proposed round-two budget is 12
genuinely new frames.

Build the second visual pack in this order:

1. all round-one cited frames and their available immediate neighbours;
2. all newly retrieved refinement frames;
3. temporally diverse round-one fresh and prior frames, balanced across every
   candidate;
4. uniform anchors filling the largest remaining temporal gaps.

Deduplicate, sort chronologically, and cap the pack at 64 frames. Drop
uncited uniform anchors first when space is needed. Displayed frame IDs are
local to each round and must be reassigned only after sorting.

Do not run the second verifier unless at least one genuinely new frame was
added. The second verifier independently reevaluates every candidate from the
revised visual pack. Keep the same X/Y[/Z] mapping, hypotheses, and
discriminative clue, but do not show it the first verdict. Set
`refinement_available=false`, so it cannot request a third pass.

Refinement is also skipped when the request is malformed, its anchors are
invalid, retrieval fails, or the wall-clock deadline has expired. Every
optional operation receives the same deadline; timeout falls back to the
majority.

#### Query-Planner Prompt

Use exactly one text-only, schema-constrained call:

```text
You create neutral visual-search queries for egocentric long-video question
answering.

Do not decide which candidate is correct. Do not use vote counts, predictor
identities, option letters, probabilities, or facts not stated in the question
and candidate answers.

Rewrite every supplied candidate as a concrete, testable visual hypothesis.
Identify the minimum observable fact or facts that distinguish them. Produce at
most two short visual retrieval queries per candidate.

For temporal questions, ensure the queries separately cover the reference
event and the candidate-specific target event. Every query must describe
something that could be visible in an individual frame or a short neighboring
sequence.

Return only JSON matching the supplied schema.
```

Planner input:

```text
Question: {question}
Candidate X: {candidate_x_text}
Candidate Y: {candidate_y_text}
Candidate Z: {candidate_z_text_if_present}
```

Planner output:

```json
{
  "candidate_x": {
    "hypothesis": "A directly observable claim.",
    "retrieval_queries": ["reference event", "candidate-specific target"]
  },
  "candidate_y": {
    "hypothesis": "A directly observable claim.",
    "retrieval_queries": ["reference event", "candidate-specific target"]
  },
  "candidate_z": {
    "hypothesis": "A directly observable claim.",
    "retrieval_queries": ["reference event", "candidate-specific target"]
  },
  "discriminative_clue": "The visible fact that distinguishes the candidates.",
  "relation": "AFTER"
}
```

Use the two-candidate schema without `candidate_z` for two-versus-one cases and
the three-candidate schema with `candidate_z` for all-different cases.

Allowed `relation` values:

```text
BEFORE
AFTER
FIRST
LAST
REVISIT
STATE_CHANGE
COUNT
IDENTITY
NONE
```

If planner JSON is invalid, do not retry. Use deterministic fallback queries:

```text
question + Candidate X text
question + Candidate Y text
question + Candidate Z text, when present
```

For the corresponding fallback report, use each candidate text as its
hypothesis, use the question's requested visible fact as the discriminative
clue, and obtain the relation from the existing deterministic temporal-program
compiler.

#### Verifier Prompt

```text
You are an evidence verifier for egocentric long-video question answering.

Evaluate every supplied candidate—X, Y, and Z when present—independently and
symmetrically using only the supplied, chronologically ordered images and frame
IDs.

SUPPORTED means the supplied evidence directly shows every decisive fact
required by the candidate.

CONTRADICTED means the supplied evidence directly shows an incompatible fact.

INSUFFICIENT means the required event or detail is missing, unclear, blurred,
occluded, or could occur inside an unsampled interval. Missing evidence is
never CONTRADICTED.

For temporal questions:
- identify both the reference event and target event;
- cite their supplied frame IDs; and
- verify their order using the supplied timestamps.

For FIRST or LAST, do not claim sufficient coverage unless the evidence
supports the required global ordering. For identity questions, verify that it
is the same object, person, or location. For multi-part candidate answers,
every decisive part must be supported.

The input states whether one evidence-refinement round is available. If it is
available and the supplied evidence does not decisively distinguish the
hypotheses, identify exactly one missing visible fact and provide one or two
neutral visual-search queries for finding it. You may anchor the search BEFORE,
AFTER, or AROUND supplied frame IDs. Never invent a timestamp.

Do not request refinement when exactly one hypothesis is fully SUPPORTED and
every competing hypothesis is CONTRADICTED. When refinement is unavailable,
evaluate the current evidence as the final pass and do not request more.

Do not use answer plausibility, option-letter priors, predictor identity, vote
count, candidate position, or unstated world knowledge.

Reason internally, then return only concise JSON matching the supplied schema.
```

Verifier input:

```text
Question: {question}

Candidate X hypothesis: {hypothesis_x}
Candidate Y hypothesis: {hypothesis_y}
Candidate Z hypothesis: {hypothesis_z_if_present}
Discriminative clue: {discriminative_clue}
Verification round: {1_or_2}
Refinement available: {true_or_false}

Evidence index:
F001 = 12.400 s
F002 = 18.100 s
...
```

Verifier output:

```json
{
  "candidate_x": {
    "verdict": "SUPPORTED",
    "supporting_frame_ids": ["F012"],
    "contradicting_frame_ids": [],
    "temporal_check": "PASS",
    "identity_check": "NOT_APPLICABLE",
    "coverage_check": "PASS",
    "brief_evidence": "The cited frame directly shows the required event."
  },
  "candidate_y": {
    "verdict": "CONTRADICTED",
    "supporting_frame_ids": [],
    "contradicting_frame_ids": ["F012"],
    "temporal_check": "FAIL",
    "identity_check": "NOT_APPLICABLE",
    "coverage_check": "PASS",
    "brief_evidence": "The visible event is incompatible with this candidate."
  },
  "candidate_z": {
    "verdict": "CONTRADICTED",
    "supporting_frame_ids": [],
    "contradicting_frame_ids": ["F012"],
    "temporal_check": "FAIL",
    "identity_check": "NOT_APPLICABLE",
    "coverage_check": "PASS",
    "brief_evidence": "The visible event is incompatible with this candidate."
  },
  "decisive_visible_fact": "The observation at F012 distinguishes the candidates.",
  "refinement": {
    "needed": false,
    "reason": "NOT_NEEDED",
    "missing_visible_fact": "",
    "retrieval_queries": [],
    "anchor_frame_ids": [],
    "temporal_region": "NONE"
  }
}
```

Use the two-candidate schema without `candidate_z` for two-versus-one cases and
the three-candidate schema with `candidate_z` for all-different cases.

Allowed values:

```text
verdict:
    SUPPORTED | CONTRADICTED | INSUFFICIENT

temporal_check / identity_check / coverage_check:
    PASS | FAIL | NOT_APPLICABLE | INSUFFICIENT

refinement.reason:
    NOT_NEEDED | MISSING_REFERENCE_EVENT | MISSING_TARGET_EVENT |
    TEMPORAL_GAP | IDENTITY_UNCLEAR | COUNT_COVERAGE |
    VISUAL_DETAIL_UNCLEAR | CONFLICTING_EVIDENCE | FINAL_ROUND

refinement.temporal_region:
    BEFORE | AFTER | AROUND | FULL_VIDEO | NONE
```

Example of a valid inconclusive first-round request:

```json
{
  "needed": true,
  "reason": "MISSING_TARGET_EVENT",
  "missing_visible_fact": "What the wearer handles immediately after placing the cup down.",
  "retrieval_queries": ["wearer handles object after placing cup down"],
  "anchor_frame_ids": ["F031"],
  "temporal_region": "AFTER"
}
```

In the strict schemas, every displayed field is required,
`additionalProperties` is false, citation lists contain at most three frame
IDs, and frame IDs must match the supplied `F001 ... F064` range.

When `refinement.needed` is true, `missing_visible_fact` must be non-empty,
`retrieval_queries` must contain one or two concrete observable queries,
`anchor_frame_ids` may contain at most two supplied frame IDs, and
`temporal_region` cannot be `NONE`. When it is false, the missing fact, queries,
and anchors must be empty and the temporal region must be `NONE`. `BEFORE`,
`AFTER`, and `AROUND` require at least one valid anchor; `FULL_VIDEO` requires
no anchor.

Code must reject a report when:

- a cited frame ID was not supplied;
- `SUPPORTED` has no supporting citation;
- `CONTRADICTED` has no contradicting citation;
- a candidate marked `SUPPORTED` has an applicable temporal, identity, or
  coverage check that is not `PASS`; or
- any required candidate record is absent;
- a refinement request violates the structured conditions above; or
- the final-round report requests further refinement.

Citation validation proves only that the frame was supplied. It does not prove
that the model interpreted the image correctly.

#### Model Configuration

| Parameter | Proposed default |
|---|---|
| Planner and verifier | `Qwen/Qwen3.5-9B` |
| Pinned revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Backend and dtype | vLLM, BF16 |
| Thinking | Disabled |
| Temperature | 0 |
| Tensor parallelism | 1 |
| Concurrency | 1 |
| Maximum images | 64 |
| Maximum pixels per image | 451,584 |
| Context length | 65,536 tokens |
| Planner maximum output | 256 tokens |
| Verifier maximum output | 640 tokens |
| Structured output | Strict JSON schema |
| Initial verifier calls | 1 |
| Conditional evidence-refinement rounds | At most 1 |
| Generic or unchanged retries | 0 |

Additional disagreement-stage call count:

| Case | Planner calls | Visual verifier calls | Follow-up retrievals |
|---|---:|---:|---:|
| Unanimous | 0 | 0 | 0 |
| Any disagreement, decisive first pass | 1 | 1 | 0 |
| Any disagreement, valid refinement | 1 | 2 | 1 |

Run the existing predictor calls in parallel or batches where possible. The
complete hidden-test dependency chain includes both Qwen3.5 evidence passes,
both Qwen3 evidence passes needed to construct the third predictor, the
conditional Qwen3 verifier, the new disagreement stage, and—only for a valid
non-decisive first report—one targeted retrieval and re-verification. Compute
shared SigLIP2 embeddings and extracted frames once per video.

If the planner, retrieval, verifier, JSON validation, citation validation, or
per-query time budget fails, return the frozen ensemble fallback. Never emit an
invalid option. Enforce the challenge's 300-second limit across the complete
per-question pipeline. Use a monotonic deadline of
`start + max_query_seconds - timeout_reserve_seconds` for every optional stage;
on expiry, cancel refinement, serialize the case-specific fallback, and leave
the reserve for cleanup and output.

#### Required CLI

Add these flags to the conditional runner:

```text
--mode hypothesis_existing|hypothesis_fresh
--disagreement-scope all_disagreements
--grounder-model
--grounder-cache-dir
--candidate-frames
--centers-per-candidate
--temporal-nms-seconds
--neighborhood-radius
--prior-quota-two-candidate
--prior-quota-three-candidate
--uniform-quota-two-candidate
--uniform-quota-three-candidate
--planner-max-new-tokens
--verifier-max-new-tokens
--max-evidence-refinements
--refinement-frame-budget
--refinement-local-candidate-frames
--refinement-full-candidate-frames
--refinement-around-seconds
--max-query-seconds
--timeout-reserve-seconds
```

Development command shape:

```bash
python baselines/longqa/run_generate_longqa_conditional.py \
  --mode hypothesis_fresh \
  --disagreement-scope all_disagreements \
  --subset-file configs/egolongqa_dev140_seed20260709.json \
  --primary-predictions <qwen35-temporal-pivot-predictions.jsonl> \
  --secondary-predictions <qwen35-uniform64-predictions.jsonl> \
  --tertiary-predictions <qwen3-verifier-predictions.jsonl> \
  --proofpack <temporal-pivot-proofpack.jsonl> \
  --proofpack-reference <qwen35-temporal-pivot-predictions.jsonl> \
  --grounder-model google/siglip2-so400m-patch14-384 \
  --grounder-cache-dir <siglip2-feature-cache> \
  --candidate-frames 128 \
  --centers-per-candidate 4 \
  --temporal-nms-seconds 10 \
  --neighborhood-radius 1 \
  --prior-quota-two-candidate 8 \
  --prior-quota-three-candidate 5 \
  --uniform-quota-two-candidate 24 \
  --uniform-quota-three-candidate 25 \
  --max-evidence-refinements 1 \
  --refinement-frame-budget 12 \
  --refinement-local-candidate-frames 64 \
  --refinement-full-candidate-frames 128 \
  --refinement-around-seconds 30 \
  --max-query-seconds 300 \
  --timeout-reserve-seconds 15 \
  --max-frames 64 \
  --planner-max-new-tokens 256 \
  --verifier-max-new-tokens 640 \
  --llm-model Qwen/Qwen3.5-9B \
  --output <run-directory>/predictions.jsonl \
  --audit-output <run-directory>/audit.jsonl \
  --eval-output <run-directory>/results.json
```

The predictor argument order is part of the frozen baseline: primary is
Qwen3.5 temporal pivot, secondary is Qwen3.5 uniform64, and tertiary is the
Qwen3 verifier. The temporal-pivot answer is retained only as the frozen
fallback when all-different verification is inconclusive.

Run the no-fresh-evidence, loop-disabled control using:

```text
--mode hypothesis_existing
--max-evidence-refinements 0
```

Run the fresh single-pass ablation using:

```text
--mode hypothesis_fresh
--max-evidence-refinements 0
```

Extend the existing test file and run:

```bash
python -m pytest \
  baselines/longqa/tests/test_run_generate_longqa_conditional.py -q
```

After generation, run the standard evaluator:

```bash
python scripts/eval_longqa_diagnostics.py \
  --predictions <run-directory>/predictions.jsonl \
  --annotations <wearable-ai-root>/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl \
  --output <run-directory>/diagnostics.json
```

#### Fallback Table

| Situation | Output |
|---|---|
| All predictors agree | Shared answer |
| Decisive first or final report supporting minority and contradicting majority | Minority answer |
| Decisive first or final report supporting majority and contradicting minority | Majority answer |
| All-different report with exactly one supported and both alternatives contradicted | Uniquely supported answer |
| Inconclusive or failed all-different verification | Temporal-pivot fallback |
| Valid non-decisive first report with a concrete missing-evidence request and sufficient time | Retrieve targeted new evidence and verify once more |
| Planner failure | Deterministic queries, then continue |
| Initial retrieval or verifier failure | Majority answer |
| Invalid JSON, citation, or refinement request | Majority answer; no retry |
| First report is inconclusive but cannot validly refine | Majority answer |
| Final report is inconclusive | Majority answer |
| Refinement retrieval fails or adds no new frame | Majority answer |
| Timeout | Existing ensemble fallback |

#### Output And Audit Contract

The normal prediction JSONL must remain compatible with
`build_prediction_row()`. Put verbose information in a separate aligned audit
JSONL containing:

```text
schema
index
sample_key
video_path
predictor_answers
vote_counts
case_type
unique candidate answers
majority_answer and minority_answer when applicable
case-specific fallback
private X/Y[/Z] mapping
planner report
retrieval queries
round-one fresh selected frames
prior selected frames
round-one evidence with frame IDs and timestamps
round-one verifier report
refinement trigger and reason
missing visible fact
refinement queries, anchors, and temporal region
round-two newly retrieved frames
round-two final evidence with frame IDs and timestamps
round-two verifier report
decision round
decision reason
selected answer
override flag
fallback flag and reason
planner / round-one retrieval / verifier-one / refinement retrieval /
verifier-two / total timing
configuration fingerprint
```

Never store gold labels or correctness inside an inference audit. Join labels
only inside the separate evaluator.

#### Evaluation Protocol

Run four systems with identical predictors and frozen fallbacks:

1. Frozen ensemble: majority for two-versus-one and temporal-pivot fallback for
   all-different.
2. `hypothesis_existing`: identical verifier without fresh retrieval.
3. `hypothesis_fresh` with `max_evidence_refinements=0`: fresh single-pass
   ablation.
4. `hypothesis_fresh` with `max_evidence_refinements=1`: bounded
   evidence-refinement Approach 1.

Use the existing fixed split without cross-validation:

```text
1. Implement, unit-test, and debug on dev140.
2. Freeze code, prompts, model revisions, budgets, and configuration hash.
3. Evaluate exactly once on the remaining 560 questions.
4. Do not modify the method after inspecting those 560 results.
5. Run all 700 only for the final headline comparison.
```

EgoLongQA validation contains one question per video, so dev140 and its
remaining-560 complement are video-disjoint.

Report:

```text
overall accuracy
two-versus-one accuracy
all-different accuracy
override count
minority-correct fixes
majority-correct regressions
all-different fixes and regressions versus temporal-pivot fallback
wrong-to-different-wrong changes
net gain = fixes - regressions
override precision = fixes / override count
decisive override precision = fixes / (fixes + regressions)
minority recovery
majority retention
invalid JSON and citation count
temporal and non-C accuracy
fresh/prior/uniform frame counts and overlap
refinement-trigger count and rate
valid and invalid refinement-request count
refinement retrievals that added genuinely new frames
round-one versus round-two final decisions
round-two minority-correct fixes
round-two majority-correct regressions
extra latency caused by refinement
p50 / p95 / maximum inference time
number of questions over 300 seconds
```

Promotion requires positive net gain on the frozen remaining-560 evaluation:

```text
two-versus-one fixes + all-different fixes
>
two-versus-one regressions + all-different regressions
```

A change of only one or two answers is weak evidence and must be reported as
such rather than treated as a robust improvement.

#### Minimum Tests

1. The frozen ensemble reproduces exactly `552/700` before intervention.
2. Unanimous rows make zero additional calls.
3. Both two-versus-one and all-different rows enter the same judge flow.
4. Predictor names, letters, votes, and gold fields never enter prompts.
5. The X/Y[/Z] mapping is deterministic and round-trips correctly; the
   two-candidate schema rejects Z and the three-candidate schema requires it.
6. X, Y, and Z when present receive equal fresh and prior evidence budgets.
7. The third predictor's derived evidence is not counted as a third source.
8. Frames are deduplicated, capped at 64, and chronologically sorted.
9. Every citation refers to a supplied frame ID.
10. Minority `SUPPORTED` plus majority `CONTRADICTED` overrides.
11. In an all-different case, exactly one `SUPPORTED` plus two `CONTRADICTED`
    selects the supported candidate.
12. Every other verdict combination uses the case-specific frozen fallback.
13. Temporal support without both event citations and valid order is rejected.
14. Multi-part answers require support for every decisive part.
15. A valid non-decisive first report with a concrete request may trigger
    exactly one refinement.
16. A decisive first report, invalid request, failed retrieval, or insufficient
    remaining time never triggers refinement.
17. Round two contains at least one new frame, remains chronological, and never
    exceeds 64 frames.
18. The second verifier cannot request or trigger a third pass.
19. A final `INSUFFICIENT` report returns the case-specific fallback.
20. Invalid JSON, invalid citations, retrieval failure, and timeout fall back.
21. Resume is accepted only when sample keys and the configuration fingerprint
    match.

## Approach 2: Evidence-Grounded Disagreement Arbitration

This is the simplest fallback when fresh evidence retrieval is unavailable. Do
not use predictor profiles.

For the controlled comparison with Approach 1, implement this as
`hypothesis_existing` from the Version 1.2 contract: use the same planner,
verifier model, first-pass prompt, and decision rule, replace the 24 fresh
frames with uniform evidence, and set `max_evidence_refinements=0`. This is
deliberately a loop-disabled single-pass control. The conceptual description
below remains the same.

Each predictor returns an answer and its selected, timestamped evidence frames.
Deduplicate these into two candidates for two-versus-one or three candidates
for all-different:

```text
P1 → A + frames
P2 → A + frames
P3 → B + frames
```

The judge checks the combined evidence frames for every unique candidate:

- If the frames clearly support **B** and contradict **A**, choose **B**.
- In that two-versus-one case, otherwise keep the majority answer **A**.
- For all-different, choose a candidate only when it is uniquely supported and
  both alternatives are contradicted; otherwise use temporal-pivot.

```text
Predictor disagreement
→ combine and chronologically order the evidence frames
→ test every unique candidate
→ select only a uniquely supported candidate whose alternatives are contradicted
→ otherwise use the case-specific frozen fallback
```

This is the entire baseline experiment. Predictor profiles, additional evidence
retrieval, calibration, and evidence-conditioned re-verification are separate
extensions.

### Simplified Decision Rule

Do not ask the judge to generate or update probabilities. This creates false
precision without calibration.

```text
Judge checks every unique candidate against the combined timestamped frames.

If exactly one candidate is supported and every alternative is contradicted:
    apply the case-specific selection rule
else:
    use majority for 2-vs-1 or temporal-pivot for all-different
```

Historical results merely explain the fallback: the majority answer is correct
more often when the evidence is inconclusive. Predictor profiles are therefore
excluded from the first experiment.

## Approach 3: Hypothesis-Driven Verification

Treat every candidate answer as a hypothesis that must be supported or
falsified using timestamped video evidence. Do not ask every verification
question on every example. Apply the core checks first, followed only by the
relevant failure-mode module. As written, this is a prompt-only single-pass
ablation; Approach 1 adds active retrieval and the bounded evidence-feedback
loop.

### Core Checks

Ask these questions for every candidate:

```text
1. What exact visible fact must be true for this answer to be correct?

2. Which timestamp directly shows that fact?
   Do not use plausibility or common sense as evidence.

3. Does this evidence distinguish the candidate from the competing answer,
   or could both answers fit the same frames?

4. Is there a frame that directly contradicts the candidate?
   Missing evidence is not contradiction.

5. Is the critical evidence clear enough?
   Check blur, occlusion, small objects, duplicate frames, and large frame gaps.
```

### Temporal Questions

Use these checks when the question asks about before, after, first, last, a
return, or a state change:

```text
6. Is the reference event itself correctly identified?

7. Are both the reference event and target event directly visible?

8. Do their timestamps satisfy the required order?

9. Could the important event occur inside an unsampled frame gap?

10. For FIRST or LAST:
    Was enough of the video inspected to rule out an earlier or later
    occurrence?

11. For RETURN:
    Is it genuinely the same location, with evidence that the wearer left and
    later returned?

12. For STATE CHANGE:
    Is it the same object before and after, and is the change directly visible?
```

### Object, Person, And Location Questions

```text
13. Is it the same object, person, or location across timestamps?

14. Does a close-up or crop preserve enough full-frame context to identify it?

15. Are the candidates distinguished by a small detail that is actually
    visible?
```

### Evidence-Selection And Ensemble Checks

```text
16. Do the majority predictors rely on the same or heavily overlapping frames?

17. Does the minority predictor provide unique evidence that the majority
    missed?

18. Is the judge favoring an answer because of vote count, predictor identity,
    option position, or language plausibility rather than video evidence?

19. Does a documented predictor failure mode genuinely occur here?
    A profile description alone is not evidence.

20. If every candidate is unsupported or contradicted, mark the case
    INSUFFICIENT instead of inventing another answer.
```

### Required Output For Each Candidate

```text
Required visible fact:
Supporting timestamps:
Contradicting timestamps:
Temporal or identity check:
Missing-evidence risk:
Relevant predictor failure mode:
Verdict: SUPPORTED / CONTRADICTED / INSUFFICIENT
```

The central questions are:

```text
What directly proves this candidate?
What directly falsifies it?
Could the evidence selector simply have missed the critical moment?
```

## Approach 4: Predictor-Specific Error Profiles

This is a training-free alternative to numerical reliability statistics. Each
predictor receives a short, qualitative error profile grounded in its previous
results.

### Build the Profiles Offline

For each predictor, inspect:

- cases where it was uniquely correct;
- cases where it was uniquely wrong;
- the evidence frames it selected in those cases; and
- recurring evidence-selection strengths, failures, or blind spots.

Summarize the findings using a fixed format:

```text
Predictor profile:
- Evidence-selection method:
- Usually succeeds when:
- Commonly fails when:
- Typical evidence blind spots:
```

Profiles must describe patterns actually observed in the labeled development
data. They should not contain invented strengths or failure modes.

### Use the Profiles During Disagreement

```text
1. The judge examines the combined, chronologically ordered evidence.

2. The judge tests the majority and minority answers against that evidence.

3. The judge checks whether each predictor's relevant documented strength or
   failure mode applies to the current question.

4. Override the majority only when:
       the minority answer has convincing visual evidence
       AND
       the majority predictors exhibit a documented failure pattern.

5. Otherwise keep the majority answer.
```

The video evidence remains the primary decision signal. Predictor profiles only
help the judge interpret a disagreement; they cannot replace or overrule clear
visual evidence.

This approach requires no probability estimates, model fine-tuning, calibration
head, or cross-validation. Build the profiles from a labeled development set,
freeze them, and evaluate the fixed prompt on the remaining data.
