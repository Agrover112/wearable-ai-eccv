# EgoLongQA Evaluation and Uncertainty Protocol

## Goal

Compare methods fairly on the 700-example validation set and show:

1. overall performance;
2. uncertainty caused by which examples are in the evaluation sample;
3. performance differences across question categories; and
4. why disagreement judging succeeds or fails.

This protocol is training-free. Bootstrap and saved-prediction analyses require
no extra VLM inference; matched text-only and single-frame controls require
one-time inference if their predictions have not already been saved.

## Required prediction table

Save one row per example:

| `example_id` | `video_id` | `category` | `gold` | `M1_prediction` | `M2_prediction` | ... |
|---|---|---|---|---|---|---|

All methods must be evaluated on exactly the same rows. Also store a Boolean
`correct` column for each method.

## Primary results

For every method, report:

- full-set accuracy on all available evaluation examples;
- macro-average accuracy across categories;
- worst-category accuracy;
- per-category accuracy.

For every important pair of methods \(M_a,M_b\), report the paired difference:

\[
\Delta_{a,b}=\operatorname{Acc}(M_a)-\operatorname{Acc}(M_b).
\]

The primary comparison is the difference \(\Delta\), not two unrelated confidence
intervals around the individual accuracies.

For the disagreement judge, additionally report:

| Quantity | Definition |
|---|---|
| Fix | Frozen case-specific fallback wrong \(\rightarrow\) final answer correct |
| Regression | Frozen case-specific fallback correct \(\rightarrow\) final answer wrong |
| Net gain | Number of fixes minus number of regressions |
| Refinement trigger rate | Questions receiving the targeted second evidence pass |
| Follow-up rescue | First pass wrong/inconclusive \(\rightarrow\) final answer correct |

Separate first-pass and follow-up fixes and regressions. Also report valid and
invalid refinement requests, follow-up retrievals that found new frames, and
the additional p50/p95 latency from the conditional round.

## Video-blind and reduced-visual-input evaluation

VideoQA accuracy can be inflated by answer-position, wording, option-length, or
commonsense shortcuts. Following the diagnostic setup used by
[Google's EgoTempo](https://arxiv.org/pdf/2503.13646), evaluate progressively
stronger inputs while keeping the model, question, options, prompt, option
order, and decoding fixed:

| Input condition | Model receives | What it tests |
|---|---|---|
| Chance | No model; uniform A/B/C/D guess | Nominal MCQ floor |
| Always-C | No video or language model | Label-position prior |
| Text only | Question and four options | Language and commonsense shortcuts |
| Single frame | Question, options, and central frame | Static visual shortcuts |
| Full method | Normal selected video evidence | Actual video contribution |

Current dataset-only reference points are:

| Baseline | Accuracy |
|---|---:|
| Uniform random choice | 25.0% |
| Always predict C | 444/700 = 63.43% |
| Select the shortest option | 40.3% |

The text-only and single-frame controls must use the same answer model and
constrained A/B/C/D decoding as the matched video run. Do not substitute black
or blank images for the text-only condition because they introduce an
out-of-distribution visual input. Save their predictions using the same
`sample_key` values as every video method.

For each condition, report:

- overall accuracy and error rate;
- accuracy when the gold answer is not C;
- macro-average accuracy over gold option letters A/B/C/D;
- predicted option-letter distribution;
- temporal-question and non-temporal-question accuracy; and
- per-category accuracy.

For a video method \(M_v\) and its matched text-only control \(M_t\), report:

\[
\text{VisualGain} =
\operatorname{Acc}(M_v)-\operatorname{Acc}(M_t)
\]

with a paired-bootstrap 95% confidence interval. Also report the transition
counts:

```text
text wrong → video correct
text correct → video wrong
both correct
both wrong
```

This reveals whether visual evidence genuinely repairs language-only errors or
introduces distractions. Because option C is unusually frequent, optionally
repeat the cheap text-only control with cyclically rotated option order to
measure position sensitivity.

### Evidence-blind judge sanity check

An **evidence-blind judge** is not a fully video-blind system: its neutral
candidate answers were still produced by video predictors. It tests only
whether the judge follows textual plausibility or candidate priors when visual
evidence is absent.

Run the verifier with the question and neutral candidate hypotheses but no
images or evidence index. For this diagnostic only, allow the verifier to
describe a missing-evidence request but do not execute retrieval. Under the
strict verification contract, every supplied candidate should be
`INSUFFICIENT` and the request should target the absent visual fact; the
harness then retains the
case-specific frozen fallback. Report:

- unsupported `SUPPORTED` or `CONTRADICTED` verdicts;
- valid, missing, and malformed requests for visual evidence;
- attempted overrides;
- invalid citations; and
- final deviation from the frozen ensemble fallback, which should be zero.

Do not present this as a competing QA method. It is a grounding sanity check
for the judge.

## Stratified paired bootstrap

Use this to estimate **dataset-sampling uncertainty**.

For \(B=10{,}000\) bootstrap repetitions:

1. Within each category \(k\), sample \(n_k\) rows **with replacement** from that
   category's \(n_k\) saved rows.
2. Concatenate the sampled rows from all categories.
3. Use the **same sampled row indices for every method**.
4. Recalculate each method's accuracy from its saved correctness values.
5. Store every paired accuracy difference \(\Delta^{(b)}_{a,b}\).

Report the 2.5th and 97.5th percentiles of the stored differences as the paired
bootstrap 95% confidence interval:

\[
\mathrm{CI}_{95\%}(\Delta)=
\left[
Q_{0.025}\!\left(\Delta^{(1:B)}\right),
Q_{0.975}\!\left(\Delta^{(1:B)}\right)
\right].
\]

Interpretation:

- interval entirely above \(0\): evidence that \(M_a\) is better;
- interval containing \(0\): the observed difference is inconclusive;
- this measures **data-sampling uncertainty**, not VLM randomness.

Stratifying preserves the original number of examples in each category. Because
EgoLongQA currently has one question per video, resampling rows is appropriate. If
future data contains multiple questions per video, resample whole videos instead.

## Category analysis

For every category, calculate:

| Category | \(n\) | \(Acc(M_a)\) | \(Acc(M_b)\) | \(\Delta_{a,b}\) | Paired 95% CI |
|---|---:|---:|---:|---:|---:|

Within a category, obtain its paired CI by resampling only that category's rows
with replacement. Also report:

\[
\text{MacroAcc}(M)=\frac{1}{K}\sum_{k=1}^{K}\operatorname{Acc}_k(M)
\]

\[
\text{WorstCategoryAcc}(M)=\min_k \operatorname{Acc}_k(M).
\]

The spread of category scores describes **category sensitivity or heterogeneity**;
it is not model uncertainty. Treat category claims as diagnostic unless multiple
comparisons are corrected.

## Qualitative analysis: Where Does the Judge Help—and Why?

Use an **outcome-conditioned evidence-flow analysis**, not a gallery of
hand-picked successes. The analysis should determine:

```text
correct candidate available?
    → decisive evidence selected?
    → first retrieval found it?
    → verifier identified what was still missing?
    → targeted follow-up found it?
    → final verifier used it correctly?
```

This separates candidate-set, evidence-retrieval, visual-perception, temporal-
reasoning, and arbitration failures.

The clearest Google precedent is
[EgoTempo](https://arxiv.org/pdf/2503.13646), which compares predictions for the
same question with 1, 8, and 64 frames, marks successes and failures, presents
failure cases across reasoning categories, and tests chronologically shuffled
frames. [GroundVQA](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Di_Grounded_Question-Answering_in_CVPR_2024_supplemental.pdf)
visualizes predicted and reference temporal evidence, while
[VideoAgent](https://arxiv.org/pdf/2403.10517) traces missing information through
retrieval to the final answer. Our corresponding comparison is:

```text
individual predictions
    → majority decision
    → judge with existing evidence
    → judge after first fresh retrieval
    → optional targeted follow-up retrieval
    → final verification
    → final answer
```

### Analysis population

The primary population is all 213 disagreements:

| Initial case | Questions | Evaluation purpose |
|---|---:|---|
| Majority candidate correct | 107 | Avoid regressions |
| Minority candidate correct | 63 | Main recoverable target |
| Neither 2-vs-1 candidate correct | 19 | Candidate-set ceiling |
| All-different, some candidate correct | 20 | Three-way arbitration |
| All-different, no candidate correct | 4 | Candidate-set ceiling |

After judging, report every row under one of these outcome strata:

| Outcome | Meaning |
|---|---|
| Successful fix | Wrong frozen fallback changed to a correct candidate |
| Safe retention | Correct frozen fallback retained |
| Missed opportunity | Correct candidate available but wrong fallback retained |
| Regression | Correct frozen fallback changed to a wrong candidate |
| All-different fix | Judge corrects the frozen temporal-pivot fallback |
| All-different regression | Judge changes a correct temporal-pivot fallback |
| Candidate-set limitation | Neither proposed candidate was correct |

When both judge controls are available, additionally distinguish a **fresh-only
rescue**—the existing-evidence judge fails but fresh retrieval enables the
correct decision—from a rescue achieved without fresh evidence. Within the
fresh method, distinguish a **first-pass rescue** from a **follow-up rescue**
that becomes correct only after the verifier requests and receives new frames.

### Main qualitative figure

Show five rows: one two-versus-one fix, one safe retention, one missed
opportunity, one regression, and one all-different case. Put candidate-set
limitations in the appendix. Use these columns:

```text
question and candidate texts
    → predictor answers
    → common chronological timeline
    → existing evidence
    → first-pass fresh evidence and verdict
    → missing-evidence request and targeted follow-up frames
    → final citations and verdicts
    → final answer versus gold
```

Example timeline:

```text
0 s ─────────────────────────────────────────────────── video end
Uniform evidence:  ●       ●       ●       ●
Temporal-pivot:          ▲ ▲
First retrieval:                             ◆ ◆
Targeted follow-up:                                ■ ■
Human decisive span:                          █████
```

Show timestamps and preserve chronological order. Display evidence sources as
`uniform`, `temporal-pivot`, `first retrieval`, and `targeted follow-up`; the
third predictor reuses uniform and pivot evidence and therefore must not be
drawn as a third independent evidence selector. Show the requested missing
fact, concise structured evidence statements, and frame citations—not private
chain-of-thought.

### Google-style diagnostic figures

The qualitative case studies should be accompanied by two aggregate diagnostic
figures inspired by EgoTempo Figures 5 and 6. Use the same method colors,
markers, category order, and accuracy scale in both figures. Limit the main
figure to at most five method lines; move additional predictors or ablations to
the appendix.

#### Accuracy across categories

Create a two-panel figure:

**Top panel — category profile**

- x-axis: the 13 EgoLongQA scene categories in a fixed order, preferably
  decreasing validation-set size;
- y-axis: accuracy from 0% to 100%;
- recommended series: matched text-only Qwen3.5, central single-frame Qwen3.5,
  Qwen3.5 uniform64, frozen ensemble baseline, and the bounded-refinement judge;
- optionally show the existing-evidence judge as a thin dashed control;
- place the fresh single-pass judge in the appendix as the direct loop
  ablation, or replace Qwen3.5 uniform64 with it when the judge contribution is
  the figure's main focus;
- move the temporal-pivot predictor to the appendix if five lines already make
  the main panel crowded;
- print each category's \(n\) below its label; and
- shorten long x-axis labels, with full names in the caption.

**Bottom panel — judge gain**

For category \(k\), plot:

\[
\Delta_k =
\operatorname{Acc}_k(\text{bounded-refinement judge})
-
\operatorname{Acc}_k(\text{frozen ensemble baseline}).
\]

Use a zero-centred axis, green for positive gains, red for regressions, and a
paired-bootstrap 95% confidence interval for every \(\Delta_k\). The bottom
panel prevents small method differences from disappearing inside the 0–100%
accuracy range.

Interpret the figure as robustness across **scene contexts**, not as an
explanation of reasoning ability. Categories contain only 19–127 examples, so
the category results are diagnostic; do not claim a category-specific
improvement from point estimates alone.

Suggested caption:

> **Accuracy across EgoLongQA scene categories.** Top: category-level accuracy
> from text-only and single-frame controls through the full video predictor,
> frozen ensemble baseline, and bounded-refinement judge. Bottom: paired accuracy change
> of the judge relative to the majority; bars show stratified paired-bootstrap
> 95% confidence intervals. Category sample counts are shown below the x-axis.

#### Performance variation with video duration

EgoTempo uses multiple duration ranges because its videos vary substantially.
EgoLongQA is different: 616/700 videos fall between 9.5 and 10.5 minutes.
Therefore, do not use equal-width bins that hide the central mode or produce
nearly empty groups. Use these pre-declared, interpretable ranges:

| Duration range | Expected \(n\) | Interpretation |
|---|---:|---|
| \(<9.5\) min | 11 | Short tail |
| \(9.5\)–\(10.5\) min | 616 | Main recording mode |
| \(>10.5\)–\(12\) min | 49 | Moderately long |
| \(>12\) min | 24 | Long tail |

Recalculate and display the exact counts after joining predictions to duration
metadata. Create another two-panel figure:

**Top panel — duration profile**

- x-axis: the four ordered duration ranges;
- y-axis: accuracy from 0% to 100%;
- use the same method series and visual styling as the category figure;
- show paired-bootstrap or Wilson 95% intervals and the number of videos in
  each bin; and
- optionally add a secondary x-axis showing effective temporal spacing,
  \(D/N_{\text{frames}}\), for a fixed frame budget.

**Bottom panel — judge gain**

Plot the paired difference between the bounded-refinement judge and the frozen
ensemble baseline inside each duration range, with a zero line and paired-bootstrap 95%
intervals. This tests whether targeted evidence seeking becomes more valuable
as a fixed frame budget becomes temporally sparser.

The short and long tails have small sample sizes, so label their estimates as
exploratory. Do not infer a duration trend unless the paired differences and
their intervals show a consistent pattern. If matched runs at several frame
budgets are available, an appendix version may reproduce EgoTempo more closely
by plotting 16, 32, and 64 frames; all other settings—model, resolution, prompt,
and decoding—must remain fixed.

Suggested caption:

> **Performance variation with video duration.** Top: accuracy within
> pre-declared duration ranges. Bottom: paired gain of bounded-refinement
> judging over the frozen ensemble baseline. EgoLongQA is concentrated near ten
> minutes, so sparse tail estimates are shown with confidence intervals and
> sample counts.

### Manual evidence audit

Audit approximately 30–40 disagreement cases using a fixed random seed. Include
all overrides if feasible and a seeded, outcome-matched sample of non-overrides.
Stratify primarily by reasoning requirement:

- cross-time ordering or state change;
- fine object identity or OCR detail;
- counting or repeated actions;
- spatial location; and
- global activity, intent, or summary.

Use the dataset's scene category only as a secondary balancing variable because
scene categories do not directly identify the reasoning failure.

For each audited case, mark the human-identified decisive evidence span or
spans and assign one primary diagnosis:

- correct answer absent from the candidate set;
- decisive evidence absent from the first selected frames;
- verifier failed to identify the missing visible fact;
- targeted follow-up retrieval missed the requested fact;
- fine-detail or visual-perception error;
- temporal order, counting, or state-tracking error;
- final judge/arbitration error despite adequate evidence; or
- ambiguous or incorrect annotation.

If possible, use two independent reviewers, resolve disagreements, and report
their agreement. Report diagnosis frequencies alongside the four main examples.
Disclose the sample seed and selection rule so the examples are reproducible
and do not appear cherry-picked. Publish video frames only when permitted by the
dataset license, and obscure personally identifying content when required.

## Judge prompt description and templates

The judge uses **neutral, evidence-grounded hypothesis verification**. It does
not receive predictor names, vote counts, option letters, category labels,
confidence scores, or gold answers. The two or three unique answer texts are
mapped deterministically to neutral `Candidate X`, `Candidate Y`, and optionally
`Candidate Z` identifiers. The mapping back to the ensemble decision happens in
code after verification.

The prompt has five important rules:

1. evaluate every supplied candidate independently and symmetrically;
2. use only directly visible evidence from the supplied frames;
3. treat missing or unclear evidence as `INSUFFICIENT`, not `CONTRADICTED`;
4. cite the frames supporting every verdict; and
5. when the first evidence pack is inconclusive, state exactly what visible
   evidence is missing before requesting one targeted follow-up.

Use one text-only planner call and one initial visual verifier call. A valid,
non-decisive first report may trigger one targeted retrieval followed by one
final visual verification. All model calls use temperature \(0\) and strict
JSON output. There are no generic retries and no third verification. The
complete implementation contract is in
[`JUDGE_APPROACHES.md`](../JUDGE_APPROACHES.md); the templates below are the
paper-facing reproducible form.

### Query-planner template

The planner rewrites the two or three unique answers as observable hypotheses
and produces balanced visual-search queries. It must not select a winner.

**System template**

```text
You create neutral visual-search queries for egocentric long-video question
answering.

Do not decide which candidate is correct. Do not use vote counts, predictor
identities, option letters, probabilities, or facts not stated in the question
and candidate answers.

Rewrite every supplied candidate as a concrete, testable visual hypothesis.
Identify the minimum observable fact or facts that distinguish them. Produce at
most two short visual retrieval queries per candidate.

For temporal questions, separately cover the reference event and the
candidate-specific target event. Every query must describe something visible in
an individual frame or short neighboring sequence.

Return only JSON matching the supplied schema.
```

**User template**

```text
Question: {{question}}
Candidate X: {{candidate_x_text}}
Candidate Y: {{candidate_y_text}}
Candidate Z: {{candidate_z_text_if_present}}
```

**Output template**

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

Use separate strict schemas: omit `candidate_z` for two-versus-one cases and
require it for all-different cases.

Allowed relations are:

```text
BEFORE | AFTER | FIRST | LAST | REVISIT | STATE_CHANGE |
COUNT | IDENTITY | NONE
```

### Evidence-verifier template

The verifier receives the question, hypotheses, clue, chronologically ordered
images, and a trusted frame-ID-to-timestamp index. Displayed frame IDs are
assigned only after evidence has been deduplicated and sorted.

**System template**

```text
You are an evidence verifier for egocentric long-video question answering.

Evaluate every supplied candidate—X, Y, and Z when present—independently and
symmetrically using only the supplied, chronologically ordered images and frame
IDs.

SUPPORTED means that the supplied evidence directly shows every decisive fact
required by the candidate.

CONTRADICTED means that the supplied evidence directly shows an incompatible
fact.

INSUFFICIENT means that a required event or detail is missing, unclear, blurred,
occluded, or could occur in an unsampled interval. Missing evidence is never
CONTRADICTED.

For temporal questions, identify and cite both the reference event and target
event, then verify their order using the supplied timestamps. For FIRST or LAST,
require evidence supporting the global ordering. For identity questions,
confirm that it is the same object, person, or location. Every decisive part of
a multi-part answer must be supported.

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

**User template**

```text
Question: {{question}}

Candidate X hypothesis: {{hypothesis_x}}
Candidate Y hypothesis: {{hypothesis_y}}
Candidate Z hypothesis: {{hypothesis_z_if_present}}
Discriminative clue: {{discriminative_clue}}
Verification round: {{1_or_2}}
Refinement available: {{true_or_false}}

Evidence images are supplied in chronological order.
Evidence index:
{{frame_id_timestamp_index}}
```

Example evidence index:

```text
F001 = 12.400 s
F002 = 18.100 s
...
F064 = 587.900 s
```

**Output template**

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

Omit `candidate_z` under the strict two-candidate schema and require it under
the strict three-candidate schema.

Allowed values are:

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

Example valid first-round request:

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

When `refinement.needed=true`, require a non-empty missing fact, one or two
observable queries, at most two valid anchor frame IDs, and a non-`NONE`
temporal region. When it is false, require empty request fields and `NONE`.
`BEFORE`, `AFTER`, and `AROUND` require an anchor; `FULL_VIDEO` does not.
Reject a final-round request for further refinement.

### Conditional evidence-refinement round

The planner runs once. If the first verifier report is valid but non-decisive,
and it provides a concrete missing fact and valid query, retrieve unseen frames
for that request. Rebuild a chronological evidence pack containing the newly
retrieved frames and the most important first-round evidence, capped at 64
images, then verify every candidate once more.

This is not a repeated sample of the same judgment: the second call must contain
at least one genuinely new frame. The second verifier does not see the first
verdict, but keeps the same neutral X/Y[/Z] mapping, hypotheses, and clue. It
cannot request a third pass. If the request is invalid, retrieval adds no new
evidence, the shared monotonic deadline expires, or the final result remains
inconclusive, use the case-specific frozen fallback. Give retrieval and final
verification the same hard deadline, leaving a fixed reserve for cleanup and
output.

### Programmatic decision rule

The prompt does not ask, “Which predictor wins?” Code maps the neutral IDs back
to the two or three unique candidates, validates every citation, and applies:

```text
deadline = monotonic_start + max_query_seconds - timeout_reserve_seconds
fallback = majority for 2-vs-1
           temporal-pivot answer for all-different

report = VERIFY(
    initial_evidence,
    refinement_available=true,
    deadline=deadline
)

if report has exactly one valid SUPPORTED candidate
   and every alternative is validly CONTRADICTED:
       if 2-vs-1:
           return minority only when minority is SUPPORTED
           otherwise return majority
       if all-different:
           return the uniquely SUPPORTED candidate

if report contains a valid missing-evidence request
   and one refinement remains
   and the monotonic deadline has not expired:
       new_evidence = RETRIEVE_MISSING_EVIDENCE(
           report.refinement,
           deadline=deadline
       )

       if at least one new frame is found:
           final_report = VERIFY(
               revised_evidence,
               refinement_available=false,
               deadline=deadline
           )

           if final_report has exactly one valid SUPPORTED candidate
              and every alternative is validly CONTRADICTED:
                  apply the same case-specific selection rule

return fallback
```

This conservative rule makes final `INSUFFICIENT`, malformed JSON, invalid
citations, failed refinement, timeouts, and verifier failures fall back to the
majority for two-versus-one cases or the frozen temporal-pivot answer for
all-different cases.

### Prompt reporting checklist

Freeze the prompts before held-out evaluation and report:

- exact system and user templates;
- prompt/schema version and configuration hash;
- model name and pinned revision;
- temperature, thinking mode, and output-token limits;
- candidate-neutralization rule;
- evidence ordering, first- and second-round frame budgets;
- refinement trigger, structured request schema, and maximum round count;
- per-round retrieval and verifier timing;
- allowed verdicts and programmatic override rule; and
- invalid-output, timeout, and fallback counts.

Store the exact rendered prompts or their hashes in the run audit, but never
place gold labels or correctness fields in an inference prompt.

## What the other procedures measure

| Procedure | What it measures | Use here |
|---|---|---|
| Paired bootstrap of saved rows | Dataset-sampling uncertainty | Primary |
| Repeated full VLM runs/seeds | Model, decoding, or retrieval randomness | Only if the pipeline is stochastic |
| Five disjoint 140-example folds | Variation between subsets of this validation set | Secondary diagnostic |
| True cross-validation | Generalization of a fitted/tuned method | Not needed for a fixed training-free method |

For a deterministic pipeline—fixed frames, fixed prompts, temperature \(0\), and
constrained answer decoding—one inference run is sufficient. If any component is
stochastic, run the complete method on all examples for 3–5 seeds and report
mean \(\pm\) standard deviation separately from the bootstrap interval.

The average accuracy over five equal, disjoint folds equals the full-700 accuracy.
Their standard deviation can show subset heterogeneity, but it should not replace
the paired bootstrap confidence interval.

## Development discipline

1. Debug and choose the method using `dev140`.
2. Freeze prompts, thresholds, frame budgets, and fallback rules.
3. Evaluate the frozen method on held-out validation examples.
4. Save every prediction once.
5. Run all bootstrap and category analyses from the saved prediction table on CPU.
6. Keep the hidden challenge test set untouched until final submission.

Do not tune a method after examining its held-out or hidden-test errors.

## Recommended paper table

| Method | Overall accuracy | Macro accuracy | Worst category | Difference vs. baseline (paired 95% CI) |
|---|---:|---:|---:|---:|
| Baseline |  |  |  | — |
| Proposed |  |  |  |  |

Add a second table with per-category accuracies and paired differences. State the
bootstrap repetition count, stratification variable, resampling unit, and random
seed in the caption or evaluation section.
