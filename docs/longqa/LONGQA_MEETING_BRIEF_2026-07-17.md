# EgoLongQA Meeting Brief - 2026-07-17

## Purpose Of This Document

This brief explains the EgoLongQA project as if the reader has just joined the
team. It summarizes the established system, then focuses on everything
implemented and evaluated since the 2026-07-13 meeting.

The main message is straightforward: the strongest competition result remains
the full pivot/uniform disagreement verifier at `539/700` (`77.00%`). The work
since July 13 has not produced a higher full-validation score, but it has
substantially improved our understanding of the remaining errors. We now have:

- several tested ways of allocating frames adaptively;
- multiple candidate-arbitration baselines;
- a frame-level evidence audit;
- a consolidated disagreement dataset with grouped cross-validation;
- a question-independent semantic event ledger;
- complete-answer likelihood scoring from Qwen itself; and
- an object-level detection augmentation currently being stabilized.

These experiments have narrowed the useful search space. Simple increases in
frame count, retrieval complexity, prompt structure, or likelihood scoring are
not enough. The remaining opportunity is to represent events and relations more
faithfully, then use that evidence without overturning already-correct answers.

## The Task In Plain Language

EgoLongQA contains 700 long first-person videos. A question may refer to events
that occur minutes apart, for example:

- what happened after a purchase;
- which object appeared first or last;
- whether the wearer returned to a previous place;
- how an object changed state; or
- how two separate encounters are related.

For each sample, the system receives:

- a video;
- a natural-language question;
- four answer options labeled A, B, C, and D; and
- a gold option used for local validation.

The final answer model is `Qwen/Qwen3-VL-8B-Instruct`. Qwen does not receive the
complete video stream. We first choose a limited set of images, put them in
chronological order, append the question and options, and ask Qwen to return one
letter.

This creates two distinct technical problems:

1. **Evidence selection:** which frames should Qwen see?
2. **Answer arbitration:** when two evidence policies disagree, which answer
   should we trust?

Most of the recent work targets one of these two problems.

## Data, Subsets, And Outputs

### Data

- Annotations:
  `data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl`
- Videos: `data/wearable-ai/egolongqa/val/`
- Full local validation set: 700 questions.

### Iteration subsets

- `configs/egolongqa_dev20_seed20260709.json`: a cheap 20-question proof-of-
  concept set.
- `configs/egolongqa_dev140_seed20260709.json`: the main development set.
- The remaining 560 rows have also been used as a held-out complement for
  selected calibration experiments.

Dev20 is not a subset of dev140. Runs and proof packs must therefore be joined
using stable sample keys rather than assuming row overlap.

### Important evaluation caveat

The gold labels are heavily imbalanced. Option C is correct for `444/700`
questions, and C is correct for `14/20` questions in dev20. Consequently, a
method can obtain apparently reasonable accuracy without using the video.

We report accuracy together with:

- accuracy of always choosing C;
- non-C accuracy;
- predicted option distribution;
- temporal-question accuracy;
- per-category accuracy; and
- paired fixes and regressions against a reference run.

### Run artifacts

Canonical completed outputs are stored under `runs/egolongqa/<run_name>/`:

- `predictions.jsonl`: raw and parsed answers;
- `results.json` and `results_summary.json`: official accuracy output;
- `diagnostics.json`: shortcut-aware metrics and per-sample diagnostics;
- `proofpack.jsonl`: selected frame metadata for grounded runs;
- method-specific files such as `semantic_scores.jsonl` or
  `event_ledger.jsonl`; and
- `slurm.out`, `slurm.err`, and `vllm_server.log`.

Large reusable caches and exported images live in scratch rather than the
repository.

## Established Pipeline Before This Meeting

The best first-pass system uses a 64-frame temporal proof pack at approximately
672x672 pixels per frame. A lightweight question compiler recognizes temporal
operators such as `AFTER`, `BEFORE`, `FIRST`, `LAST`, and `STATE_CHANGE`.
SigLIP2 similarity locates relevant candidate frames, while uniform anchors and
bridge frames preserve the surrounding story.

Two complementary full-validation runs are especially important:

- temporal pivot: `528/700`;
- uniform64/672: `514/700`.

They disagree on 122 questions. A second Qwen call on only these disagreements
produces `539/700`, the current best result. Their candidate oracle, which picks
the correct candidate whenever either run is correct, is `575/700`. The
36-answer gap between the verifier and this oracle is the clearest remaining
opportunity that does not require a larger model.

## Pipeline Vocabulary

### Candidate frame

One image sampled from the video before final selection. Grounded experiments
often begin with 128 or 256 uniformly spaced candidates.

### Embedding

A vector of numbers representing the semantic content of text or an image.
Cosine similarity between a text embedding and image embedding estimates how
well the frame matches the query.

### SigLIP2

The image-text encoder used for inexpensive relevance scoring. It proposes
evidence but does not answer the multiple-choice question.

### Proof pack

The final chronological collection of frames sent to Qwen. Frames may be
labeled as pivots, targets, local context, bridges, anchors, or coverage fill.

### Pivot

The event used to orient a temporal question. In "What happened after paying?",
paying is the pivot and the search proceeds forward from it.

### Eventlet

A small local temporal neighborhood around a selected event center. It gives
Qwen before/after context rather than an isolated image.

### Verifier

A second model call used only when two first-pass systems disagree. It inspects
combined evidence and selects the final answer.

### Log probability

The model's numerical confidence in a token or text continuation. Higher values
mean the model considers that answer more likely under the supplied evidence.

## Work Implemented Since 2026-07-13

## 1. QCA-Style Segment Allocation

### Goal

Uniform sampling gives every temporal region the same number of frames. QCA was
adapted as a training-free way to allocate more frames to segments that appear
both visually informative and relevant to the question.

The video is split into 16 temporal segments. Each segment receives a score
combining:

- average question/options relevance; and
- visual content deviation, meaning how much the segment changes internally.

Frames are then selected within each segment using relevance and diversity.

### Implemented variants

- standalone QCA for every question;
- QCA only for `GLOBAL` questions, with temporal pivot used elsewhere.

### Results

- standalone QCA: `104/140`;
- QCA-global router: `109/140`.

### Audit finding

Every sample received the same quota vector: four frames in each of 16
segments. The contribution scores varied, but the temperature, normalization,
and rounding pipeline flattened the final allocation.

The run therefore tested stratified relevance-diversity selection, not truly
dynamic QCA allocation. QCA contributes two answers missed by pivot and uniform,
but the current allocator should not be promoted without fixing quota scaling
and asserting non-uniform allocations in tests.

## 2. Multi-Event Retrieval

### Goal

Some questions describe two or three separate events. A single pivot query may
retrieve one event well while missing the other. The multi-event path splits the
question into clauses, retrieves evidence for each clause, and inserts
chronological bridges.

The router applies this policy primarily to `FIRST` and `STATE_CHANGE`
questions, retaining the stronger pivot policy elsewhere.

### Result

- multi-event router: `110/140`.

### What the audit showed

Only five routed answers differ from pivot: two fixes and three regressions.
The weak point is clause parsing. Text fragments such as "seen carrying it" do
not identify the shared object or the relationship between events.

The next meaningful version needs structured event arguments, for example:

```text
event_1: purchase(object=X, place=store_1)
event_2: carry(object=X, place=store_2)
shared_entity: X
```

Adding more event centers without entity linking is unlikely to help.

## 3. Support-And-Contradiction Verification

### Goal

The original verifier can choose any of four options. The new prompt asks it to
check visible support, visible contradiction, and required temporal order before
changing the answer. It is gated to selected operators rather than running on
all disagreements.

### Result

- support/contradiction verifier: `111/140`.

It ties pivot and the earlier verifier. It makes three fixes and three
regressions relative to pivot, so the additional reasoning prompt changes
behavior but not aggregate quality.

## 4. Frame-Level Evidence Audit

We exported representative frames for:

- two QCA-only wins;
- all five multi-event changes;
- 24 full-verifier fixes; and
- 13 full-verifier regressions.

The audit is documented in
`documentation/LONGQA_EVIDENCE_AUDIT_2026-07-14.md`. Images are stored in
scratch under the frame-audit directory.

The strongest finding concerns verifier freedom. Of eight occasions where the
full verifier selected neither pivot nor uniform, only one was correct. Most of
the verifier's value comes from choosing the better existing candidate, not
inventing a third answer.

This motivated candidate-constrained arbitration experiments.

## 5. Pairwise Order-Swap Verifier

### Method

For each pivot/uniform disagreement, the verifier sees exactly two candidate
answer texts, not all four options. It is queried twice:

1. pivot candidate shown first;
2. uniform candidate shown first.

The answer changes are mapped back to the original candidates. We accept a
change only when both orders choose the same semantic answer; otherwise we keep
pivot. This tests and suppresses position bias.

### Result

- `109/140`, with 38 verifier calls.

The method is safer than unconstrained third-option generation, but still does
not improve pivot. Candidate restriction alone does not solve evidence
ambiguity.

## 6. Option-Permutation Verifier

### Method

Each disagreement is answered four times after cyclically rotating which answer
text appears under A, B, C, and D. Predictions are mapped back to their original
semantics and voted. The final choice is restricted to the pivot or uniform
candidate, with pivot used for ties.

### Result

- `109/140` using 76 calls;
- two fixes and four regressions relative to pivot.

Eight of the 76 individual calls select a third semantic option, and option
order materially affects the model. Voting reduces but does not remove that
instability.

## 7. Q-Frame-Style Mixed Resolution

### Goal

Instead of sending every selected frame at 672px, spend resolution on the most
relevant frames and use cheaper images for broad coverage.

The implemented budget is:

- 4 high-resolution frames at 451,584 pixels;
- 8 medium-resolution frames at 200,704 pixels;
- 32 low-resolution frames at 50,176 pixels.

Frames are sampled probabilistically from relevance scores, then ranked so the
best receive the largest pixel budgets.

### Result

- `96/140`, runtime 36 minutes.

It is efficient but loses too much temporal and visual evidence. Four detailed
frames cannot compensate for reducing the total pack to 44 and heavily
compressing most images.

## 8. AdaQ Adaptive Sampling

### Method

AdaQ converts normalized frame relevance into a probability distribution. Its
temperature is adjusted by the variance of relevance scores. A top-p filter
keeps the smallest high-probability candidate set whose cumulative probability
reaches 0.95, after which 64 frames are sampled without replacement.

In plain language, it tries to concentrate effort when a few moments are
clearly relevant and spread effort when relevance is uncertain.

### Result

- c256/final64 at 448px: `96/140`;
- runtime: 4h10m40s.

The adaptive relevance distribution did not preserve enough narrative context,
and doubling candidates made it expensive.

## 9. FOCUS-Style Coarse-To-Fine Sampling

### Method

The 256-frame timeline is split into 16 temporal arms. A few frames estimate
each arm's mean relevance and uncertainty. An upper-confidence score favors
segments that are either promising or insufficiently explored. The selector
then "zooms" into the best arms and samples 64 frames.

### Result

- `93/140`;
- runtime: 4h30m35s.

The exploration logic does not overcome the loss of global narrative coverage.
It is also much slower than direct 64-frame inference.

## 10. Letter-Level Likelihood And Blind Calibration

### Method

Rather than ask Qwen to generate an answer, we inspect its next-token
probability for A, B, C, and D. We compute:

- visual scores with the video evidence;
- blind scores with no images; and
- a calibrated score that adds or subtracts a fixed amount of the blind prior.

The blind branch measures how strongly language and dataset shortcuts prefer an
option before seeing the video.

### Results

- dev140 with blind weight `+0.5`: `90/140`;
- held-out val560 with locked weight `-0.1`: `419/560`;
- visual-only on the same 560 rows: `417/560`;
- original pivot on those rows: `417/560`.

The held-out gain is real but small. A nearby weight of `-0.2` retrospectively
reaches `421/560`, showing the curve is shallow rather than providing a strong
calibration mechanism. The existing verifier reaches `428/560` on the same
rows.

## 11. Candidate Letter-Likelihood Verifier

### Method

Only the 19 pivot/uniform disagreements are scored. For each one, we obtain
letter probabilities from:

- the pivot frame context;
- the uniform frame context; and
- the video-blind context.

The original implementation averaged scores from pivot and uniform contexts,
then applied the blind adjustment.

### Result

- `107/140`.

The averaged resolver chooses only `6/19` disagreements correctly. Pivot scores
alone and uniform scores alone each select `11/19` under the earlier locked
letter-score formulation. The failure is cross-context averaging: log
probabilities from different visual prompts are not directly calibrated to the
same scale.

This motivated the later complete-answer experiment that keeps contexts
separate.

## 12. Lightweight Narrative Gate

### Method

This experiment samples 16 caption anchors, asks Qwen to create concise notes,
retrieves 12 question-relevant narrative entries, and combines them with 64
answer frames.

### Result

- `98/140`;
- 280 Qwen calls;
- runtime 1h59m24s.

Like the earlier timeline prototype, unconstrained captions omit decisive
details and do not reliably stitch events into a usable story. This led to a
more explicit schema-constrained ledger rather than another prose timeline.

## 13. Consolidated Disagreement Dataset

### Purpose

The full pivot and uniform runs disagree on 122 samples. We created one
machine-readable row per disagreement containing:

- operator and question type;
- pivot, uniform, verifier, and gold answers;
- candidate answer lengths;
- proof-pack relevance and temporal-coverage statistics;
- frame-source proportions;
- visual and blind likelihood differences where available; and
- whether pivot, uniform, or neither candidate is correct.

Artifacts:

- `analysis/egolongqa/disagreement_router_2026-07-16/disagreements.jsonl`
- `analysis/egolongqa/disagreement_router_2026-07-16/cv_summary.json`

### Why grouped nested cross-validation is necessary

There are only 108 disagreements where one of the two candidates is correct.
A flexible router can easily memorize such a small dataset. We therefore:

- keep all questions from the same video in one fold;
- train on four groups and test on an unseen group;
- choose regularization inside a second, inner cross-validation loop; and
- report only predictions made on held-out groups.

This is called nested grouped cross-validation. It reduces both video leakage
and hyperparameter-selection optimism.

### Results on 108 resolvable disagreements

- existing verifier candidate choice: `73/108`;
- metadata-only router: `86/108` (`79.63%`);
- metadata plus verifier features: `86/108`;
- metadata plus likelihood features: `88/108` (`81.48%`);
- all feature families: `85/108`.

The likelihood family is available for only 103 of 122 disagreements, so the
`88/108` result is exploratory rather than deployment-ready. Metadata-only
routing is the cleaner signal, but the sample remains too small for a confident
competition submission. This dataset is best used to formulate simple routing
rules and audit fold stability.

## 14. Schema-Constrained Semantic Event Ledger

### Motivation

A list of natural-language captions is difficult to retrieve consistently and
can become verbose. We instead ask Qwen to describe each sampled frame using a
fixed record:

- scene;
- entities;
- actions;
- attributes;
- spatial or interaction relations;
- visible text/OCR; and
- directly visible state changes.

The description prompt is question-independent. This matters because the
ledger is intended to be a reusable video memory, not another answer-biased
retriever.

### Pipeline

1. Uniformly sample 16 frames from each video.
2. Generate one strict JSON event record per frame at a 448px-equivalent cap.
3. Cache each record in scratch using video, frame, model, schema, and pixel
   fingerprints.
4. Embed the question/options and ledger entries with MiniLM.
5. Retrieve the eight most relevant records.
6. Combine their frame indices with 44 high-priority pivot frames and uniform
   coverage, capped at 64 chronological images.
7. Send both selected ledger notes and actual images to Qwen for direct MCQ
   answering.

The original run exposed output truncation when records became too verbose.
The schema now limits list lengths and phrase lengths, and only failed records
retry with a 512-token budget. The successful run generated 320 cached frame
records plus 20 final answers.

### Result

- dev20: `14/20` (`70.00%`);
- parent pivot on the same rows: `15/20`;
- uniform64 on the same rows: `17/20`;
- always-C: `14/20`.

The ledger adds no unique fix over pivot or uniform. It is an auditable semantic
memory, but this first representation does not improve answering. Scaling it to
dev140 is not justified without changing the event representation or retrieval
logic.

## 15. Complete-Answer Semantic Likelihood

### Motivation

Letter probabilities measure confidence in labels such as A or C, which are
sensitive to option position. We therefore implemented teacher-forced scoring
of the complete text of only the two candidate answers.

For each disagreement and each visual context:

1. construct the question plus images;
2. append each candidate answer as an assistant continuation;
3. obtain the log probability of every candidate token from Qwen;
4. average token log probabilities to reduce short-answer bias; and
5. keep pivot-context and uniform-context scores separate.

The final rule changes pivot only when both contexts prefer the same candidate;
otherwise it falls back to pivot.

### Engineering note

vLLM initially OOMed because prompt-logprob evaluation temporarily materialized
full-vocabulary logits for thousands of prompt tokens. We preserved the same
64 frames and 672px resolution but limited prefill chunks to 1,024 tokens and
reserved more GPU memory. This made the run feasible.

### Result

- final dev140: `110/140`;
- pivot baseline: `111/140`;
- uniform baseline: `110/140`;
- relative to pivot: five fixes and six regressions.

On the 19 disagreements:

- pivot-context complete-text scores: `9/19` correct;
- uniform-context complete-text scores: `8/19`;
- consensus/fallback decision: `9/19`;
- retaining the original pivot answer: `10/19`.

Complete-answer scoring is more semantically principled and provides useful
features, but it does not improve the final decision under this setup.

## 16. Concept-Conditioned Object Detection

### Goal

Frame captions can say that a "mug" or "person" is visible, but a structured
object layer can also preserve detector confidence and bounding boxes. The
current detector augmentation extracts short concepts from the question and
options, then asks Grounding DINO to locate those concepts in the same 16 ledger
frames.

The detection output is intended as a small semantic database:

```text
video -> frame/time -> object label -> confidence -> bounding box
```

Selected detection labels are appended to ledger evidence before the final
Qwen answer. Raw detections are cached in scratch and remain auditable.

### Current status

The first detector submission failed because FP32 pixels were passed to BF16
convolution weights. Casting image tensors exposed a second mixed-precision
issue in Grounding DINO's text-enhancement branch. The detector is small enough
to load safely in FP32, so the implementation now uses FP32 end to end.

Job `49201187`, submitted after the first dtype fix, has already failed before
producing detections. It should be rerun once using the current FP32 code. There
is still no detector accuracy result, and the detection cache remains empty.

## Result Summary Since July 13

| Direction | Run | Result | Decision |
| --- | --- | ---: | --- |
| Dynamic segment allocation | QCA | 104/140 | Allocation collapsed to uniform quotas; do not scale |
| QCA router | QCA for GLOBAL, pivot otherwise | 109/140 | Complementary only |
| Multi-event retrieval | FIRST/STATE_CHANGE router | 110/140 | Needs entity-linked event parsing |
| Evidence verification | Support/contradiction gate | 111/140 | Tie; no promotion |
| Candidate verification | Pairwise order swap | 109/140 | Candidate restriction alone is insufficient |
| Candidate verification | Option permutation voting | 109/140 | Option-order sensitivity confirmed |
| Mixed resolution | Q-Frame-style 4/8/32 | 96/140 | Too much evidence compression |
| Adaptive retrieval | AdaQ c256 | 96/140 | Slow and weak |
| Adaptive retrieval | FOCUS c256 | 93/140 | Slow and weak |
| Letter likelihood | Blind-corrected dev | 90/140 | Poor standalone decision rule |
| Held-out likelihood | Locked weight on val560 | 419/560 | Small +2 effect, below verifier |
| Disagreement likelihood | Fused candidate letter scores | 107/140 | Cross-context averaging fails |
| Narrative | Lightweight caption gate | 98/140 | Captions lose decisive details |
| Semantic likelihood | Complete candidate text | 110/140 | Valid feature, no gain |
| Semantic memory | Event ledger | 14/20 | Below pivot and uniform; do not scale yet |
| Object memory | Grounding DINO augmentation | Pending rerun | FP32 stability fix now applied |

## What We Have Learned

### 1. The strongest full result has not changed

The disagreement verifier remains `539/700`. None of the recent dev methods
justifies a full 700-row run based on standalone accuracy.

### 2. More sophisticated retrieval is not automatically better

AdaQ, FOCUS, QCA, and mixed-resolution selection all introduce reasonable
adaptivity, but they lose the broad chronological narrative that uniform64 and
anchored pivot packs preserve.

### 3. Narrative text is still lossy

Both the lightweight narrative gate and semantic event ledger underperform
their visual parents. The problem is not merely caption formatting. Independent
frame records do not naturally encode identity continuity, event completion,
or causal links across time.

### 4. Candidate arbitration remains promising but delicate

The full candidate oracle remains far above the verifier, but every tested
heuristic can overturn correct pivot answers. Position voting, letter
likelihood, complete-text likelihood, and extra reasoning prompts all produce
real fixes and real regressions of similar size.

### 5. Similarity and likelihood are useful features, not final policies

SigLIP relevance, blind log probabilities, candidate margins, and semantic text
scores help characterize uncertainty. None is sufficiently calibrated to act
as a universal selector by itself.

### 6. Small-subset interpretation must include shortcuts

The event ledger's `14/20` equals always-C because dev20 contains 14 C answers.
Without reporting this baseline, the result could be mistakenly described as a
promising 70% proof of concept.

## Remaining Bottlenecks By Pipeline Stage

### Candidate timeline

Short events can still be absent from the 128-frame candidate grid. We lack a
direct selector-recall benchmark stating whether decisive evidence was present.

### Event representation

Current text splitting and single-frame records do not link entities over time.
The missing structure is closer to an event graph:

```text
entity -> action -> object -> place -> timestamp -> next related event
```

### Evidence packing

Selectors trade off detailed evidence against global coverage. Generic adaptive
sampling frequently concentrates too much budget in visually salient regions.

### VLM reasoning

Qwen remains sensitive to option order and can choose incorrectly even when
reasonable evidence is present. Confidence from different visual contexts is
not directly calibrated.

### Routing data

There are only 108 resolvable full disagreements. A router with dozens of
features can overfit, even under grouped cross-validation. Simple and
interpretable rules are preferable until more independent data exists.

## Recommended Next Actions

### Immediate

1. Rerun the FP32 Grounding DINO augmentation once. Treat it as an incremental
   comparison against the same 20-row event ledger, not as a candidate for full
   validation.
2. Inspect detector labels and boxes before interpreting accuracy. Confirm that
   extracted question concepts correspond to visible objects rather than verbs,
   temporal words, or answer-only distractors.
3. Use the disagreement table to test a very small number of interpretable
   abstention rules, reporting every grouped fold rather than only pooled CV.

### If semantic memory is continued

4. Replace independent frame descriptions with entity-linked event records.
   Track recurring objects and people across adjacent ledger frames.
5. Represent temporal relations explicitly, such as `before`, `after`,
   `same_entity`, `state_before`, and `state_after`.
6. Require evidence coverage for every named event in multi-event questions
   before allocating remaining frames to global anchors.

### Competition-focused priorities

7. Preserve the full `539/700` verifier as the current submission baseline.
8. Defer a larger Qwen model or final multi-model ensemble until the last week,
   when the strongest candidate set and routing policy are stable.
9. Treat InternVideo and any HieraMamba output primarily as additional event
   proposals for Qwen proof packs, not as automatic replacements for the answer
   model. InternVideo is already being explored by another teammate;
   HieraMamba is not the immediate bottleneck.

## Code Orientation

### Core generation and grounding

- `data/wearable-ai/starter_kit/model.py`
  - HF and vLLM model interfaces;
  - option-letter scoring;
  - complete-answer token scoring;
  - structured JSON generation and retry handling.
- `run_generate_longqa_proofpack.py`
  - temporal compiler;
  - pivots, eventlets, bridges, QCA, multi-event, AdaQ, FOCUS, and mixed
    resolution selectors.
- `run_generate_longqa_verifier.py`
  - baseline verifier;
  - support/contradiction prompt;
  - pairwise order swap;
  - option permutation;
  - candidate letter likelihood.

### New semantic evidence paths

- `run_generate_longqa_event_ledger.py`
  - question-independent per-frame JSON records;
  - MiniLM ledger retrieval;
  - cached/resumable final answering.
- `run_generate_longqa_event_ledger_detector.py`
  - question-concept extraction;
  - Grounding DINO labels, confidence, and boxes;
  - scratch detection cache;
  - detector teardown before Qwen startup.
- `run_generate_longqa_semantic_likelihood.py`
  - complete candidate-answer scoring;
  - separate pivot and uniform contexts;
  - conservative consensus/fallback decision.

### Analysis and operations

- `scripts/build_longqa_disagreement_dataset.py`
  - 122-row feature table;
  - nested video-grouped logistic routing;
  - feature-family ablations.
- `scripts/run_longqa_disagreement_cv.sh`
  - reproducible CPU analysis command.
- `scripts/run_longqa_semantic_experiments.sh`
  - shared execution, cache, diagnostics, and archival wrapper.
- `RUN_LOG.md`
  - authoritative run status, results, and interpretation.
- `documentation/LONGQA_EVIDENCE_AUDIT_2026-07-14.md`
  - frame-level audit of QCA, multi-event, and verifier behavior.

## Commands Relevant Today

The event ledger and semantic-likelihood runs are complete and do not need to
be rerun. Only the FP32 detector augmentation remains:

```bash
sbatch slurm_longqa_qwen3_event_ledger_detector_dev20.sh
```

The launcher automatically discovers the newest completed event-ledger archive.
It reuses the 320 cached ledger records. Detection outputs are written to
scratch and the final predictions, evaluations, and diagnostics are archived
under `runs/egolongqa/`.

## Current Decision

The project should keep the `539/700` verifier as its competition baseline.
The immediate detector rerun is a bounded object-memory ablation. Beyond that,
the most defensible new research direction is an entity-linked event graph or
semantic database that preserves identity and temporal relationships, not
another isolated-frame ranking policy.

Larger Qwen models, final ensembles, and external temporal backbones remain
valid late-stage options, but they should not replace the current effort to
measure evidence coverage and make candidate arbitration safer.
