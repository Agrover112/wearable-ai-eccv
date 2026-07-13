# EgoLongQA Meeting Brief - 2026-07-13

## Executive Summary

We are solving Wearable AI Workshop Challenge 3 / EgoLongQA: multiple-choice
question answering over long egocentric videos. The hard part is not merely
recognizing objects in individual frames. Most questions refer to events that
occur far apart and ask about order, state changes, first/last occurrences, or
what happened before or after a particular event.

The project has progressed through four main stages:

1. Establish reproducible Qwen baselines and correct frame sampling.
2. Scale temporal coverage and image resolution.
3. Replace generic frame retrieval with question-compiled temporal proof packs.
4. Combine complementary predictions with a disagreement-only verifier.

The strongest completed full-validation result is currently `539/700`
(`77.00%`). It uses a Qwen3-VL verifier only when the temporal-pivot and
uniform64 models disagree. The best single-pass proof-pack models reach
`528-529/700` (`75.43-75.57%`), compared with `371/700` (`53.00%`) for the
initial effective four-frame baseline.

The central finding is that EgoLongQA needs both:

- broad chronological coverage to preserve the video's narrative; and
- question-conditioned evidence around the events named by the question.

Pure similarity retrieval tends to find visually relevant moments but can lose
the temporal relationship between them. Pure uniform sampling preserves the
timeline but can miss short or visually detailed events. The strongest methods
combine these two signals or arbitrate between them.

## Task, Data, And Evaluation

### Input

- Annotations:
  `data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl`
- Videos: `data/wearable-ai/egolongqa/val/`
- Validation samples: 700
- Each row contains a video path, question, four answer options, gold answer,
  and activity/category metadata.

The competition formulation remains direct multiple choice: Qwen receives the
question, all options, and a selected chronological set of images. It must
return one option letter. Keeping question plus options in the prompt preserves
comparability with the starter-kit baseline.

### Iteration subsets

- `configs/egolongqa_dev20_seed20260709.json`: smoke tests.
- `configs/egolongqa_dev140_seed20260709.json`: stratified development subset.

Dev140 is useful for rejecting weak ideas, but it is not precise enough to rank
small improvements. Several methods tied on dev and separated on the full set.
All promoted configurations therefore require a final 700-sample evaluation.

### Outputs

Completed runs are archived under `runs/egolongqa/<run_name>/` and generally
contain:

- `predictions.jsonl`: raw model output, parsed answer, and method metadata.
- `results.json` and `results_summary.json`: official evaluation output.
- `diagnostics.json`: shortcut, category, temporal, and answer-letter metrics.
- `grounding.jsonl` or `proofpack.jsonl`: selected frame evidence.
- `vllm_server.log`, `slurm.out`, and `slurm.err`: execution records.

Frame audits are saved outside the repository in scratch so selected evidence
can be inspected without committing images.

### Evaluation safeguards

The validation labels are strongly imbalanced: option C is correct for
`444/700` samples (`63.43%`). Accuracy is therefore reported together with:

- always-C and shortest-option baselines;
- predicted and gold answer distributions;
- macro and per-letter accuracy;
- non-C accuracy;
- temporal versus non-temporal accuracy;
- category accuracy;
- paired wins/losses against reference runs.

Stable `video_path||question` keys are used when comparing predictions from
different runs. This avoids invalid row-order comparisons and enables fair
rescoring of old full runs on dev140.

## End-To-End Pipeline

The current inference path can be understood as six stages.

### 1. Read and classify the question

The annotation supplies the question and options. For proof-pack methods, a
lightweight temporal compiler classifies the question into one of:

- `AFTER`
- `BEFORE`
- `FIRST`
- `LAST`
- `STATE_CHANGE`
- `GLOBAL` fallback

It also extracts a pivot description, direction, and target concept. For
example, "After paying for coffee, what did I do next?" becomes an `AFTER`
program with `paying for coffee` as the pivot and a forward target search.

Implementation: `run_generate_longqa_proofpack.py`.

### 2. Build a candidate timeline

Candidate frames are sampled uniformly from the complete video. Grounded and
proof-pack experiments commonly use 128 candidates. Uniform baselines skip
semantic selection and directly sample the requested number of final frames.

This stage controls the temporal search resolution. If a short event does not
appear in the candidate set, no later selector can recover it.

### 3. Compute reusable visual features

SigLIP or SigLIP2 encodes candidate images. Text queries derived from the
question/options are embedded and compared with normalized image features.

The expensive image features are cached in scratch and keyed by video,
grounder model/revision, candidate count, and preprocessing settings. Query
changes can then reuse image embeddings rather than decode and encode every
video again.

The current preferred grounder is
`google/siglip2-so400m-patch14-384`. SigLIP2 did not independently improve
dev140 accuracy over SigLIP, but retrieved substantially different frames and
is useful as a complementary evidence source.

### 4. Select and chronologically pack evidence

Different experiment families implement different policies here:

- uniform sampling;
- relevance retrieval plus global anchors;
- local eventlets;
- option-contrastive eventlets;
- temporal-pivot proof packs;
- coverage-filled proof packs;
- operator routing.

All final image lists are deduplicated, capped, and sorted chronologically
before being sent to Qwen. `proofpack.jsonl` records frame index, timestamp,
similarity score, evidence role, and selection fingerprint.

### 5. Answer with Qwen3-VL

The primary model is `Qwen/Qwen3-VL-8B-Instruct`, served through vLLM on one
H100 NVL. The strongest configurations use:

- 64 final frames;
- `QWEN_MAX_PIXELS=451584`, approximately 672x672;
- a 49,152-token context window;
- direct question-plus-options prompting.

At 64 frames and 672px, mean context use is approximately 27,884 tokens, or
56.7% of the window. Context capacity is therefore not yet the limiting factor,
although more frames did not improve accuracy.

### 6. Parse, evaluate, and optionally verify

Raw outputs are retained and parsed robustly into A/B/C/D. Generation and
grounding are resumable, with configuration fingerprints preventing accidental
reuse across incompatible settings.

For the strongest ensemble, agreement rows are copied without another model
call. Only disagreements receive a second Qwen call with a chronological union
of pivot evidence and uniform coverage. The verifier is explicitly told that
neither candidate answer is guaranteed to be correct.

Implementation: `run_generate_longqa_verifier.py`.

## Implemented Experimental Directions

### Correct temporal sampling

The starter path originally treated `--max-frames` as if it selected that many
images. It only imposed a cap; LongQA still sampled four frames by default.
Explicit `--frames-per-interval` support fixed this and created the first true
32-, 64-, and 128-frame baselines.

This was the largest early improvement: Qwen3-VL rose from `371/700` at four
frames to `450/700` at 32 low-resolution frames.

### Resolution and frame-count scaling

We evaluated the tradeoff between temporal coverage and per-frame detail:

- 224px-equivalent: 50,176 pixels, about 64 visual tokens/frame.
- 448px-equivalent: 200,704 pixels, about 256 visual tokens/frame.
- 672px-equivalent: 451,584 pixels, about 576 visual tokens/frame.
- 896px-equivalent: 802,816 pixels, about 1,024 visual tokens/frame.

Moving from 32 frames at 224px to 32 at 448px improved the full score from
`450` to `464`. On dev140, 64 frames at 672px reached `110/140`, while 96
frames at the same resolution fell to `102/140`. Uniform48/560px also
underperformed both endpoints.

Conclusion: useful information is not a smooth function of total visual
tokens. The temporal grid matters, and 64 frames at 672px is the strongest
tested uniform operating point.

### SigLIP relevance grounding

The first grounder sampled 128 candidates, selected the 24 frames most similar
to the question, added up to eight uniform anchors, and sent at most 32 frames
to Qwen. It reached `490/700`, substantially above uniform32/448 (`464/700`).

This established that relevance retrieval can recover short events missed by
coarse uniform sampling. However, it was slower and still weaker than broader
64-frame coverage because isolated relevant frames do not preserve enough
narrative continuity.

### Hybrid relevance plus global anchors

The hybrid selector combined 16 relevance-selected frames with a dense set of
uniform anchors, capped at 64 final images. It reached `498/700`, the best
result at that stage.

The purpose of anchors is not merely fallback coverage. They provide temporal
orientation around retrieved events and allow Qwen to infer transitions and
ordering rather than treating selected images as unrelated snapshots.

### Open QA and video-blind ablations

Open QA hides the options, asks Qwen for a concise answer, embeds that answer
and the four options with MiniLM, and chooses by cosine similarity. It reached
only `74/140`, close to the video-blind baseline at `73/140` and far below
direct-MCQ uniform64.

The video-blind run is nevertheless important: `73/140` (`52.14%`) shows how
much can be obtained from question/option priors without images. New methods
must beat this shortcut baseline on non-C and macro-letter metrics, not merely
raw accuracy.

### Prompt variants and structured evidence

We implemented evidence-first, option-verification, temporal-anchor,
anti-shortcut, combined, and structured-proof prompts. None reliably improved
the direct baseline. The structured temporal prompt used identical frame packs
to an unstructured pivot run but fell from `109/140` to `104/140`.

Conclusion: evidence selection currently matters more than elaborate prompt
formatting. The competition path retains a concise direct-MCQ prompt.

### Eventlets and semantic boundaries

An eventlet is a small local temporal neighborhood around a selected center.
Adjacent-frame embedding distance can additionally identify scene or content
boundaries. Eventlet hybrid reached `104/140`, tying uniform64/448 but not
improving it.

Eventlets are still useful inside stronger policies because a single frame may
show an object without showing the action immediately before or after it.
Generic semantic-boundary frames, however, consumed too much evidence budget
without being conditioned on the temporal question.

### Option-contrastive retrieval

This method built separate text hypotheses for each answer option and reserved
retrieval capacity for every option. The intention was to collect both support
and contradiction evidence rather than over-commit to the question alone.

It fell to `98/140`. Equal option quotas appear to amplify visually distinctive
distractors. Option-conditioned evidence may still help on already-identified
disagreements, but it should not drive first-pass frame selection.

### Temporal-pivot proof packs

The temporal compiler transforms the question into an operator-specific search
program. The selector retrieves pivot eventlets, searches in the permitted
direction for target eventlets, adds bridges between distant events, and keeps
global anchors. The resulting 64-frame proof is chronological.

This was the first grounded selector to beat matched uniform coverage:

- temporal pivot at 448px: `109/140`;
- temporal pivot at 672px: `111/140`;
- original temporal pivot full: `528/700` (`75.43%`).

Its gains are strongest on explicit temporal questions, especially `AFTER`,
`FIRST`, and state-change formulations.

### Coverage-filled pivot

The revised pivot removed generic boundary filler and repeatedly inserted a
frame into the largest uncovered temporal gap. It also increased anchor and
target quotas. This aimed to preserve the temporal proof while reducing blind
regions in the overall story.

It reached `529/700` (`75.57%`), only one answer above the original pivot.
Although it improved some operator categories, its more generic coverage did
not yield a meaningful aggregate gain over the simpler policy.

### Operator routing

Analysis showed that temporal pivots outperform uniform sampling for explicit
operators, while uniform coverage is often stronger on `GLOBAL` questions.
The operator router therefore uses:

- temporal-pivot selection for explicit operators;
- the exact baseline uniform64 grid for `GLOBAL` questions.

The dev run reached `111/140`; the completed full run reaches `529/700`
(`75.57%`). This is only one answer above the original pivot and ties the
coverage-filled pivot. Operator routing is therefore valid, but deterministic
first-pass routing does not capture enough of the candidate complementarity.

### Disagreement-only verification

The original pivot and uniform64/672 disagree on 122 full-validation samples.
Their candidate oracle is `575/700` (`82.14%`), showing substantial usable
complementarity.

The verifier calls Qwen only for these disagreements, using a 64-frame union of
prioritized proof-pack evidence and uniform coverage. It reaches `539/700`
(`77.00%`), the current best completed result:

- fixes 24 pivot errors;
- regresses 13 pivot-correct answers;
- net improvement of 11 over the original pivot;
- gains nine net answers on `GLOBAL` disagreements alone.

The verifier chooses the pivot candidate 76 times, uniform 38 times, and an
alternative answer eight times. It still trails the candidate oracle by 36
answers, so candidate arbitration remains a major improvement opportunity.

## Result Progression

| Stage | Method | Full accuracy | Correct |
| --- | --- | ---: | ---: |
| Initial | Effective 4-frame Qwen3 baseline | 0.5300 | 371/700 |
| Coverage | Uniform32, 224px | 0.6429 | 450/700 |
| Resolution | Uniform32, 448px | 0.6629 | 464/700 |
| Retrieval | SigLIP c128, top24 + anchors | 0.7000 | 490/700 |
| Coverage | Uniform64, 448px | 0.7071 | 495/700 |
| Hybrid | Relevance + dense anchors | 0.7114 | 498/700 |
| Resolution | Uniform64, 672px | 0.7343 | 514/700 |
| Proof pack | Original temporal pivot, 672px | 0.7543 | 528/700 |
| Proof pack | Coverage-filled pivot, 672px | 0.7557 | 529/700 |
| Verification | Pivot/uniform disagreement verifier | **0.7700** | **539/700** |

The improvement from four frames to the current verifier is 168 additional
correct answers, or 24 absolute accuracy points.

## What Did Not Work And Why

- **Naive timeline summarization:** two VLM calls, `465/700`. Natural-language
  summaries compressed away option-discriminating visual details.
- **Uniform128 at 224px:** timed out at 668 predictions and showed only 0.6976
  prefix accuracy. More low-resolution frames were not better.
- **Uniform96 at 672px:** `102/140`, below uniform64. Additional frames can add
  redundancy and make temporal reasoning harder.
- **Open QA plus embedding matching:** `74/140`. The free-form answer is often
  too vague to match reliably to nuanced options.
- **Option-contrastive eventlets:** `98/140`. Distractor-specific retrieval
  receives too much capacity.
- **Structured evidence prompt:** `104/140` using the same frames as a
  `109/140` direct prompt. Added structure did not improve reasoning.
- **Cold candidate grounding:** early c256 runs spent many hours encoding and
  timed out. Reusable image-feature caching is mandatory.

These negative results narrow the search space: improve evidence construction
and arbitration before adding larger prompts, more frames, or more expensive
backbones.

## Pipeline Improvement Map

The following identifies where future work can intervene and what failure each
stage can cause.

### Candidate timeline

**Failure:** the relevant event is absent from all 128 candidates.

**Possible improvements:** adaptive candidate density near motion/content
changes; cheap scene-change proposals; selector-recall audits against manually
identified evidence; variable candidate counts for unusually long videos.

### Question compiler

**Failure:** a multi-event question is reduced to one pivot, or a temporal
question falls into `GLOBAL`.

**Possible improvements:** multi-pivot parsing; explicit subject/object event
arguments; separate handling for comparisons, repeated actions, and two-hop
before/after chains; confidence-aware fallback when parsing is ambiguous.

### Similarity representation

**Failure:** SigLIP similarity responds to visible nouns but not actions,
relations, or subtle state changes.

**Possible improvements:** compare SigLIP2 with Qwen visual/text encoder scores;
use an action-aware video foundation model as a feature provider; combine
query relevance with adjacent-frame content deviation instead of relevance
alone.

### Segment and event allocation

**Failure:** a fixed quota wastes frames in easy regions while under-sampling a
complex or relevant segment.

**Possible improvements:** QCA-style dynamic segment budgets. QCA is
training-free: it allocates frames based on query relevance plus segment
content deviation, then greedily balances relevance and diversity inside each
segment. Our cached SigLIP2 c128 features make a lightweight adaptation
feasible. It is especially promising as the `GLOBAL` router branch.

### Evidence packing

**Failure:** relevant frames are present but do not form a coherent temporal
chain, or too many near-duplicates consume the 64-frame budget.

**Possible improvements:** multi-pivot chains; minimum/maximum per-segment
quotas; relevance-diversity packing; contradiction evidence only for uncertain
options; bridge selection based on event connectivity rather than temporal
distance alone.

### VLM reasoning

**Failure:** the correct evidence is visible, but Qwen chooses the wrong option
or follows answer priors.

**Possible improvements:** targeted support-versus-contradiction verification;
short operator-specific prompts; answer calibration against video-blind logits;
second-pass calls only when candidates disagree or confidence is low.

### Verifier and routing

**Failure:** the verifier overturns a correct specialist prediction or cannot
choose between complementary candidates.

**Possible improvements:** operator-conditioned gating; preserve the pivot on
`FIRST` and `STATE_CHANGE` unless evidence is strong; use candidate confidence,
answer margin, and evidence overlap; train-free calibration on dev without
using full-validation labels; audit the 24 fixes and 13 regressions separately.

### Evaluation loop

**Failure:** dev140 noise promotes a non-improvement, or shortcut-sensitive raw
accuracy hides poor visual grounding.

**Possible improvements:** paired bootstrap/confidence intervals; rolling
accuracy for active runs; selector recall annotations; error buckets by
operator and visual requirement; reserve full validation for candidates with
clear paired gains or strong complementarity.

## Recommended Next Work

### Immediate

1. Audit the verifier's 24 fixes and 13 regressions, grouped by operator and
   whether the decisive evidence is present in each candidate pack.
2. Run the QCA-like selector as a cached-feature experiment, initially only
   for `GLOBAL` questions rather than replacing the temporal pivot.
3. Evaluate multi-event compilation for questions containing two named events or a
   state comparison across time.

### After those results

4. Add support-and-contradiction evidence retrieval only for candidate
   disagreements.
5. Calibrate verifier routing using dev predictions, operator type, candidate
   confidence, frame-set overlap, and answer disagreement.
6. Add repetition-aware eventlets for the small subset of counting/repeated
   action questions.

### Lower priority

- Integrating HieraMamba as a temporal proposal generator.
- Replacing Qwen with another answer model before selector errors are measured.
- Larger uniform frame counts or resolutions.
- Another unconstrained natural-language timeline stage.

HieraMamba and InternVideo may eventually improve temporal/action features, but
the current candidate oracle and verifier gap show that substantial gains are
already available from better selection and arbitration without model training.

## Code Orientation

- `data/wearable-ai/starter_kit/run_evaluation.py`: standard LongQA evaluation.
- `run_generate_longqa.py`: uniform generation and resume path.
- `run_generate_longqa_grounded.py`: SigLIP/SigLIP2 retrieval and feature cache.
- `run_generate_longqa_proofpack.py`: compiler, eventlets, pivots, coverage
  fill, and operator routing.
- `run_generate_longqa_verifier.py`: disagreement-only second pass.
- `run_generate_longqa_openqa.py`: open-answer plus option-similarity ablation.
- `run_generate_longqa_video_blind.py`: language-only shortcut baseline.
- `run_generate_longqa_timeline.py`: deprecated timeline prototype.
- `scripts/analyze_longqa_disagreements.py`: paired candidate analysis.
- `scripts/eval_longqa_diagnostics.py`: shortcut-aware evaluation.
- `scripts/export_grounding_frames.py`: selected-frame audits to scratch.
- `RUN_LOG.md`: authoritative run results, status, and detailed inference.

The shareable scheduler-independent implementation is mirrored in the
`wearable-ai-eccv` repository on branch `feat/egolongqa-qwen-baselines`.

## Current Takeaway

The project is no longer searching for a basic baseline. It has established a
strong operating point and identified a specific remaining problem: selecting
and reasoning over the right temporal evidence for each question type.

The next improvement is unlikely to come from simply increasing visual input.
It should come from better dynamic allocation for global narrative questions,
better multi-event proof construction, and a verifier that captures more of
the `575/700` candidate oracle without overturning correct specialist answers.
