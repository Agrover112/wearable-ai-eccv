# EgoLongQA Meeting Brief - 2026-07-30

## Purpose Of This Document

This document is written for someone joining the project today. It explains
the task, the current pipeline, what each implemented technique actually does,
what was learned from the experiments, what is running now, and which decisions
the team should make in this meeting.

The main conclusion is that individual systems have reached a stable
performance range of approximately 75-77% on all 700 local validation
questions. The resumed Qwen3.5 uniform run reaches `538/700` (`76.86%`).
Combining three complementary completed systems with a fixed majority vote
produces the new best full-validation result: `552/700` (`78.86%`).

There is nevertheless substantial room above 78.86%. Different strong models
often answer different questions correctly. For example, the Qwen3 and
Qwen3.5 full verifiers have the same accuracy but disagree on 43 answers. If an
ideal selector always chose the correct one of these two answers, accuracy
would be `558/700` (`79.71%`). The immediate problem is therefore no longer
simply "find more frames." It is increasingly "decide which of several
plausible answers should be trusted."

## Suggested Meeting Agenda

1. Confirm which experimental branches should now be closed.
2. Review the completed Qwen3.5 results and the still-running uniform run.
3. Decide whether the next priority is a validated disagreement router.
4. Agree on how to avoid further overfitting to the repeatedly inspected
   140-question development set.
5. Decide when, if at all, to spend compute on HieraMamba, a larger VLM, or a
   final ensemble.
6. Assign the remaining work before the 7 August deadline.

## The Task In Plain Language

EgoLongQA contains 700 long first-person videos. Each question has four answer
options. A question may ask:

- what happened before or after a particular action;
- which of several events occurred first or last;
- whether the wearer returned to an earlier place;
- how an object changed over time; or
- what object, person, or location was involved in an event.

The answer model cannot practically inspect every video frame at high
resolution. Our pipeline therefore chooses a limited chronological set of
images, gives those images together with the question and all four options to a
vision-language model, and asks it to return A, B, C, or D.

This separates the task into two problems:

1. **Evidence construction:** which images should the model see?
2. **Answer selection:** when different evidence sets or models disagree, which
   answer should be used?

## Data And Evaluation

### Inputs

- Annotations:
  `data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl`
- Videos: `data/wearable-ai/egolongqa/val/`
- Full local validation set: 700 questions.
- Main development set:
  `configs/egolongqa_dev140_seed20260709.json`.
- Small proof-of-concept set:
  `configs/egolongqa_dev20_seed20260709.json`.

### Important evaluation warning

The answer distribution is imbalanced. Option C is correct for `444/700`
questions (`63.43%`). We therefore inspect more than raw accuracy:

- accuracy when the gold answer is not C;
- predicted answer distribution;
- temporal-question accuracy;
- category accuracy;
- fixes and regressions against a reference model; and
- oracle accuracy, which measures how many questions at least one candidate
  system answers correctly.

The 140-question set is useful for rejecting poor ideas, but it has now been
examined many times. Small gains on it should not be treated as reliable
without a locked comparison or the full 700-question run.

## Current End-To-End System

The strongest established pipeline can be summarized as follows:

1. Sample candidate images across the complete video.
2. Build either a broad uniform view or a question-conditioned temporal view.
3. Sort the final images chronologically.
4. Resize at most 64 images to a maximum of 451,584 pixels each, approximately
   672x672.
5. Give the images, question, and four options to Qwen.
6. Parse the returned option and save complete metadata.
7. When selected first-pass systems disagree, optionally make a second model
   call using combined evidence.

The main answer models are:

- `Qwen/Qwen3-VL-8B-Instruct`;
- `Qwen/Qwen3.5-9B`, with thinking disabled for controlled comparison.

Both are served with vLLM on one H100. At 64 frames and 672px, prompts contain
approximately 28,000 tokens, or about 57% of the configured 49,152-token
window.

## What We Actually Implemented

## 1. Uniform Temporal Coverage

### What it does

Uniform sampling divides the complete video duration into equal intervals and
takes one frame from each interval. It does not use the question to choose
frames.

The strongest uniform setting uses 64 frames at approximately 672x672. Its
purpose is to preserve the broad sequence of the recording: where the wearer
went, which activities happened, and how one part of the video followed
another.

### Why it matters

The original starter path effectively used only four frames because the frame
cap and the number of sampled frames were separate settings. Correcting this
was the largest early improvement.

### Main results

- Effective four-frame Qwen3 baseline: `371/700` (`53.00%`).
- Uniform 32 frames at low resolution: `450/700` (`64.29%`).
- Uniform 64 frames at 448px: `495/700` (`70.71%`).
- Uniform 64 frames at 672px: `514/700` (`73.43%`).
- Uniform 96 frames at 672px on dev140: `102/140`, worse than 64 frames.

More frames are not automatically better. At a fixed context budget, excessive
frames can add redundant or distracting evidence.

## 2. SigLIP And SigLIP2 Frame Retrieval

### What these models do

SigLIP and SigLIP2 are image-text matching models. They do not answer the
question. They convert an image and a text query into numerical vectors and
assign a similarity score. A higher score means the image and text appear more
semantically related.

### How we used them

The first retrieval baseline sampled 128 images uniformly across the video,
scored each image against the question and options, kept the 24 highest-scoring
images, and added up to eight uniformly distributed images for coverage.

Image embeddings are cached in scratch. Once a video has been encoded, a new
text query can reuse those image vectors rather than running the image encoder
again.

### What happened

The first SigLIP retrieval run reached `490/700`, better than the matched
32-frame uniform result of `464/700`. This showed that semantic retrieval can
find short relevant events.

However, selecting isolated high-similarity images loses parts of the timeline.
The method can find a coffee cup, counter, or shop without preserving whether
payment occurred before or after another action. It remained below uniform
64-frame coverage.

## 3. Temporal Pivoting

### The problem it addresses

A question such as "What did I do after paying?" contains two parts:

- a reference event: paying;
- a target event that must occur after it.

Plain similarity retrieval may find both actions but does not enforce their
order.

### How it is implemented

1. Sample 128 candidate frames uniformly across the complete video.
2. Read the question with a small rule-based parser.
3. Classify it as `AFTER`, `BEFORE`, `FIRST`, `LAST`, `STATE_CHANGE`, or a
   general question.
4. Extract the reference event when one exists. In the example above, the
   reference event is "paying."
5. Use SigLIP2 to score candidate frames for the reference event and select two
   well-separated reference centers.
6. Restrict the target search according to the question. `AFTER` searches
   candidate frames later than the main reference center; `BEFORE` searches
   earlier frames. `FIRST` and `LAST` favor the earliest or latest relevant
   occurrences.
7. Select eight well-separated target centers using similarity to the complete
   question and options.
8. Add the immediate neighboring candidate on either side of each selected
   center. These neighbors are approximately one candidate step away, not
   every original video frame.
9. Add up to eight bridge frames between the main reference and the strongest
   targets.
10. Add 24 uniformly distributed anchor frames across the whole video.
11. Fill any remaining budget and sort the final 64 images chronologically.

The final pack therefore contains a global timeline plus concentrated evidence
around the relevant event and the event asked about.

### Results

- Qwen3 temporal pivot at 672px on dev140: `111/140`.
- Qwen3 temporal pivot on all 700: `528/700` (`75.43%`).
- Qwen3.5 on the exact cached pivot frames: `537/700` (`76.71%`).

Temporal pivoting is the strongest single evidence-selection policy tested so
far. Its advantage over uniform sampling is modest but consistent enough to
retain.

## 4. Adaptive Frame Allocation

We implemented several methods that try to spend the frame budget unevenly.

### QCA-style allocation

The video is split into 16 chronological segments. Each segment is scored by:

- how relevant its frames are to the question and options; and
- how much visual content changes inside the segment.

The intended behavior is to allocate more frames to relevant, changing
segments. In our run, normalization and rounding produced four frames in every
segment for every question. The final behavior was therefore close to
stratified uniform sampling.

Result: `104/140` standalone and `109/140` when used only for selected question
types.

### AdaQ

AdaQ converts frame-relevance scores into probabilities. If a few frames have
much higher relevance, it concentrates the sampling distribution; when scores
are similar, it spreads probability more broadly. It then samples 64 frames
from 256 candidates.

Result: `96/140`. The probability distribution emphasized relevance but did
not preserve enough chronological context.

### FOCUS-style allocation

The candidate timeline is divided into temporal regions. A small number of
frames estimates each region's average relevance and uncertainty. Regions with
high estimated value receive more samples, after which frames are drawn within
those regions.

Result: `93/140`. It was expensive and weaker than uniform or temporal pivoting.

These experiments indicate that allocating more frames to apparently relevant
regions is not sufficient. The selected evidence must also preserve transitions
and event order.

## 5. Explicit Narrative Representations

### Timeline summary

Qwen first viewed sampled frames and wrote a chronological text summary. A
second call received the summary plus answer frames.

Result: `465/700` (`66.43%`). Generated summaries omitted details and sometimes
introduced uncertain statements that then influenced the answer.

### Question-independent event ledger

We divided the video evidence into small groups and asked Qwen to produce
structured records describing visible actions, objects, places, and changes.
The records were merged into a chronological ledger and supplied during final
answering.

The corrected dev20 prototype reached `14/20`. It did not beat the relevant
uniform or pivot references. The ledger idea is sensible, but reliable
captioning and entity tracking are themselves difficult problems.

## 6. Temporal Chain Of Thought

### What it means in this project

Temporal Chain of Thought was implemented as a frame-selection process, not
merely a request for a longer written explanation.

For single-step selection, Qwen inspected low-resolution candidate frames,
returned the identifiers of frames it considered relevant, and then answered
using those selected images at high resolution.

For dynamic-segment selection, 256 candidates were divided into four
chronological sections. Qwen selected frames independently inside each section,
the selections were merged, and the final images were given to the answer
stage. Another version added 16 uniform frames after selection.

### Results

- Single-step selection: `82/140`.
- Dynamic four-segment selection: `95/140`.
- Dynamic selection plus uniform coverage: partial `28/40`, still below the
  matched uniform prefix.
- Asking for an answer-stage temporal rationale: `110/140`, exactly matching
  the ordinary uniform result.

Qwen-based selection often chose too few frames or many ambiguous frames.
Longer reasoning also created output-format failures before strict retry was
added. These implementations did not improve accuracy and should not currently
consume full-run compute.

## 7. Grounding DINO Object Localization

### What Grounding DINO does

Grounding DINO is an open-vocabulary object detector. Given an image and text
concepts such as "cup," "bag," or "receipt," it returns bounding boxes around
matching visible regions. Unlike SigLIP2, which gives one relevance score for
the whole image, Grounding DINO identifies where an object appears inside the
image.

### How concepts were supplied

Object-like words were extracted from the question and answer options. Common
action words and uninformative terms were removed. Grounding DINO then searched
selected temporal-pivot frames for those concepts. Detection results were
cached and reused by several visual-input variants.

### Implemented variants

**Labels over full frames:** bounding boxes and object names were drawn on
selected images. This produced dense overlays and visual clutter.

Result: `96/140`.

**Full-frame and crop pairs:** for selected detail moments, the model received
the original image and an additional crop around the strongest detected object.
The final input contained 56 temporal-context images and up to eight object
crops.

Result: `109/140`. It did not beat temporal pivoting, but it supplied five
correct answers missed by both pivot and uniform.

**Evidence panels:** detected crops were arranged beside the source image in a
compact visual panel.

Result: `107/140`.

**Object ledger:** detections across time were converted into text describing
which concepts appeared at which timestamps.

Result: `102/140`.

### Main lesson

Object crops can help a small number of questions that require fine detail.
Adding labels, panels, or long detection lists broadly makes the input noisier.
Object localization is therefore a possible secondary evidence view, not a
replacement for the temporal timeline.

## 8. Uncertainty Guidance

### What uncertainty means here

For each candidate image or image group, Qwen receives that evidence together
with the multiple-choice question. Instead of immediately generating an answer,
we inspect Qwen's probability distribution over possible next tokens.

If most probability is concentrated on one answer letter, the distribution has
low Shannon entropy. We treat this as high confidence. If probability is spread
across several letters, entropy is high and the evidence is considered
ambiguous.

This is different from SigLIP2:

- SigLIP2 asks whether the image and text are semantically similar.
- Uncertainty guidance asks whether Qwen itself can make a confident answer
  from the supplied visual evidence.

Low entropy does not prove that the answer is correct. A model can be
confidently wrong. The experiments test whether this signal is still useful for
frame selection or routing.

### How scoring was implemented

1. Resize candidate evidence to 50,176 pixels, approximately 224x224, to keep
   scoring calls inexpensive.
2. Ask Qwen to answer the original question from only that candidate evidence.
3. Request the top 100 next-token probabilities.
4. Compute a lower bound on Shannon entropy from that distribution.
5. Rank evidence by low entropy.
6. Cache every score so later experiments do not repeat the calls.
7. Use the selected evidence at 451,584 pixels for the final answer.

The top-100 distributions captured effectively all probability mass in the
pilot runs, so truncation is not a meaningful source of error.

### Implemented uncertainty variants

**Segment-balanced uncertainty:** sample 64 candidates, divide them into 16
temporal segments, and keep the two lowest-entropy frames from each segment.
Uniform frames fill the remaining 64-frame budget.

Result: `15/20`.

**Uncertainty temporal pivot:** use Qwen probabilities rather than SigLIP2 to
score whether each candidate contains the reference event and whether each
candidate supports a confident final answer. The same directional pivot logic
then constructs the 64-frame pack.

Result: `16/20`, one answer below uniform on the same temporal dev20 set.

**Short-window uncertainty:** score overlapping groups of nine neighboring
candidate frames, select confident windows across the timeline, keep frames
around each selected window center, and add 16 uniform frames.

Result: `15/20`.

**Uncertainty plus dynamic temporal selection:** first use entropy to construct
a 64-frame shortlist, then ask Qwen to select evidence independently within
four chronological segments, and finally add uniform coverage.

Result: `15/20`.

**Uncertainty-ranked object crops:** Grounding DINO creates candidate
full-frame/crop pairs. Qwen entropy ranks 24 such candidates, after which the
top eight crops are added to 56 temporal-pivot frames.

The first 20 examples scored `16/20`, but the complete dev140 run fell to
`106/140`. The encouraging prefix did not generalize.

**Disagreement routing:** obtain answers from temporal pivot, uniform sampling,
and object crops. When all agree, keep the common answer. When they disagree,
score each complete 64-frame view with the same question and choose the answer
associated with the lowest-entropy view.

Result: `115/140` (`82.14%`), the best completed model-based dev140 result.

### Main lesson

Entropy is weak as a universal frame selector but useful as a disagreement
signal. The successful use is not "choose all low-entropy frames." It is
"when several completed systems disagree, prefer the answer supported by the
view on which Qwen is least uncertain."

## 9. Object And Uncertainty Combination

The Grounding DINO and uncertainty experiments were also combined directly:

1. Generate object crops from temporal-pivot frames.
2. Pair each crop with its original image.
3. Ask Qwen how confidently that pair supports an answer.
4. Keep the eight lowest-entropy pairs.
5. Combine them with 56 temporal-context frames.

This reached only `106/140`. It is a useful negative result: confidence ranking
cannot rescue a generally noisy object-crop pool. The object branch should be
retired from the main queue unless a future error audit identifies a tightly
defined object-detail subset.

## 10. Qwen Thinking Models

We tested the dedicated Qwen3-VL Thinking checkpoint with 64 uniform images.
Initial runs often spent the entire reasoning budget without producing a final
answer marker. The final implementation used:

- a 2,048-token reasoning budget;
- explicit reasoning delimiters;
- a reserved final-answer budget; and
- an answer-only retry when needed.

The protocol-valid result was `94/140` (`67.14%`), far below the ordinary
Instruct model. Longer hidden reasoning is therefore not automatically helpful
for this task.

## 11. Qwen3.5 Model Scaling

### Controlled setup

Qwen3.5-9B was run with thinking disabled so that frame evidence, prompt, and
answer format matched Qwen3-VL-8B. A Triton implementation is used for the
model's GDN prefill operation because the alternative attempted to compile an
incompatible CUDA kernel.

### Completed results

- Qwen3.5 uniform dev140: `112/140`.
- Qwen3.5 temporal pivot dev140: `113/140`.
- Qwen3.5 temporal pivot full: `537/700` (`76.71%`).
- Qwen3.5 verifier on original pivot/uniform disagreements, dev140:
  `113/140`.
- Qwen3.5 verifier on all 700: `539/700` (`77.00%`).

The Qwen3 and Qwen3.5 full verifiers both score `539/700`, but disagree on 43
answers. Each fixes nineteen errors made by the other. This is strong evidence
for complementary model behavior, but not evidence that either model should
always replace the other.

## 12. Verifiers And Ensembles

### Original disagreement verifier

The Qwen3 temporal-pivot and uniform64 systems disagree on 122 of 700
questions. On agreement rows, the common answer is copied. On disagreement
rows, Qwen receives a 64-frame chronological mixture of high-priority pivot
evidence and uniform coverage, plus all four answer options.

Qwen3 result: `539/700` (`77.00%`).

Qwen3.5 result using the same candidate systems and evidence construction:
`539/700` (`77.00%`).

### Candidate-pair verifier

A later experiment restricted the verifier to exactly two proposed answer
texts. This prevents it from inventing a third answer. Applied to disagreements
between the Qwen3.5 pivot and the entropy router, it reached `115/140`, tying
the router rather than improving it.

### Majority voting

On dev140, a fixed majority over the entropy router, Qwen3.5 pivot, and Qwen3.5
uniform reaches `118/140` (`84.29%`). This is encouraging but was observed
after repeated dev140 experimentation.

Majority voting is not always reliable. Two systems can make the same correlated
mistake and outvote a complementary correct answer. The recent full-verifier
results show that a feature-based disagreement router is more promising than
adding more votes blindly.

## Current Status At Meeting Time

Status below was refreshed at **2026-07-30 19:30 CEST**.

| Experiment | Slurm job | Status | Current evidence |
| --- | ---: | --- | --- |
| Qwen3.5 pivot/uniform verifier, dev140 | `49306485` | Completed | `113/140`; 19 verifier calls; archived |
| Qwen3.5 pivot/uniform verifier, full700 | `49306592` | Completed | `539/700`; 122 verifier calls; archived |
| Qwen3.5 uniform64/672, full700 | `49312012` | Completed resume | Resumed from 686 rows and finished at `538/700` (`76.86%`) |
| Fixed Qwen3.5 pivot/uniform + Qwen3 verifier majority | offline | Completed | New best result: `552/700` (`78.86%`) |

The first resume attempt, job `49311311`, failed before inference because
scratch storage was still unavailable. The second attempt completed the final
14 rows with images. The original interrupted attempt remains preserved
separately, and the code now aborts if video extraction returns no images.

The fixed majority also scores `436/560` (`77.86%`) outside dev140, improving
over each of its three components on that less-inspected portion. The three
inputs disagree on 213 questions; at least one is correct on 190 of them, while
the current vote is correct on 113. Choosing among existing answers is
therefore the clearest remaining opportunity.

## Results Worth Showing In The Meeting

| System | Evaluation set | Accuracy |
| --- | ---: | ---: |
| Effective four-frame Qwen3 baseline | 700 | `371/700` (`53.00%`) |
| Uniform64 at 672px, Qwen3 | 700 | `514/700` (`73.43%`) |
| Temporal pivot, Qwen3 | 700 | `528/700` (`75.43%`) |
| Temporal pivot, Qwen3.5 | 700 | `537/700` (`76.71%`) |
| Uniform64 at 672px, Qwen3.5 | 700 | `538/700` (`76.86%`) |
| Pivot/uniform verifier, Qwen3 | 700 | `539/700` (`77.00%`) |
| Pivot/uniform verifier, Qwen3.5 | 700 | `539/700` (`77.00%`) |
| Fixed Qwen3.5 pivot/uniform + Qwen3 verifier majority | 700 | **`552/700` (`78.86%`)** |
| Entropy disagreement router | dev140 | `115/140` (`82.14%`) |
| Fixed cross-model majority | dev140 | `118/140` (`84.29%`) |

## What The Results Mean

### Established positive findings

1. Correct temporal coverage is essential. Moving from four to 32 and then 64
   frames produced the largest gains.
2. 64 frames at approximately 672x672 is the strongest uniform operating point.
3. Temporal pivoting gives a modest but repeatable improvement over uniform
   sampling by preserving the relationship between a reference event and the
   requested event.
4. Model scaling from Qwen3-VL-8B to Qwen3.5-9B improves the temporal-pivot
   full run by nine answers.
5. Different models and evidence views contain many complementary correct
   answers.
6. Uncertainty is more useful for routing completed answers than for selecting
   every frame.

### Directions that should be deprioritized

1. More frames at the same high resolution.
2. Naive image-text top-k retrieval without timeline coverage.
3. Generic timeline summaries or long textual ledgers.
4. AdaQ, FOCUS, and current QCA allocation.
5. Current Temporal Chain of Thought implementations.
6. Broad Grounding DINO labels, panels, or object ledgers.
7. Uncertainty-ranked object crops.
8. Dedicated Qwen Thinking inference.
9. Another prompt-only variation.

## Decisions To Make Today

### 1. Lock the next routing evaluation

The strongest next direction is a consolidated disagreement table containing,
for each question:

- answers from Qwen3 and Qwen3.5;
- uniform, pivot, and verifier answers;
- temporal operator and question category;
- answer probabilities and entropy margins;
- whether a video-blind model was confident;
- retrieval statistics; and
- correctness, used only during cross-validation.

A small router can then be evaluated with grouped or nested cross-validation.
The routing rule must be learned on training folds and scored on unseen folds.
This is more defensible than manually selecting a rule after inspecting all
answers.

### 2. Define a promotion threshold

A new method should not receive a full expensive run because it improves
dev140 by one answer. Suggested promotion criteria are:

- at least three net fixes on dev140;
- improvement on non-C and temporal questions;
- a meaningful oracle contribution;
- no dependence on individual gold labels; and
- stable behavior under grouped cross-validation.

### 3. Decide the role of HieraMamba

HieraMamba should be treated as a source of proposed temporal segments, not as
an answer model. Its top intervals could replace or augment SigLIP2 target
centers inside the existing 64-frame temporal-pivot pack.

Given the deadline, it should be attempted only if:

- a usable checkpoint and inference pipeline are already available;
- top intervals can be cached quickly; and
- a dev20/dev140 comparison can be completed without disrupting the current
  ensemble work.

### 4. Reserve larger-model experiments for the final stage

Qwen3.5 shows that model scaling can provide a small improvement. A still
larger Qwen or Llama model may help, but it is expensive and should reuse fixed
cached evidence. It should be attempted only after the routing and ensemble
inputs are finalized, so model scaling is not confounded with another frame
selection change.

## Proposed Work Before The Next Meeting

1. Treat the completed fixed majority as the new deployment baseline.
2. Evaluate a candidate-constrained arbiter first on the 24 cases where the
   three majority inputs return three different answers.
3. Enrich the consolidated disagreement dataset with answer entropy,
   likelihood margins, verifier outputs, blind confidence, and retrieval
   statistics.
4. Evaluate a calibrated router with grouped or nested cross-validation.
5. Test uncertainty-guided dynamic TCoT only on unresolved disagreements.
6. Test crop admission based on uncertainty reduction only for object-detail
   disagreements.
7. Prepare final full-validation and competition-test launchers using cached
   evidence and resumable predictions.
8. Keep HieraMamba as a bounded parallel investigation, not the main path.

## Files To Know

- Full run history: `RUN_LOG.md`
- Previous meeting brief:
  `documentation/LONGQA_MEETING_BRIEF_2026-07-17.md`
- Temporal evidence audit:
  `documentation/LONGQA_EVIDENCE_AUDIT_2026-07-14.md`
- Temporal proof-pack implementation:
  `data/wearable-ai/starter_kit/run_generate_longqa_proofpack.py`
- Uncertainty implementation:
  `data/wearable-ai/starter_kit/run_generate_longqa_uncertainty.py`
- Grounding DINO object augmentation:
  `data/wearable-ai/starter_kit/run_generate_longqa_object_hints.py`
- Temporal Chain of Thought:
  `data/wearable-ai/starter_kit/run_generate_longqa_tcot.py`
- Disagreement verifier:
  `data/wearable-ai/starter_kit/run_generate_longqa_verifier.py`
- Full-majority builder: `scripts/build_longqa_full_majority.sh`
- Model-disagreement analysis:
  `documentation/LONGQA_MODEL_DISAGREEMENT_ANALYSIS_2026-07-30.md`
