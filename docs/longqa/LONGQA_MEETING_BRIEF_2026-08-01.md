# EgoLongQA Meeting Brief - 2026-08-01

## Purpose Of This Document

This document is for a teammate joining the project today. It explains the
task, the evidence given to the models, the exact decision rule used by our
best system, the experiments completed since the 30 July meeting, and the
remaining work before the 7 August deadline.

The current best pipeline answers `565/700` local validation questions
correctly, or `80.71%`. It is training-free and runs with models that fit on one
H100. Its improvement does not come from one model or one frame-selection
method. It combines three completed answer systems and then corrects some of
their disagreements with an order-balanced Qwen3.5 scoring step.

The most important distinction in this brief is:

- **Frame selection** decides what visual evidence a model receives.
- **Answer generation** produces an answer from one evidence view.
- **Answer arbitration** decides which answer to trust when completed systems
  disagree.

Our largest recent gain came from the third stage.

## Suggested Meeting Agenda

1. Confirm `565/700` as the frozen primary pipeline.
2. Review exactly how the three candidate answers and final arbitration are
   produced.
3. Decide whether endpoint-inclusive sampling for `FIRST` questions should be
   tested on the disjoint 560-question split.
4. Decide whether option-balanced retrieval merits a restricted held-out test.
5. Assign a manual audit of errors where no strong model proposes the correct
   answer.
6. Reserve larger-model or competition-submission compute for the final week.

## The Task In Plain Language

EgoLongQA contains long first-person videos and four-option questions. Many
questions cannot be answered from a single visually relevant image. They may
ask what happened before or after another action, compare the first and last
occurrence of an object, or require identifying the same person or item at
distant points in the recording.

Sending every video frame to a vision-language model is not practical. We
instead construct a chronological set of at most 64 images, resize each image
to at most 451,584 pixels (approximately 672x672), and provide those images
together with the complete question and all four answer options.

The local validation data contains 700 questions. We use:

- `dev140`: a fixed, stratified 140-question development split;
- `val560`: the disjoint 560-question complement; and
- `val700`: the complete local validation set.

Option C is correct for `444/700` questions, so raw accuracy alone can hide
answer-position shortcuts. We also inspect non-C accuracy, predicted answer
distribution, temporal-question accuracy, paired fixes and regressions, and
performance on the disjoint `val560` split.

## Progress From The Original Baseline

| System | Evaluation | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Effective four-frame Qwen3 baseline | 700 | 371 | 53.00% |
| Qwen3 uniform64, 672px | 700 | 514 | 73.43% |
| Qwen3 temporal pivot, 672px | 700 | 528 | 75.43% |
| Qwen3.5 temporal pivot, 672px | 700 | 537 | 76.71% |
| Qwen3.5 uniform64, 672px | 700 | 538 | 76.86% |
| Qwen3 pivot/uniform verifier | 700 | 539 | 77.00% |
| Fixed three-system majority | 700 | 552 | 78.86% |
| Rotation-averaged temporal-pivot arbitration | 700 | **565** | **80.71%** |

The largest early improvement came from fixing temporal coverage: the starter
path effectively sampled four images, while the corrected baseline samples 64.
The later improvement from `552` to `565` comes from deciding among existing
answers more reliably, without changing the three candidate systems.

## Current Best Method: Exact End-To-End Description

The current method has four stages.

## Stage 1: Produce Three Independent Candidate Answers

### Candidate A: Qwen3.5 with temporal-pivot evidence

This branch tries to preserve both a global view and focused evidence around a
question's reference event.

1. Sample 128 candidate images uniformly over the video.
2. Read simple temporal words in the question, such as `after`, `before`,
   `first`, or `last`.
3. For an `AFTER` or `BEFORE` question, identify the reference action. For
   example, in "What did I pick up after paying?", paying is the reference.
4. Use SigLIP2 image-text similarity to find two candidate images that best
   match the reference action.
5. Use similarity to the question and answer options to retrieve eight target
   event centers on the requested part of the timeline.
6. Add nearby candidate images around the retrieved centers so the model sees
   short transitions rather than isolated snapshots.
7. Add bridge images between the reference and target moments.
8. Add 24 evenly distributed images across the full recording.
9. Fill the remaining budget and sort the final 64 images chronologically.
10. Give those images, the question, and all four options to Qwen3.5-9B with
    thinking disabled.

This system scores `537/700`.

### Candidate B: Qwen3.5 with a global uniform view

This branch divides the complete recording into 64 equal temporal intervals
and samples one image from each interval. It does not use the question when
selecting images. The purpose is to preserve broad coverage, including events
that a text-image retriever might overlook or repeated event instances that
look visually similar.

The 64 chronological images, question, and four options are sent to the same
Qwen3.5-9B model and prompt used by Candidate A. This system scores `538/700`.

### Candidate C: Qwen3 disagreement verifier

This branch originates from the earlier Qwen3-VL-8B pipeline. Its underlying
uniform and temporal-pivot systems first produce answers. When they disagree,
Qwen3 receives a chronological 64-image mixture containing high-priority pivot
evidence and broad uniform coverage, then answers from all four options.

This produces a third answer with different model and evidence behavior. It
scores `539/700`. Its purpose in the final system is diversity, not because it
is individually more accurate than Qwen3.5.

## Stage 2: Construct The Fixed Base Decision

For every question, collect the three candidate letters.

- If at least two systems return the same answer, that answer is the majority.
- If all three return different answers, use the first listed candidate,
  Qwen3.5 temporal pivot, as the deterministic tie-break.

This fixed rule scores `552/700`. The three systems disagree on 213 questions,
so these rows contain most of the remaining opportunity.

## Stage 3: Rescore Disagreements With Qwen3.5

Only the 213 disagreement rows enter this stage. Agreement rows keep their
common answer and require no additional model call.

For each disagreement:

1. Reuse the 64 chronological temporal-pivot images from Candidate A.
2. Present the original question and all four answer texts to Qwen3.5.
3. Request the model's next-token log scores for answer labels A, B, C, and D
   instead of generating a free-form response.
4. Repeat the scoring four times. On each repetition, cyclically move the
   answer texts through the displayed A-D positions.
5. Map each displayed-label score back to its original answer text.
6. Average the mapped log scores over the four placements and normalize them
   into comparable probabilities.
7. Consider only answers proposed by at least one of the three candidate
   systems, and select the proposed answer with the highest averaged score.

For example, an answer text originally shown as option B appears once in each
of the A, B, C, and D display positions. A persistent preference for the label
C therefore cannot by itself make that answer win. This directly reduces the
option-order sensitivity measured in earlier experiments.

The final restriction to proposed answers is deliberate. The rescoring model
is used to select among independently produced hypotheses, not to introduce an
unsupported fourth decision during arbitration.

## Stage 4: Restore Full Validation Order

The rotation rule was first evaluated on the disjoint `val560` split. It
improved the fixed majority from `436/560` to `445/560`:

- 50 majority decisions changed;
- 27 changes fixed an error;
- 18 changes replaced a correct answer with a wrong one; and
- 5 changes moved between two wrong answers.

The independently evaluated development result is `120/140`. Merging the two
disjoint outputs in original annotation order gives `565/700` (`80.71%`). The
rule uses no ground-truth labels when making a prediction.

## What The Best Method Does Not Do

The promoted pipeline is easier to understand when its boundaries are clear.

- It does not train or fine-tune Qwen, SigLIP2, or a separate router.
- It does not generate captions, scene graphs, or object ledgers in the main
  path.
- It does not use Grounding DINO crops in the final decision.
- It does not enable long-form thinking in Qwen3.5.
- It does not choose a model from the known category label.
- It does not use validation answers during inference.
- Qwen answer generation always receives all four complete options. Candidate
  restriction occurs only in the final disagreement arbitration stage.

## Why This Combination Works

The global and temporal-pivot views fail differently.

- Uniform sampling preserves the whole sequence but can miss a short event
  between sampled times.
- Temporal pivoting concentrates images around a relevant action but can
  retrieve the wrong occurrence of a repeated event or omit useful global
  context.
- Qwen3 and Qwen3.5 also make different visual and linguistic errors even when
  their total accuracy is similar.

The final scorer benefits from these independent proposals. It only needs to
decide which proposed answer is best supported by the pivot images. Rotating
the options prevents one answer from winning mainly because it occupied a
preferred letter position.

## Remaining Errors In The Primary Pipeline

The final system has 135 errors.

| Error source | Errors | Meaning |
| --- | ---: | --- |
| Correct answer proposed but not selected | 64 | Arbitration failure |
| Correct answer absent from deployed candidates but present in another run | 10 | Candidate coverage failure |
| Correct answer absent from all six strong runs | 61 | Evidence or model-understanding failure |

The deployed three-system oracle is `629/700`: at least one deployed candidate
is correct on 629 questions. The final selector chooses correctly on 565 of
those rows. Better arbitration is therefore still valuable, but almost half of
the remaining errors cannot be fixed by rearranging the same three answers.

Cross-time ordering is the largest weak slice at `153/205` (`74.63%`). These
questions often require distinguishing repeated actions, preserving first/last
order, or carrying an object's identity between distant moments. Shopping is
also difficult because it combines repeated products, fine text, and event
order.

## Work Completed Since The 30 July Meeting

## 1. Rotation-Averaged Arbitration

This is the promoted method described above. It passed the disjoint val560 test
and improved the complete result from `552/700` to `565/700`.

## 2. Matched Shortcut Controls

Qwen3.5 with only question and option text scores `452/700`. Giving it only the
central video image scores `382/700`. The primary system's gain is therefore
not explained by language-only reasoning or a single generic image; broad
visual evidence contributes substantially.

## 3. SigLIP2 Token-Limit Audit

The original combined question-and-options retrieval query can exceed
SigLIP2's 64-position text limit, causing later text to be truncated during
frame retrieval. This does not hide answer options from Qwen, but it can make
the retrieved frame set insensitive to later options.

We implemented a token-safe variant that encodes the target and every option
separately. It scores `109/140`, below the original pivot's `113/140`. The bug
is real and corrected in new retrieval modes, but it does not explain the
remaining performance gap by itself.

## 4. Qwen3-VL Embeddings For Frame Retrieval

We replaced SigLIP2 similarity with the Qwen3-VL-Embedding-2B model while
keeping the same final Qwen3.5 answer stage. This reaches `114/140`. It provides
some complementary frames but is slower and remains below the promoted
`120/140` development result.

## 5. Structured Hypothesis Verification

On candidate disagreements, Qwen3.5 was asked to check visible support,
contradiction, temporal order, identity, and evidence coverage before choosing
between proposed answers. Existing-evidence, fresh-retrieval, and refinement
variants score `119/140`, `118/140`, and `117/140` respectively. Their detailed
reports reveal useful failure modes, but none improves the primary fallback.

## 6. Latest Retrieval And Prompt Ablations

| Experiment | Dev140 result | Main conclusion |
| --- | ---: | --- |
| Option-balanced temporal retrieval | **117/140** | Strongest new retrieval ablation |
| Pairwise answer-text tournament | 116/140 | Overrides too many correct primary answers |
| Option-verification prompt | 116/140 | Better than baseline prompt, below primary |
| Endpoint-inclusive uniform64 | 116/140 | Small sampling change has a measurable effect |
| Operator-adaptive prompt | 114/140 | No consistent gain |
| Corrected temporal pivot v2 | 112/140 | Compound-clause route is harmful |

### Option-balanced retrieval

For each temporal question, SigLIP2 reserves at least one retrieved event
center for every answer option, then uses the remaining target budget for the
question. This prevents one option from dominating the aggregate retrieval
query. It improves the old pivot from `93/113` to `98/113` on temporally
structured questions, but its full dev result remains below the primary.

### Endpoint-inclusive uniform sampling

The legacy uniform formula does not include the final video frame. The corrected
formula includes both the first and last frames while keeping 64 evenly spaced
positions. It improves matched uniform sampling from `112/140` to `116/140`.
On the 18 questions classified as `FIRST`, endpoint-inclusive sampling scores
`18/18`, compared with `16/18` for the primary system.

### Corrected temporal pivot v2

This compiler separates the requested target from the `AFTER` or `BEFORE`
reference event, applies the direction relative to both retrieved reference
occurrences, and detects compound temporal questions. The corrected
single-operator route gains one net answer over the old pivot on 110 rows. The
new compound-clause route loses two, leaving the complete method weaker.

## Current Conclusions

### Established findings

1. Sixty-four chronological images at approximately 672x672 are a strong
   operating point on one H100.
2. Global and question-conditioned views remain complementary.
3. Qwen3.5-9B improves over Qwen3-VL-8B in matched primary runs.
4. Option position materially affects model decisions; cyclic rescoring is the
   only arbitration change that has produced a convincing held-out gain.
5. Separate per-option retrieval is better than allowing a long combined query
   to truncate or average away options.
6. The remaining errors are split between arbitration failures and cases where
   the required answer is absent from all current model outputs.

### Directions to stop expanding

1. Generic prompt variations.
2. Pairwise answer tournaments without a reliable abstention rule.
3. Current compound-clause retrieval.
4. Broad object crops, panels, or detection ledgers.
5. More high-resolution frames without a targeted reason.
6. Another dev140-fitted majority or hand-written routing table.

## Recommended Next Steps

### 1. Test the endpoint rule on held-out data

The only simple new routing observation with no dev regressions is: use the
endpoint-inclusive uniform answer for questions classified as `FIRST`, and
otherwise retain the primary answer. It changes two dev140 answers and fixes
both, producing `122/140`. This must be evaluated on `val560` before promotion.

Only the `FIRST` subset needs new endpoint predictions, which limits compute
and avoids another full uniform run.

### 2. Audit evidence presence on unrecoverable errors

Inspect a balanced sample from the 61 questions missed by all six strong
systems. For each question, record whether decisive evidence is visible in the
uniform and pivot packs. If it is absent, improve evidence construction. If it
is present, the bottleneck is visual interpretation or temporal binding.

### 3. Use option-balanced retrieval as an evidence proposal, not a replacement

Its strong matched improvement on temporal questions justifies retaining its
predictions as a candidate source. It does not justify replacing the promoted
system or adding it to a naive majority, which scores only `119/140`.

### 4. Defer larger models to the final competition stage

A larger Qwen model is most defensible on the fixed disagreement or
candidate-missing subsets, using cached evidence rather than repeating frame
selection. It should not displace the held-out endpoint test or error audit.

## Reproducibility Pointers

- Primary configuration: `configs/egolongqa_primary_pipeline.json`
- Full primary predictions:
  `runs/egolongqa/qwen35_9b_vllm_rotation_avg_pivot_full_2026-07-31/`
- Disjoint val560 rotation evaluation:
  `runs/egolongqa/qwen35_9b_vllm_rotation_avg_pivot_val560_2026-07-31/`
- Primary error analysis:
  `documentation/LONGQA_ROTATION_ERROR_ANALYSIS_2026-07-31.md`
- Complete experiment ledger: `RUN_LOG.md`

The primary configuration explicitly records the three candidate prediction
files, majority policy, rotation-averaged disagreement rule, final evaluation,
and known SigLIP2 retrieval limitation.
