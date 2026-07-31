# Temporal pivot and Qwen retrieval handoff

Date: 2026-07-31
Branch observed: `exp/molmo2-er-and-2b`
Scope: research direction and experiment plan; no retrieval implementation was
made as part of this handoff

## Purpose

This document transfers the conclusions of the temporal-pivot follow-up
discussion to the main thread. It should be read alongside
`docs/longqa/TEMPORAL_PIVOT_REVIEW_2026-07-31.md`, which contains the full
prediction audit, legacy ablation results, and current implementation findings.

The central recommendation is to stop adding heuristics around weak SigLIP
scores and first establish a clean, quality-oriented Qwen-only retrieval
ceiling. Start with exhaustive Qwen3-VL-Reranker-8B scoring over a modest set of
ordered video clips. Only after measuring that result should the experiment be
made cheaper with a 2B reranker, cached embeddings, fewer candidates, or a
SigLIP shortlist.

## Decisions already made

1. Preserve the existing temporal-pivot result as a disagreement-verifier
   candidate. It is valuable for diversity even though it is not the strongest
   standalone selector.
2. Do not keep elaborating the current pivot/target-pair heuristic before
   testing a cleaner retrieval architecture. The proposed pair logic is
   defensible in isolation but makes the overall selector increasingly hard to
   reason about.
3. Evaluate a Qwen-only retriever before combining Qwen with SigLIP. The goal is
   to learn the attainable retrieval quality before optimizing compute.
4. Prioritize a naive, expensive, quality-oriented configuration first. This is
   a practical quality ceiling, not a mathematical upper bound.
5. Run a two-sample technical smoke test, screen configurations on dev20, and
   use dev140 for decisions.
6. Compare independent-frame documents with ordered-clip documents. The frame
   condition isolates the embedding-space change; the clip condition measures
   the value of local temporal context.
7. Defer the generative Qwen selector-agent idea. Record it for later, but do
   not include it in the first Qwen retrieval experiment.

## Confirmed evidence

### Temporal pivot is useful mainly as a complementary view

The supplied full-700 predictions show:

| System | Correct |
| --- | ---: |
| Qwen3 uniform | 514/700 |
| Qwen3 temporal pivot | 528/700 |
| Qwen3 pivot/uniform verifier | 539/700 |
| Qwen3.5 uniform | 538/700 |
| Qwen3.5 temporal pivot | 537/700 |
| Qwen3.5 pivot/uniform verifier | 539/700 |

The original Qwen3 pivot/uniform pair has a 575/700 candidate oracle; the
Qwen3.5 pair has a 580/700 candidate oracle. Improving standalone retrieval
must therefore be tracked together with disagreement diversity. A selector
that becomes more uniform-like may improve by itself while weakening the
verifier.

Leading `After ...` questions are the cleanest pivot success case. Embedded
`after`, `before`, and `GLOBAL` cases are mixed or negative. Consult the
existing review for the complete syntax partition.

### The current SigLIP target query is structurally lossy

The legacy target text is:

```text
Find video evidence needed to answer this multiple-choice question.
Question: ...
Options:
A. ...
B. ...
C. ...
D. ...
```

For the current SigLIP2 checkpoint, the standard text preprocessing window is
64 tokens. In the audited 700 questions, target-query token lengths have
min/median/p95/max of 50/104/200/363, and 642/700 exceed the window. After
processor truncation, all four option markers remain visible in only 73/700
rows. The retrieval input is consequently order-dependent and frequently
means “question plus the first one or two options,” not the complete MCQ.

This limitation applies to the standard SigLIP1 and fixed-resolution SigLIP2
interfaces: SigLIP2 improves visual representation quality, but it does not
turn a long MCQ into temporal reasoning. See the official
[SigLIP documentation](https://huggingface.co/docs/transformers/model_doc/siglip)
and [SigLIP2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2).

### The pixels are not obviously misformatted

The implementation:

- converts decoded OpenCV frames from BGR to RGB;
- wraps them as PIL images;
- sends them through the checkpoint's `AutoProcessor`; and
- calls `get_image_features` followed by normalization.

There is no obvious channel-order, normalization, or raw-tensor bug. The more
important mismatch is semantic:

- SigLIP receives independent still frames rather than an ordered sequence;
- it receives a long instruction and mutually exclusive options instead of a
  short visual description;
- one global dot product must flatten all alternatives into one score; and
- temporal words such as “after” and “before” do not create ordering
  constraints inside the image encoder.

The fixed image resolution may also weaken small-object, OCR, and hand-action
evidence in egocentric footage, but that is secondary to the text/task
mismatch.

### Simple truncation mitigations did not improve the screen

On the fixed first 20 rows of dev140, historical pivot scores 14/20 and uniform
scores 15/20. The following valid screens did not reach the predeclared 16/20
promotion threshold:

| Variant | Correct |
| --- | ---: |
| Exact legacy control | 14/20 |
| Question only | 13/20 |
| Question plus independent option mean | 12/20 |
| Compact temporal target plus options | 13/20 |
| Compact target plus options with per-pivot masking | 12/20 |
| Compact target plus independent option mean | 12/20 |
| Legacy query with pivot/target overlap excluded | 14/20 |

The truncation is a real defect, but removing text or averaging option scores
also removes useful semantics. Do not assume that a technically cleaner
SigLIP query will automatically improve answer accuracy.

## What the legacy temporal-pivot flow currently does

At a high level:

1. Uniformly sample 128 candidate frames.
2. Compute one normalized SigLIP2 embedding for every independent frame.
3. Compile the question into an operator such as `AFTER`, `BEFORE`, `FIRST`,
   `LAST`, `STATE_CHANGE`, or `GLOBAL`.
4. Score the long question-plus-options target query against all frame
   embeddings.
5. For non-global cases, separately score a shorter pivot query.
6. Select two pivot centers and add neighboring eventlet frames.
7. For directional questions, use the primary pivot to mask which target
   frames are legal. The second pivot is rendered but does not control the
   legacy directional mask.
8. Select target eventlets, bridges, uniform anchors, and boundary-based fill
   until the 64-frame budget is reached.
9. Chronologically sort those frames and pass them to the main Qwen answerer.

Operators are therefore treated differently already, but only partially. The
same truncated target similarity dominates most routes, and temporal relations
are approximated with masks and quotas rather than represented in the
retrieval document.

An explicit pivot-target pair proposal was discussed: retain multiple plausible
pivots, independently search the legal side of each, and score valid temporal
pairs. This is more internally consistent than allowing only the first pivot to
control direction. It was not chosen as the next experiment because it adds
more allocation, deduplication, and bridging rules around the same weak
single-frame scores.

## Why Qwen embedding and reranking models are relevant

The official Qwen3-VL retrieval family contains 2B and 8B embedding models and
2B and 8B rerankers. It accepts text, images, video, and mixed multimodal
documents and supports much longer text than the SigLIP setup. See the
[official Qwen3-VL-Embedding repository](https://github.com/QwenLM/Qwen3-VL-Embedding).

The two model types serve different purposes:

- **Embedding model:** independently maps the query and document to vectors.
  Clip embeddings can be cached and reused for all questions about a video.
  Retrieval is then a cheap similarity search.
- **Reranker:** jointly processes a query-document pair and produces a
  relevance score. Document work cannot be fully cached independently, but the
  model can express richer interactions between the complete MCQ and the
  visual clip.

The reranker is the preferred quality-ceiling experiment because the question
contains alternatives, relations, and fine-grained evidence requirements. The
embedding model is the preferred efficiency path after the ceiling is known.

Official MMEB-v2 results are not a clean apples-to-apples comparison with the
specific SigLIP checkpoint or EgoLongQA. They do, however, show that Qwen's
retrieval models are designed and evaluated for multimodal retrieval, including
video and visual-document tasks. Reported overall/video scores include:

| Model | Overall | Video |
| --- | ---: | ---: |
| Qwen3-VL-Embedding-2B | 73.4 | 53.6 |
| Qwen3-VL-Reranker-2B | 75.2 | 53.2 |
| Qwen3-VL-Embedding-8B | 77.8 | not reproduced here |
| Qwen3-VL-Reranker-8B | 79.2 | 61.0 |

These benchmark values motivate testing; they do not predict EgoLongQA
accuracy.

## Recommended clean architecture

Begin without explicit pivot parsing:

```text
video
  -> fixed temporal windows
  -> ordered low-resolution frames per window
  -> Qwen reranker scores each window against the complete MCQ
  -> temporal non-maximum suppression / diversity selection
  -> retrieved high-resolution frames plus uniform insurance frames
  -> chronological ordering
  -> main Qwen answerer
```

Use an ordered list of extracted images as the video document. Physical MP4
clip files do not need to be written. The retrieval stage should save selected
timestamps and scores to JSONL so the answer stage can be rerun without paying
the retrieval cost again.

A suitable reranking instruction is conceptually:

```text
Judge how useful this video clip is for answering the complete multiple-choice
question. Score visible supporting or contradicting evidence; do not answer the
question.
```

The query is the full question and all options. The document is one ordered
clip. This instruction is appropriate for a reranker that is designed for
query-document relevance; it was not appropriate to embed as part of a
64-token SigLIP caption.

## Quality-first experiment ladder

### U0: no-retrieval control

- Main Qwen receives 64 uniform high-resolution frames.
- Purpose: retain the answerer baseline under the exact evaluation setup.

### U1: exhaustive Qwen reranker ceiling candidate

- Model: `Qwen3-VL-Reranker-8B`.
- Partition the video into 32 equal temporal windows.
- Sample eight ordered low-resolution frames within each window.
- Score all 32 clip documents against the complete MCQ.
- Apply temporal NMS so adjacent or overlapping high-scoring windows do not
  consume the complete budget.
- Select six clips and decode the corresponding 48 high-resolution frames.
- Add 16 uniform high-resolution insurance frames.
- Deduplicate and chronologically sort; if duplicates reduce the count, fill
  from uniform coverage to 64.
- Purpose: a simple, deliberately expensive, Qwen-only quality ceiling.

The 48/16 split is a robust starting point, not a sacred setting. Also record a
pure top-eight-clip condition only if it is cheap to produce from the same
cached scores; it measures whether uniform insurance helps.

### U2: candidate-coverage ceiling

- Keep the U1 model, query, answerer, and final 64-frame budget.
- Increase temporal-window density or use half-window overlap.
- Purpose: determine whether U1 is limited by model quality or coarse candidate
  coverage.

Do not begin here. U1 should first verify that the model and document design are
promising.

### U3: smaller reranker

- Replace Reranker-8B with Reranker-2B under the winning U1/U2 configuration.
- Purpose: measure quality and latency lost by reducing model size.

### U4: embedding controls

Run both under the same candidate timestamps and final budget:

1. `Qwen3-VL-Embedding-2B` or 8B with one frame per document.
2. The same model with four- or eight-frame ordered clip documents.

The frame condition answers whether gains come from the Qwen embedding space
and complete text. The clip condition isolates the additional value of local
temporal context.

### U5: efficient two-stage Qwen retrieval

- Cache Qwen clip embeddings per video.
- Use the embedding model to shortlist roughly 8-16 clips.
- Apply Qwen3-VL-Reranker-2B or 8B only to the shortlist.
- Keep selection and answer budgets identical to the best prior condition.
- Purpose: approach the reranker ceiling within the leaderboard time budget.

### U6: optional SigLIP-Qwen hybrid

Only after U1-U5 establish the Qwen-only ceiling:

- retain SigLIP as the cheap exhaustive first-stage retriever;
- let Qwen see the full untruncated MCQ and rerank a small candidate set; and
- keep deterministic temporal selection outside SigLIP.

This may ultimately be the best latency/quality tradeoff, but testing it first
would make a negative result ambiguous: either SigLIP recall or Qwen reranking
could be the bottleneck.

## Frame-count experiments

Keep these three variables separate:

1. **Candidate density:** total video frames inspected by the retriever.
2. **Clip size:** ordered frames represented by one retrieval document.
3. **Answer budget:** high-resolution frames given to the main answerer.

Use a factorized sweep:

1. Hold the answer budget at 64 and candidate timestamps fixed; compare clip
   sizes of 1, 4, and 8 frames.
2. Choose the clip size; compare candidate densities of 128, 256, and 512
   inspected frames.
3. Choose the retrieval configuration; compare answer budgets of 32, 64, and,
   only if supported by the main model and runtime, 96 frames.
4. Finally compare uniform insurance budgets of 0, 8, and 16 frames.

Do not change candidate density, clip size, model size, and answer budget in the
same run. The resulting gain would not be attributable.

## Evaluation sequence and gates

### Two-sample smoke

This is a technical gate, not an accuracy result. Confirm:

- the model loads in the project environment;
- ordered clips reach the intended video/multi-image processor path;
- the full MCQ is not truncated unexpectedly;
- scores are finite and non-constant;
- frame timestamps remain chronological after selection;
- selection metadata and per-stage timings are written; and
- peak memory allows retrieval and answering, either sequentially or in
  separate processes.

### Dev20 screen

Use the same fixed first 20 canonical dev140 rows used by the existing pivot
ablation. Report:

- accuracy;
- corrections and regressions versus uniform and legacy pivot;
- selected timestamps for qualitative inspection;
- median, p95, and maximum retrieval and end-to-end latency; and
- whether selected windows collapse into one temporal neighborhood.

Dev20 is a screening set. A one- or two-question gain is insufficient to claim
a winner. Promote configurations that are competitive with the existing
baselines and have a plausible correction pattern.

### Dev140 decision run

For every promoted configuration report:

- overall accuracy;
- `AFTER`, `BEFORE`, `FIRST`, `LAST`, `STATE_CHANGE`, and `GLOBAL` results;
- leading versus embedded `after` where possible;
- corrections/regressions against uniform and legacy pivot;
- disagreement count and candidate-oracle score when paired with uniform;
- median, p95, and maximum latency; and
- selection overlap with uniform and temporal pivot.

Do not judge the new selector only by standalone accuracy. Because temporal
pivot feeds the disagreement verifier, retain candidate diversity and oracle
coverage as first-class metrics.

### Full evaluation

Run only after dev140 identifies a configuration. Preserve all config fields,
model revisions, candidate timestamps, selected indices, and timing data in the
output fingerprint/metadata.

## Runtime target

The leaderboard budget is 300 seconds per query. A bounded 2B retrieval stage
with batching and cached decoding is plausibly compatible with that budget on
an H100, but this has not been measured and must not be presented as
guaranteed.

Operational guidance:

- amortize model loading over the dataset;
- batch clip scoring where the reranker implementation permits;
- avoid re-decoding the same candidate frames for every stage;
- cache embedding-model clip representations by video;
- run selection and answer generation as separate resumable stages if both
  large models do not comfortably coexist in memory; and
- target p95 below roughly 240-250 seconds to leave scheduler, decoding, and
  serialization margin.

Exhaustively reranking 128 independent frames with an 8B model is unlikely to
be the final latency solution. That is why the initial ceiling uses a modest
number of clip documents and why U3-U5 exist.

## If SigLIP is revisited

Use SigLIP for short, positive visual concepts rather than the complete MCQ:

```text
a person descends a staircase
the wearer enters a playground
a red cup on a table
```

Score extracted visual atoms independently. Apply temporal ordering in code,
not by embedding words such as “after” and hoping they alter a still-image
similarity appropriately. Add only a small number of option-residual peaks:
frames strongly retrieved by one concrete option description that were not
already covered by the generic target peaks. Cap these at approximately two to
four centers so distractor options do not receive equal frame quotas.

This direction is still limited by the need to produce good visual atoms. The
failed question-only and option-mean screens show that naive decomposition is
not enough.

## Deferred generative selector-agent idea

Do not run this in the first round. Preserve it as a later experiment:

1. Show a smaller generative Qwen a low-resolution, timestamped storyboard.
2. Ask it to return structured intervals and supporting frames, for example:

   ```json
   {
     "pivot_intervals": [[120, 140]],
     "target_intervals": [[180, 220]],
     "supporting_frames": [27, 29, 42, 45]
   }
   ```

3. Decode high-resolution frames from those intervals.
4. Let the larger Qwen answer from the selected evidence.

This realizes the proposed “two Qwen agents” architecture, but it nearly
doubles multimodal inference, can hallucinate timestamps, may create correlated
selector/answerer errors, and could reduce the disagreement diversity needed by
the verifier. Existing generative/TCoT selector experiments in the repository
were also expensive and did not beat the simple pivot. Revisit only after the
embedding/reranker ceiling is known.

## Suggested implementation boundary

To keep the first experiment interpretable:

- add a new Qwen retrieval path rather than silently changing the legacy
  temporal-pivot defaults;
- reuse the existing candidate extraction, frame decoding, answer prompt, and
  evaluation code where possible;
- save a retrieval manifest containing sample key, window timestamps, clip
  scores, selected frame indices, uniform-fill indices, model ID/revision,
  query hash, and timing;
- make the answer runner consume that manifest; and
- leave the existing pivot and verifier outputs reproducible.

The branch was already dirty when this handoff was created. Inspect and
preserve all existing edits before implementing the Qwen retrieval path.

## Immediate next action

Implement only U1 far enough to run two samples:

1. Qwen3-VL-Reranker-8B.
2. Thirty-two temporal windows per video.
3. Eight ordered low-resolution frames per window.
4. Full question plus all options as the query.
5. Six retrieved clips plus 16 uniform frames for a final budget of 64.
6. Persist the scored-window and selected-frame manifest.
7. Measure retrieval time, end-to-end time, and peak GPU memory.

If the smoke is technically sound, run the fixed dev20 screen. Do not start the
optimization ladder until that quality-oriented result is understood.
