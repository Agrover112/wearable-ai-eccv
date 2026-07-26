# Qwen3.5-9B EgoLongQA Dev140 Experiment Suite

**Date:** 2026-07-26  
**Scope:** EgoLongQA deterministic dev140 split  
**Primary model:** `Qwen/Qwen3.5-9B`  
**Status:** Seven evaluations completed, one evidence-audit manifest completed,
one planned dense-burst evaluation failed before inference.

## Summary

This experiment suite tested whether replacing the answer model with
Qwen3.5-9B, changing how temporal evidence is allocated, or giving the model a
native low-FPS video would improve EgoLongQA performance. It also preserved the
existing disagreement-review structure: a temporal-pivot pass and a uniform
pass answer independently, and a third call reviews only their disagreements.

The strongest completed Qwen3.5 results all reached `113/140` (`80.71%`):

- temporal-pivot proof-pack generation;
- deterministic uniform64;
- exact replay of the saved temporal-pivot frames; and
- provenance-aware disagreement review.

These equal aggregate scores do not mean the methods produced the same answers.
The pivot and uniform passes disagreed on 17 samples. Each fixed seven errors
made by the other, giving a candidate oracle of `120/140` (`85.71%`). The
reviewer chose the secondary answer only twice, producing one fix and one
regression, so it did not close any of the seven-answer oracle gap.

Operator-adaptive frame allocation reached `109/140` (`77.86%`). Its largest
loss relative to the parent pivot occurred on `FIRST` questions. The native
video runs completed successfully after replacing raw source videos with
streamed, cached FFmpeg proxies and using two H100s. Native video reached
`101/140` (`72.14%`) at 0.25 FPS and `98/140` (`70.00%`) at 0.5 FPS. Both were
substantially below the 64-image baselines.

The dense-burst run has no accuracy result. It failed after two seconds because
the parser creates `args.burst_offset_seconds` while the runner reads
`args.burst_offsets_seconds`. This was a configuration-name bug, not an OOM.

## Motivation

The suite followed from three observations in earlier EgoLongQA work.

First, the established Qwen3-VL-8B system was already strong when it received
64 carefully selected images, but many remaining errors appeared to come from
missing brief events or incorrectly resolving temporal order. Qwen3.5-9B was
therefore tested as a stronger answer model without changing the basic
multiple-choice pipeline.

Second, uniform and temporal-pivot evidence are complementary. The previous
pipeline benefited from reviewing their disagreements, so the Qwen3.5
replacement needed to preserve that reviewer rather than comparing only
independent first-pass accuracy.

Third, the source videos are approximately ten minutes long and usually change
slowly. This suggested two alternatives to a fixed 64-image budget:

1. allocate dense local bursts around semantically retrieved event centers; or
2. stream a low-FPS video through Qwen's native video interface, preserving
   more temporal samples at much lower spatial resolution.

The experiments were organized as controls around those hypotheses. Uniform64
and exact proof-pack replay isolate answer-model behavior. Operator-adaptive
and dense-burst variants alter only frame selection. Native-video variants
alter the visual representation and vLLM input path. Provenance-aware review
tests arbitration. The evidence audit prepares a manual measurement of whether
the decisive event was present before asking the model to reason about it.

## Protocol

### Data

- Input annotations:
  `data/wearable_ai_2026_egolongqa_val_700.jsonl`
- Video directory: `data/videos`
- Evaluation subset:
  `configs/egolongqa_dev140_seed20260709.json`
- Subset construction: 140 deterministically selected samples, seed `20260709`
- Subset file SHA-1: `e16a6ca78ea8abf7fa95f07f387ab69746e4a421`
- Evaluation unit: one question and one associated video per row

The split is label-imbalanced: the gold distribution is A=1, B=39, C=91,
D=9. Always predicting C would score `91/140` (`65.00%`). For that reason,
aggregate accuracy is reported together with non-C accuracy and prediction
distribution.

### Shared answer-model settings

Unless noted otherwise, the completed GPU runs used:

- model: `Qwen/Qwen3.5-9B`;
- backend: vLLM;
- dtype: bfloat16;
- non-thinking generation;
- prompt variant: `baseline`;
- maximum new tokens: 16;
- tensor parallelism: 1;
- concurrency: 1;
- one H100;
- `QWEN_MIN_PIXELS=784`;
- `QWEN_MAX_PIXELS=451584`, approximately 672x672 pixels per image;
- vLLM model length: 65,536 tokens;
- vLLM GPU memory utilization: 0.92; and
- GDN prefill backend: Triton.

No repeated seeds were run. vLLM reports a seed internally, but the experiment
artifacts do not establish bitwise deterministic multimodal generation. The
exact parent-proofpack replay differed from the original parent run on 2/140
answers despite using the same saved frame indices, model, prompt family, and
aggregate context size. Small differences of one or two answers should
therefore be treated as within observed run-to-run variability until repeated.

### Method 1: Parent Temporal-Pivot Proof Pack

The parent run generated the evidence pack and answered in one pipeline:

1. Sample 128 uniformly distributed candidate frames.
2. Compile the question into a temporal program such as `AFTER`, `BEFORE`,
   `FIRST`, `LAST`, `STATE_CHANGE`, or `GLOBAL`.
3. Embed candidates with `Qwen/Qwen3-VL-Embedding-8B`.
4. Retrieve pivot and target centers appropriate to the temporal operator.
5. Add local context, uniform anchors, bridges, and semantic-boundary frames.
6. Sort the final evidence chronologically and cap it at 64 images.
7. Ask Qwen3.5-9B for one option letter.

The proof-pack fingerprint was `3fe37a7124ac`; its file SHA-1 was
`c118d041014b5b7240f097d9f78d252b15794344`.

### Method 2: Fixed Frame-Pack Controls

The fixed-pack runner performs no retrieval. It records exact frame indices,
extracts those frames, and sends them through the same Qwen3.5 answer path.

Two conditions were run:

- **Uniform64:** 64 deterministic frames across the complete video.
- **Parent replay:** the exact 64 saved indices from the parent temporal-pivot
  `proofpack.jsonl`.

The runner fingerprints the frame source, source proof-pack hash, model,
context, prompt, and pixel settings. These controls answer two different
questions: whether Qwen3.5 benefits from the temporal proof pack relative to a
uniform grid, and whether the parent runner itself changes inference relative
to replaying its saved images.

### Method 3: Global Anchors Plus Dense Bursts

The planned dense-burst pack retained 16 global anchors and selected up to
eight ranked event centers from the parent proof pack. Each center was expanded
at offsets `-2`, `-1`, `-0.33`, `+0.33`, `+1`, and `+2` seconds using the
video's actual FPS. Duplicate or clipped frames were removed, and remaining
slots were to be filled by the midpoint of the largest uncovered temporal gap,
up to a total of 64 frames.

This method was intended to test whether short, dense neighborhoods capture
actions that isolated semantic centers miss. The job failed while computing
its fingerprint, before frame metadata, model startup, or inference. It
therefore has no usable run artifacts and should not be interpreted as a
negative result for the method.

### Method 4: Operator-Adaptive Sampling

The operator-adaptive runner reused the parent proof pack's ranked centers but
changed the local temporal pattern according to the compiled operator. Every
condition retained exactly 64 chronological frames:

| Operator | Samples | Policy |
| --- | ---: | --- |
| `GLOBAL` | 40 | 64 uniform frames |
| `AFTER` | 52 | 16 anchors plus forward-oriented pivot and target bursts |
| `BEFORE` | 14 | 16 anchors plus backward-oriented pivot and target bursts |
| `FIRST` | 26 | 24 broad anchors plus early event neighborhoods |
| `LAST` | 3 | 24 broad anchors plus late event neighborhoods |
| `STATE_CHANGE` | 5 | 16 anchors plus paired pre-change and post-change bursts |

Directional policies used up to two pivot centers and six target centers.
`FIRST` and `LAST` used up to five edge centers. `STATE_CHANGE` used up to four
centers. The offsets were asymmetric: `AFTER` extended target evidence as far
as +4 seconds, `BEFORE` as far as -4 seconds, edge policies as far as 10
seconds, and state-change policies as far as 8 seconds on each side.

All residual slots were filled from a four-times-denser uniform grid. The
metadata records the operator, policy, source center, rank, relevance score,
offset in seconds, and whether each frame is a global anchor.

One implementation detail matters when comparing `GLOBAL` against the fixed
uniform run. The adaptive runner's uniform formula includes both endpoints
using `round(position * (total_frames - 1) / (count - 1))`. The fixed-pack
control follows the historical extractor formula
`int(position * (total_frames - 1) / count)`, which does not include the final
frame for a 64-frame request. These are both broad uniform grids, but they are
not identical frame packs.

### Method 5: Provenance-Aware Disagreement Review

The reviewer used the parent pivot predictions as candidate 1 and uniform64 as
candidate 2.

- Agreement rows were copied without a model call.
- Disagreement rows received up to 32 high-priority parent proof-pack images
  and 32 independent uniform images.
- Each image was labeled with its source group, frame index, timestamp, and
  proof-pack role where available.
- The prompt exposed the semantics of the two candidate options and asked the
  model to check visible support, contradiction, and temporal order.
- Valid outputs were `CANDIDATE_1`, `CANDIDATE_2`, or `INSUFFICIENT`.
- Invalid or insufficient output would retain candidate 1.
- Thinking was disabled, and the response budget was 16 tokens.

There were 17 model calls for 140 samples. The reviewer returned candidate 1
fifteen times and candidate 2 twice. It never returned `INSUFFICIENT` and never
violated the output contract.

### Method 6: Native Video With Cached Low-Resolution Proxies

The original native-video implementation passed the source MP4 directly to
vLLM. The long 1080p videos caused pathological decoder memory behavior. A
single-H100 array reached the server and then ended with SLURM OOM events.
Moving to two H100s increased model and KV-cache capacity but did not fix the
raw-video decode path: the first request either returned HTTP 500 or stalled.

The successful pipeline separated preprocessing from inference:

1. FFmpeg decoded each source incrementally.
2. The `fps` filter retained only the requested temporal samples.
3. Each retained frame was immediately resized with Lanczos.
4. FFmpeg encoded a silent H.264 proxy with `libx264`, CRF 18, the `veryfast`
   preset, `yuv420p`, and fast-start metadata.
5. The content-addressed proxy was cached by source identity and preprocessing
   settings.
6. vLLM received one local `video_url` pointing to the proxy, preserving its
   native-video input path.

The two proxy conditions were:

| Condition | Resolution | Frames/video min/mean/p50/p95/max | Cache size |
| --- | --- | --- | ---: |
| 0.25 FPS | 480x256 | 127 / 154.25 / 150 / 177 / 225 | 0.91 GB |
| 0.5 FPS | 352x192 | 254 / 308.53 / 301 / 354 / 450 | 1.05 GB |

Each cache contains 140 MP4 files under
`.cache/egolongqa/native_video_proxies/<variant>/`. Proxy preparation used a
12-worker CPU array and took about 32 minutes per condition. The GPU runs used
two H100s, tensor parallelism 2, a 131,072-token context, 0.90 GPU memory
utilization, one video per prompt, and no multimodal processor cache.

The native comparison is not a pure FPS ablation because resolution changes at
the same time: 0.25 FPS uses 480x256, while 0.5 FPS uses 352x192. Its result
should be read as a comparison of two total visual-budget configurations.

### Method 7: Evidence-Recall Audit

The CPU audit selected samples satisfying either of these conditions:

- pivot and uniform predictions disagreed; or
- at least one candidate was wrong.

It produced 34 annotation rows. Selection reasons overlap:

- 17 prediction disagreements;
- 27 parent errors; and
- 27 uniform errors.

Each row contains the reconstructed 128-frame candidate grid, the final
proof-pack frames, video duration, operator, both predictions, gold answer,
selection reasons, and an empty annotation template. The job intentionally did
not invent decisive-event labels. Human annotations are still required before
event recall, all-events-covered rate, or temporal-order coverage can be
reported.

## Metrics

### Accuracy

The fraction of samples whose parsed option letter matches the gold option.
All reported results contain 140 predictions and were evaluated against the
same dev140 subset.

### Non-C accuracy

Accuracy on the 49 examples whose gold answer is not C. This is included
because C is correct on 91/140 examples.

### Paired fixes and regressions

For a variant compared with the parent pivot:

- a **fix** is parent wrong and variant correct;
- a **regression** is parent correct and variant wrong;
- **both wrong, different** means both answers are incorrect but disagree.

This paired view distinguishes genuine complementarity from an equal aggregate
score produced by different errors.

### Candidate oracle

The score obtained by choosing the correct answer whenever either the pivot or
uniform candidate is correct. This is an upper bound for a reviewer restricted
to those two candidate answers, not a realizable model result.

### Context fill

vLLM prompt tokens divided by the configured context window. This measures
context pressure, not visual quality. Native video and image-pack token counts
are produced by different multimodal processing paths and are not directly
equivalent frame-for-frame.

## Runs

### Completed and terminal jobs

SLURM states and elapsed times were verified with `sacct`.

| Job | Experiment | Resources | State | Elapsed | Output |
| --- | --- | --- | --- | ---: | --- |
| `49265042` | Parent temporal pivot | 1x H100, 12 CPU, 96 GB | COMPLETED | 02:50:28 | `qwen35_parent_temporal_pivot_dev140` |
| `49265043` | Fixed uniform64 | 1x H100, 12 CPU, 96 GB | COMPLETED | 02:07:25 | `qwen35_fixed_uniform64_dev140` |
| `49265045` | Fixed parent replay | 1x H100, 12 CPU, 96 GB | COMPLETED | 00:56:05 | `qwen35_fixed_parent_pivot_dev140` |
| `49265046` | Dense bursts | 1x H100, 8 CPU, 64 GB | FAILED, exit 1 | 00:00:02 | empty run directory |
| `49265047` | Operator adaptive | 1x H100, 8 CPU, 64 GB | COMPLETED | 01:39:27 | `qwen35_operator_adaptive_dev140` |
| `49265048` | Provenance reviewer | 1x H100, 12 CPU, 96 GB | COMPLETED | 00:10:08 | `qwen35_provenance_verifier_dev140` |
| `49265049` | Evidence audit | 20 CPU, 32 GB | COMPLETED | 02:32:15 | `longqa_evidence_recall_audit_qwen35_dev140` |
| `49266069_0` | 0.25 FPS proxy preparation | 12 CPU, 48 GB | COMPLETED | 00:32:06 | proxy cache |
| `49266069_1` | 0.5 FPS proxy preparation | 12 CPU, 48 GB | COMPLETED | 00:32:24 | proxy cache |
| `49266071_0` | Native 0.25 FPS | 2x H100, 12 CPU, 128 GB | COMPLETED | 00:12:10 | `qwen35_native_video_fps0p25_dev140` |
| `49266071_1` | Native 0.5 FPS | 2x H100, 12 CPU, 128 GB | COMPLETED | 00:12:09 | `qwen35_native_video_fps0p5_dev140` |

The parent job resumed proof-pack selection from 11/140 cached rows. Uniform64
reused all 140 cached frame-pack metadata rows but regenerated all answers.
Other elapsed times include server startup and any method-specific preparation,
so this table should not be used as a controlled throughput benchmark.

### Native-video debugging history

| Job | Configuration | Outcome |
| --- | --- | --- |
| `49264658_[0-1]` | Initial 1-H100 array | Failed immediately because the job started outside the repository and could not find the environment |
| `49264748_[0-1]` | Corrected paths | Failed immediately because `python` was unavailable after the missing Conda environment |
| `49264845`, `49264858` | Launcher/debug iterations | Cancelled while correcting environment and server configuration |
| `49265044_[0-1]` | Raw source MP4, 1 H100 | SLURM OOM after 29-31 minutes |
| `49265786_0` | Raw source MP4, 2 H100 | First request failed with HTTP 500 after 21 minutes; sibling task cancelled |
| `49265867_0` | Raw source MP4, 2 H100 | Cancelled after the first raw-video request stalled |
| `49266069_[0-1]` | Streamed CPU proxy preparation | Both conditions completed |
| `49266071_[0-1]` | Cached proxies, 2 H100 | Both 140-sample evaluations completed in about 12 minutes |

The evidence indicates that the raw-source failure was not primarily a
model-weight or KV-cache capacity problem. The two-GPU vLLM server reported
ample KV capacity, yet raw decoding still failed or stalled. Streaming the
decode into small proxies removed the problematic representation and made the
same native-video API stable.

## Results

### Main accuracy

| Experiment | Correct | Accuracy | Delta vs parent |
| --- | ---: | ---: | ---: |
| Parent temporal pivot | 113/140 | **80.71%** | reference |
| Fixed uniform64 | 113/140 | **80.71%** | 0.00 pp |
| Fixed parent replay | 113/140 | **80.71%** | 0.00 pp |
| Provenance reviewer | 113/140 | **80.71%** | 0.00 pp |
| Operator adaptive | 109/140 | 77.86% | -2.86 pp |
| Native video, 0.25 FPS at 480x256 | 101/140 | 72.14% | -8.57 pp |
| Native video, 0.5 FPS at 352x192 | 98/140 | 70.00% | -10.71 pp |
| Dense bursts | n/a | n/a | failed before inference |

The controlled uniform64 comparison against the earlier
Qwen3-VL-8B uniform64/672 dev140 run (`110/140`) is +3 answers, or +2.14
percentage points. This is encouraging but remains a single dev140 run. No
Qwen3.5 full-700 evaluation was performed, so these results do not replace the
established full-validation numbers in `docs/longqa/RUN_LOG.md`.

### Shortcut-aware results

| Experiment | Non-C correct | Non-C accuracy | Predicted A/B/C/D |
| --- | ---: | ---: | --- |
| Parent temporal pivot | 42/49 | 85.71% | 7 / 46 / 72 / 15 |
| Fixed uniform64 | 45/49 | **91.84%** | 8 / 46 / 69 / 17 |
| Fixed parent replay | 43/49 | 87.76% | 7 / 45 / 71 / 17 |
| Provenance reviewer | 42/49 | 85.71% | 8 / 46 / 72 / 14 |
| Operator adaptive | 40/49 | 81.63% | 8 / 43 / 71 / 18 |
| Native 0.25 FPS | 39/49 | 79.59% | 10 / 45 / 64 / 21 |
| Native 0.5 FPS | 39/49 | 79.59% | 9 / 43 / 62 / 26 |
| Always C | 0/49 | 0.00% | 0 / 0 / 140 / 0 |

All Qwen3.5 visual runs substantially exceed the always-C shortcut and retain
useful non-C performance. Uniform64 has the best non-C score despite tying the
parent in aggregate; it loses an equal number of C examples.

### Paired comparison with the parent pivot

| Variant | Different answers | Fixes | Regressions | Both wrong, different | Net correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed uniform64 | 17 | 7 | 7 | 3 | 0 |
| Fixed parent replay | 2 | 1 | 1 | 0 | 0 |
| Operator adaptive | 18 | 6 | 10 | 2 | -4 |
| Provenance reviewer final output | 2 | 1 | 1 | 0 | 0 |
| Native 0.25 FPS | 33 | 9 | 21 | 3 | -12 |
| Native 0.5 FPS | 35 | 7 | 22 | 6 | -15 |

Pivot and uniform disagree on 17 rows. Their candidate oracle is `120/140`
(`85.71%`), compared with `113/140` for either candidate and the reviewer.
Twenty samples are wrong under both candidates; on three of those they choose
different wrong options.

The reviewer retained candidate 1 on 15 disagreements. Of its two candidate-2
choices, one corrected the parent and one overturned a correct parent answer.
The reviewer therefore changed behavior exactly as intended but provided no
net accuracy gain. The remaining reviewer opportunity is not a larger evidence
budget; each call already used 64 images. It is better calibration of when
uniform evidence is more trustworthy.

### Operator-stratified results

| Operator | n | Parent pivot | Operator adaptive | Fixed uniform64 |
| --- | ---: | ---: | ---: | ---: |
| `AFTER` | 52 | 41/52 (78.85%) | 40/52 (76.92%) | 40/52 (76.92%) |
| `BEFORE` | 14 | 12/14 (85.71%) | 12/14 (85.71%) | 12/14 (85.71%) |
| `FIRST` | 26 | **24/26 (92.31%)** | 21/26 (80.77%) | 21/26 (80.77%) |
| `GLOBAL` | 40 | 30/40 (75.00%) | 30/40 (75.00%) | **34/40 (85.00%)** |
| `LAST` | 3 | 2/3 (66.67%) | 2/3 (66.67%) | 2/3 (66.67%) |
| `STATE_CHANGE` | 5 | 4/5 (80.00%) | 4/5 (80.00%) | 4/5 (80.00%) |

Most of the operator-adaptive loss is concentrated in `FIRST`, where dense
edge neighborhoods lose three correct answers relative to the parent pivot.
The `GLOBAL` comparison also shows that the historical fixed uniform grid is
four answers stronger than the adaptive runner's end-inclusive uniform grid.
This does not establish that the endpoint formula is causal because the runs
are not bitwise deterministic, but it is large enough to control explicitly in
the next ablation.

### Native-video comparison

The two native configurations disagree on 22 answers. Relative to 0.5 FPS,
the 0.25-FPS configuration makes ten fixes and seven regressions, for a net
gain of three answers. Five of the 22 disagreements are different wrong
answers.

| Condition | Prompt tokens min/mean/p50/p95/max | Context fill mean/p95/max | Accuracy |
| --- | --- | --- | ---: |
| 0.25 FPS, 480x256 | 8,424 / 10,151 / 9,967 / 11,297 / 13,124 | 7.74% / 8.62% / 10.01% of 131K | **72.14%** |
| 0.5 FPS, 352x192 | 9,742 / 11,746 / 11,585 / 12,645 / 14,260 | 8.96% / 9.65% / 10.88% of 131K | 70.00% |

Doubling temporal density increased mean prompt tokens by about 16%, even
though spatial resolution was reduced, but accuracy fell by 2.14 percentage
points. Neither native run came close to filling the 131K context. The limiting
factor in these settings is therefore not text/KV context exhaustion.

The likely problem is evidence quality: hundreds of low-resolution frames do
not automatically give the model the same inspectable detail and explicit
temporal structure as 64 high-resolution images. Because FPS and resolution
were changed together, the current data cannot separate insufficient spatial
detail from excessive temporal redundancy.

### Category diagnostics

| Category | n | Parent pivot | Uniform64 | Operator adaptive | Native 0.25 | Native 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Daily Activities | 26 | 73.1% | 73.1% | 73.1% | 65.4% | 57.7% |
| Events | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 66.7% |
| Fashion Advice | 5 | 80.0% | 80.0% | 80.0% | 80.0% | 80.0% |
| Gardening | 7 | 85.7% | 85.7% | 71.4% | 42.9% | 85.7% |
| Hiking-Outdoors | 7 | 85.7% | 85.7% | 100.0% | 100.0% | 100.0% |
| Hobbies-Daily Activities | 7 | 71.4% | 71.4% | 71.4% | 71.4% | 57.1% |
| Outdoor Activities and Sports | 5 | 100.0% | 60.0% | 60.0% | 60.0% | 60.0% |
| Pets, social gatherings with friends and family | 5 | 40.0% | 60.0% | 40.0% | 80.0% | 60.0% |
| Shopping | 15 | 73.3% | 73.3% | 73.3% | 73.3% | 60.0% |
| Sightseeing | 20 | 75.0% | 85.0% | 80.0% | 60.0% | 60.0% |
| Travel-Sightseeing (Indoors) | 10 | 100.0% | 100.0% | 80.0% | 90.0% | 90.0% |
| Travel-Sightseeing (Outdoors) | 11 | 90.9% | 81.8% | 81.8% | 72.7% | 81.8% |
| Travel-Tourism | 19 | 89.5% | 89.5% | 89.5% | 79.0% | 79.0% |

Most categories contain too few samples for stable conclusions. The table is
diagnostic rather than inferential. The most credible native-video weakness is
the consistent drop on the larger Daily Activities, Sightseeing, and
Travel-Tourism groups.

## Interpretation

### What the suite supports

1. **Qwen3.5-9B is a useful replacement candidate.** On the controlled
   uniform64 dev140 setting it improves from the prior Qwen3-VL-8B result of
   `110/140` to `113/140`.
2. **Pivot and uniform evidence remain complementary.** Their equal accuracy
   hides 17 answer disagreements and seven unique fixes in each direction.
3. **The current reviewer leaves measurable value unused.** The two-candidate
   oracle is seven answers above the reviewer, while the reviewer changes only
   two answers.
4. **The tested operator-adaptive policy is not better than the parent pack.**
   It loses four net answers, particularly on `FIRST`.
5. **Cached proxies solve the native-video systems problem.** Two H100s plus
   raw source MP4s did not; streaming FFmpeg proxies did.
6. **More low-resolution temporal samples do not guarantee better accuracy.**
   The 0.5-FPS condition uses roughly twice as many frames and more tokens but
   scores below the 0.25-FPS condition.
7. **Context length is not the native-video bottleneck here.** Maximum context
   fill is under 11% of 131K for both successful native configurations.

### What the suite does not support

- It does not establish full-validation or competition performance for
  Qwen3.5; all accuracy results are dev140.
- It does not establish that dense bursts are ineffective; that method never
  ran.
- It does not isolate FPS from resolution in the native-video comparison.
- It does not prove operator-specific effects for `LAST`, `STATE_CHANGE`, or
  small semantic categories because their sample counts are very small.
- It does not provide evidence-recall scores because decisive-event
  annotations have not been completed.
- It does not establish statistical significance or eliminate inference
  variability because there is one run per condition.

### Recommended next experiments

1. Fix the dense-burst argument destination, add a parser-to-fingerprint test,
   and run the intended 64-frame experiment.
2. Repeat parent replay and uniform64 at least three times or force a fully
   deterministic generation configuration before interpreting one-answer
   differences.
3. Make the native ablation factorial: compare 0.25 and 0.5 FPS at the same
   resolution, then compare resolutions at a fixed FPS.
4. Test a hybrid native representation: retain the low-FPS proxy for coverage
   but add a small set of high-resolution event crops or frames.
5. Annotate the 34-sample evidence manifest before adding more retrieval
   policies. Separate "decisive event absent" from "event present but reasoning
   wrong."
6. Improve the reviewer as a calibrated selector between two candidates rather
   than asking for unconstrained extra reasoning. The 17-disagreement set and
   85.71% oracle define a compact target.
7. Promote Qwen3.5 to the full 700 samples only after the deterministic replay
   and reviewer calibration questions are resolved.

## Issues And Fixes

### Native-video host-memory failure

**Observed:** One-H100 raw-video jobs ended with SLURM OOM events. Two-H100 jobs
still failed or stalled on the first request.

**Fix applied:** Added streamed FFmpeg proxy creation, content-addressed proxy
caching, proxy settings in the inference fingerprint, a CPU preprocessing
array, and a two-H100 inference array.

**Result:** Both 140-sample native-video evaluations completed in approximately
12 minutes after approximately 32 minutes of one-time preprocessing per
condition.

### Dense-burst argument-name mismatch

**Observed:** Job `49265046` failed before inference:

```text
AttributeError: 'Namespace' object has no attribute
'burst_offsets_seconds'. Did you mean: 'burst_offset_seconds'?
```

**Status:** Not fixed or rerun in this report. The run directory is empty. The
parser option and internal attribute must be made consistent before
reproduction.

### Environment activation noise

`scripts/activate_local_env.sh` expects
`.conda/envs/wearable-ai-eccv`, which is not currently present. Launchers
continue with `source ... || true` and then activate `.venv`. Completed jobs
therefore contain a benign `EnvironmentLocationNotFound` message in stderr.
Early native submissions failed before the explicit `.venv` fallback and are
listed in the debugging history.

### Reproducibility state

The repository HEAD at report time was
`0b8b077d7e7efe9a245478ea29bb09ac45066e20`, but the Qwen3.5 runners, tests,
SLURM scripts, and related model changes were uncommitted. Artifact
fingerprints identify run inputs and settings, but the Git commit alone does
not reconstruct the executed source. The suite should be committed before it
is treated as an archival benchmark.

### Test coverage

The six experiment test modules pass:

```text
35 passed in 0.80s
```

The dense-burst failure exposes a missing integration test: unit tests covered
frame planning, but not `parse_args()` flowing into fingerprint construction.

## Reproduction

Run commands from the repository root. The launchers use the canonical dev140
split and write to `runs/egolongqa/<run_name>/`.

### Parent temporal pivot

```bash
sbatch slurm_scripts/qwen35_parent_pivot_dev140/run.sh
```

### Fixed uniform64

```bash
sbatch slurm_scripts/qwen35_fixed_pack_dev140/run.sh
```

### Exact parent-proofpack replay

```bash
sbatch slurm_scripts/qwen35_fixed_pack_dev140/run_parent_proofpack.sh
```

### Operator-adaptive pack

```bash
bash slurm_scripts/qwen35_operator_adaptive_dev140/submit.sh
```

### Provenance-aware disagreement reviewer

```bash
sbatch slurm_scripts/qwen35_provenance_verifier_dev140/run.sh
```

### Evidence-recall annotation manifest

```bash
sbatch slurm_scripts/longqa_evidence_recall_audit_dev140/run.sh
```

### Native-video proxy preparation and inference

```bash
proxy_job=$(sbatch --parsable \
  slurm_scripts/qwen35_native_video_dev140/prepare_proxies_array.sh)
sbatch --dependency=afterok:${proxy_job} \
  slurm_scripts/qwen35_native_video_dev140/run_dev140_fps_array.sh
```

The inference launcher uses `--require-proxies`, so it will fail rather than
silently decode raw source videos when a cache entry is missing.

### Dense bursts

The launcher is present at
`slurm_scripts/qwen35_dense_bursts_dev140/run.sh`, but it should not be
resubmitted until the `burst_offset_seconds` versus `burst_offsets_seconds`
attribute mismatch is fixed and covered by a parse-to-fingerprint test.

## Artifact Index

### Results and predictions

- Parent pivot:
  `runs/egolongqa/qwen35_parent_temporal_pivot_dev140/`
- Fixed uniform64:
  `runs/egolongqa/qwen35_fixed_uniform64_dev140/`
- Fixed parent replay:
  `runs/egolongqa/qwen35_fixed_parent_pivot_dev140/`
- Operator adaptive:
  `runs/egolongqa/qwen35_operator_adaptive_dev140/`
- Provenance reviewer:
  `runs/egolongqa/qwen35_provenance_verifier_dev140/`
- Native 0.25 FPS:
  `runs/egolongqa/qwen35_native_video_fps0p25_dev140/`
- Native 0.5 FPS:
  `runs/egolongqa/qwen35_native_video_fps0p5_dev140/`
- Evidence audit:
  `runs/egolongqa/longqa_evidence_recall_audit_qwen35_dev140/`

Each completed evaluation directory contains `predictions.jsonl`,
`results.json`, and `results_summary.json`. Grounded methods additionally
contain `proofpack.jsonl`, `frame_packs.jsonl`, `frame_metadata.jsonl`, or
`verifier_evidence.jsonl` as appropriate.

### Proxy caches

- `.cache/egolongqa/native_video_proxies/fps0p25_480x256/`
- `.cache/egolongqa/native_video_proxies/fps0p5_352x192/`

### SLURM launchers

- `slurm_scripts/qwen35_parent_pivot_dev140/`
- `slurm_scripts/qwen35_fixed_pack_dev140/`
- `slurm_scripts/qwen35_dense_bursts_dev140/`
- `slurm_scripts/qwen35_operator_adaptive_dev140/`
- `slurm_scripts/qwen35_provenance_verifier_dev140/`
- `slurm_scripts/qwen35_native_video_dev140/`
- `slurm_scripts/longqa_evidence_recall_audit_dev140/`

### SLURM logs

The corresponding stdout and stderr files are under matching directories in
`slurm_outputs/`. The dense-burst exception is in
`slurm_outputs/qwen35_dense_bursts_dev140/q35-bursts-dev140_49265046.err`.

