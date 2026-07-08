# A Data-Centric Analysis of the EgoLongQA Validation Split

*Section draft for the ECCV 2026 Wearable AI Challenge paper. All figures and
tables referenced below are produced by `scripts/eda_distributions.py` and
`scripts/eda_clustering.py` in publication form (vector PDF + LaTeX `booktabs`).*

## Overview

Before designing a model for long-form egocentric video question answering, we
characterise the target benchmark to (i) quantify the reasoning skills it
actually demands and (ii) audit it for answer-selection shortcuts that would let
a model inflate accuracy without attending to the video. Our analysis covers the
full `egolongqa` validation split of the `facebook/wearable-ai` dataset: 700
multiple-choice questions, each paired with a distinct first-person video and a
free-form reference answer. Every question offers four candidate options (A–D)
together with a single labelled correct choice, so a random guesser attains an
expected accuracy of 25%.

## Composition and scene taxonomy

The split contains exactly one question per video, giving 700 unique
video–question pairs spanning 13 scene categories. The category distribution is
long-tailed (Table 1): everyday indoor activity dominates (*Daily Activities*,
18.1%), followed by three tourism-oriented categories (*Sightseeing* 14.7%,
*Travel-Tourism* 13.3%, and *Shopping* 10.3%), while the tail comprises
specialised settings such as *Gardening*, *Events*, and *Fashion Advice*, each
below 5%. The benchmark is therefore biased toward ordinary daily-life and
travel footage rather than curated or staged scenarios, which is consistent with
its always-on wearable-capture premise.

**Table 1.** Scene-category composition of the EgoLongQA validation split.

| Category | Count | Share |
|---|---:|---:|
| Daily Activities | 127 | 18.1% |
| Sightseeing | 103 | 14.7% |
| Travel-Tourism | 93 | 13.3% |
| Shopping | 72 | 10.3% |
| Travel-Sightseeing (Outdoors) | 59 | 8.4% |
| Travel-Sightseeing (Indoors) | 53 | 7.6% |
| Hiking-Outdoors | 37 | 5.3% |
| Hobbies-Daily Activities | 35 | 5.0% |
| Gardening | 32 | 4.6% |
| Pets / social gatherings | 25 | 3.6% |
| Outdoor Activities and Sports | 23 | 3.3% |
| Events | 22 | 3.1% |
| Fashion Advice | 19 | 2.7% |

## Video duration and the frame-sampling budget

The recordings are long and, unusually, of near-constant length: reading the mp4
container headers of all 700 clips (no frame decoding) gives a mean duration of
**10.2 minutes** (median 10.0, s.d. 0.87; Table 1b, Fig. 1b'), tightly
concentrated at ~600 s with a thin tail to the 1.2 min minimum and 15 min
maximum. Every clip is encoded at 15 fps, so a typical video contains
**≈9,000 frames** and the split totals 119 hours of first-person footage. Because
duration is essentially fixed, it is uninformative about file size (Pearson
$r{=}0.24$); size variation instead reflects visual complexity and bitrate.

This has a direct consequence for inference. The starter-kit LongQA baseline
samples only **4 frames uniformly over the whole video** — one frame every
~2.5 minutes for these clips — which is far too sparse for a benchmark in which
79% of questions require ordering events (§"Question characteristics"). The
evaluation protocol caps sampling at 32 frames (`--max-frames`) under a 300 s
per-query timeout. We therefore adopt **32 frames** (one every ≈19 s) as the
default for our feature-space analysis and baseline models, and reserve denser
sampling (64–128 frames) for the long-video model, whose selective temporal token
compression is designed to ingest and compress many frames within the timeout.

**Table 1b.** Video-duration statistics for the EgoLongQA validation split
($N{=}700$), read from mp4 headers. See `duration_summary.tex`.

| Statistic | Value |
|---|---:|
| Videos | 700 |
| Mean duration | 10.2 min |
| Median duration | 10.0 min |
| Std. dev. | 0.87 min |
| Min / Max | 1.2 / 15.0 min |
| p25 / p75 / p95 | 10.0 / 10.2 / 11.1 min |
| Frame rate | 15 fps (all clips) |
| Total footage | 119 h |

The distribution is not perfectly uniform: 616 of 700 clips (88%) fall in the
9.5–10.5 min mode, but the residual is asymmetric — only 11 clips are shorter
than 9.5 min whereas 73 are longer than 10.5 min (24 exceed 12 min). The
recording length is thus a deliberate ~10-minute target with a modest right tail
rather than a hard cap.

**Figure 1b'.** Video-duration distribution. (a) Histogram on a log-scaled count
axis, so the ~10-min mode and the sparse short/long tails are both legible;
(b) counts per duration bucket, exposing the 11 shorter vs. 73 longer clips that
the linear-scale view hides (`dist_video_duration_tail.pdf`). A linear single-axis
version is also provided (`dist_video_duration.pdf`).

## Question characteristics and temporal grounding

Questions are moderately long and self-contained: they average 146 characters
(median 138, 90th percentile 232), with the distribution shown in
Fig. 1(a). The phrasing is strongly interrogative and event-anchored — the most
frequent opening tokens are *what* and *after* (Fig. 1b) — signalling that
questions typically reference a specific moment or object within an extended
recording rather than asking for a global summary.

Crucially, the benchmark is dominated by **temporal reasoning**. Applying a
lexicon of ordering cues (*after, before, first, then, earlier, during, …*), we
find that 554 of 700 questions (79.1%) contain at least one explicit temporal
marker, with *after* (285 occurrences) and *first* (183) the most common
(Fig. 1c). Answering the majority of EgoLongQA questions thus requires localising
and ordering events across a long video, not merely recognising static content
in a single frame — a property that directly motivates long-context video models
with temporal token compression over frame-level classifiers.

**Figure 1.** Textual structure of EgoLongQA questions: (a) question-length
distribution in characters; (b) ten most frequent opening words; (c) temporal
ordering-cue frequency; (d) length-based answer-position bias (see §
"Shortcut audit"). Each panel is also emitted as a standalone single-column
figure (`dist_*.pdf`).

## Semantic structure of the question set

To probe the latent topical structure of the questions, we embed all 700 with
two text encoders and cluster each embedding set with $k$-means ($k{=}10$),
labelling clusters by their class-based TF-IDF (c-TF-IDF) distinctive terms.
We deliberately compare a text-specialist encoder,
`sentence-transformers/all-MiniLM-L6-v2` (384-d), against the text tower of the
multimodal `google/siglip2-so400m-patch14-384` (1152-d) — the same vision–language
model family we use for frame encoding — to test whether a jointly-trained
image–text space organises the questions differently from a purpose-built
sentence encoder.

The recovered clusters correspond to coherent, recognisable activity themes:
in-store price inspection and shopping, kitchen and food preparation, urban
navigation and street signage, museum and exhibit reading, outdoor trails, and
vehicle/arrival scenes (Table 2). However, both encoders yield near-zero
silhouette scores (0.028 for MiniLM, 0.005 for the SigLIP text tower), indicating
that the questions occupy a continuous semantic space rather than well-separated
topical islands; the clusters should therefore be read as soft thematic
groupings, not a canonical taxonomy. The two encoders agree only weakly on the
partition (adjusted Rand index 0.162), and the text-specialist produces the more
separable geometry, as expected given that SigLIP's text tower is optimised for
image alignment rather than sentence semantics. Fig. 2 shows the two UMAP
projections side by side; per-encoder projections are also provided as separate
figures (`umap_clusters_minilm.pdf`, `umap_clusters_siglip.pdf`).

**Table 2.** Representative question clusters (`all-MiniLM-L6-v2`, $k{=}10$) with
c-TF-IDF distinctive terms. Full per-encoder summaries are in
`cluster_summary_{minilm,siglip}.tex`.

| Cluster | Size | Theme (distinctive terms) |
|---|---:|---|
| Retail / price inspection | 35 | price, examined, discount, store |
| Object handling & cleaning | 92 | tool, clean, table, kitchen, floor |
| Food preparation | 78 | ingredient, pot, sauce, meat, pan |
| Colour & signage | 79 | color, sign, paint, statue |
| Vehicles & arrival | 52 | dog, parked, vehicle, car |
| Urban navigation | 63 | building, sign, street, park |
| Museum / exhibit reading | 71 | exhibit, book, artist, title |
| Road signs & traffic | 72 | sign, warning, speed, intersection |
| Shopping & clothing | 99 | item, store, closet, clothing |
| Outdoor trails | 59 | trail, path, beach, walk |

**Figure 2.** UMAP projection of EgoLongQA question embeddings clustered with
$k$-means ($k{=}10$), comparing `all-MiniLM-L6-v2` (text-only) against the
`siglip2-so400m-patch14-384` text tower (multimodal). The two encoders induce
only weakly consistent partitions (adjusted Rand index 0.162).

## Shortcut audit: how far can a video-blind model get?

A recurring failure mode of multiple-choice VQA benchmarks is that superficial
answer-string or answer-position regularities allow high accuracy without any
visual grounding. We audit two such regularities and find both present.

**Answer-label prior.** The correct-option letter is severely imbalanced
(Table 3): option C is correct in 63.4% of questions, versus 25% under a uniform
prior, while option A is correct only 1.1% of the time. A constant "always answer
C" policy — which reads neither the video nor the question — therefore attains
63.4% accuracy, establishing an unusually high video-blind floor.

**Answer-length prior.** Ranking the four options of each question by character
length, the correct option is the *shortest* in 40.3% of cases, well above the
25% expected under no length bias, and monotonically less likely as options grow
longer (Table 4, Fig. 1d). A blind "always pick the shortest option" heuristic
thus scores ≈40%.

**Table 3.** Correct-answer letter distribution. A majority-class baseline
("always C") scores 63.4%.

| Letter | Count | Share |
|---|---:|---:|
| A | 8 | 1.1% |
| B | 205 | 29.3% |
| C | 444 | 63.4% |
| D | 43 | 6.1% |

**Table 4.** Position of the correct option when the four options are ranked by
length. A uniform benchmark would place 25% in each row.

| Correct option is… | Count | Share |
|---|---:|---:|
| Longest | 119 | 17.0% |
| 2nd longest | 141 | 20.1% |
| 3rd longest | 158 | 22.6% |
| Shortest | 282 | 40.3% |

These two leaks are **statistically independent**: among the 282 questions whose
correct option is the shortest, 62.8% have letter C — essentially identical to
C's 63.4% base rate — so the length bias carries no additional information about
the letter and vice versa. They are two distinct, separately exploitable
shortcuts rather than one confound.

**Table 5.** Video-blind baselines exposed by the shortcut audit (chance = 25%).

| Video-blind policy | Accuracy |
|---|---:|
| Always answer **C** | 63.4% |
| Always pick the **shortest** option | 40.3% |
| Random guess | 25.0% |

## Implications for modelling and evaluation

Read together, the three axes of this analysis — task demand, input budget, and
answer priors — describe a benchmark whose apparent difficulty and true
difficulty are sharply misaligned, and this misalignment dictates our approach.

**The task genuinely requires dense long-temporal reasoning, yet the reference
input budget precludes it.** Nearly four in five questions demand localising and
ordering events across a ~10-minute recording (≈9,000 frames at 15 fps), while
the starter-kit baseline samples only four frames — one every ~2.5 minutes. For a
question such as "what did I do *after* X?", such sampling will typically never
observe X at all: the baseline is structurally incapable of the reasoning the
benchmark measures. Because the recordings are of near-constant length (§"Video
duration"), a *fixed* frame budget maps to a *fixed* temporal resolution across
the entire split, so frame count is the single most consequential design choice
and requires no per-video adaptation. This directly motivates (i) sampling well
above the four-frame baseline — we adopt 32 frames (~1 per 19 s) as a floor and
64–128 for the long-video model — and (ii) an architecture with selective
temporal token compression, which can ingest many frames while keeping the token
sequence within the 300 s per-query budget.

**Raw accuracy is not a trustworthy measure of visual reasoning on this split.**
The 63.4% "always-C" prior, together with the independent 40.3% shortest-option
prior, means a model can score well while remaining effectively blind — and,
combined with the point above, a frame-starved model can post a respectable
number that reflects shortcut exploitation rather than video understanding. We
therefore report accuracy alongside the video-blind baselines of Table 5 and
treat the **margin above the 63.4% floor**, not the absolute score, as the
quantity of interest. Taken as a whole, EgoLongQA is a benchmark on which shallow
methods *look* competitive: strong answer priors and an under-sampled baseline
mask the fact that its core skill — temporal grounding over long egocentric video
— remains largely untested. Closing that gap, and measuring it honestly against
the blind floors, is the objective of the remainder of this work.
