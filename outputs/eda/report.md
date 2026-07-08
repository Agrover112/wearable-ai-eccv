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

Two conclusions follow directly from this analysis and shape our approach. First,
because ~79% of questions require ordering events over a long recording, the
benchmark genuinely rewards long-context temporal modelling; a strong image-QA
model applied to sparse frames is unlikely to be competitive, justifying our
choice of a long-video architecture with selective temporal token compression.
Second, and more urgently, the 63.4% "always-C" floor means that **raw
multiple-choice accuracy is not, on its own, a trustworthy measure of visual
reasoning** on this split: a model must clear this shortcut floor by a wide
margin before its gains can be attributed to genuine video understanding. We
therefore report accuracy alongside these video-blind baselines and treat the gap
above them, rather than the absolute number, as the quantity of interest.
