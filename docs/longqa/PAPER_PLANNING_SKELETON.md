# Paper planning skeleton

**Status:** planning only  
**Scope:** EgoLongQA / Wearable AI workshop project  
**Rule:** Do not turn this file into a finished paper before the team agrees on the story, research questions, evidence, and claims.

## 1. Working title and authors

**Title:** `[PLACEHOLDER: short, descriptive title]`

**Authors:** `[PLACEHOLDER: names, affiliations, order]`

**Corresponding author:** `[PLACEHOLDER]`

**Links:** `[PLACEHOLDER: code, website, supplementary material]`

## 2. Candidate stories

Choose one story after the meeting. Do not combine all experiments into one narrative.

| Choice | Central question | Evidence required | Risks / open issues | Decision |
| --- | --- | --- | --- | --- |
| A. Temporal coverage | Does preserving coverage of the whole recording help long-horizon MCQ video QA? | Matched frame-count and resolution comparisons; strict split definitions | Sampling phase may explain part of the gain | `[OPEN]` |
| B. Evidence selection | Can question-conditioned selection improve evidence quality without losing the timeline? | Selector outputs, frame audits, recall/complementarity analysis | Retrieval may omit transitions or repeated instances | `[OPEN]` |
| C. Answer arbitration | Can several independently generated answers be combined without leaking labels or overfitting a development set? | Candidate disagreement table, held-out evaluation, routing rules | Small development subsets can make rules look stronger than they are | `[OPEN]` |
| D. Reproducible system study | What does a careful training-free pipeline contribute under a fixed compute and latency budget? | Full configuration, timings, container record, ablations | This is a systems report unless a sharper question is selected | `[OPEN]` |

**Selected story:** `[OPEN]`  
**Why this story:** `[OPEN: two or three sentences after team discussion]`  
**Stories explicitly rejected:** `[OPEN]`

## 3. Research-question worksheet

Fill this in before writing the abstract or conclusion.

| RQ | Precise question | Independent variable | Outcome | Comparison | Required artifact | Status |
| --- | --- | --- | --- | --- | --- | --- |
| RQ1 | `[OPEN]` | `[OPEN]` | Accuracy / latency / evidence metric | `[OPEN]` | `[RUN_LOG or artifact path]` | Open |
| RQ2 | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` | Open |
| RQ3 | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` | Open |

For each selected RQ, record:

- What is held constant?
- What is changed?
- Which split is used for selection and which split is used for confirmation?
- What result would count as support, no effect, or evidence against the question?
- What alternative explanation must be checked?

## 4. Proposed paper structure

### Abstract

`[PLACEHOLDER: problem, approach, evaluation setting, main measured result, limitation]`

### 1. Introduction

`[PLACEHOLDER: task motivation]`

`[PLACEHOLDER: concrete gap; avoid broad claims until supported by references and experiments]`

`[PLACEHOLDER: selected contribution list, limited to contributions actually demonstrated]`

### 2. Task and data

`[PLACEHOLDER: EgoLongQA task definition, question format, answer options, video characteristics]`

`[PLACEHOLDER: split construction and the relationship between the 700-question validation set, dev140, and val560]`

`[PLACEHOLDER: evaluation metric and key-alignment safeguards]`

### 3. System

#### 3.1 Input preparation

`[PLACEHOLDER: video decoding, uniform frame sampling, endpoint handling, image resolution, timestamp representation]`

#### 3.2 Evidence selection

`[PLACEHOLDER: global sampling path]`

`[PLACEHOLDER: question-conditioned retrieval path; describe the text query and truncation-safe representation]`

`[PLACEHOLDER: temporal packing and deduplication]`

#### 3.3 Answer generation

`[PLACEHOLDER: Qwen model, prompt, options, output format, reasoning setting]`

#### 3.4 Candidate comparison and verification

`[PLACEHOLDER: which independent candidates are used]`

`[PLACEHOLDER: disagreement rule, allowed outputs, abstention or fallback policy]`

`[PLACEHOLDER: how all rules avoid using test labels or development labels improperly]`

### 4. Experiments

`[PLACEHOLDER: primary comparison and controlled ablations]`

`[PLACEHOLDER: compute and latency protocol]`

`[PLACEHOLDER: subset selection protocol and held-out confirmation]`

### 5. Results

`[PLACEHOLDER: fill only from checked result files and the final run ledger]`

### 6. Analysis

`[PLACEHOLDER: error categories, temporal coverage, distractor evidence, repeated instances, and disagreement behavior]`

### 7. Limitations, reproducibility, and ethics

`[PLACEHOLDER: limitations and unresolved implementation risks]`

`[PLACEHOLDER: privacy and egocentric-video considerations]`

`[PLACEHOLDER: release policy for videos, derived frames, logs, prompts, and credentials]`

### 8. Conclusion

`[PLACEHOLDER: only claims supported by the evidence-to-claim ledger]`

## 5. Figure placeholders

Use these as layout boxes until the team chooses the final figures. Each figure needs a source artifact and an owner.

### Figure 1: End-to-end pipeline

```text
+--------------------------------------------------------------+
| [FIGURE 1 PLACEHOLDER: video -> frame candidates -> evidence |
|  pack -> Qwen answer(s) -> optional disagreement decision]   |
+--------------------------------------------------------------+
Caption placeholder: [OPEN]
Source files: [OPEN]
Owner: [OPEN]   Status: [OPEN]
```

### Figure 2: Temporal evidence example

```text
+--------------------------------------------------------------+
| [FIGURE 2 PLACEHOLDER: one video timeline with sampled,      |
|  retrieved, retained, and distractor frames distinguished]   |
+--------------------------------------------------------------+
Caption placeholder: [OPEN]
Source audit/contact sheet: [OPEN]
Owner: [OPEN]   Status: [OPEN]
```

### Figure 3: Error or disagreement analysis

```text
+--------------------------------------------------------------+
| [FIGURE 3 PLACEHOLDER: category-level errors or candidate    |
|  disagreement outcomes; no chart until denominators are      |
|  fixed and checked]                                          |
+--------------------------------------------------------------+
Caption placeholder: [OPEN]
Data source: [OPEN]
Owner: [OPEN]   Status: [OPEN]
```

### Optional Figure 4: Reproducible deployment

```text
+--------------------------------------------------------------+
| [OPTIONAL FIGURE 4: source environment -> image build ->     |
|  local smoke test -> immutable registry reference]           |
+--------------------------------------------------------------+
Caption placeholder: [OPEN]
Source files: [OPEN]
Owner: [OPEN]   Status: [OPEN]
```

## 6. Table placeholders

### Table 1: Main comparison

| Method | Model | Frames / resolution | Split | Accuracy | Runtime | Source |
| --- | --- | --- | --- | ---: | ---: | --- |
| `[baseline]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[RUN_LOG]` |
| `[primary]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[RUN_LOG]` |

### Table 2: Controlled ablation

| Variant | Changed factor | Fixed factors | Split | Accuracy | Interpretation | Source |
| --- | --- | --- | --- | ---: | --- | --- |
| `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` |

### Table 3: Disagreement and complementarity

| Candidate A | Candidate B | Disagreements | A correct only | B correct only | Both wrong | Oracle | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` |

### Table 4: Runtime and resource use

| Stage | Hardware | Startup | Per-question / amortized time | Peak memory | Context fill | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` |

## 7. Evidence-to-claim ledger

Every sentence in the abstract, introduction, results, and conclusion that states a result or comparison should have a row here.

| Claim ID | Draft claim | Claim type | Exact source artifact | Split / denominator | Analysis or script | Status / reviewer |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | `[OPEN]` | Result / method / limitation | `[PATH]` | `[OPEN]` | `[COMMAND or script]` | Unchecked |
| C02 | `[OPEN]` | Result / method / limitation | `[PATH]` | `[OPEN]` | `[COMMAND or script]` | Unchecked |

**Rules:**

- Do not transfer a dev140 result to the full validation set without a separately identified evaluation.
- Record whether a number is standalone, an oracle, a routed result, or a retrospective diagnostic.
- Record whether any rule was chosen using labels from the same split on which it is reported.
- Keep historical or failed runs available as provenance, but do not present them as successful experiments.

## 8. Experiment selection template

Use this table to decide what belongs in the paper. A large experiment list is not a contribution by itself.

| Experiment | RQ | Control | Changed component | Split | Artifact complete? | Include? | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `[OPEN]` | `[RQ]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | No | `[OPEN]` | `[OPEN]` |
| `[OPEN]` | `[RQ]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | No | `[OPEN]` | `[OPEN]` |

## 9. Reproducibility checklist

- [ ] Dataset version, paths, split manifests, and sample-key convention recorded.
- [ ] Exact model identifiers and revisions recorded.
- [ ] Prompt text and decoding settings stored.
- [ ] Frame count, sampling formula, endpoint policy, resolution, and timestamps recorded.
- [ ] Retrieval model, query construction, normalization, and cache fingerprint recorded.
- [ ] GPU type/count, software environment, and timing boundaries recorded.
- [ ] Resume and failure behavior documented.
- [ ] Container build inputs, immutable image digest, archive hash, and smoke-test result recorded.
- [ ] Commands for reproducing each reported table are listed.
- [ ] Secrets, access tokens, private URLs, and raw personal data excluded.

## 10. Limitations and ethics placeholders

### Limitations

`[PLACEHOLDER: temporal sampling can miss short events]`

`[PLACEHOLDER: question-conditioned retrieval can be biased by query construction]`

`[PLACEHOLDER: answer arbitration may fail when all candidates share the same error]`

`[PLACEHOLDER: development-set selection and distribution shift]`

`[PLACEHOLDER: latency, memory, and hardware constraints]`

### Ethics and responsible use

`[PLACEHOLDER: consent, privacy, and handling of egocentric recordings]`

`[PLACEHOLDER: risks of incorrect answers in assistive or safety-relevant use]`

`[PLACEHOLDER: dataset and model licenses]`

## 11. Friday/Saturday meeting agenda

1. Agree on the single main story and the two or three research questions.
2. Separate established results from hypotheses and unfinished experiments.
3. Select the primary table and the minimum supporting ablations.
4. Decide which error analysis is needed before making each claim.
5. Choose Figure 1 and one analysis figure; assign owners and data sources.
6. Review split integrity, label use, timing, and reproducibility risks.
7. Agree on paper contributions, limitations, ethics language, and what not to claim.
8. Assign writing, figure, code, website, and review tasks with dates.

**Meeting decisions:** `[OPEN]`  
**Owners and deadlines:** `[OPEN]`  
**Next review date:** `[OPEN]`

## 12. Source anchors to verify before writing

These are pointers, not permission to copy conclusions without checking the underlying artifacts:

- `RUN_LOG.md`
- `documentation/LONGQA_EVIDENCE_AUDIT_2026-07-14.md`
- `documentation/LONGQA_FINAL_DAY_ARBITRATION_2026-08-07.md`
- `documentation/LONGQA_FINAL_DAY_ENSEMBLE_AUDIT_2026-08-07.md`
- `documentation/LONGQA_MEETING_BRIEF_2026-07-30.md`
- `documentation/LONGQA_MEETING_BRIEF_2026-08-04.md`
- `documentation/LONGQA_SUBMISSION_QWEN35_ENDPOINT_2026-08-07.md`
- `documentation/reproducibility_manifests/final-image-receipt.txt`
- `test_submission/qwen35_27b_dual_view_fusion/SUBMISSION_RECEIPT.md`

Before publication, replace each source pointer with the exact result file, command, and split used.
