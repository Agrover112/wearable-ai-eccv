# Project website content map

**Status:** planning only  
**Purpose:** map the future project website without writing a finished narrative or inventing results.

## 1. Site principles

- Lead with the task and the system, not a marketing claim.
- Show only results that have a checked source artifact and a clear denominator.
- Keep exploratory, failed, and unfinished experiments visibly separate from the final comparison.
- Do not publish raw videos, private logs, credentials, registry tokens, or personally identifying data.
- Every result card, chart, and interactive view must link to its source file and status.

## 2. Page and section map

| Page / section | Purpose | Content slots | Source / asset | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| Home | State the task and project scope in a few lines | `[TITLE]`, `[ONE-SENTENCE TASK DESCRIPTION]`, `[HERO PLACEHOLDER]` | `[OPEN]` | `[OPEN]` | Planned |
| Task | Explain the input, question format, options, and evaluation | `[DIAGRAM]`, `[DATASET DESCRIPTION]`, `[METRIC]` | `RUN_LOG.md`, dataset documentation | `[OPEN]` | Planned |
| Method | Show the selected end-to-end pipeline | `[PIPELINE FIGURE]`, `[SHORT STEP LABELS]` | `[OPEN: FIGURE 1 SOURCE]` | `[OPEN]` | Planned |
| Experiments | Present the controlled comparison | `[RESULT TABLE]`, `[ABLATION TABLE]`, `[SPLIT NOTE]` | `RUN_LOG.md`, checked result files | `[OPEN]` | Planned |
| Evidence analysis | Explain what frames and errors were inspected | `[TIMELINE VIEW]`, `[ERROR CATEGORIES]`, `[CONTACT SHEET PLACEHOLDER]` | Evidence audit and scratch assets | `[OPEN]` | Planned |
| Reproducibility | Explain how to rerun the software and image | `[REPOSITORY LINK]`, `[ENVIRONMENT]`, `[CONTAINER DIGEST]`, `[COMMANDS]` | Build provenance and submission receipt | `[OPEN]` | Planned |
| Research document | Host the paper or extended abstract | `[PDF LINK]`, `[HTML TEXT]`, `[BIBLIOGRAPHY]` | Paper workspace | `[OPEN]` | Planned |
| Team / acknowledgements | List contributors and permitted acknowledgements | `[NAMES]`, `[AFFILIATIONS]` | Team-provided text | `[OPEN]` | Planned |

## 3. Home page skeleton

```text
[PROJECT TITLE PLACEHOLDER]
[ONE-SENTENCE DESCRIPTION OF EGO-LONG-HORIZON VIDEO QA]

[HERO IMAGE OR PIPELINE DIAGRAM PLACEHOLDER]

[LINK: Method] [LINK: Results] [LINK: Reproducibility] [LINK: Paper]
```

Do not add a performance number here until the team selects the evaluation split, denominator, and source artifact.

## 4. Method page skeleton

```text
[VIDEO INPUT]
       |
       v
[FRAME CANDIDATES]
       |
       +--> [GLOBAL COVERAGE PATH]
       |
       +--> [QUESTION-CONDITIONED EVIDENCE PATH, IF SELECTED]
       |
       v
[CHRONOLOGICAL EVIDENCE PACK]
       |
       v
[QWEN ANSWER GENERATION]
       |
       v
[OPTIONAL CANDIDATE COMPARISON / VERIFICATION]
       |
       v
[ANSWER]
```

For each box, fill only:

| Step | Plain-language description | Exact implementation file | Artifact | Status |
| --- | --- | --- | --- | --- |
| Input | `[OPEN]` | `[PATH]` | `[PATH]` | Open |
| Candidates | `[OPEN]` | `[PATH]` | `[PATH]` | Open |
| Selection | `[OPEN]` | `[PATH]` | `[PATH]` | Open |
| Packing | `[OPEN]` | `[PATH]` | `[PATH]` | Open |
| Answering | `[OPEN]` | `[PATH]` | `[PATH]` | Open |
| Verification | `[OPEN]` | `[PATH]` | `[PATH]` | Open |

## 5. Results page slots

### Main result table

| Method | Model | Sampling / evidence | Split | Accuracy | Runtime | Link to source |
| --- | --- | --- | --- | ---: | ---: | --- |
| `[BASELINE]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` |
| `[PRIMARY]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` |

### Ablation table

| Ablation | What changes | What stays fixed | Split | Result | Source |
| --- | --- | --- | --- | ---: | --- |
| `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[OPEN]` | `[PATH]` |

### Results copy placeholder

`[PLACEHOLDER: two or three sentences describing only the checked table. Include split, denominator, and whether the result is standalone, routed, or diagnostic.]`

### Status labels

- **Checked:** source artifact, split, denominator, and computation reviewed.
- **Provisional:** result exists but still needs review.
- **Exploratory:** useful for context; not a promoted result.
- **Failed / incomplete:** retained for provenance, not compared as a completed method.

## 6. Evidence analysis page

### Timeline viewer placeholder

```text
[VIDEO ID / QUESTION]
[TIMELINE AXIS WITH TIMESTAMPS]
[GLOBAL FRAMES]
[RETRIEVED FRAMES]
[FINAL PACK]
[ANSWER / GOLD LABEL: only where sharing is approved]
```

**Backing artifact:** `[OPEN: evidence audit manifest or approved derived asset]`  
**Privacy review:** `[OPEN]`  
**Do not publish:** raw video, unreviewed frames, private paths, or personal information.

### Error analysis slots

| Analysis | Question | Required data | Output | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| Temporal coverage | `[OPEN]` | `[OPEN]` | `[PLOT PLACEHOLDER]` | `[OPEN]` | Planned |
| Distractor evidence | `[OPEN]` | `[OPEN]` | `[CONTACT SHEET PLACEHOLDER]` | `[OPEN]` | Planned |
| Repeated instances | `[OPEN]` | `[OPEN]` | `[TIMELINE PLACEHOLDER]` | `[OPEN]` | Planned |
| Candidate disagreement | `[OPEN]` | `[OPEN]` | `[TABLE PLACEHOLDER]` | `[OPEN]` | Planned |

## 7. Reproducibility page

### Environment slots

| Item | Value / link | Verification status | Owner |
| --- | --- | --- | --- |
| Repository revision | `[OPEN]` | `[OPEN]` | `[OPEN]` |
| Dataset version | `[OPEN]` | `[OPEN]` | `[OPEN]` |
| Model identifiers | `[OPEN]` | `[OPEN]` | `[OPEN]` |
| Python / CUDA / framework | `[OPEN]` | `[OPEN]` | `[OPEN]` |
| GPU configuration | `[OPEN]` | `[OPEN]` | `[OPEN]` |
| Container digest | `[OPEN]` | `[OPEN]` | `[OPEN]` |
| Local smoke-test record | `[OPEN]` | `[OPEN]` | `[OPEN]` |

### Reproduction command slots

```text
[COMMAND 1: environment setup]
[COMMAND 2: cache or asset preparation]
[COMMAND 3: evaluation]
[COMMAND 4: scoring / table generation]
```

Commands should be checked on a clean environment before publication. Never put access tokens or private registry credentials in the website.

## 8. Asset inventory

| Asset | Type | Description | Existing artifact | Web-safe copy needed? | License / privacy check | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pipeline diagram | SVG / PNG | `[OPEN]` | `[PATH]` | Yes | `[OPEN]` | `[OPEN]` | Planned |
| Result chart | SVG / PNG | `[OPEN]` | `[PATH]` | Yes | `[OPEN]` | `[OPEN]` | Planned |
| Evidence contact sheet | PNG | `[OPEN]` | Scratch audit directory | Review first | `[OPEN]` | `[OPEN]` | Planned |
| Timeline example | PNG / interactive | `[OPEN]` | `[PATH]` | Review first | `[OPEN]` | `[OPEN]` | Planned |
| Container diagram | SVG / PNG | `[OPEN]` | `[PATH]` | Yes | `[OPEN]` | `[OPEN]` | Planned |
| Paper PDF | PDF | `[OPEN]` | `[PATH]` | Yes | `[OPEN]` | `[OPEN]` | Planned |

## 9. Interactive ideas backed by existing artifacts

These are ideas only. Build them only after the referenced artifact is checked and approved for publication.

| Interactive | Existing backing | Minimum safe implementation | Open question | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| Timeline frame scrubber | Evidence-audit manifest and derived frame sheets | Scrub approved derived images with timestamps; no raw video | Which examples are safe to publish? | `[OPEN]` | Candidate |
| Candidate comparison viewer | Archived prediction JSONL and aligned keys | Select a question and show candidate answers plus approved evidence metadata | Can answer labels be shown publicly? | `[OPEN]` | Candidate |
| Result-table filters | Checked run ledger | Filter by method, split, resolution, or frame count | Which fields are stable across runs? | `[OPEN]` | Candidate |
| Runtime breakdown | Diagnostics and latency summaries | Display stage timings with hardware and timing scope | Are all stages measured comparably? | `[OPEN]` | Candidate |
| Container provenance panel | Build provenance, archive hash, and immutable digest | Show hashes and reproducibility metadata, never credentials | Which digest is final and public? | `[OPEN]` | Candidate |

Do not implement an interactive demo that implies causal conclusions not established by the corresponding experiment.

## 10. Accessibility checklist

- [ ] Every image has meaningful alt text or is marked decorative.
- [ ] Pipeline and timeline graphics have a text equivalent.
- [ ] Charts include a data table or downloadable tabular representation.
- [ ] Color is not the only way to distinguish frame roles or outcomes.
- [ ] Keyboard navigation works for menus, filters, viewers, and tables.
- [ ] Focus states are visible.
- [ ] Text and controls meet contrast requirements.
- [ ] Motion can be paused or disabled.
- [ ] Captions or transcripts are provided for any video or audio.
- [ ] Tables have headers and remain readable on small screens.
- [ ] Links and buttons have descriptive names.
- [ ] No private paths, secrets, or raw personal data are exposed in accessible text.

## 11. Content review checklist

- [ ] Main story approved in the Friday/Saturday meeting.
- [ ] Every numerical claim appears in the paper evidence ledger.
- [ ] Split and denominator are visible beside every result.
- [ ] Development-set tuning is separated from held-out evaluation.
- [ ] Failed and incomplete runs are labelled accurately.
- [ ] Figures have source artifacts, captions, and owners.
- [ ] Model and dataset licenses are checked.
- [ ] Privacy review is complete for every image or video-derived asset.
- [ ] Website text has been reviewed for plain language and unsupported claims.

## 12. Ownership and status board

| Work item | Owner | Reviewer | Dependency | Due date | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Choose paper story | `[OPEN]` | `[OPEN]` | Meeting | `[OPEN]` | Not started | `[OPEN]` |
| Verify main result table | `[OPEN]` | `[OPEN]` | Run artifacts | `[OPEN]` | Not started | `[OPEN]` |
| Select public evidence examples | `[OPEN]` | `[OPEN]` | Privacy review | `[OPEN]` | Not started | `[OPEN]` |
| Draw pipeline figure | `[OPEN]` | `[OPEN]` | Method decision | `[OPEN]` | Not started | `[OPEN]` |
| Build results page | `[OPEN]` | `[OPEN]` | Result table | `[OPEN]` | Not started | `[OPEN]` |
| Build reproducibility page | `[OPEN]` | `[OPEN]` | Environment audit | `[OPEN]` | Not started | `[OPEN]` |
| Review accessibility | `[OPEN]` | `[OPEN]` | Website draft | `[OPEN]` | Not started | `[OPEN]` |

## 13. Decisions to make at the meeting

1. What is the one-sentence project story?
2. Which research questions are in scope?
3. Which results are sufficiently checked to publish?
4. Which two or three figures will answer those questions?
5. Which derived frames, tables, and logs may be shared publicly?
6. Who owns the paper, website, figures, artifact release, and final review?
