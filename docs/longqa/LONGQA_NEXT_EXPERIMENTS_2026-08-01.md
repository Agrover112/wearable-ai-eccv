# EgoLongQA Next Experiment Suite - 2026-08-01

## Purpose

This suite tests eight agreed directions without modifying the frozen
`565/700` primary result. All new model runs use Qwen3.5-9B as the answerer.
Specialist models only select or organize visual evidence.

## Experiment Map

| # | Experiment | Scope | Launcher |
| ---: | --- | --- | --- |
| 1 | Candidate-C contribution audit | dev140, CPU | `bash scripts/run_longqa_candidate_c_audit.sh` |
| 2 | Independent uncertainty-grounded Candidate C | all dev140 | `sbatch slurm_longqa_qwen35_ug_candidate_c_dev.sh` |
| 3 | Uncertainty evidence for the existing final judge | existing ensemble disagreements | `sbatch slurm_longqa_qwen35_ug_original_multiview_rotation_dev.sh` |
| 4 | Pivot + uniform + uncertainty multi-view arbitration | same job as #3 | same launcher; inspect `rotation_multiview_summary.json` |
| 5 | Hierarchical occurrence-aware pivot | dev20 | `sbatch slurm_longqa_qwen35_hierarchical_pivot_dev20.sh` |
| 6 | HieraMamba interval grounding | dev20 | sequential chain below |
| 7 | Targeted Qwen OCR evidence | 49 gated dev140 rows | `sbatch slurm_longqa_qwen35_targeted_ocr_dev.sh` |
| 8 | SigLIP2 object re-identification | 39 gated dev140 rows | `sbatch slurm_longqa_qwen35_object_reid_dev.sh` |

Experiment 2 deliberately runs on every dev140 row. A third candidate that is
evaluated only when pivot and uniform disagree cannot recover cases where both
existing branches agree on the same wrong answer.

## Completed CPU Audit

The Candidate-C audit is already stored at
`analysis/egolongqa/candidate_c_replacement_audit_dev140_2026-08-01.json`.

- Qwen3.5 pivot + uniform majority: `115/140`; oracle: `120/140`.
- Adding the current nested Qwen3 verifier: majority `116/140`; oracle
  `128/140`.
- The current Candidate C uniquely supplies eight correct answers.
- The promoted final scorer selects an answer unique to Candidate C seven
  times, five of which are correct.
- Qwen-embedding pivot raises majority to `117/140`, but adds no oracle recall.
- Endpoint uniform raises oracle to `123/140`, below the current Candidate C's
  `128/140`.

The replacement Candidate C must therefore be judged by unique candidate
recall as well as standalone accuracy.

## What Each New Branch Does

### Uncertainty Candidate And Arbitration

The uncertainty branch samples 128 candidate frames. Qwen3.5 measures its own
answer entropy on each candidate and separately estimates whether the temporal
reference event is visible. It uses these scores to construct a 64-frame
temporal pack and produces an independent answer for every dev140 row.

The selection file from this run is reused by both arbitration jobs. Each
disagreement is scored under four cyclic option placements using:

- temporal-pivot evidence alone;
- uncertainty-selected evidence alone; and
- the mean log score from pivot, uniform, and uncertainty evidence.

### Hierarchical Pivot

The video is first divided into 16 coarse windows represented by four low-cost
frames each. Qwen3.5 ranks windows for the reference and target events. Only
the selected windows are searched frame by frame. Multiple temporally separated
occurrences are retained, before/after constraints are enforced, and the final
pack contains up to 48 local frames plus 16 global endpoint frames.

### HieraMamba

HieraMamba remains a proposal generator, not the answerer. Its stages are:

1. One-time setup: `sbatch slurm_hieramamba_bootstrap_h100.sh`
2. `sbatch slurm_hieramamba_queries_dev20.sh`
3. `sbatch slurm_hieramamba_extract_dev20.sh`
4. `sbatch slurm_hieramamba_infer_dev20.sh`
5. Audit `proposals.jsonl` and several contact sheets.
6. Run the following Qwen3.5 answer jobs in parallel:
   - `sbatch slurm_longqa_qwen35_hieramamba_top1_uniform_dev20.sh`
   - `sbatch slurm_longqa_qwen35_hieramamba_top3_uniform_dev20.sh`
   - `sbatch slurm_longqa_qwen35_hieramamba_top1_pivot_dev20.sh`

The one-time environment bootstrap in `documentation/HIERAMAMBA_EXPERIMENTS.md`
must be completed before stage 2.

### Targeted OCR

The deterministic gate selects 49 dev140 questions involving signs, names,
prices, numbers, books, labels, or other written information. Existing
question-conditioned Grounding DINO detections propose high-resolution regions.
Qwen3.5 explicitly transcribes only those regions, then receives the
transcription, full chronological frames, and detail crops before answering.
The complete transcription is cached in each prediction for inspection.

This is targeted visual transcription, not dense OCR over every frame.

### Object Re-Identification

The deterministic gate selects 39 questions involving repeated objects,
reappearances, or state changes. Grounding DINO supplies candidate boxes.
SigLIP2 embeds the object crops, and same-concept detections are clustered by
appearance. The answer prompt receives stable track IDs and the first, peak,
and last observation of each track, together with neighbouring full frames.

This is a training-free tracklet/re-identification baseline. It does not claim
that a track remains continuous through long occlusions.

## Dependency-Aware Schedule

### Can Start Immediately And Run Simultaneously

These jobs have no dependency on another new result and may run on separate
GPUs at the same time:

```bash
sbatch slurm_longqa_qwen35_ug_candidate_c_dev.sh
sbatch slurm_longqa_qwen35_hierarchical_pivot_dev20.sh
sbatch slurm_longqa_qwen35_targeted_ocr_dev.sh
sbatch slurm_longqa_qwen35_object_reid_dev.sh
sbatch slurm_hieramamba_queries_dev20.sh
```

The OCR and re-identification jobs read the same archived detections but write
to separate output directories and caches.

### Must Wait For Uncertainty Candidate C

After `slurm_longqa_qwen35_ug_candidate_c_dev.sh` completes, these two jobs can
run simultaneously:

```bash
sbatch slurm_longqa_qwen35_ug_original_multiview_rotation_dev.sh
sbatch slurm_longqa_qwen35_ug_candidate_c_multiview_rotation_dev.sh
```

The first retains the current Qwen3 verifier as Candidate C and changes only
the judge evidence. The second replaces Candidate C with the uncertainty
branch and evaluates the resulting majority and arbitration policies.

### HieraMamba Must Remain Sequential Until Proposals Exist

Query conversion must finish before EgoVLP extraction. Extraction must finish
before HieraMamba inference. The three Qwen3.5 answer variants may run together
only after normalized proposals have been produced and visually audited.

## Promotion Checks

- Candidate C: compare unique correct proposals and three-candidate oracle to
  the current `128/140` oracle, not only majority accuracy.
- Arbitration: compare fixes and regressions against the current `120/140`
  promoted dev result.
- Hierarchical and HieraMamba pilots: inspect interval quality before any
  dev140 expansion.
- OCR and re-identification: compare against Qwen3.5 pivot on exactly the same
  gated rows, then inspect whether their unique fixes improve an overlay or
  candidate oracle.
- No branch moves to val560 unless its rule is frozen before that evaluation.
