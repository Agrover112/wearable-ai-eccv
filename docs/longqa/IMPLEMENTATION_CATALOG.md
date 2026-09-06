# Implementation catalog

This catalog maps each implementation family to its code. It supplements
`REPRODUCIBILITY_GUIDE.md`, which contains environment, model, dataset, and
container instructions. A file appearing here does not mean that its experiment
completed or improved accuracy; consult `RUN_LOG.md` and the archived run before
reporting a result.

## Shared evaluation path

| File | Responsibility |
|---|---|
| `data/wearable-ai/starter_kit/run_evaluation.py` | Unified command-line entry point for LongQA, ConvQA, and Proactive generation and evaluation. |
| `data/wearable-ai/starter_kit/model.py` | Model registry, video decoding, frame extraction, prompts, Qwen/Llama backends, vLLM startup, answer parsing, and diagnostics. |
| `data/wearable-ai/starter_kit/longqa_utils.py` | Stable sample keys, answer parsing, timing, context statistics, and common LongQA helpers. |
| `data/wearable-ai/starter_kit/run_generate_longqa.py` | Standard multiple-choice LongQA generation loop. |
| `data/wearable-ai/starter_kit/analyze_longqa_predictions.py` | Accuracy, rolling accuracy, and category summaries for complete or partial runs. |
| `data/wearable-ai/starter_kit/slurm_runner.py` | Shards generation across Slurm nodes and merges the completed shards. |

The task-specific generators for ConvQA and Proactive are
`run_generate_convqa.py` and `run_generate_proactive.py`. They were not part of
the final EgoLongQA submission.

## Final model

`longqa_dual_view_fusion.py` implements the submitted method. It builds two
64-frame views from 128 endpoint-inclusive candidates:

1. The global view is the exact endpoint-inclusive 64-frame grid.
2. The question-conditioned view uses SigLIP2 to retrieve likely evidence,
   adds temporal neighbors and global context, removes duplicates, and restores
   chronological order.
3. Qwen3.5-27B answers both views with the same question, options, and prompt.
4. Matching answers are returned directly. On disagreement, the model obtains
   an A-D probability distribution from each view, averages the two
   distributions, and returns the highest-scoring option.

`model.py` registers this implementation as `qwen35_dual_view_fusion`. The
container recovery scripts under
`test_submission/qwen35_27b_dual_view_fusion_v2/` stage the source and the exact
Qwen3.5-27B and SigLIP2 revisions, repair the compiler dependency, test the
image, and push the verified archive.

## Frame selection and temporal evidence

| File | Method tested |
|---|---|
| `run_generate_longqa_grounded.py` | CLIP, SigLIP, or SigLIP2 image-text retrieval followed by temporal pivots, neighbors, anchors, and global frames. |
| `run_generate_longqa_proofpack.py` | Cached structured evidence packs, including event windows, option-contrastive retrieval, anchors, and temporal operators. |
| `run_generate_longqa_hierarchical_pivot.py` | Coarse-to-fine retrieval over windows, repeated occurrences, and local frame expansion. |
| `run_generate_longqa_uncertainty.py` | Qwen-native frame or segment uncertainty measured from next-token entropy, then used for evidence allocation. |
| `run_generate_longqa_tcot.py` | Single-step and dynamic-segment Temporal Chain of Thought, where Qwen selects temporal regions before answering. |
| `run_generate_longqa_timeline.py` | A generated chronological summary combined with visual evidence. |
| `run_generate_longqa_narrative_gate.py` | Timestamped narrative anchors used to gate evidence. |
| `run_generate_longqa_event_ledger.py` | Cached chronological event descriptions used as a searchable intermediate representation. |
| `run_generate_longqa_event_ledger_detector.py` | Event ledger augmented with concept-conditioned detections. |
| `run_generate_longqa_verified_bursts.py` | Coarse Qwen-scored windows expanded into short event bursts without using SigLIP as the anchor scorer. |
| `run_generate_longqa_density_ablation.py` | Matched comparison of raw and visually deduplicated evidence packs. |
| `run_answer_longqa_selection.py` | Runs the answerer on frame indices produced by a separate selector. |

`run_generate_longqa_hieramamba.py` remains as an adapter for normalized
HieraMamba proposals. Its packages and checkpoints were intentionally removed;
the code is retained to document the attempted integration.

## Object, crop, OCR, and reranking paths

| File | Method tested |
|---|---|
| `run_generate_longqa_object_hints.py` | Adds open-vocabulary object detections, crops, panels, or object ledgers to cached evidence. |
| `run_generate_longqa_specialist_evidence.py` | Runs targeted OCR or repeated-object/re-identification evidence branches only for relevant questions. |
| `run_generate_longqa_qwen_rerank.py` | Uses SigLIP2 for broad recall, then asks a Qwen vision model to rerank short temporal windows. |
| `run_generate_longqa_nvila.py` | Experimental NVILA-HD-Video and AutoGaze path. Its environment is preserved as a manifest, not as a live environment. |
| `export_grounding_frames.py` | Exports retrieved frames and metadata for human evidence audits without rerunning the VLM. |

## Answering, controls, and ablations

| File | Method tested |
|---|---|
| `run_generate_longqa_video_blind.py` | Language-only baseline using the question and options without frames. |
| `run_generate_longqa_single_frame.py` | Central-frame-only visual baseline. |
| `run_generate_longqa_openqa.py` | Generates an answer without showing options, then maps it back to an option. |
| `run_generate_longqa_likelihood.py` | Scores each option by video-conditioned likelihood minus video-blind likelihood. |
| `run_generate_longqa_conditional.py` | Runs an expensive intervention only on examples where strong candidates disagree. |

## Verification and fusion

| File | Method tested |
|---|---|
| `run_generate_longqa_verifier.py` | Reconsiders disagreements between two completed answer runs. |
| `run_generate_longqa_multicandidate_judge.py` | Gives a larger multimodal model several candidate answers and common visual evidence. |
| `run_generate_longqa_hypothesis_judge.py` | Scores whether fixed candidate hypotheses are visibly supported by the supplied evidence. |
| `run_generate_longqa_videojudge.py` | Uses the VideoJudge checkpoint to score candidate answers against video evidence. |
| `run_generate_longqa_semantic_likelihood.py` | Scores complete candidate answer text instead of only option letters. |
| `run_generate_longqa_pairwise_tournament.py` | Compares candidates pairwise in both presentation orders. |
| `run_score_longqa_option_rotation.py` | Scores each semantic answer under cyclic option rotations to measure and reduce position bias. |
| `run_score_longqa_disagreements.py` | Produces evidence-view scores for disagreements among independent candidates. |
| `run_score_longqa_evidence_rank_fusion.py` | Combines option rankings or normalized option probabilities from endpoint and option-quota views. |
| `run_score_longqa_segment_fusion.py` | Scores separate chronological frame blocks before combining their option evidence. |
| `run_score_longqa_temporal_contrast.py` | Penalizes an answer when its support appears on the wrong side of a before/after pivot. |
| `run_score_longqa_sparse_option_support.py` | Scores each option against a small option-specific evidence pack. |
| `run_score_longqa_clause_support.py` | Scores complete options or clauses using identical frames. |
| `run_score_longqa_candidate_text_disagreements.py` | Compares complete candidate texts on a fixed disagreement set. |
| `run_score_longqa_full_evidence_pairwise.py` | Compares one baseline and one challenger using their exact recorded evidence. |
| `run_score_longqa_third_evidence_view.py` | Adds an independent third view to cached endpoint and option-quota features. |

These scripts include retrospective diagnostics, direct model runs, and
label-free fusion methods. They must not be described interchangeably. The
paper evidence ledger should mark each reported number as standalone, fused,
oracle, routed, or retrospective.

## Configuration and launch records

- `configs/` contains split definitions, including the development subsets.
- Historical root-level `slurm_*.sh` launch records are preserved in
  `archives/workspace_cleanup_2026-09-01/root_slurm_scripts.tar.gz`. Their
  original paths are listed in the adjacent manifest, and their command-line
  flags are explained in `documentation/LONGQA_EXPERIMENT_FLAGS.md`.
- `RUN_LOG.md` is the chronological experiment ledger.
- `runs/egolongqa/` is the human-facing archive of predictions and diagnostics.
  Historical vLLM text logs were compressed separately; structured diagnostics
  and the final v3 smoke logs remain unpacked.
- `data/wearable-ai/starter_kit/output/egolongqa/` contains working outputs and
  caches; some experiments exist in both locations.
- `analysis/` contains comparisons, disagreement studies, and error audits.

## Reproduction records

The generated files under `documentation/reproducibility_manifests/` capture:

- the exact Conda and pip state of the main environment;
- the removed NVILA/AutoGaze environment;
- local model revisions before cache pruning;
- dataset file inventory and metadata checksums;
- hashes of starter-kit and container source files;
- final container package, command, environment, and GPU diagnostics;
- the archive receipt, smoke marker, and immutable ECR reference.

Run `scripts/maintenance/capture_wearable_ai_reproducibility.sh` before a future
environment or cache change. It records metadata only and does not copy tokens
or private registry credentials.
