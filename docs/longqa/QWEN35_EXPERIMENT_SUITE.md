# Qwen3.5 EgoLongQA Experiment Suite

This suite evaluates Qwen3.5-9B without changing the established LongQA
baselines. Each experiment owns a runner, fingerprints its inputs and inference
configuration, writes resumable JSONL artifacts, and has a scheduler launcher
under `slurm_scripts/<experiment>/`.

All GPU launchers use one H100. Qwen3.5 runs in non-thinking mode unless the
provenance verifier is explicitly launched with `THINKING=1`.

## Experiment Map

| Experiment | Runner | SLURM launcher | Main artifacts |
| --- | --- | --- | --- |
| Exact frame-pack replay | `run_generate_longqa_fixed_pack.py` | `qwen35_fixed_pack_dev140/run.sh` | `predictions.jsonl`, `frame_packs.jsonl`, `results.json` |
| Native MP4 input | `run_generate_longqa_native_video.py` | `qwen35_native_video_dev140/submit_dev140_fps_array.sh` | one run directory per FPS |
| Global anchors plus dense bursts | `run_generate_longqa_dense_bursts.py` | `qwen35_dense_bursts_dev140/run.sh` | `frame_metadata.jsonl`, predictions and results |
| Operator-adaptive sampling | `run_generate_longqa_operator_adaptive.py` | `qwen35_operator_adaptive_dev140/submit.sh` | `frame_metadata.jsonl`, predictions and results |
| Provenance-aware disagreement review | `run_generate_longqa_provenance_verifier.py` | `qwen35_provenance_verifier_dev140/run.sh` | `verifier_evidence.jsonl`, predictions and results |
| Evidence-recall audit | `run_longqa_evidence_recall_audit.py` | `longqa_evidence_recall_audit_dev140/run.sh` | annotation manifest/schema, previews, recall results |

Paths in the table are relative to `baselines/longqa/` for runners and
`slurm_scripts/` for launchers.

## 1. Exact Frame-Pack Replay

Run Qwen3.5 on the deterministic uniform64 grid:

```bash
sbatch slurm_scripts/qwen35_fixed_pack_dev140/run.sh
```

Replay an existing proof pack without changing its saved frame indices:

```bash
FRAME_SOURCE=proofpack \
PROOFPACK=/absolute/path/to/proofpack.jsonl \
RUN_NAME=qwen35_9b_fixed_parent_pivot_dev140 \
sbatch slurm_scripts/qwen35_fixed_pack_dev140/run.sh
```

The proof pack must contain the same rows as the dev140 input. The runner hashes
the source file and writes a separate frame-pack audit JSONL.

## 2. Native Video

Submit the `0.25` and `0.5` FPS native-video runs as a sequential SLURM array:

```bash
bash slurm_scripts/qwen35_native_video_dev140/submit_dev140_fps_array.sh
```

The runner sends a local `file://` MP4 URL to vLLM and lets Qwen3.5 perform
native video sampling. The server uses a 128K context and resolves the
`data/videos` symlinks into an explicit local-media allowlist.

## 3. Dense Temporal Bursts

```bash
PROOFPACK=/absolute/path/to/proofpack.jsonl \
RUN_NAME=qwen35_dense_bursts_dev140 \
sbatch slurm_scripts/qwen35_dense_bursts_dev140/run.sh
```

The default pack contains 16 global anchors and up to eight parent event
centers expanded at `-2, -1, -0.33, +0.33, +1, +2` seconds. Collisions and
boundary clipping are filled deterministically to preserve a 64-frame budget.

## 4. Operator-Adaptive Sampling

```bash
PARENT_PROOFPACK=/absolute/path/to/proofpack.jsonl \
bash slurm_scripts/qwen35_operator_adaptive_dev140/submit.sh
```

The runner keeps all policy logic local:

- `GLOBAL`: uniform64;
- `AFTER` and `BEFORE`: 16 anchors plus directional pivot/target bursts;
- `FIRST` and `LAST`: 24 broad anchors plus early/late event neighborhoods;
- `STATE_CHANGE`: paired pre/post bursts.

Every saved frame records its policy, source center, temporal offset, and FPS.

## 5. Provenance-Aware Review

This run requires two complete candidate prediction files and the primary
candidate's proof pack:

```bash
PRIMARY_PREDICTIONS=/absolute/path/to/pivot/predictions.jsonl \
SECONDARY_PREDICTIONS=/absolute/path/to/uniform/predictions.jsonl \
PRIMARY_PROOFPACK=/absolute/path/to/pivot/proofpack.jsonl \
RUN_NAME=qwen35_provenance_verifier_dev140 \
sbatch slurm_scripts/qwen35_provenance_verifier_dev140/run.sh
```

Agreement rows are copied without a model call. A disagreement request contains
up to 32 primary proof-pack images and 32 secondary uniform images. The two
groups remain separately labelled with frame indices and timestamps. The
reviewer may return only `CANDIDATE_1`, `CANDIDATE_2`, or `INSUFFICIENT`;
insufficient or invalid output keeps the primary answer.

Set `THINKING=1` only for the reviewer-thinking ablation:

```bash
THINKING=1 PRIMARY_PREDICTIONS=... SECONDARY_PREDICTIONS=... \
PRIMARY_PROOFPACK=... RUN_NAME=qwen35_provenance_thinking_dev140 \
sbatch slurm_scripts/qwen35_provenance_verifier_dev140/run.sh
```

## 6. Evidence-Recall Audit

Create the manual annotation manifest without inventing event labels:

```bash
PROOFPACK=/absolute/path/to/proofpack.jsonl \
PRIMARY_PREDICTIONS=/absolute/path/to/pivot/predictions.jsonl \
SECONDARY_PREDICTIONS=/absolute/path/to/uniform/predictions.jsonl \
PREVIEW_MODE=contact_sheets \
sbatch slurm_scripts/longqa_evidence_recall_audit_dev140/run.sh
```

The artifact directory contains:

- `annotation_manifest.jsonl`;
- `annotation_schema.json`;
- optional one-FPS contact sheets or thumbnails.

After human decisive-event intervals have been written to an annotation JSONL,
score candidate-grid and final-pack recall with:

```bash
PROOFPACK=/absolute/path/to/proofpack.jsonl \
FINAL_PACK_METADATA=/absolute/path/to/frame_metadata.jsonl \
ANNOTATIONS=/absolute/path/to/decisive_event_annotations.jsonl \
RUN_NAME=longqa_evidence_recall_scored_dev140 \
sbatch slurm_scripts/longqa_evidence_recall_audit_dev140/run.sh
```

The resulting `recall_results.json` reports event recall, all-events-covered
rate, temporal-order coverage, and operator-stratified metrics.

## Recommended Order

1. Run uniform and parent-proofpack replay to isolate the Qwen3.5 model change.
2. Run the native-video FPS array independently.
3. Run dense-burst and operator-adaptive packs from the same parent proof pack.
4. Use the strongest two completed candidates in provenance-aware review.
5. Generate and annotate the recall audit for disagreements and remaining
   errors before adding another retrieval policy.

