# Open-VLM LongQA Evaluation Runbook (10 August 2026)

## Purpose

These jobs test whether a video-specialized open model adds genuinely different
correct answers to the current Qwen3.5-27B endpoint baseline. Every model first
runs on the same five examples. A larger balanced fold is allowed only when all
five responses parse correctly and each model call stays below 300 seconds.

All vLLM jobs use endpoint-inclusive chronological sampling, native video input,
the unchanged baseline multiple-choice prompt, deterministic decoding, and one
H100. This isolates model choice from prompt and sampling changes.

## Smoke Tests

Run these first. They are independent and may be queued simultaneously when GPUs
are available:

```bash
sbatch slurm_longqa_minicpm_v45_video192_smoke5.sh
sbatch slurm_longqa_molmo2_8b_video256_smoke5.sh
sbatch slurm_longqa_internvl35_30b_a3b_flash_video64_smoke5.sh
sbatch slurm_longqa_cosmos_reason2_8b_video128_smoke5.sh
```

MiniCPM-V is the first priority because it is explicitly designed for many-frame
video input. Molmo2 is second because it supports long native-video inputs and
may supply useful diversity. InternVL Flash tests a sparse 30B-scale alternative.
Cosmos-Reason2 is lower priority because it is derived from Qwen and is therefore
less likely to produce complementary errors.

NVILA uses a separate AutoGaze environment:

```bash
sbatch slurm_setup_nvila_autogaze.sh
# Submit only after setup_nvila_autogaze_<job>.out reports cuda=True.
sbatch slurm_longqa_nvila_hd_video_smoke5.sh
```

NVILA's model card uses non-commercial Creative Commons terms. Confirm that this
license is eligible for the competition before using its predictions in a test
submission.

## Promotion to Balanced Fold 2

Only promote models whose smoke log ends with `SMOKE PASS`:

```bash
sbatch --export=ALL,OPEN_VLM_KEY=minicpm slurm_longqa_open_vlm_fold2.sh
sbatch slurm_longqa_molmo2_8b_video256_fold2.sh
sbatch --export=ALL,OPEN_VLM_KEY=internvl slurm_longqa_open_vlm_fold2.sh
sbatch --export=ALL,OPEN_VLM_KEY=cosmos slurm_longqa_open_vlm_fold2.sh
sbatch slurm_longqa_nvila_hd_video_fold2.sh
```

The launchers verify the corresponding smoke artifact before starting. Fold-2
predictions are written under `runs/egolongqa/<run-name>/predictions.jsonl`, with
accuracy and category diagnostics beside them.

## Decision Rule

Do not replace the 618/700 endpoint baseline from a five-example result. On the
140-example fold, require either a clear standalone improvement or complementary
correct answers on existing endpoint errors. Only then is a full 700-example run
or a constrained disagreement ensemble justified.
