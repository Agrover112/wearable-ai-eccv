# HieraMamba EgoLongQA Pipeline

HieraMamba is used only for timestamp proposals. Qwen still receives the
original multiple-choice question and produces the answer. The pipeline keeps
query conversion, dense feature extraction, grounding, and answering as
separate resumable stages.

## One-Time Setup

Run the bootstrap on an H100 compute node with internet access:

```bash
bash hieramamba/bootstrap_h100.sh
```

The script builds `causal-conv1d`, `mamba-ssm`, and HieraMamba's NMS extension
for the active H100 stack. The old Colab T4 wheels must not be reused.

Record the three paths printed by the script:

```bash
export HIERAMAMBA_ENV_PREFIX=/scratch/inf0/user/agaur/wai-26/external/envs/hieramamba-h100
export HIERAMAMBA_ROOT=/scratch/inf0/user/agaur/wai-26/external/hieramamba
export EGOVLP_ROOT=/scratch/inf0/user/agaur/wai-26/external/EgoVLP
```

The Slurm scripts accept either the environment name through
`HIERAMAMBA_ENV` or the full prefix through an activated shell. If a prefix is
used, set `HIERAMAMBA_ENV` to that prefix.

## Grounding Stages

Run these sequentially:

```bash
sbatch slurm_hieramamba_queries_dev20.sh
sbatch --export=ALL,HIERAMAMBA_ENV="$HIERAMAMBA_ENV_PREFIX",EGOVLP_ROOT="$EGOVLP_ROOT" \
  slurm_hieramamba_extract_dev20.sh
sbatch --export=ALL,HIERAMAMBA_ENV="$HIERAMAMBA_ENV_PREFIX",HIERAMAMBA_ROOT="$HIERAMAMBA_ROOT" \
  slurm_hieramamba_infer_dev20.sh
```

1. Query conversion sees only the question and creates one to four declarative
   visual-event queries.
2. EgoVLP extracts the official dense 32-frame/8-frame-stride feature timeline.
   `--target-clips` is intentionally not used.
3. HieraMamba predicts five intervals per query. The final stage writes a
   row-aligned `proposals.jsonl` with intervals in seconds and relation
   metadata.

Before running Qwen, audit contact sheets for at least five samples. In
particular, reject a setup where top-five intervals cover most of the video.

## Initial Answer Experiments

After `proposals.jsonl` has been audited, run:

```bash
sbatch slurm_longqa_hieramamba_top1_uniform_dev20.sh
sbatch slurm_longqa_hieramamba_top3_uniform_dev20.sh
sbatch slurm_longqa_hieramamba_top1_pivot_dev20.sh
```

These use 48 frames from grounded intervals and 16 full-video anchors. Top-1
versus top-3 measures interval recall without changing the final frame budget.
The pivot experiment prioritizes frames from the existing cached temporal-pivot
pack that fall inside HieraMamba intervals.

Run Dynamic TCoT only if interval localization is credible:

```bash
sbatch slurm_longqa_hieramamba_top1_dynamic_tcot_dev20.sh
```

It selects from 128 candidates inside top-1 intervals, then supplies at most 48
selected frames plus 16 global anchors. It is a conditional ablation, not the
default HieraMamba answer path.
