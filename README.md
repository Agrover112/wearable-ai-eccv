# Wearable AI — ECCV 2026 Challenge (egolongqa)

Analysis + modelling for the **EgoLongQA** task (long egocentric video → multiple-choice QA)
of the gated [`facebook/wearable-ai`](https://huggingface.co/datasets/facebook/wearable-ai)
dataset. Code and small derived artifacts live in git; videos are **streamed** from Hugging
Face and never stored (the split is ~203 GB).

- Analysis plan: [`docs/analysis_plan.md`](docs/analysis_plan.md)
- Findings (paper-draft prose): [`outputs/eda/report.md`](outputs/eda/report.md)
- Project conventions: [`CLAUDE.md`](CLAUDE.md)
- LongQA baseline implementation: [`baselines/longqa/README_PROJECT.md`](baselines/longqa/README_PROJECT.md)
- Baseline results and diagnostics: [`docs/longqa/RUN_LOG.md`](docs/longqa/RUN_LOG.md)
- Submitted dual-view method: [`docs/longqa/IMPLEMENTATION_CATALOG.md`](docs/longqa/IMPLEMENTATION_CATALOG.md#final-model)
- Full reproduction guide: [`docs/longqa/REPRODUCIBILITY_GUIDE.md`](docs/longqa/REPRODUCIBILITY_GUIDE.md)
- Final container source: [`test_submission/qwen35_27b_dual_view_fusion_v2/`](test_submission/qwen35_27b_dual_view_fusion_v2/)

## Setup

```bash
bash scripts/bootstrap.sh        # deps; on Colab also mounts Drive + sets HF cache
export HF_TOKEN=hf_xxx           # token WITH access to the gated dataset
python scripts/load_env.py       # or: resolve tokens from .env / Colab Secrets
set -a; source .env; set +a
```

Everything below reads the annotations directly from the dataset's parquet export —
no bulk video download is ever required.

## InternVideo3 on EgoLongQA

InternVideo3 plugs into the same frame sampling, direct-MCQ prompt, prediction schema, resume
logic, and evaluation code as Qwen. It uses Hugging Face/PyTorch because vLLM 0.19.1 does not
support the custom InternVideo3 architecture. Its checkpoint needs Transformers 4.57.3, so it
runs in a separate project-local environment and does not change the Qwen/vLLM installation.

Create and activate that environment from the repository root:

```bash
bash scripts/setup_internvideo3_env.sh
source scripts/activate_internvideo3_env.sh
```

The setup uses Python 3.10, PyTorch 2.10.0 with CUDA 12.8, Transformers 4.57.3, and the pinned
InternVideo3 checkpoint revision `c4602918b65225650d152db2850fe34e01d21fcd`. The Conda
environment lives in `.conda/`; package and model caches live in `.cache/`.

On the project cluster, reproduce the dev140 experiment with:

```bash
ssh slurm
cd /CT/RGCAHead3/nobackup/data/wearable-ai/wearable-ai-eccv
sbatch scripts/slurm_internvideo3_dev140.sh
```

This evaluates `yanziang/InternVideo3-8B-Instruct` on the deterministic dev140 split using one
H100, 64 uniformly sampled frames, a per-frame pixel cap of 451,584, BF16 weights, PyTorch
SDPA, batch size 1, and the baseline direct-MCQ prompt. Videos are read from
`/scratch/inf0/user/kkumar/val`. The run writes resumable predictions and the final evaluation
under `runs/egolongqa/internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13/`.

For a quick functional check, add `--max-samples 2` to the Python command in the Slurm script
and write to a separate output directory. See
[`baselines/longqa/README_PROJECT.md`](baselines/longqa/README_PROJECT.md) for the equivalent
direct command.

## Reproducing the embeddings (`features/`)

| file / dir | content | produced by |
|---|---|---|
| `question_emb_minilm.npy` | 700 questions × 384-d MiniLM | `eda_clustering.py` |
| `question_emb_siglip.npy` | 700 questions × 1152-d SigLIP-2 text tower | `eda_clustering.py` |
| `mcq_emb_minilm.npy` | 700 `mcq_options` strings × 384-d MiniLM | `eda_clustering_mcq.py` |
| `frames_siglip_32f/<video_id>.npy` | per-video **32 × 1152** SigLIP-2 frame features | `eda_frame_features.py` |
| `video_emb_siglip_32f.npy` | mean-pooled 1-vector-per-video matrix | `eda_frame_features.py` |
| `video_emb_siglip_32f_index.csv` | row → `video_id` mapping for the matrix above | `eda_frame_features.py` |

Text embeddings (CPU, minutes) — cached copies are committed, so these scripts skip
re-encoding unless the `.npy` files are deleted:

```bash
python scripts/eda_clustering.py       # questions: MiniLM + SigLIP text, clustering figs
python scripts/eda_clustering_mcq.py   # mcq_options: MiniLM, clustering figs
```

Frame features (GPU recommended; streams each mp4, samples frames, encodes, deletes the mp4):

```bash
# full split, all 700 videos (~9–11 h on a T4; safe to interrupt at ANY time —
# each finished video is its own features/frames_siglip_32f/<video_id>.npy and
# re-running skips everything already cached):
python scripts/eda_frame_features.py

# category-stratified subset (deterministic, seed 42):
python scripts/select_videos.py 150 .cache/videos150.txt
python scripts/eda_frame_features.py --video-list .cache/videos150.txt

# useful flags:
#   --frames 64            frames per video (default 32 = the challenge --max-frames cap)
#   --n-videos 100         only the first N rows (quick tests)
#   --time-budget-min 110  stop extracting after N minutes, still render figures
```

`<video_id>` = the mp4 basename without extension (e.g. `00cdb8ca10c069f8`), so features
for new videos can be added by anyone by dropping in `<video_id>.npy` files of shape
`(frames, 1152)`.

Changing `--frames` writes to a separate cache dir (`frames_siglip_<N>f/`), so different
frame counts coexist.

## Analysis scripts (each writes figures → `outputs/eda/figures/`, tables → `outputs/eda/tables/`)

| script | what it does |
|---|---|
| `eda_egolongqa.py` | Steps 1–5: structure, letter bias (always-C 63.4%), option-length bias (shortest 40.3%), temporal cues (79.1%), writes `report.md` skeleton |
| `eda_distributions.py` | question-length / first-word / temporal-cue / length-rank figures |
| `eda_clustering.py` / `eda_clustering_mcq.py` | Step 6: semantic clustering, MiniLM vs SigLIP text |
| `eda_durations.py` | video durations from mp4 **headers only** (no download, no GPU): ~10 min clips, 15 fps |
| `plot_duration_tail.py` | log-scale + bucketed duration figure (re-plots from the cached CSV) |
| `eda_shortcuts_by_category.py` | per-category strength of the C / shortest-option leaks |
| `eda_text_overlap.py` | blind-baseline ladder: odd-one-out 45.4%, lexical overlap 13.3%, answer↔option match 99.9% |
| `eda_frame_features.py` | Step 7: frame features + UMAP Fig A/B + video→question retrieval (R@1 43% vs 0.93% chance) |
| `plot_frame_features.py` | re-render Step 7 figures from cached features only (`--frames 32`) |
| `progress.sh` | one-shot progress box for a running extraction (`bash scripts/progress.sh`) |

## Repo layout

```
scripts/          all runnable analysis + infra scripts (pubstyle.py = shared figure style)
src/data.py       dataset loading + MCQ parsing helpers
features/         committed embeddings (small); per-video frame features keyed by video id
outputs/eda/      report.md + publication figures (PDF+PNG) + tables (CSV + LaTeX booktabs)
docs/             analysis plan
data/ models/     gitignored — never store videos (stream + delete)
```
