# Wearable AI — ECCV 2026 Challenge (egolongqa only)

This repo is the **portable source of truth** for my work on the ECCV 2026 Wearable AI
Challenge. It carries code + config + the recipe to rebuild the environment on any machine
(Colab, local, another Claude environment). It deliberately does **not** carry heavy binaries
(videos, model weights) — those are re-pulled from Hugging Face and cached on Google Drive.

## Scope
**Only the `egolongqa` task** (long egocentric video → multiple-choice QA). Ignore the
`egoconv` and `egoproactive` configs of the dataset.

## Task
Given a long first-person (head-mounted camera) video and a question about it, choose the
correct multiple-choice option. A free-form reference `answer` is also provided.

## Dataset — `facebook/wearable-ai`, config `egolongqa`, split `val`
- 🔒 **Gated** — needs access approval at https://huggingface.co/datasets/facebook/wearable-ai
  and an HF token with that access on every environment.
- 700 rows. Schema:
  | column | meaning |
  |---|---|
  | `video_path` | video filename, e.g. `00cdb8ca10c069f8.mp4` |
  | `question` | natural-language question |
  | `answer` | free-form reference answer |
  | `mcq_options` | `"A. ... B. ... C. ... D. ..."` |
  | `mcq_answer` | correct letter, e.g. `C` |
  | `category` | e.g. `Sightseeing`, `Travel-Tourism` |
- The dataset repo also ships a `starter_kit/` with baseline + eval scripts.

## Models (stack)
Pipeline: `frames → vision encoder → (compressor) → LLM → answer`. Pluggable encoders/models.

- **BIMBA** (primary long-video-QA model): CVPR 2025, Mamba selective-scan token compression for
  long-form VQA (built for EgoSchema/LongVideoBench/Video-MME — same space as egolongqa).
  - Code: https://github.com/md-mohaiminul/BIMBA  | Weights: `mmiemon/BIMBA-LLaVA-Qwen2-7B` (7B)
  - Base LLM: `lmms-lab/LLaVA-Video-7B-Qwen2`. Vision encoder: **SigLIP so400m**. Frames via `decord`.
  - Hard deps: `mamba-ssm` + `causal-conv1d` (CUDA-compiled, finicky on Colab) + LLaVA-NeXT.
  - VRAM: bf16 7B ≈ 15 GB → L4 (24 GB) ideal; T4 (16 GB) only with 4-bit.
- **SigLIP 2** (`google/siglip2-so400m-patch14-384`) — standalone frame feature extraction/analysis.
- **Hiera** (`facebook/hiera_base_224.mae_in1k`) — alternative image/video encoder.
- **Qwen3-VL** — all-in-one baseline to compare against. May need `transformers` from source.

NOTE: "HieraMamba" and "Bamba" were mistaken names — the intended model is **BIMBA** (above).
Real Mamba-video alternatives if needed: VideoMamba (2403.06977), MambaVision (2407.08083).

## Layout
```
scripts/bootstrap.sh      # rebuild env on ANY machine (Colab or local)
scripts/download_data.py  # pull ONLY egolongqa annotations + its videos from HF
src/data.py               # load egolongqa, resolve video paths
src/infer_qwen3vl.py      # Qwen3-VL inference over sampled frames
src/evaluate.py           # MCQ accuracy
data/  models/  outputs/  # gitignored — live on Drive cache in Colab
```

## Rebuild on a new environment
```bash
bash scripts/bootstrap.sh          # installs deps, mounts Drive (Colab), sets HF cache
export HF_TOKEN=hf_xxx             # token with gated access
python scripts/download_data.py    # fetch egolongqa split + referenced videos
```

## Storage strategy
- **In git:** everything under `scripts/`, `src/`, `notebooks/`, this file, `requirements.txt`.
- **Cached on Google Drive (Colab):** `HF_HOME` points to Drive so datasets + model weights
  persist across runtime restarts. Set by `scripts/bootstrap.sh`.
- **Never committed:** `data/`, `models/`, `outputs/`, `*.mp4`, tokens.

## Response style (important)
The user has **ADHD**. Always respond crisp and scannable — lead with the answer, use short
bullets / numbered steps, never dump dense paragraphs. Long output is fine only if chunked into
clear sections.

## Vocabulary (important)
Never invent new/made-up terms, names, or jargon. Always use existing, standard vocabulary —
the real names for tools, concepts, APIs, and dataset/model terminology — so everything stays
correct and verifiable. No coining your own labels.

## Conventions
- Keep everything egolongqa-specific; do not add egoconv/egoproactive code paths.
- Never hardcode or commit the HF token — read it from `HF_TOKEN` env or Colab Secrets.
