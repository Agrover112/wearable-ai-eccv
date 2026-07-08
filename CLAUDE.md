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

## Model
**Qwen3-VL** family (video-capable VLM). May require `transformers` from source. See
`requirements.txt`. Frame sampling over long clips is the key design choice — see
`src/infer_qwen3vl.py`.

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

## Conventions
- Keep everything egolongqa-specific; do not add egoconv/egoproactive code paths.
- Never hardcode or commit the HF token — read it from `HF_TOKEN` env or Colab Secrets.
