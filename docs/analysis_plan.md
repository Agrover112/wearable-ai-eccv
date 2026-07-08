# EgoLongQA — Data Analysis Plan

Analysis of the `egolongqa/val` annotations (700 rows). Scope = **analysis only**
(understanding the data). No VQA training/inference. Encoders (SigLIP) are used only as
analysis probes.

## 🟢 Text/label analysis — no video, no models  ✅ DONE
- **Step 1 — Structure:** rows, unique videos, questions/video, category distribution
- **Step 2 — Answer balance:** MCQ letter bias, #options
- **Step 3 — Text stats:** question/answer/option lengths, first words
- **Step 4 — Temporal reasoning:** % needing event ordering (after/before/first…)
- **Step 5 — Text shortcut checks:** option-length bias

**Findings so far (video-blind shortcuts):**
| shortcut | blind baseline |
|---|---:|
| Always answer **C** | 63.4% |
| Always pick **shortest** option | 40.3% |
| (chance) | 25% |
Also: 79.1% of questions are temporal; length & letter shortcuts are ~independent.
→ code: `scripts/eda_egolongqa.py`, `scripts/eda_distributions.py` · outputs: `outputs/eda/`

## 🟡 Semantic clustering — small text embedder (CPU)  ⏳ TODO
- **Step 6:** embed 700 questions → UMAP → cluster → name themes (objects/places/actions…).
  Shows what skills the task really tests.

## 🟠 Feature-space visualization — SigLIP + frames  ⏳ TODO
- **Step 7:** EgoCross Fig-4 style. Sample ~8 frames/video → SigLIP → pool → 1 vec/video;
  Q/A → SigLIP text. UMAP, color by category. Shows category separability + text↔video alignment.
  Streams video from HF (never stored); subset (~150) on CPU, full run on GPU.

## 🔴 Video-side shortcut probes — SigLIP (encoder only)  ⏳ TODO  ← Step 8
Can the answer be guessed from the **video** without reasoning? (video analog of Steps 2/5)
- **Option-matches-video:** SigLIP similarity of each MCQ option to the frames — does the
  highest-similarity option = correct, above 25%? (answer without reading the question)
- **Single-frame bias:** does 1 frame score ≈ all frames? (is the "long video" decorative?)
- **Temporal necessity:** shuffle frames — does accuracy drop? If not, order doesn't matter.
- **First/last-frame bias:** does last-frame-only match the answer (egocentric end-position)?
Reference: "single-frame / atemporal" video-QA bias literature. Encoder-only = still analysis.

---
**Status:** Steps 1–5 ✅ · Steps 6, 7, 8 queued. Build Step 7 (SigLIP infra) → reuse it for Step 8.
