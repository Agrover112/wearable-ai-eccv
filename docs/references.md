# References

✅ = verified this session (HF/web) · ⚠️ = named but not yet pinned/fetched

## Dataset
- ✅ **facebook/wearable-ai** (ECCV 2026 Wearable AI, `egolongqa`) — https://huggingface.co/datasets/facebook/wearable-ai

## Models
| Model | Use | Link |
|---|---|---|
| ✅ BIMBA (weights, 7B) | primary long-video QA | https://huggingface.co/mmiemon/BIMBA-LLaVA-Qwen2-7B |
| ✅ BIMBA base LLM | — | https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2 |
| ✅ SigLIP 2 | frame/text features (Step 7/8) | https://huggingface.co/google/siglip2-so400m-patch14-384 |
| ✅ Hiera | alt vision encoder | https://huggingface.co/facebook/hiera_base_224.mae_in1k |
| ✅ MiniLM (SBERT) | question clustering (Step 6) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| ⚠️ Qwen3-VL | baseline VLM | family — specific repo not pinned yet |

## Papers
- ✅ **BIMBA: Selective-Scan Compression for Long-Range Video QA** (CVPR 2025) — https://arxiv.org/abs/2503.09590
- ✅ **EgoCross: Benchmarking MLLMs for Cross-Domain Egocentric Video QA** (Fig 4 t-SNE idea) — https://arxiv.org/abs/2508.10729
- ✅ **Revisiting the "Video" in Video-Language Understanding** — ATP / single-frame bias, Buch et al., **CVPR 2022 (Oral)** — https://arxiv.org/abs/2206.01720 · code https://github.com/StanfordVL/atp-video-language  *(basis for Step 8)*
- ✅ VideoMamba — https://arxiv.org/abs/2403.06977
- ✅ Video Mamba Suite — https://arxiv.org/abs/2403.09626
- ✅ MambaVision — https://arxiv.org/abs/2407.08083

## Code
- ✅ BIMBA — https://github.com/md-mohaiminul/BIMBA · homepage https://sites.google.com/view/bimba-mllm

## Libraries / methods
- **Libraries:** sentence-transformers (SBERT), umap-learn, scikit-learn, matplotlib, huggingface_hub,
  datasets, pandas/pyarrow, transformers, torch, decord, mamba-ssm, causal-conv1d, wandb
- **Methods:** UMAP, t-SNE, KMeans, TF-IDF / c-TF-IDF (cluster naming)

## Corrected (named but not used)
- **Bamba** (IBM hybrid Mamba-2 text LLM) — real, but text-only → wrong for frame encoding
- **"HieraMamba"** — no such model/paper found; the intended model was **BIMBA**
