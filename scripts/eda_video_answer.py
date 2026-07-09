#!/usr/bin/env python
"""EgoLongQA — video vs. free-form ANSWER in the shared SigLIP space (Fig C).

Companion to eda_frame_features.py (video vs. question). Encodes the free-form
`answer` field with the SigLIP-2 text tower (CPU is fine), then:
  - joint UMAP of pooled video vectors + answer vectors, colored by category
  - video->answer retrieval (R@1/5/10, median rank)

Caches features/answer_emb_siglip.npy (700 rows, annotation order).

    python scripts/eda_video_answer.py --frames 32
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_frame_features as S
from pubstyle import set_style, df_to_booktabs

OUT = S.OUT


def encode_answers_siglip(texts):
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(S.SIGLIP)
    model = AutoModel.from_pretrained(S.SIGLIP).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            inp = proc(text=texts[i:i+32], return_tensors="pt", padding="max_length",
                       max_length=64, truncation=True)
            tf = model.get_text_features(**inp)
            if not torch.is_tensor(tf):
                po = getattr(tf, "pooler_output", None)
                tf = po if po is not None else tf.last_hidden_state.mean(1)
            out.append(torch.nn.functional.normalize(tf, dim=-1).cpu().numpy())
            print(f"\r  encoding answers {min(i+32, len(texts))}/{len(texts)}",
                  end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    a = ap.parse_args()
    S.FRAMES = a.frames
    S.CACHE_DIR = S.FEAT / f"frames_siglip_{a.frames}f"

    set_style()
    df = S.load_annotations()

    # answers: encode once, cache in annotation order
    cache = S.FEAT / "answer_emb_siglip.npy"
    if cache.exists():
        ans_all = np.load(cache)
        S.log(f"loaded cached answer embeddings {ans_all.shape}")
    else:
        S.log("encoding 700 free-form answers with SigLIP text tower (CPU)...")
        ans_all = encode_answers_siglip(df["answer"].tolist())
        np.save(cache, ans_all)
        S.log(f"cached -> {cache}")

    # videos: pooled vectors from the per-video cache
    files = sorted(S.CACHE_DIR.glob("*.npy"))
    names = [f.stem + ".mp4" for f in files]
    frames = np.stack([np.load(f) for f in files]).astype(np.float32)
    vpool = frames.mean(axis=1)
    vpool = vpool / (np.linalg.norm(vpool, axis=1, keepdims=True) + 1e-8)
    S.log(f"pooled {len(names)} cached videos, {a.frames} frames each")

    idx = {os.path.basename(p): i for i, p in enumerate(df["video_path"])}
    keep = [n for n in names if n in idx]
    vpool = np.stack([vpool[names.index(n)] for n in keep])
    ans = np.stack([ans_all[idx[n]] for n in keep])
    ans = ans / (np.linalg.norm(ans, axis=1, keepdims=True) + 1e-8)
    cats = [df["category"].iloc[idx[n]] for n in keep]

    import umap
    nn = max(2, min(15, vpool.shape[0] - 1))
    S.log("UMAP: joint video+answer projection ...")
    xy = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=S.SEED) \
             .fit_transform(np.vstack([vpool, ans]))

    S.log("wrote " + S.fig_modality_umap(xy, len(keep), cats, text_name="answer",
                                         outname="umap_video_answer_modality"))

    # video -> answer retrieval
    sim = vpool @ ans.T
    N = sim.shape[0]
    ranks = np.array([int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1
                      for i in range(N)])
    row = {"N": N,
           "R@1 (%)": round(100 * float(np.mean(ranks <= 1)), 1),
           "R@5 (%)": round(100 * float(np.mean(ranks <= 5)), 1),
           "R@10 (%)": round(100 * float(np.mean(ranks <= 10)), 1),
           "median rank": int(np.median(ranks)),
           "chance R@1 (%)": round(100 / N, 2)}
    t = pd.DataFrame([row])
    t.to_csv(OUT/"tables/video_answer_retrieval.csv", index=False)
    df_to_booktabs(t, OUT/"tables/video_answer_retrieval.tex", float_fmt="%.1f",
        caption=f"Video$\\to$free-form-answer retrieval in the shared SigLIP space "
        f"({S.SIGLIP}, {a.frames} frames/video, N={N}).", label="tab:va_retrieval")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
