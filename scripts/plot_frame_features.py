#!/usr/bin/env python
"""Re-plot Step 7 figures from CACHED frame features (no video streaming).

Loads features/frame_feats_siglip_{FRAMES}f.npz produced by eda_frame_features.py
and renders Fig A (category UMAP) + Fig B (video/text modality UMAP) + the
video->question retrieval table. Use this to iterate on the figures without
re-downloading videos.

    python scripts/plot_frame_features.py --frames 32
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_frame_features as S           # reuse constants + plotting/retrieval fns
from pubstyle import set_style

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Re-render Step 7 figures from cached "
                                 "per-video SigLIP features (no streaming).")
    ap.add_argument("--frames", type=int, default=S.FRAMES,
                    help="which frame-count cache to plot (default: %(default)s)")
    return ap.parse_args()


def main():
    a = parse_args()
    S.FRAMES = a.frames
    S.CACHE_DIR = S.FEAT / f"frames_siglip_{a.frames}f"
    files = sorted(S.CACHE_DIR.glob("*.npy"))
    if not files:
        sys.exit(f"!! no per-video features in {S.CACHE_DIR} — run eda_frame_features.py first")
    set_style()
    names = [f.stem + ".mp4" for f in files]
    frames = np.stack([np.load(f) for f in files]).astype(np.float32)
    S.log(f"loaded {len(names)} cached videos from {S.CACHE_DIR}, per-frame feats {frames.shape}")

    df = pd.read_parquet(hf_hub_download(S.REPO, "egolongqa/val/0000.parquet",
        repo_type="dataset", revision="refs/convert/parquet", token=S.TOK))
    name2cat = {os.path.basename(p): c for p, c in zip(df["video_path"], df["category"])}
    # keep only cached videos, preserve their order
    keep = [i for i, n in enumerate(names) if n in name2cat]
    names = [names[i] for i in keep]; frames = frames[keep]
    cats = [name2cat[n] for n in names]

    vpool = frames.mean(axis=1)
    vpool = vpool / (np.linalg.norm(vpool, axis=1, keepdims=True) + 1e-8)

    # reuse cached SigLIP-text question embeddings aligned to these videos
    qcache = S.FEAT / "question_emb_siglip.npy"
    qtext = None
    if qcache.exists():
        qall = np.load(qcache)
        idx = {os.path.basename(p): i for i, p in enumerate(df["video_path"])}
        qtext = np.stack([qall[idx[n]] for n in names]).astype(np.float32)
        qtext = qtext / (np.linalg.norm(qtext, axis=1, keepdims=True) + 1e-8)

    import umap
    def project(X):
        nn = max(2, min(15, X.shape[0] - 1))
        return umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=S.SEED).fit_transform(X)

    S.log("Fig A: category UMAP ...")
    S.log("wrote " + S.fig_category_umap(project(vpool), cats))
    if qtext is not None:
        S.log("Fig B: video/text modality UMAP ...")
        S.log("wrote " + S.fig_modality_umap(project(np.vstack([vpool, qtext])), len(names), cats))
        S.log("video->question retrieval:")
        print(S.retrieval_table(vpool, qtext).to_string(index=False))
    S.log("done.")


if __name__ == "__main__":
    main()
