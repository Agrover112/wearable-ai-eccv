#!/usr/bin/env python
"""EgoLongQA — Step 6: semantic clustering of questions (CPU).

Runs TWO text encoders and compares them:
  - MiniLM  (sentence-transformers/all-MiniLM-L6-v2)  — text-specialist
  - SigLIP-text (google/siglip2-so400m-patch14-384)   — multimodal (shared w/ Step 7/8)

Per encoder: embed -> KMeans(k) -> c-TF-IDF distinctive terms (auto-name) -> UMAP 2D.
Comparison: silhouette (cluster separation) + Adjusted Rand Index (do the two agree?).

Publication-quality (ECCV / LNCS) via scripts/pubstyle.py: PDF+PNG figures,
colorblind-safe palette, serif fonts; tables as CSV + LaTeX booktabs.

Outputs (outputs/eda/):
  figures/umap_clusters_minilm.{pdf,png}     (all-MiniLM-L6-v2 alone)
  figures/umap_clusters_siglip.{pdf,png}     (siglip2-so400m alone)
  figures/umap_clusters_compare.{pdf,png}    (side-by-side comparison)
  tables/cluster_summary_{minilm,siglip}.{csv,tex}
  tables/encoder_comparison.{csv,tex}
  features/question_emb_{minilm,siglip}.npy   (gitignored)
"""
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubstyle import set_style, savefig, df_to_booktabs, CB, COL, DBL

K = int(os.environ.get("N_CLUSTERS", "10")); SEED = 42
SIGLIP = os.environ.get("SIGLIP_MODEL", "google/siglip2-so400m-patch14-384")

# Exact model ids + one-line role, used verbatim in every figure title/caption.
ENCODER_META = {
    "minilm": {"id": "sentence-transformers/all-MiniLM-L6-v2",
               "role": "text-only sentence encoder"},
    "siglip": {"id": SIGLIP,
               "role": "SigLIP-2 multimodal text tower"},
}
OUT = Path("outputs/eda"); (OUT/"figures").mkdir(parents=True, exist_ok=True); (OUT/"tables").mkdir(exist_ok=True)
FEAT = Path("features"); FEAT.mkdir(exist_ok=True)
GENERIC = {"what","after","before","first","did","i","my","the","was","were","when","which","while",
           "earlier","last","then","during","that","this","see","saw","just","right","left","time",
           "get","got","go","went","me","do","does","doing","later","video","near","use","large"}

def load_questions():
    pq = hf_hub_download("facebook/wearable-ai","egolongqa/val/0000.parquet",
        repo_type="dataset",revision="refs/convert/parquet",token=os.environ.get("HF_TOKEN") or True)
    return pd.read_parquet(pq)

def embed_minilm(qs):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return m.encode(qs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

def embed_siglip(qs):
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(qs), 32):
            batch = qs[i:i+32]
            inp = proc(text=batch, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            tf = model.get_text_features(**inp)
            if not torch.is_tensor(tf):  # transformers 5.x may return an output object
                tf = getattr(tf, "pooler_output", None)
                if tf is None: tf = getattr(tf, "last_hidden_state").mean(1)
            tf = torch.nn.functional.normalize(tf, dim=-1)
            out.append(tf.cpu().numpy())
    return np.concatenate(out)

def cluster_and_name(df, emb, tag):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    labels = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(emb)
    docs = [" ".join(df.question[labels==c]) for c in range(K)]
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z]{3,}", max_features=4000)
    X = vec.fit_transform([d.lower() for d in docs]); vocab = np.array(vec.get_feature_names_out())
    rows=[]
    print(f"\n=== {tag}: {K} clusters ===")
    for c in range(K):
        row = X[c].toarray().ravel(); order = row.argsort()[::-1]
        terms = [vocab[i] for i in order if vocab[i] not in GENERIC][:6]
        sub = df[labels==c]
        rows.append({"cluster":c,"size":int((labels==c).sum()),"top_terms":", ".join(terms),
                     "example":sub.question.iloc[0][:80]})
        print(f"[{c}] n={int((labels==c).sum()):>3} | {', '.join(terms)}")
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT/f"tables/cluster_summary_{tag}.csv", index=False)
    df_to_booktabs(summ[["cluster","size","top_terms"]], OUT/f"tables/cluster_summary_{tag}.tex",
        float_fmt="%.0f", caption=f"EgoLongQA question clusters ({tag} encoder, k={K}): "
        "size and c-TF-IDF distinctive terms.", label=f"tab:clusters_{tag}")
    return labels

def main():
    set_style()
    df = load_questions(); qs = df.question.tolist()
    encoders = {"minilm": embed_minilm, "siglip": embed_siglip}
    embs, labs = {}, {}
    for tag, fn in encoders.items():
        cache = FEAT/f"question_emb_{tag}.npy"
        if cache.exists():
            print(f"\n>>> Loading cached {tag} embeddings ...")
            e = np.load(cache)
        else:
            print(f"\n>>> Encoding with {tag} ...")
            e = np.asarray(fn(qs)); np.save(cache, e)
        embs[tag]=e
        labs[tag] = cluster_and_name(df, e, tag)

    # comparison metrics
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    import umap
    print("\n=== ENCODER COMPARISON ===")
    cmp=[]
    for tag in encoders:
        sil = silhouette_score(embs[tag], labs[tag])
        cmp.append({"encoder":tag,"dim":embs[tag].shape[1],"silhouette":round(float(sil),4)})
        print(f"  {tag:8} dim={embs[tag].shape[1]:>4}  silhouette={sil:.4f} (higher=better separated)")
    ari = adjusted_rand_score(labs["minilm"], labs["siglip"])
    print(f"  agreement (Adjusted Rand Index) MiniLM vs SigLIP: {ari:.4f} (1=identical, 0=random)")
    cmp_df = pd.DataFrame(cmp)
    cmp_df.to_csv(OUT/"tables/encoder_comparison.csv", index=False)
    df_to_booktabs(cmp_df, OUT/"tables/encoder_comparison.tex", float_fmt="%.4f",
        caption=f"Text-encoder comparison on EgoLongQA questions (k={K}). Silhouette: "
        f"cluster separation (higher better). MiniLM vs SigLIP-text agreement: "
        f"adjusted Rand index = {ari:.4f}.", label="tab:encoder_cmp")

    # ---- UMAP projections (one per encoder, computed once & reused) --------
    cmap = ListedColormap(CB[:K])
    sil_of = {d["encoder"]: d["silhouette"] for d in cmp}
    xys = {tag: umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(embs[tag])
           for tag in encoders}

    def draw(ax, tag):
        xy = xys[tag]
        ax.scatter(xy[:,0], xy[:,1], c=labs[tag], cmap=cmap, vmin=0, vmax=K-1, s=10,
                   alpha=0.85, linewidths=0)
        for c in range(K):
            cx, cy = xy[labs[tag]==c].mean(0)
            ax.text(cx, cy, str(c), fontsize=8, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="0.3", lw=0.6, alpha=0.85))
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.grid(False)

    # (a) one standalone figure per encoder, titled with the EXACT model id ---
    for tag in encoders:
        m = ENCODER_META[tag]
        fig, ax = plt.subplots(figsize=(COL, COL))
        draw(ax, tag)
        ax.set_title(f"{m['id']}\n{m['role']} · dim={embs[tag].shape[1]} · "
                     f"silhouette={sil_of[tag]:.3f}", fontsize=6.5)
        savefig(fig, OUT/f"figures/umap_clusters_{tag}")
        print(f"  wrote figures/umap_clusters_{tag}.{{pdf,png}}  ({m['id']})")

    # (b) single comparison figure across all encoders -----------------------
    fig, axes = plt.subplots(1, len(encoders), figsize=(DBL, DBL*0.52))
    for ax, tag in zip(axes, encoders):
        m = ENCODER_META[tag]
        draw(ax, tag)
        ax.set_title(f"{m['id']}\n({m['role']}, silhouette={sil_of[tag]:.3f})", fontsize=6.5)
    fig.suptitle(f"EgoLongQA question clusters (KMeans k={K}, UMAP projection) — "
                 f"text-only vs multimodal encoder\n"
                 f"cluster agreement: adjusted Rand index = {ari:.2f} "
                 f"(1 = identical partition, 0 = random)", fontsize=8)
    fig.tight_layout()
    savefig(fig, OUT/"figures/umap_clusters_compare")
    print(f"\n>> Wrote figures/umap_clusters_{{minilm,siglip,compare}}.{{pdf,png}} "
          f"+ tables/cluster_summary_*.{{csv,tex}} + encoder_comparison.{{csv,tex}}")

if __name__ == "__main__":
    main()
