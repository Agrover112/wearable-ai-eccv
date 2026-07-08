#!/usr/bin/env python
"""EgoLongQA — semantic clustering of the MCQ option text (CPU, MiniLM only).

Companion to eda_clustering.py (which clusters the QUESTION text). Here we embed
the full multiple-choice option strings `mcq_options` (the A-D candidate answers,
letters stripped) with the best text-only encoder from that comparison,
sentence-transformers/all-MiniLM-L6-v2, then KMeans(k) -> c-TF-IDF distinctive
terms (auto-name) -> UMAP 2D.

Publication-quality (ECCV / LNCS) via scripts/pubstyle.py: PDF+PNG figure,
colorblind-safe palette, serif fonts; table as CSV + LaTeX booktabs.

Outputs (outputs/eda/):
  figures/umap_clusters_mcq_minilm.{pdf,png}
  tables/cluster_summary_mcq_minilm.{csv,tex}
  features/mcq_emb_minilm.npy   (gitignored)
"""
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubstyle import set_style, savefig, df_to_booktabs, CB, COL

K = int(os.environ.get("N_CLUSTERS", "10")); SEED = 42
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("outputs/eda"); (OUT/"figures").mkdir(parents=True, exist_ok=True); (OUT/"tables").mkdir(exist_ok=True)
FEAT = Path("features"); FEAT.mkdir(exist_ok=True)
GENERIC = {"what","after","before","first","did","i","my","the","was","were","when","which","while",
           "earlier","last","then","during","that","this","see","saw","just","right","left","time",
           "get","got","go","went","me","do","does","doing","later","video","near","use","large","none"}

def load():
    pq = hf_hub_download("facebook/wearable-ai","egolongqa/val/0000.parquet",
        repo_type="dataset",revision="refs/convert/parquet",token=os.environ.get("HF_TOKEN") or True)
    return pd.read_parquet(pq)

def strip_letters(opts):
    """Drop the 'A. ', 'B. ' ... option markers, keep only the answer wording."""
    return re.sub(r"\b[A-D]\.\s*", " ", str(opts)).strip()

def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    return m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

def cluster_and_name(texts, emb):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    labels = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(emb)
    docs = [" ".join(np.array(texts)[labels==c]) for c in range(K)]
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z]{3,}", max_features=4000)
    X = vec.fit_transform([d.lower() for d in docs]); vocab = np.array(vec.get_feature_names_out())
    rows=[]
    print(f"\n=== MCQ options ({MODEL}): {K} clusters ===")
    for c in range(K):
        row = X[c].toarray().ravel(); order = row.argsort()[::-1]
        terms = [vocab[i] for i in order if vocab[i] not in GENERIC][:6]
        label = ", ".join(terms[:2])          # short data-driven cluster name
        example = np.array(texts)[labels==c][0][:80]
        rows.append({"cluster":c,"label":label,"size":int((labels==c).sum()),
                     "top_terms":", ".join(terms),"example":example})
        print(f"[{c}] n={int((labels==c).sum()):>3} | {label:<24} | {', '.join(terms)}")
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT/"tables/cluster_summary_mcq_minilm.csv", index=False)
    df_to_booktabs(summ[["cluster","label","size","top_terms"]], OUT/"tables/cluster_summary_mcq_minilm.tex",
        float_fmt="%.0f", caption=f"EgoLongQA MCQ-option clusters "
        f"({MODEL.split('/')[-1]} encoder, k={K}). The label is the two most distinctive "
        f"c-TF-IDF terms of each cluster.", label="tab:clusters_mcq")
    return labels, {r["cluster"]: r["label"] for r in rows}

def main():
    set_style()
    df = load(); texts = [strip_letters(o) for o in df.mcq_options.tolist()]

    cache = FEAT/"mcq_emb_minilm.npy"
    if cache.exists():
        print(f">>> Loading cached MCQ embeddings ...")
        emb = np.load(cache)
    else:
        print(f">>> Encoding {len(texts)} MCQ-option strings with {MODEL} ...")
        emb = np.asarray(embed(texts)); np.save(cache, emb)

    labels, names = cluster_and_name(texts, emb)

    from sklearn.metrics import silhouette_score
    import umap
    sil = float(silhouette_score(emb, labels))
    print(f"\nsilhouette = {sil:.4f} (higher = better separated)")

    cmap = ListedColormap(CB[:K])
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(COL*1.7, COL))
    ax.scatter(xy[:,0], xy[:,1], c=labels, cmap=cmap, vmin=0, vmax=K-1, s=10, alpha=0.85, linewidths=0)
    for c in range(K):
        cx, cy = xy[labels==c].mean(0)
        ax.text(cx, cy, str(c), fontsize=8, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="0.3", lw=0.6, alpha=0.85))
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.grid(False)
    ax.set_title(f"{MODEL}\nMCQ option text · k={K} · silhouette={sil:.3f}", fontsize=6.5)
    # legend: cluster number + color -> data-driven name
    handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=CB[c],markersize=6,
                    label=f"{c}: {names[c]}") for c in range(K)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01,0.5), fontsize=6,
              handletextpad=0.2, labelspacing=0.35, borderpad=0.4, frameon=False,
              title="cluster (top terms)", title_fontsize=6)
    savefig(fig, OUT/"figures/umap_clusters_mcq_minilm")
    print(f"\n>> Wrote figures/umap_clusters_mcq_minilm.{{pdf,png}} + tables/cluster_summary_mcq_minilm.{{csv,tex}}")

if __name__ == "__main__":
    main()
