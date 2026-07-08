#!/usr/bin/env python
"""EgoLongQA — Step 7: SigLIP frame feature-space visualization (EgoCross Fig-4 style).

For every video: STREAM it from HF -> sample FRAMES frames uniformly (decord) ->
SigLIP image features -> mean-pool -> 1 vector/video -> DELETE the file. Videos are
never bulk-stored (the split is ~203 GB); each mp4 is removed right after decoding.

Text side reuses the cached SigLIP-text question embeddings from Step 6
(features/question_emb_siglip.npy) so the video and text vectors share one encoder.

Outputs:
  UMAP of video vectors colored by `category`         -> category separability
  Joint UMAP of video + question vectors (by modality) -> text<->video alignment
  Video->question retrieval table (R@1/5/10, median rank) in the shared SigLIP space
  Per-frame + pooled features cached to features/frame_feats_siglip.npz  (Step 8 reuses)

Publication-quality (ECCV/LNCS) via scripts/pubstyle.py: PDF+PNG, colorblind-safe, booktabs.

Usage:
  # FULL split — all 700 videos (~9-11 h wall time; fully resumable, so it can
  # be spread over several sessions — rerunning skips every cached video):
  python scripts/eda_frame_features.py

  # stratified subset (e.g. 150), list produced by select_videos.py:
  python scripts/select_videos.py 150 .cache/videos150.txt
  python scripts/eda_frame_features.py --video-list .cache/videos150.txt

  # flags (env vars FRAMES/N_VIDEOS/VIDEO_LIST/TIME_BUDGET_MIN work as fallback):
  #   --frames 64             frames sampled per video (default 32)
  #   --n-videos 100          only the first N rows of the split (quick tests)
  #   --video-list PATH|a.mp4,b.mp4   subset: file with comma-separated names, or inline
  #   --time-budget-min 110   stop extracting after this many minutes, still
  #                           render figures from whatever is cached
"""
import os, sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubstyle import set_style, savefig, df_to_booktabs, CB, COL, DBL


def log(msg):
    """Timestamped step log to stderr (survives tqdm bar redraws)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

REPO = "facebook/wearable-ai"
SIGLIP = os.environ.get("SIGLIP_MODEL", "google/siglip2-so400m-patch14-384")
FRAMES = int(os.environ.get("FRAMES", "32"))         # LongQA baseline=4; we use the max-frames cap
N_VIDEOS = int(os.environ.get("N_VIDEOS", "0"))      # 0 = all 700
VIDEO_LIST = [n for n in os.environ.get("VIDEO_LIST", "").split(",") if n]  # explicit subset
TIME_BUDGET_MIN = float(os.environ.get("TIME_BUDGET_MIN", "0"))  # 0 = no limit; else stop
                                                     # extraction after this many minutes
                                                     # and continue to figures with what we have
SEED = 42
TOK = os.environ.get("HF_TOKEN") or True

OUT = Path("outputs/eda"); (OUT/"figures").mkdir(parents=True, exist_ok=True); (OUT/"tables").mkdir(exist_ok=True)
FEAT = Path("features"); FEAT.mkdir(exist_ok=True)
# One .npy per video, named by the video id (mp4 basename without extension), so
# anyone can add features for new videos later by dropping in <video_id>.npy.
CACHE_DIR = FEAT / f"frames_siglip_{FRAMES}f"; CACHE_DIR.mkdir(exist_ok=True)
SCRATCH = Path(os.environ.get("VIDEO_TMP", ".cache/vidtmp")) # LOCAL disk, not Drive
SCRATCH.mkdir(parents=True, exist_ok=True)


def load_annotations():
    log("downloading egolongqa/val annotations (parquet) ...")
    pq = hf_hub_download(REPO, "egolongqa/val/0000.parquet", repo_type="dataset",
                         revision="refs/convert/parquet", token=TOK)
    df = pd.read_parquet(pq)
    log(f"annotations loaded: {len(df)} rows, {df['category'].nunique()} categories")
    return df


# ---- SigLIP image encoder ---------------------------------------------------
class SiglipImage:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(SIGLIP)
        self.model = AutoModel.from_pretrained(SIGLIP).eval().to(self.dev)
        print(f">> SigLIP image encoder on {self.dev}")

    def encode(self, pil_frames):
        """List[PIL] -> (n_frames, D) L2-normalized image features."""
        torch = self.torch
        inp = self.proc(images=pil_frames, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            f = self.model.get_image_features(**inp)
        if not torch.is_tensor(f):        # transformers 5.x returns an output object
            f = getattr(f, "pooler_output", None)
            if f is None:
                f = getattr(f, "image_embeds", None)
            if f is None:
                f = self.model.get_image_features(**inp).last_hidden_state.mean(1)
        f = torch.nn.functional.normalize(f, dim=-1)
        return f.cpu().numpy().astype(np.float32)


def sample_frames(path, k):
    """Uniformly sample k frames from a video file -> list[PIL.Image]."""
    import decord
    from PIL import Image
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(path), num_threads=2)
    n = len(vr)
    idx = np.linspace(0, n - 1, k).astype(int)
    batch = vr.get_batch(idx).asnumpy()          # (k, H, W, 3) uint8 RGB
    del vr
    return [Image.fromarray(f) for f in batch]


def stream_and_encode(name, repo_path, enc):
    """Download one video to local scratch, sample+encode frames, delete it.
    Returns (feats [FRAMES,D], stats dict with MB + per-stage seconds)."""
    local = None
    try:
        t = time.time()
        local = Path(hf_hub_download(REPO, repo_path, repo_type="dataset",
                                     token=TOK, local_dir=str(SCRATCH)))
        mb = os.path.getsize(local) / 1e6
        t_dl = time.time() - t
        t = time.time(); frames = sample_frames(local, FRAMES); t_dec = time.time() - t
        t = time.time(); feats = enc.encode(frames);           t_enc = time.time() - t
        stats = {"MB": mb, "dl": t_dl, "dec": t_dec, "enc": t_enc,
                 "MBps": mb / t_dl if t_dl else 0}
        return feats, stats
    finally:
        if local and local.exists():
            try: os.remove(local)
            except OSError: pass


def vid_id(name):
    """mp4 filename -> video id used for the per-video feature file."""
    return Path(name).stem


def feat_path(name):
    return CACHE_DIR / f"{vid_id(name)}.npy"


def extract(df):
    """Return (names, per_frame [N,FRAMES,D]) for all videos, resuming from cache."""
    log("listing mp4 files in the dataset repo ...")
    api = HfApi()
    mp4 = {os.path.basename(f): f for f in api.list_repo_files(REPO, repo_type="dataset", token=TOK)
           if f.lower().endswith(".mp4")}
    log(f"repo has {len(mp4)} mp4 files")

    if VIDEO_LIST:                                # explicit subset (e.g. length-spread test)
        wanted = [n for n in VIDEO_LIST]
        log(f"VIDEO_LIST set -> {len(wanted)} explicit videos")
    else:
        wanted = [os.path.basename(p) for p in df["video_path"]]
        if N_VIDEOS:
            wanted = wanted[:N_VIDEOS]

    done = {p.stem for p in CACHE_DIR.glob("*.npy")}
    todo = [n for n in wanted if vid_id(n) not in done and n in mp4]
    missing = [n for n in wanted if n not in mp4]
    if missing:
        log(f"!! {len(missing)} requested videos absent from repo listing: {missing[:3]}...")
    log(f"{len(done)} cached in {CACHE_DIR}, {len(todo)} to stream | FRAMES={FRAMES} | scratch={SCRATCH}")
    enc = SiglipImage() if todo else None

    t0 = time.time(); tot_mb = 0.0
    bar = tqdm(todo, desc="stream+encode", unit="vid", dynamic_ncols=True)
    for i, name in enumerate(bar, 1):
        if TIME_BUDGET_MIN and (time.time() - t0) / 60 >= TIME_BUDGET_MIN:
            log(f"TIME BUDGET ({TIME_BUDGET_MIN:.0f} min) reached — stopping extraction "
                f"at {i-1}/{len(todo)}; continuing to figures with what is cached.")
            break
        try:
            feats, st = stream_and_encode(name, mp4[name], enc)
            np.save(feat_path(name), feats)          # checkpoint = the file itself
            tot_mb += st["MB"]
            rate = (time.time() - t0) / i
            eta_min = rate * (len(todo) - i) / 60
            bar.set_postfix_str(f"{st['MB']:.0f}MB {st['MBps']:.0f}MB/s "
                                f"dl{st['dl']:.1f} dec{st['dec']:.1f} enc{st['enc']:.1f}s "
                                f"| {tot_mb/1e3:.1f}GB | ETA {eta_min:.0f}min")
            log(f"[{i}/{len(todo)}] {vid_id(name)}  {st['MB']:.0f}MB @ {st['MBps']:.0f}MB/s  "
                f"dl={st['dl']:.1f}s dec={st['dec']:.1f}s enc={st['enc']:.1f}s  "
                f"ETA {eta_min:.0f} min")
            print("\a", end="", flush=True)          # 🔔 terminal bell per finished video
        except Exception as e:
            log(f"[{i}/{len(todo)}] !! {name}: {type(e).__name__}: {e}")
            continue
        gc.collect()
    bar.close()
    log(f"extraction done: {len({p.stem for p in CACHE_DIR.glob('*.npy')})} videos cached, "
        f"{tot_mb/1e3:.1f} GB streamed this run")

    # assemble in annotation order from the per-video files
    names = [n for n in wanted if feat_path(n).exists()]
    frames = np.stack([np.load(feat_path(n)) for n in names])   # (N, FRAMES, D)
    return names, frames


# ---- category color+marker (colorblind-safe: hue x shape) -------------------
MARKERS = ["o", "^", "s", "D"]
def cat_style(cats):
    uniq = list(pd.Series(cats).value_counts().index)   # frequent first
    style = {c: (CB[i % len(CB)], MARKERS[i // len(CB)]) for i, c in enumerate(uniq)}
    return uniq, style


def fig_category_umap(xy, cats):
    uniq, style = cat_style(cats)
    fig, ax = plt.subplots(figsize=(DBL*0.62, DBL*0.55))
    for c in uniq:
        m = np.array(cats) == c
        col, mk = style[c]
        ax.scatter(xy[m, 0], xy[m, 1], s=14, c=col, marker=mk, alpha=0.8, linewidths=0)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.grid(False)
    handles = [Line2D([0], [0], marker=style[c][1], color="w", markerfacecolor=style[c][0],
                      markersize=6, label=f"{c} ({int((np.array(cats)==c).sum())})") for c in uniq]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6,
              frameon=False, title="category (n)", title_fontsize=6.5, labelspacing=0.35)
    ax.set_title(f"SigLIP image features per video ({FRAMES} frames, mean-pooled)\n"
                 f"{SIGLIP} · N={len(cats)} videos · colored by category", fontsize=7)
    return savefig(fig, OUT/"figures/umap_video_features_by_category")


def fig_modality_umap(xy_joint, n_vid, cats):
    """Joint video+question UMAP. color+shape = category (shared by each pair),
    fill = modality (video filled, question hollow)."""
    uniq, style = cat_style(cats)
    v, t = xy_joint[:n_vid], xy_joint[n_vid:]
    ca = np.array(cats)
    fig, ax = plt.subplots(figsize=(DBL*0.78, DBL*0.5))
    for c in uniq:
        col, mk = style[c]; m = ca == c
        ax.scatter(v[m, 0], v[m, 1], marker=mk, s=32, facecolors=col, edgecolors=col,
                   linewidths=0.6, zorder=3)                 # video = filled
        ax.scatter(t[m, 0], t[m, 1], marker=mk, s=36, facecolors="none", edgecolors=col,
                   linewidths=0.9, zorder=3)                 # question = hollow
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.grid(False)
    cat_h = [Line2D([0], [0], marker=style[c][1], color="w", markerfacecolor=style[c][0],
                    markeredgecolor=style[c][0], markersize=6, label=c) for c in uniq]
    mod_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.45",
                    markeredgecolor="0.45", markersize=6, label="video (filled)"),
             Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                    markeredgecolor="0.45", markersize=6, label="question (hollow)")]
    leg1 = ax.legend(handles=cat_h, loc="center left", bbox_to_anchor=(1.01, 0.65),
                     fontsize=6, frameon=False, title="category", title_fontsize=6.5,
                     labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=mod_h, loc="center left", bbox_to_anchor=(1.01, 0.12), fontsize=6,
              frameon=False, title="modality", title_fontsize=6.5, labelspacing=0.3)
    ax.set_title(f"Shared SigLIP space: video vs. question, colored by category\n{SIGLIP}",
                 fontsize=7)
    return savefig(fig, OUT/"figures/umap_video_text_modality")


def retrieval_table(vid, qtext):
    """Video->question retrieval in the shared SigLIP space (both L2-normalized)."""
    sim = vid @ qtext.T                       # (N, N) cosine
    N = sim.shape[0]
    ranks = np.empty(N, int)
    for i in range(N):
        order = np.argsort(-sim[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1   # 1-based rank of the true question
    row = {
        "N": N,
        "R@1 (%)": round(100 * np.mean(ranks <= 1), 1),
        "R@5 (%)": round(100 * np.mean(ranks <= 5), 1),
        "R@10 (%)": round(100 * np.mean(ranks <= 10), 1),
        "median rank": int(np.median(ranks)),
        "chance R@1 (%)": round(100 / N, 2),
    }
    df = pd.DataFrame([row])
    df.to_csv(OUT/"tables/video_text_retrieval.csv", index=False)
    df_to_booktabs(df, OUT/"tables/video_text_retrieval.tex", float_fmt="%.1f",
        caption=f"Video$\\to$question retrieval in the shared SigLIP space "
        f"({SIGLIP}, {FRAMES} frames/video, N={N}). Rank of the true question when ranking "
        f"all questions by cosine similarity to each video; higher R@k / lower median rank = "
        f"stronger text$\\leftrightarrow$video alignment.", label="tab:vt_retrieval")
    return df


def main():
    log(f"=== Step 7: SigLIP frame features | model={SIGLIP} | FRAMES={FRAMES} ===")
    set_style()
    df = load_annotations()
    names, frames = extract(df)                      # frames: (N, FRAMES, D)
    if len(names) == 0:
        log("!! no videos processed — nothing to visualize"); return
    log(f"stacked per-frame features: {frames.shape}  (N, FRAMES, D)")
    sub = df.set_index(df["video_path"].map(os.path.basename)).loc[names].reset_index(drop=True)
    cats = sub["category"].tolist()

    # pooled video vector (mean over frames, renormalized)
    vpool = frames.mean(axis=1)
    vpool = vpool / (np.linalg.norm(vpool, axis=1, keepdims=True) + 1e-8)
    np.save(FEAT/f"video_emb_siglip_{FRAMES}f.npy", vpool)
    pd.DataFrame({"row": range(len(names)), "video_id": [vid_id(n) for n in names]}) \
        .to_csv(FEAT/f"video_emb_siglip_{FRAMES}f_index.csv", index=False)
    log(f"pooled video vectors: {vpool.shape} -> features/video_emb_siglip_{FRAMES}f.npy "
        f"(+ _index.csv mapping row -> video_id)")

    # reuse cached SigLIP-text question embeddings from Step 6, aligned to `names`
    qcache = FEAT/"question_emb_siglip.npy"
    qtext = None
    if qcache.exists():
        qall = np.load(qcache)                        # 700 rows in original order
        idx = {os.path.basename(p): i for i, p in enumerate(df["video_path"])}
        qtext = np.stack([qall[idx[n]] for n in names]).astype(np.float32)
        qtext = qtext / (np.linalg.norm(qtext, axis=1, keepdims=True) + 1e-8)
        log(f"reused cached SigLIP-text question embeddings: {qtext.shape}")

    import umap
    def project(X):                              # n_neighbors must be < n_samples
        nn = max(2, min(15, X.shape[0] - 1))
        return umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=SEED).fit_transform(X)
    log("UMAP: projecting video vectors (category figure) ...")
    xy = project(vpool)
    log("wrote " + fig_category_umap(xy, cats))

    if qtext is not None:
        log("UMAP: projecting joint video+question vectors (modality figure) ...")
        joint = np.vstack([vpool, qtext])
        xyj = project(joint)
        log("wrote " + fig_modality_umap(xyj, len(names), cats))
        log("computing video->question retrieval ...")
        rt = retrieval_table(vpool, qtext)
        print(rt.to_string(index=False))
    else:
        log("!! question_emb_siglip.npy missing — skip alignment figure (run Step 6 first)")

    log("=== Step 7 done ===")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Stream egolongqa videos, extract SigLIP "
                                 "frame features, render UMAP figures. Resumable.")
    ap.add_argument("--frames", type=int, default=FRAMES,
                    help="frames sampled uniformly per video (default: %(default)s)")
    ap.add_argument("--n-videos", type=int, default=N_VIDEOS,
                    help="only the first N rows of the split; 0 = all (default: %(default)s)")
    ap.add_argument("--video-list", default="",
                    help="subset: path to a file with comma-separated mp4 names, "
                         "or the names inline")
    ap.add_argument("--time-budget-min", type=float, default=TIME_BUDGET_MIN,
                    help="stop extracting after this many minutes and render figures "
                         "from the cache; 0 = no limit (default: %(default)s)")
    return ap.parse_args()


if __name__ == "__main__":
    _a = parse_args()
    FRAMES = _a.frames
    N_VIDEOS = _a.n_videos
    TIME_BUDGET_MIN = _a.time_budget_min
    if _a.video_list:
        _p = Path(_a.video_list)
        _raw = _p.read_text() if _p.exists() else _a.video_list
        VIDEO_LIST = [n.strip() for n in _raw.replace("\n", ",").split(",") if n.strip()]
    CACHE_DIR = FEAT / f"frames_siglip_{FRAMES}f"; CACHE_DIR.mkdir(exist_ok=True)
    main()
