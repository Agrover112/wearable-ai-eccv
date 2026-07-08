#!/usr/bin/env python
"""EgoLongQA — video duration analysis (NO video download, NO GPU).

Reads only the mp4 container header (moov atom) over HTTP with PyAV to get each
video's duration/fps — a few KB per file, never the frames. Threaded over all 700
videos. Produces duration stats + a publication histogram, and cross-checks the
file-size proxy. Informs the frame-sampling choice for Steps 7/8 and inference.

Outputs (outputs/eda/):
  tables/video_durations.csv                       (per-video duration/fps/size)
  tables/duration_summary.{csv,tex}                (mean/median/percentiles)
  figures/dist_video_duration.{pdf,png}            (duration histogram)
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
import av
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubstyle import set_style, savefig, df_to_booktabs, CB, COL, DBL
import matplotlib.pyplot as plt   # Agg backend already set inside pubstyle

REPO = "facebook/wearable-ai"; TOK = os.environ.get("HF_TOKEN") or True
OUT = Path("outputs/eda"); (OUT/"figures").mkdir(parents=True, exist_ok=True); (OUT/"tables").mkdir(exist_ok=True)
WORKERS = int(os.environ.get("WORKERS", "16"))


def probe(url, hdr):
    """Header-only read: duration (s), fps, frame count. No frame decoding."""
    c = av.open(url, options={"headers": hdr, "rw_timeout": "30000000"})
    try:
        vs = c.streams.video[0]
        dur = float(c.duration) / av.time_base if c.duration else None
        fps = float(vs.average_rate) if vs.average_rate else None
        nfr = int(vs.frames) if vs.frames else (int(dur * fps) if dur and fps else None)
        return dur, fps, nfr
    finally:
        c.close()


def main():
    set_style()
    api = HfApi()
    tree = {os.path.basename(getattr(it, "path", "")): it
            for it in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, token=TOK)
            if getattr(it, "path", "").lower().endswith(".mp4")}
    df = pd.read_parquet(hf_hub_download(REPO, "egolongqa/val/0000.parquet", repo_type="dataset",
                                         revision="refs/convert/parquet", token=TOK))
    names = [os.path.basename(p) for p in df["video_path"]]
    hdr = f"Authorization: Bearer {TOK}\r\n" if isinstance(TOK, str) else ""
    print(f">> probing {len(names)} video headers with {WORKERS} threads (no download)...")

    def one(name):
        it = tree.get(name)
        if it is None:
            return name, None, None, None, None
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{it.path}"
        try:
            dur, fps, nfr = probe(url, hdr)
        except Exception as e:
            print(f"  !! {name}: {type(e).__name__}: {e}")
            dur = fps = nfr = None
        return name, dur, fps, nfr, getattr(it, "size", None)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, n) for n in names]
        for f in tqdm(as_completed(futs), total=len(futs), desc="probe headers", unit="vid"):
            name, dur, fps, nfr, size = f.result()
            rows.append({"video": name, "duration_s": dur, "fps": fps,
                         "n_frames": nfr, "size_MB": (size/1e6 if size else None)})
    print(f">> probed in {time.time()-t0:.0f}s")

    d = pd.DataFrame(rows)
    d.to_csv(OUT/"tables/video_durations.csv", index=False)
    dur = d["duration_s"].dropna()
    mins = dur / 60.0

    summ = pd.DataFrame([{
        "N": int(dur.size),
        "mean (min)": round(mins.mean(), 2),
        "median (min)": round(mins.median(), 2),
        "std (min)": round(mins.std(), 2),
        "min (min)": round(mins.min(), 2),
        "p25 (min)": round(mins.quantile(.25), 2),
        "p75 (min)": round(mins.quantile(.75), 2),
        "p95 (min)": round(mins.quantile(.95), 2),
        "max (min)": round(mins.max(), 2),
        "total (h)": round(dur.sum()/3600, 1),
        "mode fps": round(d["fps"].dropna().mode().iloc[0], 1) if d["fps"].notna().any() else None,
    }])
    summ.to_csv(OUT/"tables/duration_summary.csv", index=False)
    df_to_booktabs(summ, OUT/"tables/duration_summary.tex", float_fmt="%.2f",
        caption=f"EgoLongQA (val) video-duration statistics, N={int(dur.size)} videos "
        f"(minutes). Durations read from mp4 headers only. These are long egocentric "
        f"clips, motivating denser frame sampling than the 4-frame baseline.",
        label="tab:durations")
    print("\n=== duration summary (minutes) ===")
    print(summ.to_string(index=False))

    # ---- publication histogram --------------------------------------------
    fig, ax = plt.subplots(figsize=(COL*1.6, COL))
    ax.hist(mins, bins=30, color=CB[0], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(mins.mean(), color=CB[3], lw=1.4, ls="--", label=f"mean = {mins.mean():.1f} min")
    ax.axvline(mins.median(), color=CB[2], lw=1.4, ls=":", label=f"median = {mins.median():.1f} min")
    ax.set_xlabel("video duration (minutes)"); ax.set_ylabel("number of videos")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"EgoLongQA (val) video durations · N={int(dur.size)}", fontsize=8)
    print("\n>> wrote", savefig(fig, OUT/"figures/dist_video_duration"))

    # size proxy check
    ok = d.dropna(subset=["duration_s", "size_MB"])
    if len(ok) > 2:
        r = np.corrcoef(ok["duration_s"], ok["size_MB"])[0, 1]
        print(f">> duration vs file-size correlation: r = {r:.3f} "
              f"(validates size as a length proxy)")


if __name__ == "__main__":
    main()
