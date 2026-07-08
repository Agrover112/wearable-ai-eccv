#!/usr/bin/env python
"""Duration figure that exposes the tail (reads cached video_durations.csv; no GPU/network).

The plain histogram is swamped by the ~10-min mode (616/700 clips), hiding how many
videos are shorter/longer. Two panels fix this without discarding any data:
  (a) histogram with a LOG-scaled count axis  -> mode spike AND sparse tail both visible
  (b) coarse duration buckets with exact counts -> how many shorter / around / longer

Output: outputs/eda/figures/dist_video_duration_tail.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubstyle import set_style, savefig, CB, DBL
import matplotlib.pyplot as plt

OUT = Path("outputs/eda")
BUCKETS = [(-1, 5, "<5"), (5, 9, "5–9"), (9, 9.5, "9–9.5"),
           (9.5, 10.5, "9.5–10.5\n(mode)"), (10.5, 11, "10.5–11"),
           (11, 12, "11–12"), (12, 99, "≥12")]


def main():
    set_style()
    d = pd.read_csv(OUT / "tables/video_durations.csv")
    m = d["duration_s"].dropna() / 60.0
    N = int(m.size)

    fig, (a, b) = plt.subplots(1, 2, figsize=(DBL, DBL * 0.4))

    # (a) log-scaled histogram — nothing cut off, tail now visible
    a.hist(m, bins=np.arange(0, 15.5, 0.25), color=CB[0], alpha=0.85,
           edgecolor="white", linewidth=0.3)
    a.set_yscale("log")
    a.axvline(m.mean(), color=CB[3], lw=1.3, ls="--", label=f"mean {m.mean():.1f}")
    a.axvline(m.median(), color=CB[2], lw=1.3, ls=":", label=f"median {m.median():.1f}")
    a.set_xlabel("video duration (minutes)")
    a.set_ylabel("number of videos (log scale)")
    a.legend(frameon=False, fontsize=7)
    a.set_title(f"(a) Duration histogram, log count axis · N={N}", fontsize=8)

    # (b) coarse buckets with exact counts
    labels = [lab for _, _, lab in BUCKETS]
    counts = [int(((m > lo) & (m <= hi)).sum()) for lo, hi, _ in BUCKETS]
    colors = [CB[2] if "mode" in l else (CB[1] if i < 3 else CB[3])
              for i, l in enumerate(labels)]
    y = np.arange(len(labels))[::-1]
    b.barh(y, counts, color=colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    b.set_yticks(y); b.set_yticklabels(labels, fontsize=7)
    b.set_xlabel("number of videos")
    b.set_xlim(0, max(counts) * 1.18)
    for yi, c in zip(y, counts):
        b.text(c + max(counts) * 0.015, yi, str(c), va="center", fontsize=7)
    b.grid(False)
    b.set_title("(b) Counts by duration bucket (min)", fontsize=8)

    short = int((m < 9.5).sum()); long = int((m > 10.5).sum())
    fig.suptitle(f"EgoLongQA video durations: {counts[3]}/{N} clips are ~10 min; "
                 f"{short} shorter, {long} longer", fontsize=8.5, y=1.02)
    fig.tight_layout()
    print(">> wrote", savefig(fig, OUT / "figures/dist_video_duration_tail"))


if __name__ == "__main__":
    main()
