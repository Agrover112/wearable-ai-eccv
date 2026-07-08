#!/usr/bin/env python
"""Select a category-stratified, shuffled subset of egolongqa videos.

Proportional to category frequency (seed 42, deterministic), then shuffled so
that stopping the extraction early keeps the subset stratified. Writes a
comma-separated list consumed by eda_frame_features.py via VIDEO_LIST.

    python scripts/select_videos.py 150 .cache/videos150.txt
"""
import os, random, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "facebook/wearable-ai"; TOK = os.environ.get("HF_TOKEN") or True
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".cache/videos150.txt")


def main():
    pq = hf_hub_download(REPO, "egolongqa/val/0000.parquet", repo_type="dataset",
                         revision="refs/convert/parquet", token=TOK)
    df = pd.read_parquet(pq)
    df["video"] = df["video_path"].map(os.path.basename)

    frac = N / len(df)
    parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=42)
             for _, g in df.groupby("category")]
    sel = pd.concat(parts)
    if len(sel) > N:
        sel = sel.sample(n=N, random_state=42)
    elif len(sel) < N:
        extra = df[~df.video.isin(sel.video)].sample(n=N - len(sel), random_state=42)
        sel = pd.concat([sel, extra])

    vids = sel.video.tolist()
    random.Random(42).shuffle(vids)           # early stop stays stratified
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(",".join(vids))
    print(f"wrote {len(vids)} videos over {sel.category.nunique()} categories -> {OUT}")
    print(sel.category.value_counts().to_string())


if __name__ == "__main__":
    main()
