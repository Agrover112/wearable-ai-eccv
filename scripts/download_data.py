#!/usr/bin/env python
"""Download ONLY the egolongqa split + the videos it references.

The facebook/wearable-ai dataset is gated: set HF_TOKEN (with access) first.
Annotations are tiny; videos are large, so we fetch only the mp4s referenced
by egolongqa/val and cache everything under HF_HOME (Drive on Colab).
"""
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

REPO = "facebook/wearable-ai"
CONFIG, SPLIT = "egolongqa", "val"
TOKEN = os.environ.get("HF_TOKEN") or True  # True -> use cached login from bootstrap

# Where resolved videos land (gitignored). On Colab, prefer the Drive cache.
OUT = Path(os.environ.get("VIDEO_DIR", "data/videos"))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print(f">> Loading {REPO}:{CONFIG}/{SPLIT} annotations...")
    ds = load_dataset(REPO, CONFIG, split=SPLIT, token=TOKEN)
    print(f"   {len(ds)} rows. Columns: {ds.column_names}")

    wanted = sorted({os.path.basename(p) for p in ds["video_path"]})
    print(f">> {len(wanted)} unique videos referenced.")

    # Discover where the videos live inside the repo, then fetch only ours.
    api = HfApi()
    all_files = api.list_repo_files(REPO, repo_type="dataset", token=TOKEN)
    by_name = {os.path.basename(f): f for f in all_files if f.lower().endswith(".mp4")}

    missing, got = [], 0
    for name in wanted:
        repo_path = by_name.get(name)
        if repo_path is None:
            missing.append(name)
            continue
        local = OUT / name
        if local.exists():
            got += 1
            continue
        p = hf_hub_download(REPO, repo_path, repo_type="dataset", token=TOKEN)
        # symlink/copy into a flat, predictable dir
        if not local.exists():
            os.symlink(p, local)
        got += 1
        if got % 25 == 0:
            print(f"   {got}/{len(wanted)} videos ready")

    print(f">> Done. {got} videos in {OUT.resolve()}.")
    if missing:
        print(f"!! {len(missing)} referenced videos not found in repo listing "
              f"(check dataset layout): {missing[:5]}{'...' if len(missing) > 5 else ''}")


if __name__ == "__main__":
    main()
