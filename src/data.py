"""Load the egolongqa split and resolve local video paths."""
import os
from pathlib import Path

from datasets import load_dataset

REPO = "facebook/wearable-ai"
CONFIG, SPLIT = "egolongqa", "val"
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "data/videos"))


def load_egolongqa(split: str = SPLIT):
    """Return the egolongqa dataset; each row's `video_path` is resolved to a
    local file under VIDEO_DIR when available."""
    ds = load_dataset(REPO, CONFIG, split=split,
                      token=os.environ.get("HF_TOKEN") or True)

    def _resolve(row):
        local = VIDEO_DIR / os.path.basename(row["video_path"])
        row["local_video"] = str(local) if local.exists() else None
        return row

    return ds.map(_resolve)


def parse_mcq(options: str):
    """Split an 'A. ... B. ... C. ... D. ...' string into {letter: text}."""
    import re
    parts = re.split(r"(?=[A-D]\.\s)", options.strip())
    out = {}
    for p in parts:
        p = p.strip()
        if len(p) >= 2 and p[0] in "ABCD" and p[1] == ".":
            out[p[0]] = p[2:].strip()
    return out


if __name__ == "__main__":
    ds = load_egolongqa()
    print(ds)
    r = ds[0]
    print("Q:", r["question"])
    print("Options:", parse_mcq(r["mcq_options"]))
    print("Gold:", r["mcq_answer"], "| local video:", r["local_video"])
