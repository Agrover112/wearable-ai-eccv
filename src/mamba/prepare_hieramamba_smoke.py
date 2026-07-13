#!/usr/bin/env python3
"""Prepare one extracted EgoLongQA row for HieraMamba inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--hieramamba-root", default="/content/hieramamba")
    parser.add_argument("--name", default="egolongqa_smoke")
    args = parser.parse_args()

    features = Path(args.features).resolve()
    root = Path(args.hieramamba_root).resolve()
    data_root = root / "data" / args.name
    video_dir, text_dir = data_root / "video", data_root / "text"
    video_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    video_id = args.video_id
    query_id = f"{video_id}_q0"
    video_source = features / "video" / f"{video_id}.npy"
    text_source = features / "text" / f"{video_id}.npy"
    metadata = json.loads((features / "metadata" / f"{video_id}.json").read_text())
    for source, destination in (
        (video_source, video_dir / f"{video_id}.npy"),
        (text_source, text_dir / f"{query_id}.npy"),
    ):
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)

    n_features = int(np.load(video_source, mmap_mode="r").shape[0])
    fps = float(metadata["fps"])
    duration = float(metadata["duration_seconds"])
    centers = np.asarray(metadata["feature_center_seconds"], dtype=float)
    stride_seconds = float(np.median(np.diff(centers))) if len(centers) > 1 else duration
    annotation = {
        "val": {
            video_id: {
                "fps": fps,
                "duration": duration,
                "num_clips": n_features,
                "annotations": [
                    {
                        # Dummy target is required by the inherited evaluator;
                        # it is never used to choose the predicted spans.
                        "segment": [0.001, min(duration, 1.0)],
                        "sentence": metadata["question"],
                        "sentence_id": query_id,
                    }
                ],
            }
        }
    }
    annotation_path = data_root / "annotations.json"
    annotation_path.write_text(json.dumps(annotation, indent=2))

    official = root / "experiments" / "hieramamba_ego4d_ckpt"
    experiment = root / "experiments" / args.name
    experiment.mkdir(parents=True, exist_ok=True)
    opt = yaml.safe_load((official / "opt.yaml").read_text())
    train_data = opt["train"]["data"]
    train_data.update(
        {
            "anno_file": str(annotation_path),
            "vid_feat_dir": str(video_dir),
            "text_feat_dir": str(text_dir),
            "clip_size": 32,
            # Preserve seconds when the smoke test uses a coarse timeline.
            "clip_stride": stride_seconds * fps,
            "downsample_rate": 1,
            "max_num_text": 1,
        }
    )
    opt["train"]["num_workers"] = 0
    opt["eval"]["data"]["split"] = "val"
    opt["eval"]["nms"]["max_num_segs"] = 5
    (experiment / "opt.yaml").write_text(yaml.safe_dump(opt, sort_keys=False))
    models_link = experiment / "models"
    if models_link.exists() or models_link.is_symlink():
        if models_link.is_symlink():
            models_link.unlink()
        elif models_link.resolve() != (official / "models").resolve():
            shutil.rmtree(models_link)
    if not models_link.exists():
        models_link.symlink_to(official / "models")
    print(experiment)


if __name__ == "__main__":
    main()
