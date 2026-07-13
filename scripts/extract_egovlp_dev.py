#!/usr/bin/env python3
"""Extract HieraMamba-compatible EgoVLP features for a dev manifest.

The official EgoVLP NLQ recipe uses a 32/30-second window and an 8/30-second
stride. Its TimeSformer consumes 16 frames sampled within each window. Outputs
are resumable: one video array, one text array, and one metadata JSON per row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


WINDOW_SECONDS = 32 / 30
STRIDE_SECONDS = 8 / 30
FRAMES_PER_CLIP = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/egolongqa_dev20_seed20260709.json")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", default="features/egovlp_dev20")
    parser.add_argument("--egovlp-root", default="/content/EgoVLP")
    parser.add_argument("--checkpoint", default="/content/EgoVLP/pretrained/egovlp.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-clips", type=int, help="Limit clips for throughput benchmarks.")
    parser.add_argument(
        "--target-clips",
        type=int,
        help="Uniformly cover the full video with N clips (pipeline smoke tests only).",
    )
    return parser.parse_args()


def load_model(root: Path, checkpoint: Path):
    import torch

    sys.path.insert(0, str(root))
    previous_cwd = Path.cwd()
    os.chdir(root)
    original_load = torch.load

    def trusted_official_load(*args, **kwargs):
        # The official 2022 checkpoint contains its ConfigParser object.
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_official_load
    try:
        from model.model import FrozenInTime

        os.environ.setdefault("LOCAL_RANK", "0")
        model = FrozenInTime(
            video_params={
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": FRAMES_PER_CLIP,
                "pretrained": True,
                "time_init": "zeros",
            },
            text_params={
                "model": "distilbert-base-uncased",
                "pretrained": True,
                "input": "text",
            },
            projection_dim=256,
            load_checkpoint=str(checkpoint),
        )
    finally:
        torch.load = original_load
        os.chdir(previous_cwd)

    model.eval().cuda()
    return model


def preprocess_frames(frames):
    """uint8 [N,H,W,C] -> normalized float [N,C,224,224]."""
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(frames, device="cuda").permute(0, 3, 1, 2).float().div_(255)
    _, _, h, w = x.shape
    scale = 256 / min(h, w)
    nh, nw = round(h * scale), round(w * scale)
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    top, left = (nh - 224) // 2, (nw - 224) // 2
    x = x[:, :, top : top + 224, left : left + 224]
    mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (x - mean) / std


def extract_video(
    model,
    video_path: Path,
    batch_size: int,
    max_clips: int | None = None,
    target_clips: int | None = None,
):
    import decord
    import torch

    reader = decord.VideoReader(str(video_path), num_threads=2)
    fps = float(reader.get_avg_fps())
    duration = len(reader) / fps
    starts = np.arange(0, max(duration - WINDOW_SECONDS, 0) + 1e-9, STRIDE_SECONDS)
    if not len(starts):
        starts = np.array([0.0])
    if target_clips is not None and target_clips < len(starts):
        starts = np.linspace(0, max(duration - WINDOW_SECONDS, 0), target_clips)
    if max_clips is not None:
        starts = starts[:max_clips]
    offsets = np.linspace(0, WINDOW_SECONDS, FRAMES_PER_CLIP, endpoint=False)
    all_features: list[np.ndarray] = []

    for begin in range(0, len(starts), batch_size):
        batch_starts = starts[begin : begin + batch_size]
        indices = np.rint((batch_starts[:, None] + offsets[None, :]) * fps).astype(int)
        indices = np.clip(indices, 0, len(reader) - 1)
        unique_indices, inverse = np.unique(indices.reshape(-1), return_inverse=True)
        decoded_batch = reader.get_batch(unique_indices)
        decoded = (
            decoded_batch.asnumpy()
            if hasattr(decoded_batch, "asnumpy")
            else decoded_batch.cpu().numpy()
        )
        unique_frames = preprocess_frames(decoded)
        inverse_tensor = torch.as_tensor(inverse, device="cuda")
        clips = unique_frames[inverse_tensor].reshape(
            len(batch_starts), FRAMES_PER_CLIP, 3, 224, 224
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = model.compute_video(clips)
        all_features.append(features.float().cpu().numpy())
        if begin == 0 or begin + batch_size >= len(starts) or begin % (batch_size * 10) == 0:
            print(f"    clips {min(begin + batch_size, len(starts))}/{len(starts)}", flush=True)

    centers = starts + WINDOW_SECONDS / 2
    return np.concatenate(all_features), centers, fps, duration


def extract_text(model, question: str):
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokens = tokenizer(question, return_tensors="pt", truncation=True, max_length=24)
    tokens = {key: value.cuda() for key, value in tokens.items()}
    with torch.inference_mode():
        hidden = model.text_model(**tokens).last_hidden_state[0]
    length = int(tokens["attention_mask"][0].sum())
    # Match the SnAG token feature convention: omit CLS and SEP.
    return hidden[1 : max(length - 1, 1)].float().cpu().numpy()


def main() -> None:
    args = parse_args()
    root = Path(args.egovlp_root).resolve()
    output = Path(args.output_dir).resolve()
    video_out, text_out, meta_out = output / "video", output / "text", output / "metadata"
    for directory in (video_out, text_out, meta_out):
        directory.mkdir(parents=True, exist_ok=True)

    samples = json.loads(Path(args.manifest).read_text())["samples"]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    model = load_model(root, Path(args.checkpoint).resolve())

    for position, sample in enumerate(samples, start=1):
        stem = Path(sample["video_path"]).stem
        video_file = Path(args.video_dir) / sample["video_path"]
        video_file_out = video_out / f"{stem}.npy"
        text_file_out = text_out / f"{stem}.npy"
        metadata_file_out = meta_out / f"{stem}.json"
        if video_file_out.exists() and text_file_out.exists() and metadata_file_out.exists():
            print(f"[{position}/{len(samples)}] {stem}: cached", flush=True)
            continue
        if not video_file.exists():
            raise FileNotFoundError(video_file)

        print(f"[{position}/{len(samples)}] {stem}: extracting", flush=True)
        features, centers, fps, duration = extract_video(
            model, video_file, args.batch_size, args.max_clips, args.target_clips
        )
        text_features = extract_text(model, sample["question"])
        np.save(video_file_out, features.astype(np.float32))
        np.save(text_file_out, text_features.astype(np.float32))
        metadata_file_out.write_text(
            json.dumps(
                {
                    "video_path": sample["video_path"],
                    "question": sample["question"],
                    "fps": fps,
                    "duration_seconds": duration,
                    "window_seconds": WINDOW_SECONDS,
                    "stride_seconds": STRIDE_SECONDS,
                    "smoke_test_uniform_timeline": args.target_clips is not None,
                    "feature_center_seconds": centers.tolist(),
                    "video_feature_shape": list(features.shape),
                    "text_feature_shape": list(text_features.shape),
                },
                indent=2,
            )
        )
        print(f"    saved video={features.shape}, text={text_features.shape}", flush=True)


if __name__ == "__main__":
    main()
