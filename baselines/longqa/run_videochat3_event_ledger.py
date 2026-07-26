#!/usr/bin/env python3
"""Generate a question-independent temporal event ledger with VideoChat3."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import types
from pathlib import Path
from typing import Any


MODEL_ID = "MCG-NJU/VideoChat3-4B"
DEFAULT_VIDEO_FOLDER = "/scratch/inf0/user/kkumar/val"
DEFAULT_SUBSET_FILE = "../../configs/egolongqa_dev140_video_ids_seed20260709.json"

LEDGER_PROMPT = """You are annotating one chronological interval from an egocentric video.
The supplied images are ordered and labelled with their timestamps. Describe only what
is visibly supported within this interval. Return exactly one JSON object with this schema:
{
  "location": "short place description or unknown",
  "entities": ["salient visible entities"],
  "caption": "a rich two-to-four sentence description of this interval",
  "action": "one concise description of what the wearer does", 
  "state_changes": ["object or scene state transition"],
  "confidence": 0.0
}
Use an empty list for state_changes when no transition is visible. confidence must be a
number from 0 to 1 reflecting how well the images support the record. Do not mention
frames, timestamps, or information outside this interval."""


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def window_bounds(duration: float, window_seconds: float) -> list[tuple[float, float]]:
    return [
        (start, min(start + window_seconds, duration))
        for start in range(0, math.ceil(duration), math.ceil(window_seconds))
    ]


def frame_indices_for_window(
    start: float,
    end: float,
    fps: float,
    total_frames: int,
    count: int,
) -> list[int]:
    first = max(0, min(int(start * fps), total_frames - 1))
    last = max(first, min(int(end * fps) - 1, total_frames - 1))
    if count == 1 or first == last:
        return [first]
    return sorted(
        {
            round(first + (last - first) * position / (count - 1))
            for position in range(count)
        }
    )


def parse_event_record(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    record = json.loads(text)
    return {
        "location": str(record.get("location", "unknown")).strip() or "unknown",
        "entities": [str(value).strip() for value in record.get("entities", []) if str(value).strip()],
        "caption": str(record.get("caption", "")).strip(),
        "action": str(record.get("action", "")).strip(),
        "state_changes": [
            str(value).strip()
            for value in record.get("state_changes", [])
            if str(value).strip()
        ],
        "confidence": float(record.get("confidence", 0.0)),
    }


def load_video_ids(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text())
    return [str(video_id) for video_id in payload["video_ids"]]


def video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return fps, total_frames


def extract_window_frames(
    video_path: str,
    indices: list[int],
) -> tuple[list[object], float, int]:
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames, fps, total_frames


def _prepare_videochat3_inputs(
    self: object,
    input_ids: object,
    past_key_values: object = None,
    inputs_embeds: object = None,
    pixel_values: object = None,
    image_grid_thw: object = None,
    pixel_values_videos: object = None,
    video_grid_thw: object = None,
    attention_mask: object = None,
    cache_position: object = None,
    logits_to_keep: object = None,
    **kwargs: object,
) -> dict[str, object]:
    """Adapt VideoChat3's 4.57 generation hook to Transformers 5.8."""
    model_inputs = super(type(self), self).prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )
    input_embeds = model_inputs.get("inputs_embeds")
    prepared_ids = model_inputs.get("input_ids")
    is_decoding_step = (
        input_embeds is not None and input_embeds.shape[1] == 1
    ) or (prepared_ids is not None and prepared_ids.shape[1] == 1)
    if cache_position is not None and cache_position[0] != 0 and is_decoding_step:
        model_inputs["pixel_values"] = None
        model_inputs["pixel_values_videos"] = None
    return model_inputs


class VideoChat3:
    def __init__(self, model_id: str, min_pixels: int, max_pixels: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.prepare_inputs_for_generation = types.MethodType(
            _prepare_videochat3_inputs, self.model
        )
        # VideoChat3 defaults its vision tower to FlashAttention 2, which is
        # unavailable for this torch/CUDA build. SDPA is slower but runs on the 3090.
        vision_tower = self.model.model.vision_tower
        vision_tower.config.attn_impl = "sdpa"
        for block in vision_tower.encoder.blocks:
            block.attn_impl = "sdpa"
        self.model.eval()

    def generate(
        self, frames: list[object], timestamps: list[float], max_new_tokens: int
    ) -> str:
        from qwen_vl_utils import process_vision_info

        content: list[dict[str, object]] = []
        for frame, timestamp in zip(frames, timestamps):
            content.append(
                {
                    "type": "image",
                    "image": frame,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
            content.append({"type": "text", "text": f"Timestamp: {timestamp:.2f}s"})
        content.append({"type": "text", "text": LEDGER_PROMPT})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=14,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        ).to(self.model.device)
        # Cast visual tensors only: casting input_ids to BF16 destroys the
        # VideoChat3 image-placeholder token IDs.
        if hasattr(self.model, "dtype"):
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            if "pixel_values_videos" in inputs:
                inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                    self.model.dtype
                )
        # VideoChat3's generation hook targets Transformers 4.57. Run greedy
        # decoding directly so its image placeholders remain in the prefill.
        import torch

        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=True, return_dict=True)
            generated_ids = []
            attention_mask = inputs["attention_mask"]
            past_key_values = outputs.past_key_values
            next_ids = outputs.logits[:, -1:].argmax(dim=-1)
            for _ in range(max_new_tokens):
                token_id = int(next_ids[0, 0])
                if token_id == self.processor.tokenizer.eos_token_id:
                    break
                generated_ids.append(token_id)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_ids)], dim=1
                )
                outputs = self.model(
                    input_ids=next_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=torch.tensor(
                        [attention_mask.shape[1] - 1], device=self.model.device
                    ),
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_ids = outputs.logits[:, -1:].argmax(dim=-1)
        return self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-folder", default=DEFAULT_VIDEO_FOLDER)
    parser.add_argument("--subset-file", default=DEFAULT_SUBSET_FILE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--window-seconds", type=float, default=7.0)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Sample frames at this rate within each caption window instead of using a fixed count.",
    )
    parser.add_argument(
        "--select-shortest",
        action="store_true",
        help="Run only the shortest video in the selected subset.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_folder = _resolve_path(args.video_folder)
    subset_file = _resolve_path(args.subset_file)
    output_path = Path(_resolve_path(args.output))
    video_ids = [args.video_id] if args.video_id else load_video_ids(subset_file)
    if args.select_shortest:
        video_ids = [
            min(
                video_ids,
                key=lambda video_id: (
                    lambda metadata: metadata[1] / metadata[0]
                )(video_metadata(os.path.join(video_folder, video_id))),
            )
        ]
    if args.max_videos is not None:
        video_ids = video_ids[: args.max_videos]

    completed: set[tuple[str, float, float]] = set()
    if not args.no_resume and output_path.exists():
        for line in output_path.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["video_id"], row["start_time"], row["end_time"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = VideoChat3(args.model, args.min_pixels, args.max_pixels)
    run_started = time.perf_counter()
    with output_path.open("a" if completed else "w") as handle:
        for video_number, video_id in enumerate(video_ids, start=1):
            video_path = os.path.join(video_folder, video_id)
            fps, total_frames = video_metadata(video_path)
            duration = total_frames / fps
            for start, end in window_bounds(duration, args.window_seconds):
                key = (video_id, start, end)
                if key in completed:
                    continue
                if args.sample_fps is None:
                    indices = frame_indices_for_window(
                        start, end, fps, total_frames, args.frames_per_window
                    )
                else:
                    stride = max(1, round(fps / args.sample_fps))
                    first = int(start * fps)
                    last = min(int(end * fps), total_frames)
                    indices = list(range(first, last, stride))
                frames, _fps, _total_frames = extract_window_frames(video_path, indices)
                timestamps = [index / fps for index in indices]
                started = time.perf_counter()
                response = model.generate(frames, timestamps, args.max_new_tokens)
                wall_time_seconds = time.perf_counter() - started
                record = parse_event_record(response)
                row = {
                    "video_id": video_id,
                    "start_time": start,
                    "end_time": end,
                    "frame_indices": indices,
                    "frame_timestamps": timestamps,
                    "record": record,
                    "raw_response": response,
                    "model": args.model,
                    "sampling_fps": args.sample_fps,
                    "wall_time_seconds": round(wall_time_seconds, 3),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            print(f"Completed {video_number}/{len(video_ids)}: {video_id}")
    print(f"Total wall time: {time.perf_counter() - run_started:.1f}s")


if __name__ == "__main__":
    main()
