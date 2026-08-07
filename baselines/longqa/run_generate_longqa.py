#!/usr/bin/env python3
"""Generate MCQ predictions for the LongQA dataset.

For each question, feeds the video and MCQ options to the model
and writes the predicted answer letter to the output file.

Usage:
  python run_generate_longqa.py --video-folder /path/to/videos
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from longqa_utils import (
    PROMPT_VARIANTS,
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    normalize_answer,
    sample_key,
)

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_if_exists(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        return []
    return load_jsonl(path)


def _build_prompt_with_timestamps(
    row: dict[str, object],
    prompt_variant: str,
    timestamps: list[float],
) -> str:
    prompt = build_longqa_prompt(
        row["question"], row["mcq_options"], prompt_variant=prompt_variant
    )
    if not timestamps:
        return prompt
    timestamp_index = ", ".join(
        f"image {index}={timestamp:.1f}s"
        for index, timestamp in enumerate(timestamps, start=1)
    )
    return (
        "The images are in chronological order. Their timestamps from the "
        f"start of the video are: {timestamp_index}. Use them to distinguish "
        "repeated events and verify temporal order.\n\n"
        + prompt
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LongQA MCQ predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
        help=(
            "Input JSONL file (relative to script dir; default points at the "
            "egolongqa split next to starter_kit/ in the HF dataset repo)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/egolongqa/predictions.jsonl",
        help="Output prediction JSONL file (relative to script dir).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="../egolongqa/val",
        help=(
            "Folder containing the video files (relative to script dir; "
            "default mirrors the HF repo layout)."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama4",
        choices=["llama4", "qwen"],
        help="Model type to use.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="HuggingFace model ID override (default: per model type).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum number of frames to sample from each video. Frames are sampled uniformly across the video duration. Higher values give the model more visual context but use more GPU memory. For a 3-min video, 32 frames ≈ 1 frame every 5.6 seconds.",
    )
    parser.add_argument(
        "--frames-per-interval",
        type=int,
        default=4,
        help=(
            "Number of frames sampled uniformly from the LongQA full-video "
            "interval before max-frames downsampling."
        ),
    )
    parser.add_argument(
        "--uniform-sampling",
        choices=["legacy", "endpoint_inclusive", "midpoint"],
        default="legacy",
        help="Uniform frame-position policy for full-video sampling.",
    )
    parser.add_argument(
        "--include-frame-timestamps",
        action="store_true",
        help="List the timestamp corresponding to every sampled image in the prompt.",
    )
    parser.add_argument(
        "--media-mode",
        choices=("images", "video"),
        default="images",
        help="Send sampled frames as separate images or one chronological video payload.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only first N samples (for debugging).",
    )
    parser.add_argument(
        "--subset-file",
        type=str,
        default=None,
        help="Optional JSON subset file containing sample keys to run.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="baseline",
        help="LongQA prompt variant (default: baseline).",
    )
    parser.add_argument(
        "--longqa-max-new-tokens",
        type=int,
        default=16,
        help="Maximum answer tokens per LongQA generation call (default: 16).",
    )
    parser.add_argument(
        "--require-final-answer-marker",
        action="store_true",
        help=(
            "Require `Final Answer: X`; if absent, append the first response as "
            "assistant context and request a short final-answer continuation."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: auto per model type).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic evaluation after generation.",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Output path for evaluation results JSON (default: output/<name>_results.json).",
    )
    parser.add_argument(
        "--no-resume-predictions",
        action="store_true",
        help="Ignore an existing output file and regenerate predictions from the start.",
    )
    # --- vLLM backend args (forwarded from run_evaluation.py via SLURM) ---
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' or 'vllm' (default: hf).",
    )
    parser.add_argument(
        "--tp", type=int, default=None, help="Tensor parallel size (vllm only)."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent HTTP requests (vllm only).",
    )
    from slurm_runner import add_slurm_args

    add_slurm_args(parser)
    args = parser.parse_args()
    if args.media_mode == "video" and args.backend != "vllm":
        parser.error("--media-mode video currently requires --backend vllm")
    if args.media_mode == "video" and args.require_final_answer_marker:
        parser.error("answer-marker retries are not implemented for video media mode")

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    video_folder = _resolve_path(args.video_folder)

    if args.slurm_nodes > 0:
        _submit_slurm(args, input_path, output_path, video_folder)
        return

    _run_local(args, input_path, output_path, video_folder)


def _submit_slurm(
    args: argparse.Namespace,
    input_path: str,
    output_path: str,
    video_folder: str,
) -> None:
    from slurm_runner import submit

    extra = [
        "--model-type",
        args.model_type,
        "--video-folder",
        video_folder,
        "--max-frames",
        str(args.max_frames),
        "--frames-per-interval",
        str(args.frames_per_interval),
        "--prompt-variant",
        args.prompt_variant,
        "--uniform-sampling",
        args.uniform_sampling,
        "--longqa-max-new-tokens",
        str(getattr(args, "longqa_max_new_tokens", 16)),
    ]
    if getattr(args, "require_final_answer_marker", False):
        extra.append("--require-final-answer-marker")
    if getattr(args, "include_frame_timestamps", False):
        extra.append("--include-frame-timestamps")
    if getattr(args, "media_mode", "images") != "images":
        extra.extend(["--media-mode", args.media_mode])
    if args.subset_file:
        extra.extend(["--subset-file", args.subset_file])
    if args.llm_model:
        extra.extend(["--llm-model", args.llm_model])
    if args.max_samples:
        extra.extend(["--max-samples", str(args.max_samples)])
    if args.batch_size:
        extra.extend(["--batch-size", str(args.batch_size)])
    if args.num_gpus is not None:
        extra.extend(["--num-gpus", str(args.num_gpus)])
    if getattr(args, "backend", "hf") != "hf":
        extra.extend(["--backend", args.backend])
    if getattr(args, "tp", None) is not None:
        extra.extend(["--tp", str(args.tp)])
    if getattr(args, "concurrency", 16) != 16:
        extra.extend(["--concurrency", str(args.concurrency)])
    submit(
        script=os.path.abspath(__file__),
        input_path=input_path,
        output_path=output_path,
        num_nodes=args.slurm_nodes,
        extra_args=extra,
        partition=args.slurm_partition,
        reservation=args.slurm_reservation,
        conda_env=args.conda_env,
        conda_base=args.conda_base,
        gpus_per_node=args.slurm_gpus,
        time_limit=args.slurm_time,
    )


def _run_local(
    args: argparse.Namespace,
    input_path: str,
    output_path: str,
    video_folder: str,
) -> None:
    from model import DEFAULT_GPU_COUNTS, detect_gpu_count

    all_data = load_jsonl(input_path)
    all_data = apply_subset(all_data, getattr(args, "subset_file", None))
    if args.max_samples is not None:
        all_data = all_data[: args.max_samples]

    available = detect_gpu_count()
    num_gpus = args.num_gpus if args.num_gpus is not None else available
    if num_gpus > available:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only {available} available. "
            f"Check --num-gpus or CUDA_VISIBLE_DEVICES."
        )
    backend = getattr(args, "backend", "hf")
    # vllm: each worker is an independent server with TP=args.tp (defaults to
    # model.DEFAULT_TP_SIZES[model_type]). Use that as the per-worker GPU count
    # so we can data-parallel across the remaining GPUs on the node.
    if backend == "vllm":
        from model import DEFAULT_TP_SIZES

        gpus_per_model = (
            args.tp
            if getattr(args, "tp", None)
            else DEFAULT_TP_SIZES.get(args.model_type, 1)
        )
    else:
        gpus_per_model = DEFAULT_GPU_COUNTS.get(args.model_type, 1)
        if num_gpus < gpus_per_model:
            raise RuntimeError(
                f"{args.model_type} requires at least {gpus_per_model} GPUs but only "
                f"{num_gpus} available. Allocate more GPUs or choose a smaller model "
                f"(e.g. qwen)."
            )
    num_workers = max(1, num_gpus // gpus_per_model)
    if num_workers <= 1:
        _run_single(args, all_data, output_path, video_folder)
    else:
        _run_parallel(
            args,
            all_data,
            output_path,
            video_folder,
            num_workers,
            gpus_per_model,
        )

    if not args.no_eval and os.path.exists(output_path):
        _run_longqa_eval(input_path, output_path, args.eval_output)


def _run_longqa_eval(
    input_path: str, output_path: str, eval_output: str | None
) -> None:
    from run_evaluation import (
        _filter_subset,
        evaluate_longqa,
        load_jsonl as load_eval_jsonl,
    )

    golden = load_eval_jsonl(input_path)
    preds = load_eval_jsonl(output_path)
    if len(golden) != len(preds):
        logger.warning(
            "Golden (%d) and predictions (%d) have different lengths",
            len(golden),
            len(preds),
        )
    golden, preds = _filter_subset(golden, preds, "longqa")
    if not preds:
        raise RuntimeError("No LongQA predictions matched the annotation rows")
    results = evaluate_longqa(golden, preds)

    if not eval_output:
        base = os.path.splitext(os.path.basename(output_path))[0]
        eval_output = os.path.join(
            os.path.dirname(output_path) or ".",
            "..",
            "output",
            f"{base}_results.json",
        )
    eval_output = os.path.normpath(_resolve_path(eval_output))
    os.makedirs(os.path.dirname(eval_output) or ".", exist_ok=True)

    with open(eval_output, "w") as f:
        json.dump(results, f, indent=2)

    accuracy = results.get("accuracy", 0.0)
    correct = results.get("correct", 0)
    total = results.get("total", 0)
    print(f"LongQA Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results written to {eval_output}")


def _run_single(args: object, data: list, output_path: str, video_folder: str) -> None:
    from model import (
        create_model,
        DEFAULT_BATCH_SIZES,
        extract_frames,
        reset_prompt_token_stats,
        setup_gpus,
        summarize_prompt_token_stats,
        uniform_full_video_indices,
    )

    backend = getattr(args, "backend", "hf")
    media_mode = getattr(args, "media_mode", "images")
    if media_mode == "video" and backend != "vllm":
        raise ValueError("video media mode requires the vLLM backend")
    if args.model_type == "qwen" and backend == "vllm":
        os.environ["VLLM_QWEN_MEDIA_MODE"] = media_mode
    if backend != "vllm":
        setup_gpus(args.num_gpus, args.model_type)
    model = create_model(
        args.model_type,
        args.llm_model,
        backend=backend,
        tp_size=getattr(args, "tp", None),
        concurrency=getattr(args, "concurrency", 16),
        max_frames=args.max_frames,
    )
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES.get(args.model_type, 1)
    reset_prompt_token_stats()

    print(
        f"Generating predictions for {len(data)} samples "
        f"(1 worker, batch_size={batch_size}, backend={backend})..."
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    existing_predictions = (
        load_jsonl_if_exists(output_path)
        if not getattr(args, "no_resume_predictions", False)
        else []
    )
    resume_count = 0
    for row_idx, pred in enumerate(existing_predictions[: len(data)]):
        if sample_key(pred) != sample_key(data[row_idx]):
            break
        if not str(pred.get("mcq_answer", "")).strip():
            break
        if str(pred.get("uniform_sampling", "legacy")) != str(
            getattr(args, "uniform_sampling", "legacy")
        ):
            break
        if bool(pred.get("timestamps_in_prompt", False)) != bool(
            getattr(args, "include_frame_timestamps", False)
        ):
            break
        if str(pred.get("media_mode", "images")) != str(
            getattr(args, "media_mode", "images")
        ):
            break
        resume_count += 1
    if len(existing_predictions) != resume_count:
        with open(output_path, "w") as cached_f:
            for pred in existing_predictions[:resume_count]:
                cached_f.write(json.dumps(pred) + "\n")
    if resume_count:
        print(f"Resuming predictions from {resume_count}/{len(data)} cached rows")
    mode = "a" if resume_count else "w"
    with model, open(output_path, mode) as out_f:
        for batch_start in range(resume_count, len(data), batch_size):
            batch = data[batch_start : batch_start + batch_size]
            batch_frames = []
            batch_timestamps = []
            for row in batch:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                if not os.path.isfile(video_path):
                    raise RuntimeError(
                        f"Video is unavailable; refusing video-blind prediction: "
                        f"{video_path}"
                    )
                frames = extract_frames(
                    video_path,
                    frames_per_interval=args.frames_per_interval,
                    max_frames=args.max_frames,
                    sampling_mode=getattr(args, "uniform_sampling", "legacy"),
                )
                if not frames:
                    raise RuntimeError(
                        f"No frames extracted; refusing video-blind prediction: "
                        f"{video_path}"
                    )
                batch_frames.append(frames)
                timestamps: list[float] = []
                if getattr(args, "include_frame_timestamps", False):
                    import cv2

                    cap = cv2.VideoCapture(video_path)
                    try:
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    finally:
                        cap.release()
                    indices = uniform_full_video_indices(
                        total_frames,
                        args.frames_per_interval,
                        args.max_frames,
                        getattr(args, "uniform_sampling", "legacy"),
                    )
                    timestamps = [index / fps for index in indices] if fps > 0 else []
                    if len(frames) != len(timestamps):
                        raise RuntimeError(
                            f"Extracted {len(frames)}/{len(timestamps)} timestamped "
                            f"frames for {video_path}"
                        )
                batch_timestamps.append(timestamps)
            batch_messages = [
                [
                    {
                        "role": "user",
                        "content": _build_prompt_with_timestamps(
                            row, args.prompt_variant, timestamps
                        ),
                    }
                ]
                for row, timestamps in zip(batch, batch_timestamps)
            ]
            if getattr(args, "media_mode", "images") == "video":
                responses = [
                    model.generate_video_frames(
                        frames,
                        messages,
                        max_new_tokens=getattr(args, "longqa_max_new_tokens", 16),
                    )
                    for frames, messages in zip(batch_frames, batch_messages)
                ]
            else:
                responses = model.generate_batch(
                    batch_frames,
                    batch_messages,
                    max_new_tokens=getattr(args, "longqa_max_new_tokens", 16),
                )
            responses = _complete_missing_final_answers(
                model,
                batch_frames,
                batch_messages,
                responses,
                getattr(args, "require_final_answer_marker", False),
            )
            for row, response, timestamps in zip(batch, responses, batch_timestamps):
                pred = build_prediction_row(
                    row,
                    response,
                    prompt_variant=args.prompt_variant,
                )
                pred["longqa_max_new_tokens"] = getattr(
                    args, "longqa_max_new_tokens", 16
                )
                pred["uniform_sampling"] = getattr(
                    args, "uniform_sampling", "legacy"
                )
                pred["timestamps_in_prompt"] = bool(
                    getattr(args, "include_frame_timestamps", False)
                )
                pred["frame_timestamps"] = [round(value, 3) for value in timestamps]
                pred["media_mode"] = getattr(args, "media_mode", "images")
                out_f.write(json.dumps(pred) + "\n")
                out_f.flush()
            done = min(batch_start + batch_size, len(data))
            print(f"  Progress: {done}/{len(data)}")
    print(f"Predictions written to {output_path}")
    _print_context_summary(summarize_prompt_token_stats())


def _worker_fn(
    rank: int,
    gpus_per_model: int,
    args: object,
    shard: list,
    out_file: str,
    video_folder: str,
) -> None:
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        visible = [x for x in parent_cvd.split(",") if x.strip()]
        gpu_ids = visible[rank * gpus_per_model : (rank + 1) * gpus_per_model]
    else:
        gpu_ids = [
            str(g) for g in range(rank * gpus_per_model, (rank + 1) * gpus_per_model)
        ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    media_mode = getattr(args, "media_mode", "images")
    if media_mode == "video" and getattr(args, "backend", "hf") != "vllm":
        raise ValueError("video media mode requires the vLLM backend")
    if args.model_type == "qwen" and getattr(args, "backend", "hf") == "vllm":
        os.environ["VLLM_QWEN_MEDIA_MODE"] = media_mode

    from model import (
        create_model,
        DEFAULT_BATCH_SIZES,
        extract_frames,
        reset_prompt_token_stats,
        summarize_prompt_token_stats,
        uniform_full_video_indices,
    )

    model = create_model(
        args.model_type,
        args.llm_model,
        backend=getattr(args, "backend", "hf"),
        tp_size=getattr(args, "tp", None),
        concurrency=getattr(args, "concurrency", 16),
        max_frames=args.max_frames,
    )
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES.get(args.model_type, 1)
    reset_prompt_token_stats()

    # `with model:` is required so VLLMModel.__enter__ starts the vllm
    # subprocess and assigns self._port; without it generate_batch fails
    # with "nonnumeric port: 'None'". HF models tolerate no-op __enter__,
    # so the context wrapper is safe for both backends.
    with model, open(out_file, "w") as out_f:
        for batch_start in range(0, len(shard), batch_size):
            batch = shard[batch_start : batch_start + batch_size]
            batch_frames = []
            batch_timestamps = []
            for row in batch:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                frames = extract_frames(
                    video_path,
                    frames_per_interval=args.frames_per_interval,
                    max_frames=args.max_frames,
                    sampling_mode=getattr(args, "uniform_sampling", "legacy"),
                )
                batch_frames.append(frames)
                timestamps: list[float] = []
                if getattr(args, "include_frame_timestamps", False):
                    import cv2

                    cap = cv2.VideoCapture(video_path)
                    try:
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    finally:
                        cap.release()
                    indices = uniform_full_video_indices(
                        total_frames,
                        args.frames_per_interval,
                        args.max_frames,
                        getattr(args, "uniform_sampling", "legacy"),
                    )
                    timestamps = [index / fps for index in indices] if fps > 0 else []
                    if len(frames) != len(timestamps):
                        raise RuntimeError(
                            f"Extracted {len(frames)}/{len(timestamps)} timestamped "
                            f"frames for {video_path}"
                        )
                batch_timestamps.append(timestamps)
            batch_messages = [
                [
                    {
                        "role": "user",
                        "content": _build_prompt_with_timestamps(
                            row, args.prompt_variant, timestamps
                        ),
                    }
                ]
                for row, timestamps in zip(batch, batch_timestamps)
            ]
            if getattr(args, "media_mode", "images") == "video":
                responses = [
                    model.generate_video_frames(
                        frames,
                        messages,
                        max_new_tokens=getattr(args, "longqa_max_new_tokens", 16),
                    )
                    for frames, messages in zip(batch_frames, batch_messages)
                ]
            else:
                responses = model.generate_batch(
                    batch_frames,
                    batch_messages,
                    max_new_tokens=getattr(args, "longqa_max_new_tokens", 16),
                )
            responses = _complete_missing_final_answers(
                model,
                batch_frames,
                batch_messages,
                responses,
                getattr(args, "require_final_answer_marker", False),
            )
            for row, response, timestamps in zip(batch, responses, batch_timestamps):
                pred = build_prediction_row(
                    row,
                    response,
                    prompt_variant=args.prompt_variant,
                )
                pred["longqa_max_new_tokens"] = getattr(
                    args, "longqa_max_new_tokens", 16
                )
                pred["uniform_sampling"] = getattr(
                    args, "uniform_sampling", "legacy"
                )
                pred["timestamps_in_prompt"] = bool(
                    getattr(args, "include_frame_timestamps", False)
                )
                pred["frame_timestamps"] = [round(value, 3) for value in timestamps]
                pred["media_mode"] = getattr(args, "media_mode", "images")
                out_f.write(json.dumps(pred) + "\n")
                out_f.flush()
            done = min(batch_start + batch_size, len(shard))
            print(f"  [Worker {rank}] Progress: {done}/{len(shard)}")
    _print_context_summary(summarize_prompt_token_stats(), prefix=f"[Worker {rank}] ")
    print(f"  [Worker {rank}] Done: {out_file}")


def has_final_answer_marker(response: object) -> bool:
    return bool(
        re.search(
            r"\bfinal\s+answer\s*[:.]?\s*\(?[A-D]\)?\b",
            str(response),
            re.IGNORECASE,
        )
    )


def has_unambiguous_final_answer(response: object) -> bool:
    """Accept either the requested marker or an answer-only option letter."""
    if has_final_answer_marker(response):
        return True
    return bool(
        re.fullmatch(
            r"\s*\(?[A-D]\)?[\.:]?\s*",
            str(response),
            re.IGNORECASE,
        )
    )


def _complete_missing_final_answers(
    model: object,
    batch_frames: list[list[object]],
    batch_messages: list[list[dict[str, str]]],
    responses: list[str],
    required: bool,
) -> list[str]:
    if not required:
        return responses
    missing = [
        index
        for index, response in enumerate(responses)
        if not has_unambiguous_final_answer(response)
    ]
    if not missing:
        return responses
    retry_messages = [
        batch_messages[index]
        + [{"role": "assistant", "content": responses[index]}]
        + [
            {
                "role": "user",
                "content": (
                    "Using the reasoning above, end now with exactly "
                    "`Final Answer: X`, where X is A, B, C, or D."
                ),
            }
        ]
        for index in missing
    ]
    retry_frames = [batch_frames[index] for index in missing]
    final_answer_batch = getattr(model, "generate_final_answer_batch", None)
    if callable(final_answer_batch):
        retry_responses = final_answer_batch(
            retry_frames,
            retry_messages,
            max_new_tokens=64,
        )
    else:
        retry_responses = model.generate_batch(
            retry_frames,
            retry_messages,
            max_new_tokens=64,
        )
    completed = list(responses)
    for index, retry in zip(missing, retry_responses):
        retry_text = str(retry)
        if not has_final_answer_marker(retry_text) and has_unambiguous_final_answer(
            retry_text
        ):
            retry_text = f"Final Answer: {normalize_answer(retry_text)}"
        completed[index] = f"{responses[index]}\n\n{retry_text}"
    still_missing = [
        index
        for index, response in enumerate(completed)
        if not has_unambiguous_final_answer(response)
    ]
    if still_missing:
        raise RuntimeError(
            "Final-answer retry failed for "
            f"{len(still_missing)} response(s): indices {still_missing[:8]}. "
            "Refusing to write parser-dependent predictions."
        )
    return completed


def _print_context_summary(stats: dict[str, object], prefix: str = "") -> None:
    count = int(stats.get("count", 0) or 0)
    if count == 0:
        print(f"{prefix}Context fill: unavailable (no prompt token usage recorded)")
        return

    context_window = stats.get("context_window")
    token_part = (
        f"samples={count}, tokens min/mean/p50/p95/max="
        f"{stats['min']}/{stats['mean']}/{stats['p50']}/{stats['p95']}/{stats['max']}"
    )
    if context_window:
        print(
            f"{prefix}Context fill: {token_part}, window={context_window}, "
            f"fill mean/p95/max="
            f"{stats['mean_fill_pct']}%/{stats['p95_fill_pct']}%/{stats['max_fill_pct']}%"
        )
    else:
        print(f"{prefix}Context fill: {token_part}, window=unknown")


def _predownload_model(args: object) -> None:
    """Pre-download model weights so workers load from cache."""
    from model import DEFAULT_MODEL_IDS

    model_id = args.llm_model or DEFAULT_MODEL_IDS.get(args.model_type, "")
    if model_id and not os.path.isdir(model_id):
        print(f"Pre-downloading model {model_id}...")
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(model_id)
        from huggingface_hub import snapshot_download

        snapshot_download(model_id)


def _spawn_workers(
    data: list,
    output_path: str,
    video_folder: str,
    num_workers: int,
    gpus_per_model: int,
    args: object,
) -> tuple[list[str], list]:
    """Create shards and spawn one worker process per shard."""
    import torch.multiprocessing as mp

    shard_files = []
    processes = []
    for rank in range(num_workers):
        shard = data[rank::num_workers]
        base, ext = os.path.splitext(output_path)
        shard_file = f"{base}.shard{rank}{ext}"
        shard_files.append(shard_file)
        p = mp.Process(
            target=_worker_fn,
            args=(rank, gpus_per_model, args, shard, shard_file, video_folder),
        )
        p.start()
        processes.append(p)
    return shard_files, processes


def _join_workers(processes: list) -> None:
    """Wait for all worker processes and terminate any that exceed the timeout."""
    for p in processes:
        p.join(timeout=3600)
        if p.is_alive():
            logger.error(
                "Worker pid=%d still alive after 3600s timeout, terminating", p.pid
            )
            p.terminate()
            p.join(timeout=30)
    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Workers {failed} failed. Check logs above.")


def _merge_shards(
    shard_files: list[str],
    data: list,
    output_path: str,
    num_workers: int,
) -> None:
    """Merge per-worker shard files back into a single output in original order."""
    shard_data: dict[int, list[dict[str, object]]] = {}
    for rank, f in enumerate(shard_files):
        try:
            shard_data[rank] = load_jsonl(f)
        except FileNotFoundError:
            logger.warning(
                "Shard file %s not found (worker %d exited 0 but produced no output)",
                f,
                rank,
            )
            shard_data[rank] = []
    missing_count = 0
    with open(output_path, "w") as out_f:
        for idx in range(len(data)):
            rank = idx % num_workers
            shard_idx = idx // num_workers
            if shard_idx < len(shard_data[rank]):
                out_f.write(json.dumps(shard_data[rank][shard_idx]) + "\n")
            else:
                missing_count += 1
                placeholder = {
                    "video_path": data[idx].get("video_path", ""),
                    "mcq_answer": "",
                }
                out_f.write(json.dumps(placeholder) + "\n")
                logger.warning(
                    "Missing prediction for sample %d (worker %d, shard_idx %d): "
                    "shard has %d items, expected at least %d — wrote placeholder",
                    idx,
                    rank,
                    shard_idx,
                    len(shard_data[rank]),
                    shard_idx + 1,
                )
    if missing_count > 0:
        logger.warning(
            "Total missing predictions: %d / %d — output may be incomplete",
            missing_count,
            len(data),
        )
    for f in shard_files:
        Path(f).unlink(missing_ok=True)


def _run_parallel(
    args: object,
    data: list,
    output_path: str,
    video_folder: str,
    num_workers: int,
    gpus_per_model: int,
) -> None:
    import torch.multiprocessing as mp

    print(
        f"Generating predictions for {len(data)} samples "
        f"({num_workers} workers, {gpus_per_model} GPU(s)/worker)..."
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _predownload_model(args)

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    shard_files, processes = _spawn_workers(
        data, output_path, video_folder, num_workers, gpus_per_model, args
    )
    _join_workers(processes)
    _merge_shards(shard_files, data, output_path, num_workers)
    print(f"Predictions written to {output_path} (merged from {num_workers} shards)")


if __name__ == "__main__":
    main()
