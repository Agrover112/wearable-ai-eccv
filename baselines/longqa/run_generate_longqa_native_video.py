#!/usr/bin/env python3
"""Run Qwen3.5 LongQA on cached low-resolution video proxies through vLLM.

FFmpeg streams each source video into a proxy sampled at ``--video-fps``.
vLLM still receives one native ``video_url`` per question, but never needs to
materialize every full-resolution source frame in host memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from longqa_utils import (
    PROMPT_VARIANTS,
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    load_jsonl,
    sample_key,
)
from model import VLLMModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_VIDEO_FPS = 0.25
DEFAULT_MAX_MODEL_LEN = 131072
DEFAULT_PROXY_WIDTH = 480
DEFAULT_PROXY_HEIGHT = 256
NATIVE_VIDEO_FINGERPRINT_SCHEMA = 2
PROXY_CACHE_SCHEMA = 1


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def build_video_content(video_url: str, prompt: str) -> list[dict[str, object]]:
    """Build the OpenAI-compatible multimodal content for one local MP4."""
    return [
        {"type": "video_url", "video_url": {"url": video_url}},
        {"type": "text", "text": prompt},
    ]


def build_native_video_payload(
    model_id: str,
    video_url: str,
    prompt: str,
    max_new_tokens: int,
    video_fps: float,
) -> dict[str, object]:
    """Return the vLLM chat-completions payload for one native-video sample."""
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": build_video_content(video_url, prompt),
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "mm_processor_kwargs": {
            "fps": video_fps,
            "do_sample_frames": True,
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _common_media_root(video_paths: list[str]) -> str:
    """Return a real directory accepted by vLLM for every supplied MP4."""
    if not video_paths:
        raise ValueError("No video paths available to derive allowed media path")
    directories = [os.path.dirname(os.path.realpath(path)) for path in video_paths]
    root = os.path.commonpath(directories)
    if not os.path.isdir(root):
        raise ValueError(f"Derived allowed media path is not a directory: {root}")
    return root


def resolve_video_paths(
    rows: list[dict[str, Any]],
    video_folder: str,
    allowed_local_media_path: str | None = None,
) -> tuple[list[str], str]:
    """Resolve symlinked videos and validate vLLM local-media access.

    ``data/videos`` contains symlinks into the Hugging Face cache. Using the
    resolved file URI and its common real parent keeps vLLM's local-media
    allowlist aligned with the path that the server actually opens.
    """
    paths = [
        os.path.realpath(os.path.join(video_folder, str(row["video_path"])))
        for row in rows
    ]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Native video input does not exist: {path}")

    allowed = (
        os.path.realpath(allowed_local_media_path)
        if allowed_local_media_path
        else _common_media_root(paths)
    )
    if not os.path.isdir(allowed):
        raise ValueError(f"Allowed local media path is not a directory: {allowed}")
    for path in paths:
        if os.path.commonpath([allowed, path]) != allowed:
            raise ValueError(
                f"Video {path} is outside --allowed-local-media-path {allowed}"
            )
    return paths, allowed


def proxy_video_path(source_path: str, args: argparse.Namespace) -> str:
    """Return the content-addressed proxy path for one source video."""
    source_stat = os.stat(source_path)
    payload = {
        "schema": PROXY_CACHE_SCHEMA,
        "source": os.path.realpath(source_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "fps": args.video_fps,
        "width": args.proxy_width,
        "height": args.proxy_height,
        "codec": args.proxy_codec,
        "crf": args.proxy_crf,
        "preset": args.proxy_preset,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
    fps_tag = str(args.video_fps).replace(".", "p")
    variant = f"fps{fps_tag}_{args.proxy_width}x{args.proxy_height}"
    filename = f"{Path(source_path).stem}_{digest}.mp4"
    return os.path.join(args.proxy_cache_dir, variant, filename)


def _build_proxy_video(
    source_path: str,
    proxy_path: str,
    args: argparse.Namespace,
) -> str:
    if os.path.isfile(proxy_path):
        return proxy_path
    if args.require_proxies:
        raise FileNotFoundError(f"Native-video proxy does not exist: {proxy_path}")

    os.makedirs(os.path.dirname(proxy_path), exist_ok=True)
    temporary_path = f"{os.path.splitext(proxy_path)[0]}.{os.getpid()}.partial.mp4"
    video_filter = (
        f"fps={args.video_fps},"
        f"scale={args.proxy_width}:{args.proxy_height}:flags=lanczos"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            source_path,
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            args.proxy_codec,
            "-preset",
            args.proxy_preset,
            "-crf",
            str(args.proxy_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            temporary_path,
        ],
        check=True,
    )
    os.replace(temporary_path, proxy_path)
    return proxy_path


def prepare_video_proxies(
    source_paths: list[str],
    args: argparse.Namespace,
) -> list[str]:
    """Build or reuse streamed proxy videos while preserving input order."""
    from concurrent.futures import ThreadPoolExecutor

    proxy_paths = [proxy_video_path(source_path, args) for source_path in source_paths]
    work = zip(source_paths, proxy_paths)
    with ThreadPoolExecutor(max_workers=args.proxy_workers) as executor:
        futures = [
            executor.submit(_build_proxy_video, source_path, proxy_path, args)
            for source_path, proxy_path in work
        ]
        for index, future in enumerate(futures, start=1):
            future.result()
            print(f"  Proxy progress: {index}/{len(proxy_paths)}")
    return proxy_paths


def native_video_fingerprint(args: argparse.Namespace) -> str:
    """Hash all settings that change native-video inference or its evidence."""
    payload = {
        "schema": NATIVE_VIDEO_FINGERPRINT_SCHEMA,
        "input": os.path.abspath(args.input),
        "subset_file": os.path.abspath(args.subset_file) if args.subset_file else None,
        "video_folder": os.path.abspath(args.video_folder),
        "allowed_local_media_path": (
            os.path.abspath(args.allowed_local_media_path)
            if args.allowed_local_media_path
            else None
        ),
        "llm_model": args.llm_model,
        "video_fps": args.video_fps,
        "proxy_cache_schema": PROXY_CACHE_SCHEMA,
        "proxy_width": args.proxy_width,
        "proxy_height": args.proxy_height,
        "proxy_codec": args.proxy_codec,
        "proxy_crf": args.proxy_crf,
        "proxy_preset": args.proxy_preset,
        "max_model_len": args.max_model_len,
        "prompt_variant": args.prompt_variant,
        "max_new_tokens": args.max_new_tokens,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def resume_position(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    fingerprint: str,
) -> int:
    """Return the contiguous compatible prediction prefix for resumable runs."""
    start = 0
    for row, prediction in zip(rows, predictions):
        if sample_key(prediction) != sample_key(row):
            break
        if not str(prediction.get("mcq_answer", "")).strip():
            break
        if prediction.get("native_video_fingerprint") != fingerprint:
            break
        start += 1
    return start


class _NativeVideoVLLMModelMixin:
    """Native-video behaviour mixed into the existing vLLM lifecycle wrapper."""

    def _native_video_init(
        self,
        allowed_local_media_path: str,
        video_fps: float,
        max_model_len: int,
        gpu_memory_utilization: float,
    ) -> None:
        self.allowed_local_media_path = allowed_local_media_path
        self.video_fps = video_fps
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

    def server_args(self) -> list[str]:
        processor_kwargs = {
            "fps": self.video_fps,
            "do_sample_frames": True,
        }
        return [
            "--model",
            self.model_id,
            "--host",
            "127.0.0.1",
            "--tensor-parallel-size",
            str(self.tp_size),
            "--port",
            str(self._port),
            "--trust-remote-code",
            "--enforce-eager",
            "--gdn-prefill-backend",
            "triton",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--allowed-local-media-path",
            self.allowed_local_media_path,
            "--limit-mm-per-prompt",
            '{"video": 1}',
            "--mm-processor-cache-gb",
            "0",
            "--mm-processor-kwargs",
            json.dumps(processor_kwargs, separators=(",", ":")),
            "--media-io-kwargs",
            '{"video":{"num_frames":-1}}',
        ]

    def _start_server(self) -> None:
        import subprocess
        import sys

        logger.info("Native-video vLLM server log: %s", self._log.name)
        self._context_window = self.max_model_len
        env = os.environ.copy()
        env.setdefault("VLLM_USE_V1", "1")
        env.setdefault("LLM_DISABLE_COMPILE_CACHE", "1")
        env.setdefault("VLLM_FLASH_ATTN_VERSION", "3")
        env.setdefault("PYTHONNOUSERSITE", "1")
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        command.extend(self.server_args())
        self._proc = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=self._log,
            start_new_session=True,
            env=env,
        )

    def generate_video(
        self,
        video_path: str,
        prompt: str,
        max_new_tokens: int = 16,
    ) -> str:
        import urllib.request

        from model import record_prompt_token_counts

        video_url = Path(video_path).resolve().as_uri()
        payload = json.dumps(
            build_native_video_payload(
                self.model_id,
                video_url,
                prompt,
                max_new_tokens,
                self.video_fps,
            )
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://localhost:{self._port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            result = json.loads(response.read())
        if "error" in result:
            raise RuntimeError(f"vLLM returned error: {result['error']}")
        usage = result.get("usage", {})
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            record_prompt_token_counts(
                [int(usage["prompt_tokens"])], self._context_window
            )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as error:
            raise RuntimeError(f"Unexpected vLLM response structure: {result}") from error
        if content is None:
            raise RuntimeError("vLLM returned null content for native-video request")
        return str(content)


class NativeVideoVLLMModel(_NativeVideoVLLMModelMixin, VLLMModel):
    """vLLM server wrapper scoped to native-video LongQA evaluation."""

    def __init__(
        self,
        model_id: str,
        allowed_local_media_path: str,
        video_fps: float,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.90,
        tp_size: int = 1,
        concurrency: int = 1,
        request_timeout: int = 3600,
    ) -> None:
        super().__init__(
            model_id=model_id,
            tp_size=tp_size,
            concurrency=concurrency,
            max_frames=0,
            model_type="qwen",
            request_timeout=request_timeout,
        )
        self._native_video_init(
            allowed_local_media_path=allowed_local_media_path,
            video_fps=video_fps,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 16,
    ) -> str:
        raise RuntimeError("NativeVideoVLLMModel only accepts generate_video() requests")


def _run_eval(input_path: str, output_path: str, eval_output: str | None) -> None:
    from run_evaluation import _filter_subset, evaluate_longqa, load_jsonl as load_eval_jsonl
    from run_evaluation import write_results

    golden = load_eval_jsonl(input_path)
    predictions = load_eval_jsonl(output_path)
    if len(golden) != len(predictions):
        golden, predictions = _filter_subset(golden, predictions, "longqa")
    results = evaluate_longqa(golden, predictions)
    target = eval_output or os.path.join(os.path.dirname(output_path), "results.json")
    summary_path = write_results(target, results)
    print(
        f"LongQA Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    print(f"Results written to {target}")
    print(f"Summary written to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5 LongQA directly on local MP4 video URLs through vLLM."
    )
    parser.add_argument(
        "--input",
        default="../../data/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--video-folder", default="../../data/videos")
    parser.add_argument("--output", default="../../runs/egolongqa/qwen35_native_video/predictions.jsonl")
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume-predictions", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--allowed-local-media-path", default=None)
    parser.add_argument(
        "--proxy-cache-dir",
        default="../../.cache/egolongqa/native_video_proxies",
    )
    parser.add_argument("--proxy-width", type=int, default=DEFAULT_PROXY_WIDTH)
    parser.add_argument("--proxy-height", type=int, default=DEFAULT_PROXY_HEIGHT)
    parser.add_argument("--proxy-codec", default="libx264")
    parser.add_argument("--proxy-crf", type=int, default=18)
    parser.add_argument("--proxy-preset", default="veryfast")
    parser.add_argument("--proxy-workers", type=int, default=1)
    parser.add_argument("--prepare-proxies-only", action="store_true")
    parser.add_argument("--require-proxies", action="store_true")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="baseline")
    return parser.parse_args()


def main() -> None:
    import time

    from model import reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    video_folder = _resolve_path(args.video_folder)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    args.input = input_path
    args.output = output_path
    args.video_folder = video_folder
    args.proxy_cache_dir = _resolve_path(args.proxy_cache_dir)
    args.subset_file = _resolve_path(args.subset_file) if args.subset_file else None
    args.allowed_local_media_path = (
        _resolve_path(args.allowed_local_media_path)
        if args.allowed_local_media_path
        else None
    )

    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    source_video_paths, _ = resolve_video_paths(
        rows,
        video_folder,
    )
    video_paths = prepare_video_proxies(source_video_paths, args)
    if args.prepare_proxies_only:
        print(
            f"Prepared {len(video_paths)} proxies at {args.video_fps} FPS, "
            f"{args.proxy_width}x{args.proxy_height}"
        )
        return

    _, allowed_media_path = resolve_video_paths(
        [{"video_path": path} for path in video_paths],
        "",
        args.allowed_local_media_path,
    )
    args.allowed_local_media_path = allowed_media_path
    fingerprint = native_video_fingerprint(args)
    print(
        "Native-video LongQA config: "
        f"rows={len(rows)}, model={args.llm_model}, video_fps={args.video_fps}, "
        f"proxy={args.proxy_width}x{args.proxy_height}, "
        f"max_model_len={args.max_model_len}, fingerprint={fingerprint}"
    )
    print(f"vLLM allowed local media path: {allowed_media_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    existing = (
        load_jsonl(output_path)
        if not args.no_resume_predictions and os.path.exists(output_path)
        else []
    )
    start = resume_position(rows, existing, fingerprint)
    if start:
        print(f"Resuming native-video predictions from {start}/{len(rows)} cached rows")

    reset_prompt_token_stats(args.max_model_len)
    begun = time.time()
    mode = "a" if start else "w"
    model = NativeVideoVLLMModel(
        model_id=args.llm_model,
        allowed_local_media_path=allowed_media_path,
        video_fps=args.video_fps,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tp_size=args.tp,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
    )
    with model, open(output_path, mode) as handle:
        for index, (row, video_path) in enumerate(
            zip(rows[start:], video_paths[start:]), start=start
        ):
            prompt = build_longqa_prompt(
                row["question"],
                row["mcq_options"],
                prompt_variant=args.prompt_variant,
            )
            response = model.generate_video(
                video_path,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            prediction = build_prediction_row(
                row,
                response,
                prompt_variant=args.prompt_variant,
            )
            prediction["native_video_fingerprint"] = fingerprint
            prediction["native_video_fps"] = args.video_fps
            prediction["native_video_num_frames"] = -1
            prediction["native_video_proxy_width"] = args.proxy_width
            prediction["native_video_proxy_height"] = args.proxy_height
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Native-video progress: {index + 1}/{len(rows)}")

    print(f"Predictions written to {output_path}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
