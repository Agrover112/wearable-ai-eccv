#!/usr/bin/env python3
"""Model interface for video question answering.

Provides:
  - extract_frames(): extract video frames for model input
  - VideoQAModel: abstract base class — subclass to plug in your own model
  - Llama4ScoutModel: default implementation using Llama 4 Scout
  - Qwen2VLModel: Qwen2.5-VL implementation
  - create_model(): factory to instantiate by model type name

Setup:
  pip install -r requirements.txt
  huggingface-cli login
"""

from __future__ import annotations

import logging
import math
import os
import threading
from abc import ABC, abstractmethod

logger: logging.Logger = logging.getLogger(__name__)

_PROMPT_TOKEN_COUNTS: list[int] = []
_CONTEXT_WINDOW: int | None = None
_PROMPT_TOKEN_LOCK = threading.Lock()


def reset_prompt_token_stats(context_window: int | None = None) -> None:
    """Reset prompt token accounting for a generation run."""
    global _CONTEXT_WINDOW
    with _PROMPT_TOKEN_LOCK:
        _PROMPT_TOKEN_COUNTS.clear()
        _CONTEXT_WINDOW = context_window


def record_prompt_token_counts(
    counts: list[int],
    context_window: int | None = None,
) -> None:
    """Record prompt token counts observed during generation."""
    global _CONTEXT_WINDOW
    clean_counts = [int(c) for c in counts if int(c) >= 0]
    if not clean_counts:
        return
    with _PROMPT_TOKEN_LOCK:
        _PROMPT_TOKEN_COUNTS.extend(clean_counts)
        if context_window is not None:
            _CONTEXT_WINDOW = context_window


def summarize_prompt_token_stats() -> dict[str, object]:
    """Return aggregate prompt token/context-fill statistics."""
    with _PROMPT_TOKEN_LOCK:
        counts = sorted(_PROMPT_TOKEN_COUNTS)
        context_window = _CONTEXT_WINDOW
    if not counts:
        return {"count": 0, "context_window": context_window}

    def percentile(p: float) -> int:
        if len(counts) == 1:
            return counts[0]
        idx = round((len(counts) - 1) * p)
        return counts[max(0, min(idx, len(counts) - 1))]

    mean_tokens = sum(counts) / len(counts)
    summary: dict[str, object] = {
        "count": len(counts),
        "min": counts[0],
        "mean": round(mean_tokens, 2),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": counts[-1],
        "context_window": context_window,
    }
    if context_window:
        summary["mean_fill_pct"] = round((mean_tokens / context_window) * 100, 2)
        summary["p95_fill_pct"] = round((summary["p95"] / context_window) * 100, 2)
        summary["max_fill_pct"] = round((counts[-1] / context_window) * 100, 2)
    return summary


def _infer_context_window_from_model(model: object, processor: object) -> int | None:
    """Best-effort context window inference for HF models."""
    candidates: list[object] = []
    config = getattr(model, "config", None)
    tokenizer = getattr(processor, "tokenizer", None)
    for attr in (
        "max_position_embeddings",
        "max_seq_len",
        "seq_length",
        "model_max_length",
    ):
        if config is not None:
            candidates.append(getattr(config, attr, None))
        if tokenizer is not None:
            candidates.append(getattr(tokenizer, attr, None))
    for candidate in candidates:
        try:
            value = int(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        # Tokenizers often use giant sentinel values when no limit is known.
        if 0 < value < 10_000_000:
            return value
    return None


def _attention_lengths(inputs: object) -> list[int]:
    """Extract per-sample non-padding token counts from HF processor outputs."""
    mask = None
    if isinstance(inputs, dict):
        mask = inputs.get("attention_mask")
    else:
        mask = getattr(inputs, "attention_mask", None)
    if mask is not None:
        return [int(x) for x in mask.sum(dim=1).detach().cpu().tolist()]

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    batch_size, seq_len = input_ids.shape[:2]
    return [int(seq_len)] * int(batch_size)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, raw, default)
        return default


def _env_optional_nonnegative_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; ignoring it", name, raw)
        return None
    if value < 0:
        logger.warning("%s must be non-negative; ignoring %d", name, value)
        return None
    return value


def _env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean for %s=%r; ignoring it", name, raw)
    return None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %.2f", name, raw, default)
        return default


def extract_frames(
    video_path: str,
    intervals: list[tuple[float, float]] | None = None,
    frames_per_interval: int = 4,
    max_frames: int = 32,
    sampling_mode: str = "legacy",
) -> list[object]:
    """Extract frames from a video file as PIL Images.

    Args:
        video_path: Path to the video file.
        intervals: List of (start_sec, end_sec) to sample from.
            If None, samples uniformly from the entire video.
        frames_per_interval: Number of frames per interval.
        max_frames: Maximum total frames returned.

    Returns:
        List of PIL.Image.Image objects.
    """
    import cv2
    from PIL import Image

    if not os.path.exists(video_path):
        logger.warning("Video not found: %s", video_path)
        return []

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logger.warning("Could not open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            return []

        duration = total_frames / fps

        if intervals is None:
            intervals = [(0.0, duration)]

        frame_indices: list[int] = []
        for start, end in intervals:
            start_frame = int(start * fps)
            end_frame = min(int(end * fps), total_frames - 1)
            if end_frame <= start_frame:
                continue
            n = min(frames_per_interval, end_frame - start_frame + 1)
            if sampling_mode == "endpoint_inclusive" and n > 1:
                step = (end_frame - start_frame) / (n - 1)
                frame_indices.extend(
                    round(start_frame + i * step) for i in range(n)
                )
            elif sampling_mode == "endpoint_inclusive":
                frame_indices.append(round((start_frame + end_frame) / 2))
            elif sampling_mode == "legacy":
                step = (end_frame - start_frame) / n
                frame_indices.extend(int(start_frame + i * step) for i in range(n))
            else:
                raise ValueError(f"Unknown frame sampling mode: {sampling_mode}")

        frame_indices = sorted(set(frame_indices))

        if len(frame_indices) > max_frames:
            stride = len(frame_indices) / max_frames
            frame_indices = [frame_indices[int(i * stride)] for i in range(max_frames)]

        frames: list[object] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.info["source_frame_index"] = idx
                image.info["source_fps"] = fps
                frames.append(image)

        return frames
    finally:
        cap.release()


def flatten_batch_images(batch_frames: list[list[object]]) -> list[object]:
    """Flatten per-conversation frame lists into a single list for batch processing.

    The HuggingFace processor expects all images in a flat list, ordered by
    conversation then by frame within each conversation. This ordering must
    match the ``<image>`` placeholder tokens emitted by
    ``apply_chat_template``.

    Args:
        batch_frames: List of per-conversation frame lists, where each inner
            list contains PIL.Image.Image objects.

    Returns:
        Flat list of images in batch-then-frame order.
    """
    return [img for frames in batch_frames for img in frames]


class VideoQAModel(ABC):
    """Abstract base class for video QA models.

    To plug in your own model:
      1. Subclass VideoQAModel
      2. Override generate()
      3. Instantiate your model in the generation scripts

    This class implements the context manager protocol with no-op defaults.
    Subclasses that manage external resources (e.g., VLLMModel) can override
    __enter__ and __exit__ to handle setup/teardown.
    """

    @abstractmethod
    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a text response given video frames and conversation.

        Args:
            frames: Video frames as PIL.Image.Image objects.
            messages: Chat history as [{"role": "user"/"assistant"/"system",
                "content": str}].
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        ...

    def generate_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Generate responses for a batch of inputs.

        Default implementation falls back to sequential generate().
        Subclasses should override for true batched inference.
        """
        return [
            self.generate(frames, messages, max_new_tokens)
            for frames, messages in zip(batch_frames, batch_messages)
        ]

    def __enter__(self) -> "VideoQAModel":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        return False


class Llama4ScoutModel(VideoQAModel):
    """Llama 4 Scout implementation via HuggingFace transformers.

    Requires:
      - GPU with sufficient VRAM (1x A100 80GB recommended)
      - huggingface-cli login (with access to meta-llama models)
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Llama4ForConditionalGeneration

        logger.info("Loading model: %s ...", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info("Model loaded.")

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        import torch

        mm_messages = self._to_multimodal_messages(frames, messages)

        text = self.processor.apply_chat_template(
            mm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            images=frames if frames else None,
            return_tensors="pt",
        )
        record_prompt_token_counts(
            _attention_lengths(inputs),
            _infer_context_window_from_model(self.model, self.processor),
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 256,
    ) -> list[str]:
        import torch

        batch_mm = [
            self._to_multimodal_messages(f, m)
            for f, m in zip(batch_frames, batch_messages)
        ]
        texts = [
            self.processor.apply_chat_template(
                mm, tokenize=False, add_generation_prompt=True
            )
            for mm in batch_mm
        ]
        flat_images = flatten_batch_images(batch_frames)
        if len(flat_images) > 64:
            logger.warning(
                "Large batch: %d images in single forward pass "
                "(batch_size=%d x frames). Risk of OOM — consider "
                "reducing --batch-size or --max-frames.",
                len(flat_images),
                len(batch_frames),
            )

        inputs = self.processor(
            text=texts,
            images=flat_images if flat_images else None,
            return_tensors="pt",
            padding=True,
        )
        record_prompt_token_counts(
            _attention_lengths(inputs),
            _infer_context_window_from_model(self.model, self.processor),
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(len(texts)):
            new_tokens = output_ids[i][prompt_len:]
            results.append(
                self.processor.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return results

    def _to_multimodal_messages(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        """Convert plain messages to HF multimodal format with image placeholders."""
        mm_messages: list[dict[str, object]] = []
        images_inserted = False

        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            if role == "user" and not images_inserted and frames:
                content: list[dict[str, str]] = [{"type": "image"} for _ in frames]
                content.append({"type": "text", "text": text})
                mm_messages.append({"role": "user", "content": content})
                images_inserted = True
            else:
                mm_messages.append({"role": role, "content": text})

        return mm_messages


class Qwen2VLModel(VideoQAModel):
    """Qwen2.5-VL implementation via HuggingFace transformers.

    Requires:
      - GPU with sufficient VRAM (1x A100 80GB for 7B, 1x consumer GPU for 3B)
      - pip install qwen-vl-utils
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading model: %s ...", model_id)
        self.min_pixels = _env_int("QWEN_MIN_PIXELS", 784)
        self.max_pixels = _env_int("QWEN_MAX_PIXELS", 50176)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info("Model loaded.")

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        mm_messages = self._to_multimodal_messages(frames, messages)

        text = self.processor.apply_chat_template(
            mm_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(mm_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        record_prompt_token_counts(
            _attention_lengths(inputs),
            _infer_context_window_from_model(self.model, self.processor),
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 256,
    ) -> list[str]:
        import torch
        from qwen_vl_utils import process_vision_info

        batch_mm = [
            self._to_multimodal_messages(f, m)
            for f, m in zip(batch_frames, batch_messages)
        ]
        texts = [
            self.processor.apply_chat_template(
                mm, tokenize=False, add_generation_prompt=True
            )
            for mm in batch_mm
        ]

        all_images = []
        all_videos = []
        for mm in batch_mm:
            img, vid = process_vision_info(mm)
            all_images.extend(img if img else [])
            all_videos.extend(vid if vid else [])

        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            padding=True,
            return_tensors="pt",
        )
        record_prompt_token_counts(
            _attention_lengths(inputs),
            _infer_context_window_from_model(self.model, self.processor),
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(len(texts)):
            new_tokens = output_ids[i][prompt_len:]
            results.append(
                self.processor.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return results

    def _to_multimodal_messages(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        """Convert plain messages to Qwen VL multimodal format."""
        mm_messages: list[dict[str, object]] = []
        images_inserted = False

        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            if role == "user" and not images_inserted and frames:
                content: list[dict[str, object]] = [
                    {
                        "type": "image",
                        "image": frame,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                    for frame in frames
                ]
                content.append({"type": "text", "text": text})
                mm_messages.append({"role": "user", "content": content})
                images_inserted = True
            else:
                mm_messages.append({"role": role, "content": text})

        return mm_messages


class InternVideo3Model(VideoQAModel):
    """InternVideo3 inference through its Hugging Face checkpoint code."""

    REVISION = "c4602918b65225650d152db2850fe34e01d21fcd"

    def __init__(
        self,
        model_id: str = "yanziang/InternVideo3-8B-Instruct",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("Loading model: %s ...", model_id)
        revision = os.environ.get("INTERNVIDEO3_REVISION", self.REVISION)
        self.min_pixels = _env_int("VISION_MIN_PIXELS", 262144)
        self.max_pixels = _env_int("VISION_MAX_PIXELS", 524288)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            revision=revision,
            trust_remote_code=True,
        )
        logger.info("Model loaded.")

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
    ) -> str:
        import torch
        from transformers.video_utils import VideoMetadata

        mm_messages = self._to_multimodal_messages(frames, messages)
        processor_kwargs: dict[str, object] = {"do_sample_frames": False}
        if frames:
            processor_kwargs["video_metadata"] = VideoMetadata(
                total_num_frames=len(frames),
                fps=float(frames[0].info["source_fps"]),
                frames_indices=[
                    int(frame.info["source_frame_index"]) for frame in frames
                ],
            )
            self.processor.video_processor.size = {
                "shortest_edge": self.min_pixels * len(frames),
                "longest_edge": self.max_pixels * len(frames),
            }
        inputs = self.processor.apply_chat_template(
            mm_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **processor_kwargs,
        )
        record_prompt_token_counts(
            _attention_lengths(inputs),
            _infer_context_window_from_model(self.model, self.processor),
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _to_multimodal_messages(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        """Insert the sampled frames as one ordered video in the first user turn."""
        mm_messages: list[dict[str, object]] = []
        video_inserted = False

        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "user" and not video_inserted and frames:
                content = [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": text},
                ]
                mm_messages.append({"role": "user", "content": content})
                video_inserted = True
            else:
                mm_messages.append({"role": role, "content": text})

        return mm_messages


def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one.

    Note: there is an inherent TOCTOU (time-of-check to time-of-use) race
    between the moment this socket is closed and the moment vLLM binds to
    the returned port.  In practice the window is very small and collisions
    are rare on multi-GPU nodes.  ``_verify_served_model()`` provides a
    post-startup check that detects port collisions when they do occur.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _merge_reasoning_content(message: dict[str, object]) -> str | None:
    """Combine parsed reasoning with final content across vLLM schema versions."""
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        return f"{reasoning}\n\n{content}" if content else str(reasoning)
    return str(content) if content is not None else None


class VLLMModel(VideoQAModel):
    """vLLM backend that auto-manages an OpenAI-compatible vLLM server.

    Launches a vLLM server in a separate conda env on __enter__, sends
    concurrent HTTP requests for inference, and kills the server on __exit__.
    """

    def __init__(
        self,
        model_id: str,
        tp_size: int = 1,
        concurrency: int = 16,
        max_frames: int = 32,
        model_type: str = "qwen",
        request_timeout: int = 3600,
    ) -> None:
        self.model_id = model_id
        self.tp_size = tp_size
        self.concurrency = concurrency
        self.max_frames = max_frames
        self.model_type = model_type
        self.request_timeout = request_timeout
        self._context_window: int | None = None
        self._thinking_token_budget = _env_optional_nonnegative_int(
            "VLLM_THINKING_TOKEN_BUDGET"
        )
        self._is_qwen35 = "qwen3.5" in model_id.lower()
        self._enable_thinking = _env_optional_bool("QWEN_ENABLE_THINKING")
        if self._is_qwen35 and self._enable_thinking is None:
            self._enable_thinking = False
        self._gdn_prefill_backend = os.environ.get("VLLM_GDN_PREFILL_BACKEND")
        self._qwen_media_mode = os.environ.get(
            "VLLM_QWEN_MEDIA_MODE", "images"
        ).strip().lower()
        if self._qwen_media_mode not in {"images", "video"}:
            raise ValueError("VLLM_QWEN_MEDIA_MODE must be `images` or `video`")
        if self._is_qwen35 and not self._gdn_prefill_backend:
            self._gdn_prefill_backend = "triton"
        if self._gdn_prefill_backend not in {None, "triton", "flashinfer"}:
            raise ValueError(
                "VLLM_GDN_PREFILL_BACKEND must be `triton` or `flashinfer`"
            )
        self._proc: object | None = None
        self._port: int | None = None
        self._log: object | None = None

    def __enter__(self) -> "VLLMModel":
        import tempfile

        self._port = find_free_port()
        log_dir = os.environ.get("VLLM_LOG_DIR", os.getcwd())
        os.makedirs(log_dir, exist_ok=True)
        self._log = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="vllm_server_",
            suffix=".log",
            delete=False,
            dir=log_dir,
        )
        try:
            self._start_server()
            self._wait_for_health()
        except BaseException:
            self._kill_server()
            raise
        return self

    def _start_server(self) -> None:
        import subprocess

        log_path = self._log.name
        logger.info("vLLM server log: %s", log_path)

        server_args = [
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
        ]
        if self.model_type != "llama4":
            self._context_window = _env_int("VLLM_QWEN_MAX_MODEL_LEN", 16384)
            gpu_memory_utilization = _env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.90)
            server_args.extend(
                [
                    "--max-model-len",
                    str(self._context_window),
                    "--gpu-memory-utilization",
                    str(gpu_memory_utilization),
                ]
            )
            server_args.extend(["--dtype", "bfloat16"])
            reasoning_parser = os.environ.get("VLLM_REASONING_PARSER")
            if self._is_qwen35 and not reasoning_parser:
                reasoning_parser = "qwen3"
            if reasoning_parser:
                server_args.extend(["--reasoning-parser", reasoning_parser])
            if self._gdn_prefill_backend:
                server_args.extend(
                    ["--gdn-prefill-backend", self._gdn_prefill_backend]
                )
            if self._thinking_token_budget is not None:
                server_args.extend(
                    [
                        "--reasoning-config",
                        (
                            '{"reasoning_start_str":"<think>",'
                            '"reasoning_end_str":"</think>"}'
                        ),
                    ]
                )
        else:
            self._context_window = 131072
            # Minimal llama4 config (minimal subset of the Maverick judge
            # config that runs cleanly on 8x H100 — the Maverick path also
            # sets `--gpu-memory-utilization 0.8`, `--kv-cache-dtype fp8`,
            # `--max-model-len 32768`, etc.; we deliberately drop those
            # here): online FP8 quantization from the bf16 source
            # checkpoint (HF model ID
            # `meta-llama/Llama-4-Scout-17B-16E-Instruct`, downloaded to
            # `<path-to-bf16-scout-checkpoint>`).
            # Avoid the NVIDIA pre-quantized FP8 + modelopt loader path —
            # it hangs post-MoE-init in this stack. Avoid the kitchen-sink
            # of recipe flags too (`--kv-cache-dtype fp8`, `--async-scheduling`,
            # `--no-enable-prefix-caching`, `--max-num-batched-tokens`,
            # `--safetensors-load-strategy prefetch`); none were validated
            # to actually help and several stall warmup. `--enforce-eager`
            # is already in the shared base list above.
            #
            # `--max-model-len` must be capped: Scout's default model_max_length
            # is 10,485,760 (10M tokens — the new long-context feature). KV
            # cache for that is ~60 GiB/GPU and fails engine init with
            # `ValueError: KV cache larger than available` on 8x H100 80 GB.
            # 131,072 (128K) covers ConvQA prompts with 32 video frames +
            # multi-turn dialog history (observed max ~38K tokens). KV cache
            # at 128K is ~16 GiB/GPU, well within the 8x80GB H100 budget.
            server_args.extend(
                ["--quantization", "fp8", "--max-model-len", str(self._context_window)]
            )
        if self.max_frames > 0:
            if self.model_type == "qwen" and self._qwen_media_mode == "video":
                limit_json = '{"video": 1}'
            else:
                limit_json = f'{{"image": {self.max_frames}}}'
            server_args.extend(["--limit-mm-per-prompt", limit_json])
            # --mm-processor-kwargs is Qwen-specific (controls pixel resolution);
            # other models (e.g., Scout) do not support this flag.
            if self.model_type == "qwen":
                qwen_min_pixels = _env_int("QWEN_MIN_PIXELS", 784)
                qwen_max_pixels = _env_int("QWEN_MAX_PIXELS", 50176)
                server_args.extend(
                    [
                        "--mm-processor-kwargs",
                        (
                            '{"min_pixels": '
                            f"{qwen_min_pixels}, "
                            '"max_pixels": '
                            f"{qwen_max_pixels}"
                            "}"
                        ),
                    ]
                )
                if self._qwen_media_mode == "video":
                    server_args.extend(
                        [
                            "--media-io-kwargs",
                            (
                                '{"video": {"num_frames": '
                                f"{self.max_frames}, \"fps\": 1.0}}}}"
                            ),
                        ]
                    )
        max_num_batched_tokens = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS")
        if max_num_batched_tokens:
            server_args.extend(
                ["--max-num-batched-tokens", str(int(max_num_batched_tokens))]
            )
        max_logprobs = os.environ.get("VLLM_MAX_LOGPROBS")
        if max_logprobs:
            server_args.extend(["--max-logprobs", str(int(max_logprobs))])

        import sys

        # vLLM env vars matched to the _VllmJudgeServer path so the same
        # models behave identically on the generation vs judge sides. Without
        # VLLM_USE_V1=1, the server may silently fall back to the V0 engine
        # instead of the validated V1 path.
        env = os.environ.copy()
        env.setdefault("VLLM_USE_V1", "1")
        env.setdefault("LLM_DISABLE_COMPILE_CACHE", "1")
        env.setdefault("VLLM_FLASH_ATTN_VERSION", "3")
        env.setdefault("PYTHONNOUSERSITE", "1")

        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        cmd.extend(server_args)
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log,
            stderr=self._log,
            start_new_session=True,
            env=env,
        )

    def _apply_chat_template_options(
        self, request_data: dict[str, object]
    ) -> dict[str, object]:
        if self._enable_thinking is not None:
            request_data["chat_template_kwargs"] = {
                "enable_thinking": self._enable_thinking
            }
        return request_data

    def _wait_for_health(self) -> None:
        import time
        import urllib.request

        log_path = self._log.name if self._log else "unknown"
        start_ts = time.time()
        deadline = start_ts + self.request_timeout
        last_log_size = 0
        last_heartbeat = start_ts
        logger.info(
            "vLLM server warming up (timeout %ds, log %s)",
            self.request_timeout,
            log_path,
        )
        while time.time() < deadline:
            if self._proc.poll() is not None:
                returncode = self._proc.returncode
                log_tail = self._read_log_tail(log_path)
                self._kill_server()
                raise RuntimeError(
                    f"vLLM server exited with code {returncode}. "
                    f"Log: {log_path}\n{log_tail}"
                )
            try:
                req = urllib.request.Request(
                    f"http://localhost:{self._port}/health", method="GET"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        self._verify_served_model()
                        logger.info(
                            "vLLM server ready on port %d (pid %d, warmup took %ds)",
                            self._port,
                            self._proc.pid,
                            int(time.time() - start_ts),
                        )
                        return
            except (urllib.error.URLError, OSError):
                pass
            try:
                if self._log is not None:
                    self._log.flush()
                log_size = os.path.getsize(log_path) if self._log else last_log_size
            except (OSError, ValueError):
                # Log file may briefly disappear during vllm subprocess
                # teardown on Lustre or other network filesystems; flush() on
                # a closed handle raises ValueError. Skip progress print on
                # this tick.
                log_size = last_log_size
            now = time.time()
            # Heartbeat at least every 30 s so silent warmup phases (e.g.
            # llama4 online FP8 quant + MoE init) don't look like a hang.
            if log_size > last_log_size + 1024 * 1024 or now - last_heartbeat >= 30:
                logger.info(
                    "vLLM server still warming up (elapsed %ds, log %d KiB)",
                    int(now - start_ts),
                    log_size // 1024,
                )
                last_log_size = log_size
                last_heartbeat = now
            time.sleep(5)
        log_tail = self._read_log_tail(log_path)
        self._kill_server()
        raise RuntimeError(
            f"vLLM server failed to start within {self.request_timeout}s. "
            f"Log: {log_path}\n{log_tail}"
        )

    @staticmethod
    def _read_log_tail(log_path: str, lines: int = 30) -> str:
        try:
            with open(log_path, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception:
            return "(could not read log)"

    def _verify_served_model(self) -> None:
        import json as json_mod
        import time as _time
        import urllib.request

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"http://localhost:{self._port}/v1/models", method="GET"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json_mod.loads(resp.read())
                    served = [m["id"] for m in data.get("data", [])]
                    if self.model_id not in served:
                        raise RuntimeError(
                            f"Port {self._port} serves {served}, expected "
                            f"{self.model_id} — port collision detected. "
                            f"Another vLLM server may be running on this port."
                        )
                return
            except (
                urllib.error.URLError,
                json_mod.JSONDecodeError,
                KeyError,
                OSError,
            ) as e:
                last_exc = e
                if attempt < 2:
                    _time.sleep(5)
        raise RuntimeError(
            f"Could not verify served model on port {self._port} after "
            f"3 attempts: {last_exc} — possible port collision with "
            f"another vLLM server on this node"
        )

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self._kill_server()
        return False

    def _kill_server(self) -> None:
        import signal
        import subprocess

        try:
            if self._proc is not None:
                try:
                    pgid = os.getpgid(self._proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        self._proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        try:
                            self._proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning(
                                "vLLM server (pid %d) did not exit after SIGKILL",
                                self._proc.pid,
                            )
                except OSError:
                    pass
                self._proc = None
        finally:
            if self._log:
                self._log.close()
                self._log = None

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 4096,
        thinking_token_budget: int | None = None,
    ) -> str:
        import base64
        import io
        import json
        import urllib.error
        import urllib.request

        image_content: list[dict[str, object]] = []
        for frame in frames:
            buf = io.BytesIO()
            frame.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        openai_messages: list[dict[str, object]] = []
        images_inserted = False
        for msg in messages:
            if msg["role"] == "user" and not images_inserted and frames:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": image_content
                        + [{"type": "text", "text": msg["content"]}],
                    }
                )
                images_inserted = True
            else:
                openai_messages.append(msg)

        request_data: dict[str, object] = {
            "model": self.model_id,
            "messages": openai_messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        effective_thinking_budget = (
            self._thinking_token_budget
            if thinking_token_budget is None
            else thinking_token_budget
        )
        if effective_thinking_budget is not None:
            request_data["thinking_token_budget"] = effective_thinking_budget
        self._apply_chat_template_options(request_data)
        payload = json.dumps(request_data).encode()

        req = urllib.request.Request(
            f"http://localhost:{self._port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP {error.code}: {body[:2000]}"
            ) from error
        if "error" in result:
            raise RuntimeError(f"vLLM returned error: {result['error']}")
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if prompt_tokens is not None:
            record_prompt_token_counts([int(prompt_tokens)], self._context_window)
        try:
            message = result["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected vLLM response structure: {e}. "
                f"Response keys: {list(result.keys())}"
            ) from e
        content = _merge_reasoning_content(message)
        if content is None:
            raise RuntimeError(
                "vLLM returned null content (possible content-filter or empty "
                f"generation). Model: {self.model_id}, port: {self._port}"
            )
        return content

    def generate_video_frames(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 4096,
    ) -> str:
        """Generate with selected frames encoded as one chronological video."""
        import base64
        import io
        import json
        import urllib.error
        import urllib.request

        if self._qwen_media_mode != "video":
            raise RuntimeError(
                "generate_video_frames requires VLLM_QWEN_MEDIA_MODE=video"
            )
        encoded_frames: list[str] = []
        for frame in frames:
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=90)
            encoded_frames.append(base64.b64encode(buffer.getvalue()).decode())
        video_url = "data:video/jpeg;base64," + ",".join(encoded_frames)

        openai_messages: list[dict[str, object]] = []
        video_inserted = False
        for message in messages:
            if message["role"] == "user" and not video_inserted and frames:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_url}},
                            {"type": "text", "text": message["content"]},
                        ],
                    }
                )
                video_inserted = True
            else:
                openai_messages.append(message)

        request_data: dict[str, object] = {
            "model": self.model_id,
            "messages": openai_messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        self._apply_chat_template_options(request_data)
        request = urllib.request.Request(
            f"http://localhost:{self._port}/v1/chat/completions",
            data=json.dumps(request_data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout
            ) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {error.code}: {body[:2000]}") from error
        if "error" in result:
            raise RuntimeError(f"vLLM returned error: {result['error']}")
        usage = result.get("usage", {})
        if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
            record_prompt_token_counts(
                [int(usage["prompt_tokens"])], self._context_window
            )
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError) as error:
            raise RuntimeError(f"Unexpected vLLM response: {result}") from error
        content = _merge_reasoning_content(message)
        if content is None:
            raise RuntimeError("vLLM returned null VideoJudge content")
        return content

    def score_choice_letters(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        letters: tuple[str, ...] = ("A", "B", "C", "D"),
    ) -> dict[str, float]:
        """Return next-token log probabilities for constrained MCQ letters."""
        import base64
        import io
        import json
        import urllib.request

        image_content: list[dict[str, object]] = []
        for frame in frames:
            buf = io.BytesIO()
            frame.save(buf, format="JPEG")
            encoded = base64.b64encode(buf.getvalue()).decode()
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        openai_messages: list[dict[str, object]] = []
        images_inserted = False
        for message in messages:
            if message["role"] == "user" and not images_inserted and frames:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": image_content
                        + [{"type": "text", "text": message["content"]}],
                    }
                )
                images_inserted = True
            else:
                openai_messages.append(message)
        top_logprobs = max(20, int(os.environ.get("VLLM_MAX_LOGPROBS", "20")))
        request_data: dict[str, object] = {
                "model": self.model_id,
                "messages": openai_messages,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": top_logprobs,
        }
        self._apply_chat_template_options(request_data)
        payload = json.dumps(request_data).encode()
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
            record_prompt_token_counts([int(usage["prompt_tokens"])], self._context_window)
        try:
            candidates = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"vLLM response did not contain token logprobs: {result}") from error
        scores = {letter: float("-inf") for letter in letters}
        for candidate in candidates:
            token = str(candidate.get("token", "")).strip().upper()
            if token in scores:
                scores[token] = max(scores[token], float(candidate["logprob"]))
        missing = [letter for letter, score in scores.items() if not math.isfinite(score)]
        if missing:
            raise RuntimeError(
                "Requested option letters were absent from vLLM top_logprobs: "
                f"{missing}; returned tokens={[item.get('token') for item in candidates]}"
            )
        return scores

    def score_token_uncertainty(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        top_logprobs: int = 100,
    ) -> dict[str, object]:
        """Estimate next-token entropy from the vLLM top-K distribution.

        The returned lower bound treats all probability mass outside the
        returned top-K tokens as one aggregate outcome. This is exact when the
        tail mass is zero and remains explicitly distinguishable from the
        full-vocabulary entropy used by the UG paper.
        """
        import base64
        import io
        import json
        import urllib.request

        if top_logprobs <= 0:
            raise ValueError("top_logprobs must be positive")
        image_content: list[dict[str, object]] = []
        for frame in frames:
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        openai_messages: list[dict[str, object]] = []
        images_inserted = False
        for message in messages:
            if message["role"] == "user" and not images_inserted and frames:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": image_content
                        + [{"type": "text", "text": message["content"]}],
                    }
                )
                images_inserted = True
            else:
                openai_messages.append(message)
        request_data: dict[str, object] = {
                "model": self.model_id,
                "messages": openai_messages,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": top_logprobs,
        }
        self._apply_chat_template_options(request_data)
        payload = json.dumps(request_data).encode()
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
            record_prompt_token_counts([int(usage["prompt_tokens"])], self._context_window)
        try:
            content = result["choices"][0]["logprobs"]["content"][0]
            candidates = content["top_logprobs"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"vLLM response did not contain token logprobs: {result}"
            ) from error
        probabilities = [
            math.exp(float(candidate["logprob"]))
            for candidate in candidates
            if math.isfinite(float(candidate["logprob"]))
        ]
        observed_mass = min(1.0, float(sum(probabilities)))
        tail_mass = max(0.0, 1.0 - observed_mass)
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        if tail_mass > 0.0:
            entropy -= tail_mass * math.log(tail_mass)
        normalized_entropy = 0.0
        if observed_mass > 0.0:
            normalized = [probability / observed_mass for probability in probabilities]
            normalized_entropy = -sum(
                probability * math.log(probability)
                for probability in normalized
                if probability > 0.0
            )
        return {
            "entropy_lower_bound": float(entropy),
            "topk_normalized_entropy": float(normalized_entropy),
            "observed_probability_mass": observed_mass,
            "tail_probability_mass": tail_mass,
            "top_logprobs_requested": top_logprobs,
            "top_logprobs_returned": len(candidates),
            "predicted_token": str(content.get("token", "")),
            "predicted_logprob": float(content.get("logprob", float("-inf"))),
            "distribution": [
                {
                    "token": str(candidate.get("token", "")),
                    "logprob": float(candidate["logprob"]),
                }
                for candidate in candidates
            ],
        }

    def score_token_uncertainty_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        top_logprobs: int = 100,
    ) -> list[dict[str, object]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if len(batch_frames) != len(batch_messages):
            raise ValueError("uncertainty batch frames and messages must have equal lengths")
        results: list[dict[str, object] | None] = [None] * len(batch_frames)
        failures: list[tuple[int, Exception]] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(
                    self.score_token_uncertainty,
                    frames,
                    messages,
                    top_logprobs,
                ): index
                for index, (frames, messages) in enumerate(
                    zip(batch_frames, batch_messages)
                )
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as error:
                    failures.append((index, error))
        if failures:
            index, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} uncertainty requests failed; "
                f"first failure at batch index {index}: {error}"
            ) from error
        if any(result is None for result in results):
            raise RuntimeError("uncertainty batch completed with missing responses")
        return [result for result in results if result is not None]

    def score_candidate_texts(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        candidates: dict[str, str],
        assistant_prefix: str = "Answer:\n",
    ) -> dict[str, dict[str, object]]:
        """Score complete candidate answers as assistant-message continuations.

        vLLM returns prompt log probabilities for the teacher-forced candidate
        tokens. The mean score is length-normalized so short options do not win
        solely because they contain fewer tokens.
        """
        import base64
        import io
        import json
        import urllib.request

        image_content: list[dict[str, object]] = []
        for frame in frames:
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        openai_messages: list[dict[str, object]] = []
        images_inserted = False
        for message in messages:
            if message["role"] == "user" and not images_inserted and frames:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": image_content
                        + [{"type": "text", "text": message["content"]}],
                    }
                )
                images_inserted = True
            else:
                openai_messages.append(message)

        def request_prompt_logprobs(assistant_text: str) -> dict[str, object]:
            payload_messages = openai_messages + [
                {"role": "assistant", "content": assistant_text}
            ]
            request_data: dict[str, object] = {
                    "model": self.model_id,
                    "messages": payload_messages,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": 0,
                    "return_token_ids": True,
                    "add_generation_prompt": False,
                    "continue_final_message": True,
            }
            self._apply_chat_template_options(request_data)
            payload = json.dumps(request_data).encode()
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
                record_prompt_token_counts([int(usage["prompt_tokens"])], self._context_window)
            if result.get("prompt_token_ids") is None or result.get("prompt_logprobs") is None:
                raise RuntimeError("vLLM response omitted prompt token IDs or log probabilities")
            return result

        base = request_prompt_logprobs(assistant_prefix)
        base_ids = [int(token_id) for token_id in base["prompt_token_ids"]]
        scored: dict[str, dict[str, object]] = {}
        for label, candidate in candidates.items():
            result = request_prompt_logprobs(assistant_prefix + str(candidate).strip())
            token_ids = [int(token_id) for token_id in result["prompt_token_ids"]]
            common = 0
            for base_id, candidate_id in zip(base_ids, token_ids):
                if base_id != candidate_id:
                    break
                common += 1
            suffix_ids = token_ids[common:]
            prompt_logprobs = result["prompt_logprobs"]
            if not suffix_ids or len(prompt_logprobs) != len(token_ids):
                raise RuntimeError(
                    f"Could not isolate candidate token span for {label}: "
                    f"base={len(base_ids)}, candidate={len(token_ids)}, lcp={common}"
                )
            token_scores: list[float] = []
            for position, token_id in enumerate(suffix_ids, start=common):
                values = prompt_logprobs[position]
                selected = values.get(str(token_id)) if isinstance(values, dict) else None
                if not isinstance(selected, dict) or "logprob" not in selected:
                    raise RuntimeError(
                        f"Missing selected-token logprob for {label} at prompt position {position}"
                    )
                token_scores.append(float(selected["logprob"]))
            total = float(sum(token_scores))
            scored[label] = {
                "text": str(candidate).strip(),
                "total_logprob": total,
                "mean_logprob": total / len(token_scores),
                "token_count": len(token_scores),
                "token_logprobs": token_scores,
                "candidate_token_ids": suffix_ids,
                "prompt_prefix_tokens": common,
            }
        return scored

    def generate_json(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        schema: dict[str, object],
        schema_name: str = "structured_output",
        max_new_tokens: int = 256,
    ) -> dict[str, object]:
        """Generate a JSON object constrained by an OpenAI JSON schema."""
        import base64
        import io
        import json
        import urllib.request

        image_content: list[dict[str, object]] = []
        for frame in frames:
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        openai_messages: list[dict[str, object]] = []
        images_inserted = False
        for message in messages:
            if message["role"] == "user" and not images_inserted and frames:
                openai_messages.append(
                    {
                        "role": "user",
                        "content": image_content
                        + [{"type": "text", "text": message["content"]}],
                    }
                )
                images_inserted = True
            else:
                openai_messages.append(message)
        request_data: dict[str, object] = {
                "model": self.model_id,
                "messages": openai_messages,
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
        }
        self._apply_chat_template_options(request_data)
        payload = json.dumps(request_data).encode()
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
            record_prompt_token_counts([int(usage["prompt_tokens"])], self._context_window)
        try:
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"vLLM did not return valid schema JSON: {result}") from error
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected a JSON object, received: {parsed!r}")
        return parsed

    def generate_json_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        schema: dict[str, object],
        schema_name: str = "structured_output",
        max_new_tokens: int = 256,
    ) -> list[dict[str, object]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if len(batch_frames) != len(batch_messages):
            raise ValueError("JSON batch frames and messages must have equal lengths")
        results: list[dict[str, object] | None] = [None] * len(batch_frames)
        failed: list[tuple[int, Exception]] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(
                    self.generate_json,
                    frames,
                    messages,
                    schema,
                    schema_name,
                    max_new_tokens,
                ): index
                for index, (frames, messages) in enumerate(
                    zip(batch_frames, batch_messages)
                )
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as error:
                    failed.append((index, error))
        for index, error in failed:
            logger.warning(
                "Structured request %d failed at %d tokens; retrying at %d: %s",
                index,
                max_new_tokens,
                max(max_new_tokens * 2, 512),
                error,
            )
            results[index] = self.generate_json(
                batch_frames[index],
                batch_messages[index],
                schema,
                schema_name,
                max(max_new_tokens * 2, 512),
            )
        if any(result is None for result in results):
            raise RuntimeError("JSON batch completed with missing responses")
        return [result for result in results if result is not None]

    def generate_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 4096,
        thinking_token_budget: int | None = None,
    ) -> list[str]:
        from concurrent.futures import as_completed, ThreadPoolExecutor

        if len(batch_frames) != len(batch_messages):
            raise ValueError(
                f"batch_frames ({len(batch_frames)}) and batch_messages "
                f"({len(batch_messages)}) must have the same length"
            )
        results: list[str | None] = [None] * len(batch_frames)
        errors: list[tuple[int, Exception]] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(
                    self.generate,
                    frames,
                    msgs,
                    max_new_tokens,
                    thinking_token_budget,
                ): i
                for i, (frames, msgs) in enumerate(zip(batch_frames, batch_messages))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("vLLM request %d failed: %s", idx, exc)
                    errors.append((idx, exc))
        if errors:
            error_msgs = "; ".join(f"[{i}] {e}" for i, e in errors[:5])
            raise RuntimeError(
                f"{len(errors)} / {len(batch_frames)} vLLM requests failed. "
                f"First failures: {error_msgs}"
            )
        return [r if r is not None else "" for r in results]

    def generate_final_answer_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 64,
    ) -> list[str]:
        """Generate answer-only retries without reopening a reasoning block."""
        budget = 0 if self._thinking_token_budget is not None else None
        return self.generate_batch(
            batch_frames,
            batch_messages,
            max_new_tokens=max_new_tokens,
            thinking_token_budget=budget,
        )


MODEL_REGISTRY: dict[str, type[VideoQAModel]] = {
    "llama4": Llama4ScoutModel,
    "qwen": Qwen2VLModel,
    "internvideo3": InternVideo3Model,
}

DEFAULT_MODEL_IDS: dict[str, str] = {
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvideo3": "yanziang/InternVideo3-8B-Instruct",
}

MODEL_TYPES = tuple(DEFAULT_MODEL_IDS)
VLLM_MODEL_TYPES = ("llama4", "qwen")

DEFAULT_BATCH_SIZES: dict[str, int] = {
    "llama4": 4,
    "qwen": 8,
    "internvideo3": 1,
}

DEFAULT_GPU_COUNTS: dict[str, int] = {
    "llama4": 8,
    "qwen": 1,
    "internvideo3": 1,
}

DEFAULT_TP_SIZES: dict[str, int] = {
    "llama4": 8,
    "qwen": 1,
    "internvideo3": 1,
}


def detect_gpu_count() -> int:
    """Count available GPUs without initializing the CUDA runtime.

    CUDA_VISIBLE_DEVICES is read by the driver at init time — calling
    torch.cuda.device_count() first locks in the full device list and
    makes later env-var changes a no-op.  This helper avoids that trap.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return 0 if cvd.strip() == "" else len([x for x in cvd.split(",") if x.strip()])
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [
                line
                for line in result.stdout.strip().split("\n")
                if line.startswith("GPU ")
            ]
            return len(lines)
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        pass
    return 0


def setup_gpus(num_gpus: int | None = None, model_type: str = "llama4") -> int:
    """Configure CUDA_VISIBLE_DEVICES based on requested GPU count.

    Sets the env var *before* any torch.cuda call so the CUDA runtime
    sees the restricted device list on first init.

    Args:
        num_gpus: Number of GPUs to use. None = auto-detect.
        model_type: Model type for default GPU count.

    Returns:
        Actual number of GPUs configured.

    Raises:
        RuntimeError: If num_gpus is below model minimum or exceeds available.
    """
    available = detect_gpu_count()
    if num_gpus is None or num_gpus <= 0:
        num_gpus = available
    min_gpus = DEFAULT_GPU_COUNTS.get(model_type, 1)
    if num_gpus < min_gpus:
        raise RuntimeError(
            f"{model_type} requires at least {min_gpus} GPUs but only "
            f"{num_gpus} {'available' if num_gpus == available else 'requested'}. "
            f"Allocate more GPUs or choose a smaller model (e.g. qwen)."
        )
    if num_gpus > available:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only {available} available. "
            f"Check --num-gpus or CUDA_VISIBLE_DEVICES."
        )
    if num_gpus < available:
        existing_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if existing_cvd:
            visible_ids = [x for x in existing_cvd.split(",") if x.strip()]
            gpu_ids = ",".join(visible_ids[:num_gpus])
        else:
            gpu_ids = ",".join(str(i) for i in range(num_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        logger.info(
            "Set CUDA_VISIBLE_DEVICES=%s (%d/%d GPUs for %s)",
            gpu_ids,
            num_gpus,
            available,
            model_type,
        )
    else:
        logger.info("Using all %d GPUs for %s", available, model_type)
    return num_gpus


def create_model(
    model_type: str,
    model_id: str | None = None,
    backend: str = "hf",
    tp_size: int | None = None,
    concurrency: int = 16,
    max_frames: int = 32,
) -> VideoQAModel:
    """Factory to create a model by type name.

    Args:
        model_type: One of "llama4", "qwen", "internvideo3".
        model_id: HuggingFace model ID override. If None, uses the default
            for the given model_type.
        backend: "hf" for HuggingFace, "vllm" for vLLM server backend.
        tp_size: Tensor parallel size (vllm only). None = auto per model type.
        concurrency: Max concurrent HTTP requests (vllm only).
        max_frames: Max frames per video (used to set vLLM image limit).

    Returns:
        Instantiated VideoQAModel.
    """
    if backend == "vllm":
        if model_type not in DEFAULT_MODEL_IDS and model_id is None:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(DEFAULT_MODEL_IDS.keys())}"
            )
        if model_type not in VLLM_MODEL_TYPES:
            raise ValueError(
                f"vLLM does not support model type '{model_type}'. "
                f"Supported vLLM model types: {list(VLLM_MODEL_TYPES)}"
            )
        effective_id = model_id or DEFAULT_MODEL_IDS[model_type]
        effective_tp = (
            tp_size if tp_size is not None else DEFAULT_TP_SIZES.get(model_type, 1)
        )
        return VLLMModel(
            model_id=effective_id,
            tp_size=effective_tp,
            concurrency=concurrency,
            max_frames=max_frames,
            model_type=model_type,
        )

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"HuggingFace backend requires model type in MODEL_REGISTRY. "
            f"'{model_type}' is not registered. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    effective_id = model_id or DEFAULT_MODEL_IDS[model_type]
    cls = MODEL_REGISTRY[model_type]
    return cls(effective_id)
