"""Self-contained LongQA dual-view probability-fusion submission model."""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

import numpy as np

from longqa_utils import normalize_answer
from run_generate_longqa_grounded import CandidateFrame, create_text_image_grounder
from run_generate_longqa_proofpack import (
    build_budgeted_option_queries,
    combine_balanced_retrieval_scores,
    compile_temporal_program_v2,
    select_temporal_pivot_pack,
)

logger = logging.getLogger(__name__)
LETTERS = "ABCD"


def _parse_baseline_prompt(prompt: str) -> dict[str, str]:
    match = re.search(
        r"Question:\s*(.*?)\n\nOptions:\s*\n(.*?)\n\nAnswer with ONLY",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Dual-view model received an unrecognized LongQA prompt")
    return {
        "question": match.group(1).strip(),
        "mcq_options": match.group(2).strip(),
    }


def _probabilities(logprobs: dict[str, float]) -> dict[str, float]:
    maximum = max(logprobs.values())
    weights = {letter: math.exp(logprobs[letter] - maximum) for letter in LETTERS}
    total = sum(weights.values())
    return {letter: weights[letter] / total for letter in LETTERS}


def _answer(response: str) -> str:
    answer = normalize_answer(response)
    return answer if answer in LETTERS else ""


class Qwen35DualViewFusionModel:
    """Qwen3.5-27B with endpoint and SigLIP2 option-conditioned views.

    The evaluator supplies 128 endpoint-inclusive candidate frames followed by
    the exact 64-frame endpoint-inclusive global view. Both direct Qwen answers
    are generated using the baseline MCQ prompt. On disagreement, the four
    answer letters are scored under each view and their normalized probabilities
    are averaged. This is the label-free 618/700 validation policy.
    """

    def __init__(self, model_id: str = "/models/qwen35-27b") -> None:
        self.model_id = model_id
        self.siglip_id = os.environ.get(
            "WAI_SIGLIP2_MODEL", "/models/siglip2-so400m-patch14-384"
        )
        self.grounder: Any | None = None
        self.qwen: Any | None = None
        self._startup_report: str | None = None
        self._siglip_gpu_index: int | None = None

    def _write_startup_state(
        self,
        status: str,
        qwen_tp_size: int,
        siglip_gpu_index: int,
        visible_gpus: int,
        error: str | None = None,
    ) -> None:
        import json
        import time

        log_dir = os.environ.get("VLLM_LOG_DIR", os.getcwd())
        os.makedirs(log_dir, exist_ok=True)
        self._startup_report = os.path.join(log_dir, "dual_view_startup.json")
        payload = {
            "status": status,
            "qwen_tensor_parallel_size": qwen_tp_size,
            "qwen_gpu_indices": list(range(qwen_tp_size)),
            "siglip_gpu_index": siglip_gpu_index,
            "visible_gpu_count": visible_gpus,
            "updated_unix_time": time.time(),
        }
        if error:
            payload["error"] = error
        with open(self._startup_report, "w") as report:
            json.dump(payload, report, indent=2)
            report.write("\n")

    def __enter__(self) -> "Qwen35DualViewFusionModel":
        from model import VLLMModel
        import torch

        os.environ.setdefault("QWEN_ENABLE_THINKING", "0")
        os.environ.setdefault("QWEN_MIN_PIXELS", "784")
        os.environ.setdefault("QWEN_MAX_PIXELS", "451584")
        os.environ.setdefault("VLLM_QWEN_MAX_MODEL_LEN", "32768")
        os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
        os.environ.setdefault("VLLM_MAX_LOGPROBS", "100")
        os.environ.setdefault("VLLM_GDN_PREFILL_BACKEND", "triton")
        qwen_tp_size = int(os.environ.get("WAI_QWEN_TP_SIZE", "1"))
        siglip_gpu_index = int(os.environ.get("WAI_SIGLIP_GPU_INDEX", "1"))
        required_gpus = max(qwen_tp_size, siglip_gpu_index + 1)
        visible_gpus = torch.cuda.device_count()
        self._write_startup_state(
            "starting", qwen_tp_size, siglip_gpu_index, visible_gpus
        )
        if siglip_gpu_index < 0 or visible_gpus < required_gpus:
            self._write_startup_state(
                "failed",
                qwen_tp_size,
                siglip_gpu_index,
                visible_gpus,
                f"only {visible_gpus}/{required_gpus} required GPUs are visible",
            )
            raise RuntimeError(
                "Dual-view fusion requires "
                f"{required_gpus} visible GPUs for Qwen TP={qwen_tp_size} and "
                f"SigLIP2 on GPU {siglip_gpu_index}, but only "
                f"{visible_gpus} are visible"
            )
        self._siglip_gpu_index = siglip_gpu_index
        self.qwen = VLLMModel(
            self.model_id,
            tp_size=qwen_tp_size,
            concurrency=1,
            max_frames=64,
            model_type="qwen",
            # Model startup happens once before evaluation and is outside the
            # organizer's per-generation limit. Keep a generous warmup timeout;
            # individual turns remain bounded by the evaluation harness.
            request_timeout=1800,
        )
        try:
            self.qwen.__enter__()
            self._write_startup_state(
                "qwen_ready", qwen_tp_size, siglip_gpu_index, visible_gpus
            )
            self.grounder = create_text_image_grounder(
                self.siglip_id,
                device=f"cuda:{siglip_gpu_index}",
                batch_size=8,
                dtype="bfloat16",
            )
            self._write_startup_state(
                "ready", qwen_tp_size, siglip_gpu_index, visible_gpus
            )
        except BaseException as exc:
            self._write_startup_state(
                "failed",
                qwen_tp_size,
                siglip_gpu_index,
                visible_gpus,
                repr(exc),
            )
            if self.qwen is not None:
                self.qwen.__exit__(None, None, None)
                self.qwen = None
            raise
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if self.qwen is not None:
            self.qwen.__exit__(exc_type, exc_val, exc_tb)
            self.qwen = None
        if self.grounder is not None:
            del self.grounder
            self.grounder = None
        return False

    @staticmethod
    def _split_views(frames: list[object]) -> tuple[list[CandidateFrame], list[object]]:
        candidates: list[CandidateFrame] = []
        global_frames: list[object] = []
        for fallback_index, frame in enumerate(frames):
            info = getattr(frame, "info", {})
            role = info.get("wai_view")
            if role == "candidate":
                candidates.append(
                    CandidateFrame(
                        index=int(info.get("wai_frame_index", fallback_index)),
                        timestamp=float(info.get("wai_timestamp", fallback_index)),
                        image=frame,
                    )
                )
            elif role == "global":
                global_frames.append(frame)
        if len(candidates) != 128 or len(global_frames) != 64:
            raise RuntimeError(
                "Expected 128 retrieval candidates plus 64 global frames, got "
                f"{len(candidates)} plus {len(global_frames)}"
            )
        return candidates, global_frames

    def _option_view(self, candidates: list[CandidateFrame], row: dict[str, str]) -> list[object]:
        assert self.grounder is not None
        image_features = self.grounder.encode_images(candidates)
        queries = build_budgeted_option_queries(row, self.grounder.processor.tokenizer)
        component_scores = {
            query.label: self.grounder.score_embeddings(query.text, image_features)
            for query in queries
        }
        target_scores = combine_balanced_retrieval_scores(component_scores)
        program = compile_temporal_program_v2(row["question"])
        pivot_query = (
            "Find this temporal reference event in the video.\n"
            f"Reference event: {program.pivot or row['question']}"
        )
        pivot_scores = self.grounder.score_embeddings(pivot_query, image_features)
        selected, _ = select_temporal_pivot_pack(
            candidates,
            pivot_scores,
            target_scores,
            np.asarray(image_features),
            program,
            pivot_centers=2,
            target_centers=8,
            eventlet_radius=1,
            anchor_k=24,
            bridge_k=8,
            final_max_frames=64,
            temporal_nms_seconds=10.0,
            fill_mode="semantic_boundary",
            per_pivot_direction=True,
            target_component_scores=component_scores,
            target_centers_per_option=1,
        )
        option_frames = [item.candidate.image for item in selected]
        del image_features, component_scores, target_scores, pivot_scores
        if self._siglip_gpu_index is not None:
            import torch

            with torch.cuda.device(self._siglip_gpu_index):
                torch.cuda.empty_cache()
        if len(option_frames) != 64:
            raise RuntimeError(f"Option-conditioned selector returned {len(option_frames)}/64 frames")
        return option_frames

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 16,
    ) -> str:
        if self.qwen is None or self.grounder is None:
            raise RuntimeError("Dual-view model must be used as a context manager")
        if not messages:
            raise ValueError("LongQA messages are empty")
        prompt = messages[-1]["content"]
        row = _parse_baseline_prompt(prompt)
        candidates, global_frames = self._split_views(frames)
        option_frames = self._option_view(candidates, row)
        qwen_messages = [{"role": "user", "content": prompt}]

        endpoint_response = self.qwen.generate(global_frames, qwen_messages, max_new_tokens=16)
        option_response = self.qwen.generate(option_frames, qwen_messages, max_new_tokens=16)
        endpoint_answer = _answer(endpoint_response)
        option_answer = _answer(option_response)
        if endpoint_answer and endpoint_answer == option_answer:
            return endpoint_answer

        endpoint_scores = self.qwen.score_choice_letters(global_frames, qwen_messages)
        option_scores = self.qwen.score_choice_letters(option_frames, qwen_messages)
        endpoint_prob = _probabilities(endpoint_scores)
        option_prob = _probabilities(option_scores)
        mean_prob = {
            letter: (endpoint_prob[letter] + option_prob[letter]) / 2.0
            for letter in LETTERS
        }
        fallback = endpoint_answer or option_answer or "A"
        best = max(mean_prob.values())
        tied = [letter for letter in LETTERS if abs(mean_prob[letter] - best) < 1e-12]
        return fallback if fallback in tied else tied[0]

    def generate_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 16,
    ) -> list[str]:
        return [
            self.generate(frames, messages, max_new_tokens)
            for frames, messages in zip(batch_frames, batch_messages)
        ]
