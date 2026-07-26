#!/usr/bin/env python3
"""Resolve prediction disagreements without erasing their evidence provenance.

The primary pass contributes its saved proof-pack frames, while the secondary
pass contributes an independently sampled uniform view.  The two sources stay
as separately labelled, interleaved text/image groups in one vLLM request.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
import hashlib
import io
import json
import os
import re
import urllib.request
from typing import Any

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    index_row_aligned_metadata,
    parse_mcq_options,
    sample_key,
)
from model import VLLMModel, record_prompt_token_counts
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import baseline_uniform_indices

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
PROVENANCE_VERIFIER_SCHEMA = 1
DEFAULT_PRIMARY_FRAME_COUNT = 32
DEFAULT_SECONDARY_FRAME_COUNT = 32
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_THINKING_MAX_NEW_TOKENS = 1024

_PRIMARY_SOURCE_PRIORITY = {
    "pivot": 0,
    "directional_target": 0,
    "bridge": 1,
    "pivot_context": 1,
    "directional_target_context": 1,
    "event_center": 2,
    "event_center_context": 2,
    "anchor": 3,
    "coverage_fill": 4,
    "semantic_boundary": 5,
}


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("mcq_answer_parsed") or row.get("mcq_answer", "")).strip().upper()


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, total_frames


def select_primary_proofpack_evidence(
    selected: list[dict[str, Any]], max_frames: int = DEFAULT_PRIMARY_FRAME_COUNT
) -> list[dict[str, Any]]:
    """Keep the best saved proof-pack frames, then display them chronologically."""
    ranked = sorted(
        selected,
        key=lambda item: (
            _PRIMARY_SOURCE_PRIORITY.get(str(item.get("source", "")), 6),
            -(float(item["score"]) if item.get("score") is not None else -1e9),
            float(item.get("timestamp", 0.0)),
            int(item["frame_index"]),
        ),
    )
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in ranked:
        frame_index = int(item["frame_index"])
        if frame_index in seen:
            continue
        seen.add(frame_index)
        chosen.append(
            {
                "frame_index": frame_index,
                "timestamp": float(item.get("timestamp", 0.0)),
                "source": str(item.get("source", "proofpack")),
                "source_score": item.get("score"),
            }
        )
        if len(chosen) >= max_frames:
            break
    return sorted(chosen, key=lambda item: (item["timestamp"], item["frame_index"]))


def select_secondary_uniform_evidence(
    total_frames: int,
    fps: float,
    max_frames: int = DEFAULT_SECONDARY_FRAME_COUNT,
) -> list[dict[str, Any]]:
    """Build deterministic uniform evidence owned by the secondary pass."""
    return [
        {
            "frame_index": frame_index,
            "timestamp": frame_index / fps,
            "source": "uniform_secondary",
            "source_score": None,
        }
        for frame_index in baseline_uniform_indices(total_frames, max_frames)
    ]


def select_provenance_evidence(
    proofpack_selected: list[dict[str, Any]],
    total_frames: int,
    fps: float,
    primary_max_frames: int = DEFAULT_PRIMARY_FRAME_COUNT,
    secondary_max_frames: int = DEFAULT_SECONDARY_FRAME_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select two independent evidence groups without cross-source deduplication."""
    primary = select_primary_proofpack_evidence(proofpack_selected, primary_max_frames)
    secondary = select_secondary_uniform_evidence(total_frames, fps, secondary_max_frames)
    return primary, secondary


def _frame_pairs(
    video_path: str, evidence: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], object]]:
    frames = extract_frames_by_indices(
        video_path, [int(item["frame_index"]) for item in evidence]
    )
    by_frame_index = {int(frame.info["source_frame_index"]): frame for frame in frames}
    return [
        (item, by_frame_index[int(item["frame_index"])])
        for item in evidence
        if int(item["frame_index"]) in by_frame_index
    ]


def _encode_image(frame: object) -> dict[str, object]:
    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


def _evidence_header(group: str, count: int) -> str:
    return (
        f"=== {group} ({count} frames) ===\n"
        "This is an evidence provenance label, not a vote for either candidate."
    )


def build_provenance_content(
    row: dict[str, Any],
    primary_answer: str,
    secondary_answer: str,
    primary_frames: list[tuple[dict[str, Any], object]],
    secondary_frames: list[tuple[dict[str, Any], object]],
) -> list[dict[str, object]]:
    """Build one chat content list with source labels adjacent to every image."""
    options = parse_mcq_options(row["mcq_options"])
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Two independent video passes disagreed. Decide only between their "
                "candidate option semantics using the labelled visual evidence.\n\n"
                f"Question: {row['question']}\n\n"
                f"Candidate 1 (primary pass): option {primary_answer}: "
                f"{options[primary_answer]}\n"
                f"Candidate 2 (secondary pass): option {secondary_answer}: "
                f"{options[secondary_answer]}\n\n"
                "Check visible support, contradiction, and temporal order. Do not "
                "introduce a third option."
            ),
        },
        {"type": "text", "text": _evidence_header("PRIMARY PROOFPACK EVIDENCE", len(primary_frames))},
    ]
    for position, (metadata, frame) in enumerate(primary_frames, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"PRIMARY frame {position:02d} | t={metadata['timestamp']:.3f}s | "
                    f"frame={metadata['frame_index']} | source={metadata['source']}"
                ),
            }
        )
        content.append(_encode_image(frame))

    content.append(
        {"type": "text", "text": _evidence_header("SECONDARY UNIFORM EVIDENCE", len(secondary_frames))}
    )
    for position, (metadata, frame) in enumerate(secondary_frames, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"SECONDARY frame {position:02d} | t={metadata['timestamp']:.3f}s | "
                    f"frame={metadata['frame_index']} | source={metadata['source']}"
                ),
            }
        )
        content.append(_encode_image(frame))

    content.append(
        {
            "type": "text",
            "text": (
                "Return exactly one label: CANDIDATE_1, CANDIDATE_2, or INSUFFICIENT. "
                "Return INSUFFICIENT when the supplied evidence cannot distinguish "
                "the two candidates."
            ),
        }
    )
    return content


def build_provenance_payload(
    model_id: str,
    content: list[dict[str, object]],
    max_new_tokens: int,
    thinking: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if thinking:
        payload.update(
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 1.5,
            }
        )
    else:
        payload["temperature"] = 0.0
    return payload


def request_vllm_chat(model: VLLMModel, payload: dict[str, object]) -> str:
    """Issue an interleaved-content request through an existing VLLMModel server."""
    request = urllib.request.Request(
        f"http://localhost:{model._port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=model.request_timeout) as response:
        result = json.loads(response.read())
    if "error" in result:
        raise RuntimeError(f"vLLM returned error: {result['error']}")
    usage = result.get("usage", {})
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        record_prompt_token_counts([int(usage["prompt_tokens"])], model._context_window)
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected vLLM response structure: {result}") from error
    if content is None:
        raise RuntimeError("vLLM returned null content for provenance verification")
    return str(content)


class ProvenanceVerifierVLLMModel(VLLMModel):
    """Small vLLM extension that preserves text-image ordering in a request."""

    def generate_provenance(
        self,
        content: list[dict[str, object]],
        max_new_tokens: int,
        thinking: bool,
    ) -> str:
        payload = build_provenance_payload(
            self.model_id, content, max_new_tokens, thinking
        )
        return request_vllm_chat(self, payload)


def parse_verifier_choice(response: object) -> str | None:
    """Parse only the three allowed verifier decisions."""
    text = str(response).strip().upper()
    if "</THINK>" in text:
        text = text.rsplit("</THINK>", 1)[-1].strip()
    normalized = re.sub(r"[\s-]+", "_", text)
    if normalized in {"CANDIDATE_1", "CANDIDATE_2", "INSUFFICIENT"}:
        return normalized.lower()
    final_match = re.search(
        r"(?:FINAL(?:_ANSWER|_DECISION)?|ANSWER|DECISION)\s*[:=]?\s*"
        r"(CANDIDATE[\s_-]*[12]|INSUFFICIENT)",
        text,
    )
    if final_match:
        return re.sub(r"[\s-]+", "_", final_match.group(1)).lower()
    choices = re.findall(r"\b(CANDIDATE[\s_-]*[12]|INSUFFICIENT)\b", text)
    if len(choices) == 1:
        return re.sub(r"[\s-]+", "_", choices[0]).lower()
    return None


def choose_candidate(
    choice: str | None, primary_answer: str, secondary_answer: str
) -> tuple[str, bool]:
    """Use primary whenever the verifier declines or violates its output contract."""
    if choice == "candidate_1":
        return primary_answer, False
    if choice == "candidate_2":
        return secondary_answer, False
    return primary_answer, True


def should_call_verifier(primary_answer: str, secondary_answer: str) -> bool:
    return primary_answer != secondary_answer


def provenance_verifier_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "schema": PROVENANCE_VERIFIER_SCHEMA,
        "input": os.path.abspath(args.input),
        "subset_file": os.path.abspath(args.subset_file) if args.subset_file else None,
        "video_folder": os.path.abspath(args.video_folder),
        "primary_predictions": os.path.abspath(args.primary_predictions),
        "secondary_predictions": os.path.abspath(args.secondary_predictions),
        "primary_proofpack": os.path.abspath(args.primary_proofpack),
        "primary_frame_count": args.primary_frame_count,
        "secondary_frame_count": args.secondary_frame_count,
        "llm_model": args.llm_model,
        "tp": args.tp,
        "concurrency": args.concurrency,
        "thinking": args.thinking,
        "max_new_tokens": args.max_new_tokens,
        "thinking_max_new_tokens": args.thinking_max_new_tokens,
        "qwen_min_pixels": os.environ.get("QWEN_MIN_PIXELS", "784"),
        "qwen_max_pixels": os.environ.get("QWEN_MAX_PIXELS", "50176"),
        "vllm_max_model_len": os.environ.get("VLLM_QWEN_MAX_MODEL_LEN", "16384"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def resume_position(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    fingerprint: str,
) -> int:
    start = 0
    for row, prediction, evidence_row in zip(rows, predictions, evidence):
        if sample_key(prediction) != sample_key(row):
            break
        if sample_key(evidence_row) != sample_key(row):
            break
        if not str(prediction.get("mcq_answer", "")).strip():
            break
        if prediction.get("provenance_verifier_fingerprint") != fingerprint:
            break
        if evidence_row.get("provenance_verifier_fingerprint") != fingerprint:
            break
        start += 1
    return start


def _rewrite_prefix(path: str, rows: list[dict[str, Any]], count: int) -> None:
    with open(path, "w") as handle:
        for row in rows[:count]:
            handle.write(json.dumps(row) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve LongQA prediction disagreements with source-separated evidence."
    )
    parser.add_argument(
        "--input", default="../../data/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../../data/videos")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--secondary-predictions", required=True)
    parser.add_argument("--primary-proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", default=None)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-resume-predictions", action="store_true")
    parser.add_argument("--primary-frame-count", type=int, default=DEFAULT_PRIMARY_FRAME_COUNT)
    parser.add_argument("--secondary-frame-count", type=int, default=DEFAULT_SECONDARY_FRAME_COUNT)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen thinking only for disagreement arbitration.",
    )
    parser.add_argument(
        "--thinking-max-new-tokens", type=int, default=DEFAULT_THINKING_MAX_NEW_TOKENS
    )
    args = parser.parse_args(argv)
    if not 0 <= args.primary_frame_count <= DEFAULT_PRIMARY_FRAME_COUNT:
        parser.error("--primary-frame-count must be between 0 and 32")
    if not 0 <= args.secondary_frame_count <= DEFAULT_SECONDARY_FRAME_COUNT:
        parser.error("--secondary-frame-count must be between 0 and 32")
    if args.primary_frame_count + args.secondary_frame_count > 64:
        parser.error("provenance evidence is limited to 64 images")
    return args


def main() -> None:
    import time

    from model import reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    primary_path = _resolve_path(args.primary_predictions)
    secondary_path = _resolve_path(args.secondary_predictions)
    proofpack_path = _resolve_path(args.primary_proofpack)
    output_path = _resolve_path(args.output)
    evidence_output = _resolve_path(
        args.evidence_output
        or os.path.join(os.path.dirname(output_path), "verifier_evidence.jsonl")
    )
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    fingerprint = provenance_verifier_fingerprint(args)

    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    primary_rows = load_jsonl(primary_path)
    primary = {sample_key(row): row for row in primary_rows}
    secondary = {sample_key(row): row for row in load_jsonl(secondary_path)}
    proofpack = index_row_aligned_metadata(
        load_jsonl(proofpack_path), primary_rows, "primary proof pack"
    )
    missing = [
        sample_key(row)
        for row in rows
        if sample_key(row) not in primary
        or sample_key(row) not in secondary
        or sample_key(row) not in proofpack
    ]
    if missing:
        raise RuntimeError(f"Required primary, secondary, or proof-pack rows missing: {len(missing)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(evidence_output) or ".", exist_ok=True)
    existing_predictions = (
        [] if args.no_resume_predictions or not os.path.exists(output_path) else load_jsonl(output_path)
    )
    existing_evidence = (
        []
        if args.no_resume_predictions or not os.path.exists(evidence_output)
        else load_jsonl(evidence_output)
    )
    start = resume_position(rows, existing_predictions, existing_evidence, fingerprint)
    if len(existing_predictions) != start:
        _rewrite_prefix(output_path, existing_predictions, start)
    if len(existing_evidence) != start:
        _rewrite_prefix(evidence_output, existing_evidence, start)
    if start:
        print(f"Resuming provenance verifier from {start}/{len(rows)} rows")

    remaining = rows[start:]
    need_model = any(
        should_call_verifier(_answer(primary[sample_key(row)]), _answer(secondary[sample_key(row)]))
        for row in remaining
    )
    model = (
        ProvenanceVerifierVLLMModel(
            model_id=args.llm_model,
            tp_size=args.tp,
            concurrency=args.concurrency,
            max_frames=args.primary_frame_count + args.secondary_frame_count,
            model_type="qwen",
            request_timeout=args.request_timeout,
        )
        if need_model
        else None
    )
    request_max_tokens = (
        args.thinking_max_new_tokens if args.thinking else args.max_new_tokens
    )
    reset_prompt_token_stats()
    started = time.time()
    verified = 0
    with (model if model is not None else nullcontext()), open(
        output_path, "a" if start else "w"
    ) as predictions_handle, open(evidence_output, "a" if start else "w") as evidence_handle:
        for index, row in enumerate(remaining, start=start):
            key = sample_key(row)
            primary_answer = _answer(primary[key])
            secondary_answer = _answer(secondary[key])
            applied = should_call_verifier(primary_answer, secondary_answer)
            primary_evidence: list[dict[str, Any]] = []
            secondary_evidence: list[dict[str, Any]] = []
            raw_response: str | None = None
            choice: str | None = None
            if applied:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                fps, total_frames = _video_metadata(video_path)
                primary_evidence, secondary_evidence = select_provenance_evidence(
                    proofpack[key]["selected"],
                    total_frames,
                    fps,
                    args.primary_frame_count,
                    args.secondary_frame_count,
                )
                primary_pairs = _frame_pairs(video_path, primary_evidence)
                secondary_pairs = _frame_pairs(video_path, secondary_evidence)
                content = build_provenance_content(
                    row,
                    primary_answer,
                    secondary_answer,
                    primary_pairs,
                    secondary_pairs,
                )
                raw_response = model.generate_provenance(
                    content, request_max_tokens, args.thinking
                )
                choice = parse_verifier_choice(raw_response)
                answer, fallback_primary = choose_candidate(
                    choice, primary_answer, secondary_answer
                )
                verified += 1
            else:
                answer = primary_answer
                fallback_primary = False
                total_frames = None
                fps = None

            prediction = build_prediction_row(
                row, answer, prompt_variant="provenance_disagreement_verifier"
            )
            prediction.update(
                {
                    "candidate_answers": [primary_answer, secondary_answer],
                    "provenance_verifier_applied": applied,
                    "provenance_verifier_choice": choice,
                    "provenance_verifier_fallback_primary": fallback_primary,
                    "provenance_verifier_fingerprint": fingerprint,
                }
            )
            evidence_record = {
                "schema": PROVENANCE_VERIFIER_SCHEMA,
                "index": index,
                "sample_key": key,
                "video_path": row["video_path"],
                "candidate_answers": [primary_answer, secondary_answer],
                "provenance_verifier_applied": applied,
                "primary_proofpack_evidence": primary_evidence,
                "secondary_uniform_evidence": secondary_evidence,
                "primary_proofpack_frame_count": len(primary_evidence),
                "secondary_uniform_frame_count": len(secondary_evidence),
                "total_image_count": len(primary_evidence) + len(secondary_evidence),
                "fps": fps,
                "total_frames": total_frames,
                "raw_response": raw_response,
                "parsed_choice": choice,
                "selected_answer": answer,
                "fallback_primary": fallback_primary,
                "thinking": args.thinking,
                "request_max_tokens": request_max_tokens,
                "provenance_verifier_fingerprint": fingerprint,
            }
            predictions_handle.write(json.dumps(prediction) + "\n")
            evidence_handle.write(json.dumps(evidence_record) + "\n")
            predictions_handle.flush()
            evidence_handle.flush()
            print(f"  Provenance verifier progress: {index + 1}/{len(rows)}")

    print(f"Provenance verifier calls this invocation: {verified}")
    print(f"Runtime seconds: {time.time() - started:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
