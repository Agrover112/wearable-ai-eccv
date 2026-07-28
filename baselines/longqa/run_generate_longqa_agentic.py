#!/usr/bin/env python3
"""Run a training-free multi-agent Qwen3.5 verifier on saved proof packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from typing import Any

from agents.evidence import load_proofpack_map, select_evidence
from agents.hypothesis_agent import (
    PROMPT_VERSION as HYPOTHESIS_PROMPT_VERSION,
)
from agents.hypothesis_agent import run_hypothesis_agent
from agents.orchestration import run_specialists_parallel
from agents.organizer_agent import (
    PROMPT_VERSION as ORGANIZER_PROMPT_VERSION,
)
from agents.organizer_agent import run_organizer_agent
from agents.temporal_agent import PROMPT_VERSION as TEMPORAL_PROMPT_VERSION
from agents.visual_agent import PROMPT_VERSION as VISUAL_PROMPT_VERSION
from longqa_utils import (
    apply_subset,
    build_prediction_row,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_fixed_pack import load_jsonl, load_jsonl_if_exists
from run_generate_longqa_grounded import extract_frames_by_indices

logger = logging.getLogger(__name__)

AGENTIC_SCHEMA = 1
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_MAX_EVIDENCE_FRAMES = 64
DEFAULT_QUESTION_TIME_LIMIT = 300


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, total_frames


def agentic_fingerprint(
    args: argparse.Namespace,
    input_path: str,
    video_folder: str,
    subset_path: str | None,
    proofpack_sha1: str,
) -> str:
    payload = {
        "schema": AGENTIC_SCHEMA,
        "input": input_path,
        "input_sha1": _file_sha1(input_path),
        "video_folder": video_folder,
        "subset": subset_path,
        "subset_sha1": _file_sha1(subset_path) if subset_path else None,
        "proofpack_sha1": proofpack_sha1,
        "model": args.llm_model,
        "model_revision": args.llm_revision,
        "tp": args.tp,
        "concurrency": args.concurrency,
        "max_evidence_frames": args.max_evidence_frames,
        "hypothesis_max_new_tokens": args.hypothesis_max_new_tokens,
        "specialist_max_new_tokens": args.specialist_max_new_tokens,
        "organizer_max_new_tokens": args.organizer_max_new_tokens,
        "question_time_limit_seconds": args.question_time_limit_seconds,
        "prompt_versions": {
            "hypothesis": HYPOTHESIS_PROMPT_VERSION,
            "temporal": TEMPORAL_PROMPT_VERSION,
            "visual": VISUAL_PROMPT_VERSION,
            "organizer": ORGANIZER_PROMPT_VERSION,
        },
        "qwen_min_pixels": os.environ.get("QWEN_MIN_PIXELS", "784"),
        "qwen_max_pixels": os.environ.get("QWEN_MAX_PIXELS", "50176"),
        "vllm_max_model_len": os.environ.get("VLLM_QWEN_MAX_MODEL_LEN", "16384"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def resume_position(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    fingerprint: str,
) -> int:
    start = 0
    for row, prediction, trace in zip(rows, predictions, traces):
        if sample_key(prediction) != sample_key(row):
            break
        if str(trace.get("sample_key", "")) != sample_key(row):
            break
        if prediction.get("agentic_fingerprint") != fingerprint:
            break
        if trace.get("agentic_fingerprint") != fingerprint:
            break
        if str(prediction.get("mcq_answer_parsed", "")) not in {"A", "B", "C", "D"}:
            break
        start += 1
    return start


def _rewrite_prefix(path: str, records: list[dict[str, Any]], count: int) -> None:
    with open(path, "w") as handle:
        for record in records[:count]:
            handle.write(json.dumps(record) + "\n")


def _set_request_budget(
    model: Any,
    inference_started: float,
    limit_seconds: float,
    configured_timeout: int,
) -> None:
    elapsed = time.perf_counter() - inference_started
    remaining = math.floor(limit_seconds - elapsed)
    if remaining <= 0:
        raise TimeoutError(
            f"Agentic inference exceeded the {limit_seconds}s per-question budget"
        )
    model.request_timeout = min(configured_timeout, remaining)


def _evaluate_subset(
    rows: list[dict[str, Any]],
    predictions_path: str,
    traces_path: str,
    output_path: str,
) -> dict[str, Any]:
    from run_evaluation import evaluate_longqa, write_results

    predictions = load_jsonl(predictions_path)
    traces = load_jsonl(traces_path)
    if [sample_key(row) for row in predictions] != [sample_key(row) for row in rows]:
        raise RuntimeError("Prediction rows do not align with the selected evaluation rows")
    if [str(row["sample_key"]) for row in traces] != [
        sample_key(row) for row in rows
    ]:
        raise RuntimeError("Trace rows do not align with the selected evaluation rows")
    results = evaluate_longqa(rows, predictions)
    budgeted_times = [
        float(trace["timing"]["budgeted_inference_seconds"]) for trace in traces
    ]
    retrieval_times = [
        float(trace["timing"]["proofpack_retrieval_seconds"]) for trace in traces
    ]
    orchestration_times = [
        float(trace["timing"]["orchestration_seconds"]) for trace in traces
    ]
    preparation_times = [
        float(trace["timing"]["evidence_preparation_seconds"]) for trace in traces
    ]
    full_pipeline_times = [
        float(trace["timing"]["full_pipeline_seconds"]) for trace in traces
    ]
    ordered = sorted(budgeted_times)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    results["timing"] = {
        "mean_budgeted_inference_seconds": round(
            sum(budgeted_times) / len(budgeted_times), 3
        ),
        "p95_budgeted_inference_seconds": round(ordered[p95_index], 3),
        "max_budgeted_inference_seconds": round(max(budgeted_times), 3),
        "mean_proofpack_retrieval_seconds": round(
            sum(retrieval_times) / len(retrieval_times), 3
        ),
        "mean_evidence_preparation_seconds": round(
            sum(preparation_times) / len(preparation_times), 3
        ),
        "mean_orchestration_seconds": round(
            sum(orchestration_times) / len(orchestration_times), 3
        ),
        "mean_full_pipeline_seconds": round(
            sum(full_pipeline_times) / len(full_pipeline_times), 3
        ),
        "max_full_pipeline_seconds": round(max(full_pipeline_times), 3),
        "over_budget": sum(bool(trace["timing"]["over_budget"]) for trace in traces),
    }
    summary_path = write_results(output_path, results)
    print(
        f"LongQA Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    print(
        "Budgeted inference seconds (retrieval + agents): "
        f"mean={results['timing']['mean_budgeted_inference_seconds']:.1f}, "
        f"p95={results['timing']['p95_budgeted_inference_seconds']:.1f}, "
        f"max={results['timing']['max_budgeted_inference_seconds']:.1f}"
    )
    print(
        "Full pipeline seconds including frame preparation: "
        f"mean={results['timing']['mean_full_pipeline_seconds']:.1f}, "
        f"max={results['timing']['max_full_pipeline_seconds']:.1f}"
    )
    print(f"Results written to {output_path}")
    print(f"Summary written to {summary_path}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-free hypothesis/specialist/organizer LongQA experiment."
    )
    parser.add_argument(
        "--input", default="../../data/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../../data/videos")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-revision", default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument(
        "--max-evidence-frames", type=int, default=DEFAULT_MAX_EVIDENCE_FRAMES
    )
    parser.add_argument("--hypothesis-max-new-tokens", type=int, default=768)
    parser.add_argument("--specialist-max-new-tokens", type=int, default=1024)
    parser.add_argument("--organizer-max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--question-time-limit-seconds",
        type=int,
        default=DEFAULT_QUESTION_TIME_LIMIT,
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_evidence_frames <= 64:
        parser.error("--max-evidence-frames must be between 1 and 64")
    if args.concurrency < 2:
        parser.error("--concurrency must be at least 2 for parallel specialists")
    return args


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    subset_path = _resolve_path(args.subset_file) if args.subset_file else None
    proofpack_path = _resolve_path(args.proofpack)
    output_path = _resolve_path(args.output)
    trace_output = _resolve_path(
        args.trace_output
        or os.path.join(os.path.dirname(output_path), "agent_traces.jsonl")
    )
    eval_output = _resolve_path(
        args.eval_output or os.path.join(os.path.dirname(output_path), "results.json")
    )

    rows = apply_subset(load_jsonl(input_path), subset_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    proofpacks = load_proofpack_map(proofpack_path)
    missing = [sample_key(row) for row in rows if sample_key(row) not in proofpacks]
    if missing:
        raise RuntimeError(f"Proof pack is missing {len(missing)} selected samples")
    for row in rows:
        if set(parse_mcq_options(row["mcq_options"])) != set("ABCD"):
            raise RuntimeError(f"Expected four A-D options for {sample_key(row)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(trace_output) or ".", exist_ok=True)
    fingerprint = agentic_fingerprint(
        args,
        input_path,
        video_folder,
        subset_path,
        _file_sha1(proofpack_path),
    )
    existing_predictions = (
        [] if args.no_resume else load_jsonl_if_exists(output_path)
    )
    existing_traces = [] if args.no_resume else load_jsonl_if_exists(trace_output)
    start = resume_position(
        rows,
        existing_predictions,
        existing_traces,
        fingerprint,
    )
    if len(existing_predictions) != start:
        _rewrite_prefix(output_path, existing_predictions, start)
    if len(existing_traces) != start:
        _rewrite_prefix(trace_output, existing_traces, start)
    if start:
        print(f"Resuming agentic generation from {start}/{len(rows)} rows")

    print(
        f"Agentic config: model={args.llm_model}, frames={args.max_evidence_frames}, "
        f"parallel_specialists=2, time_limit={args.question_time_limit_seconds}s, "
        f"fingerprint={fingerprint}"
    )
    reset_prompt_token_stats()
    wall_started = time.time()
    if start < len(rows):
        model = VLLMModel(
            model_id=args.llm_model,
            tp_size=args.tp,
            concurrency=args.concurrency,
            max_frames=args.max_evidence_frames,
            model_type="qwen",
            request_timeout=args.request_timeout,
            revision=args.llm_revision,
        )
        mode = "a" if start else "w"
        with model, open(output_path, mode) as prediction_handle, open(
            trace_output, mode
        ) as trace_handle:
            for index, row in enumerate(rows[start:], start=start):
                preparation_started = time.perf_counter()
                video_path = os.path.join(video_folder, str(row["video_path"]))
                fps, total_frames = _video_metadata(video_path)
                evidence = select_evidence(
                    row,
                    proofpacks[sample_key(row)],
                    fps,
                    args.max_evidence_frames,
                )
                frame_indices = [int(item["frame_index"]) for item in evidence]
                frames = extract_frames_by_indices(video_path, frame_indices)
                extracted = [
                    int(frame.info["source_frame_index"]) for frame in frames
                ]
                if extracted != frame_indices:
                    raise RuntimeError(
                        f"Frame extraction changed evidence order for {sample_key(row)}"
                    )
                evidence_preparation_seconds = (
                    time.perf_counter() - preparation_started
                )
                options = parse_mcq_options(row["mcq_options"])
                retrieval_seconds = float(
                    proofpacks[sample_key(row)]["selection_seconds"]
                )
                orchestration_limit = (
                    args.question_time_limit_seconds - retrieval_seconds
                )
                if orchestration_limit <= 0:
                    raise TimeoutError(
                        "Proof-pack retrieval consumed the complete "
                        f"{args.question_time_limit_seconds}s query budget for "
                        f"{sample_key(row)}"
                    )

                inference_started = time.perf_counter()
                _set_request_budget(
                    model,
                    inference_started,
                    orchestration_limit,
                    args.request_timeout,
                )
                phase_started = time.perf_counter()
                hypotheses = run_hypothesis_agent(
                    model,
                    row,
                    options,
                    args.hypothesis_max_new_tokens,
                )
                hypothesis_seconds = time.perf_counter() - phase_started

                _set_request_budget(
                    model,
                    inference_started,
                    orchestration_limit,
                    args.request_timeout,
                )
                phase_started = time.perf_counter()
                temporal_report, visual_report = run_specialists_parallel(
                    model,
                    frames,
                    row,
                    options,
                    hypotheses,
                    evidence,
                    args.specialist_max_new_tokens,
                )
                specialist_seconds = time.perf_counter() - phase_started

                _set_request_budget(
                    model,
                    inference_started,
                    orchestration_limit,
                    args.request_timeout,
                )
                phase_started = time.perf_counter()
                organizer_report = run_organizer_agent(
                    model,
                    row,
                    options,
                    hypotheses,
                    temporal_report,
                    visual_report,
                    args.organizer_max_new_tokens,
                )
                organizer_seconds = time.perf_counter() - phase_started
                orchestration_seconds = time.perf_counter() - inference_started
                budgeted_inference_seconds = (
                    retrieval_seconds + orchestration_seconds
                )
                full_pipeline_seconds = (
                    budgeted_inference_seconds + evidence_preparation_seconds
                )
                over_budget = (
                    budgeted_inference_seconds
                    > args.question_time_limit_seconds
                )
                if over_budget:
                    logger.warning(
                        "Sample %d exceeded the inference budget: %.1fs",
                        index,
                        budgeted_inference_seconds,
                    )

                answer = str(organizer_report["selected_option"])
                prediction = build_prediction_row(
                    row,
                    answer,
                    prompt_variant="agentic_hypothesis_specialists_organizer",
                )
                prediction.update(
                    {
                        "agentic_fingerprint": fingerprint,
                        "agentic_orchestration_seconds": round(
                            orchestration_seconds, 3
                        ),
                        "agentic_budgeted_inference_seconds": round(
                            budgeted_inference_seconds, 3
                        ),
                    }
                )
                trace = {
                    "schema": AGENTIC_SCHEMA,
                    "index": index,
                    "sample_key": sample_key(row),
                    "video_path": row["video_path"],
                    "fps": fps,
                    "total_frames": total_frames,
                    "evidence": evidence,
                    "hypothesis_report": hypotheses,
                    "temporal_report": temporal_report,
                    "visual_report": visual_report,
                    "organizer_report": organizer_report,
                    "specialist_views_share_model": True,
                    "selected_answer": answer,
                    "timing": {
                        "hypothesis_seconds": round(hypothesis_seconds, 3),
                        "parallel_specialists_seconds": round(
                            specialist_seconds, 3
                        ),
                        "organizer_seconds": round(organizer_seconds, 3),
                        "proofpack_retrieval_seconds": round(
                            retrieval_seconds, 3
                        ),
                        "evidence_preparation_seconds": round(
                            evidence_preparation_seconds, 3
                        ),
                        "orchestration_seconds": round(
                            orchestration_seconds, 3
                        ),
                        "budgeted_inference_seconds": round(
                            budgeted_inference_seconds, 3
                        ),
                        "full_pipeline_seconds": round(
                            full_pipeline_seconds, 3
                        ),
                        "limit_seconds": args.question_time_limit_seconds,
                        "limit_excludes_frame_preparation": True,
                        "over_budget": over_budget,
                    },
                    "agentic_fingerprint": fingerprint,
                }
                prediction_handle.write(json.dumps(prediction) + "\n")
                trace_handle.write(json.dumps(trace) + "\n")
                prediction_handle.flush()
                trace_handle.flush()
                print(
                    f"  Agentic progress: {index + 1}/{len(rows)} "
                    f"answer={answer} "
                    f"budgeted={budgeted_inference_seconds:.1f}s "
                    f"full_pipeline={full_pipeline_seconds:.1f}s"
                )

    print(f"Runtime seconds including model startup: {time.time() - wall_started:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _evaluate_subset(rows, output_path, trace_output, eval_output)


if __name__ == "__main__":
    main()
