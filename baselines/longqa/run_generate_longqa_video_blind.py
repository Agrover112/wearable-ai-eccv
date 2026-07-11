#!/usr/bin/env python3
"""Run a matched LongQA ablation using Qwen's language branch only."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from longqa_utils import (
    PROMPT_VARIANTS,
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    query_hash,
    sample_key,
)
from run_generate_longqa import _print_context_summary

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_evaluation(input_path: str, output_path: str, eval_output: str) -> None:
    from run_evaluation import _filter_subset, evaluate_longqa, write_results

    golden = _load_jsonl(input_path)
    predictions = _load_jsonl(output_path)
    if len(golden) != len(predictions):
        golden, predictions = _filter_subset(golden, predictions, "longqa")
    results = evaluate_longqa(golden, predictions)
    summary = write_results(eval_output, results)
    print(
        f"LongQA Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    print(f"Results written to {eval_output}")
    print(f"Summary written to {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video-blind LongQA generation.")
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--output", default="output/egolongqa_video_blind/predictions.jsonl")
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-type", default="qwen", choices=["qwen", "llama4"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="baseline")
    parser.add_argument("--no-resume-predictions", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def video_blind_config_fingerprint(args: argparse.Namespace) -> str:
    from model import DEFAULT_MODEL_IDS

    payload = {
        "model_type": args.model_type,
        "llm_model": args.llm_model or DEFAULT_MODEL_IDS[args.model_type],
        "backend": args.backend,
        "prompt_variant": args.prompt_variant,
        "ablation": "video_blind",
    }
    return query_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main() -> None:
    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(
        args.eval_output or os.path.join(os.path.dirname(args.output), "results.json")
    )
    rows = apply_subset(_load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    config_fingerprint = video_blind_config_fingerprint(args)
    cached = [] if args.no_resume_predictions else _load_jsonl(output_path)
    valid: list[dict[str, Any]] = []
    for idx, pred in enumerate(cached[: len(rows)]):
        if sample_key(pred) != sample_key(rows[idx]):
            break
        if not str(pred.get("mcq_answer", "")).strip():
            break
        if pred.get("ablation") != "video_blind":
            break
        if str(pred.get("video_blind_config_fingerprint", "")) != config_fingerprint:
            break
        valid.append(pred)
    if len(valid) != len(cached):
        with open(output_path, "w") as f:
            for pred in valid:
                f.write(json.dumps(pred) + "\n")
    if valid:
        print(f"Resuming predictions from {len(valid)}/{len(rows)} cached rows")
    if len(valid) == len(rows):
        print("Video-blind prediction cache complete; skipping model load")
        if not args.no_eval:
            _write_evaluation(input_path, output_path, eval_output)
        return

    model = create_model(
        args.model_type,
        args.llm_model,
        backend=args.backend,
        tp_size=args.tp,
        concurrency=args.concurrency,
        max_frames=0,
    )
    reset_prompt_token_stats()
    mode = "a" if valid else "w"
    with model, open(output_path, mode) as pred_f:
        for start in range(len(valid), len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            messages = [
                [
                    {
                        "role": "user",
                        "content": build_longqa_prompt(
                            row["question"],
                            row["mcq_options"],
                            prompt_variant=args.prompt_variant,
                        ),
                    }
                ]
                for row in batch
            ]
            responses = model.generate_batch(
                [[] for _ in batch],
                messages,
                max_new_tokens=16,
            )
            for row, response in zip(batch, responses):
                pred = build_prediction_row(row, response, args.prompt_variant)
                pred.update(
                    {
                        "ablation": "video_blind",
                        "frames": 0,
                        "video_blind_config_fingerprint": config_fingerprint,
                    }
                )
                pred_f.write(json.dumps(pred) + "\n")
                pred_f.flush()
            print(f"  Progress: {min(start + len(batch), len(rows))}/{len(rows)}")
    print(f"Predictions written to {output_path}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _write_evaluation(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
