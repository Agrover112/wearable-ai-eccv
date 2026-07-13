#!/usr/bin/env python3
"""Generate LongQA predictions with a question-aware timeline scaffold.

This experimental entry point tests whether an explicit temporal narrative helps
LongQA. For each video it:

  1. samples frames uniformly and asks the VLM for compact chronological notes
     relevant to the question/options,
  2. samples answer frames and asks the VLM to answer with the timeline notes in
     the text context.

It intentionally does not replace the current baseline or SigLIP entry points.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

TIMELINE_PROMPT_TEMPLATE = (
    "You are helping answer a long egocentric video multiple-choice question.\n"
    "From the frames, write a compact chronological timeline of only the events, "
    "objects, people, signs/text, places, and before/after relations that may be "
    "useful for answering the question.\n\n"
    "Question: {question}\n\n"
    "Options:\n{mcq_options}\n\n"
    "Use at most {max_bullets} short bullets. Keep the order temporal. Do not "
    "answer the multiple-choice question yet."
)

ANSWER_PROMPT_TEMPLATE = (
    "Watch the video frames and use the chronological timeline notes to answer "
    "the multiple-choice question.\n\n"
    "Timeline notes:\n{timeline}\n\n"
    "Question: {question}\n\n"
    "Options:\n{mcq_options}\n\n"
    "Answer with ONLY the single letter of the correct option (A, B, C, or D). "
    "Do not include any other text."
)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_eval(input_path: str, output_path: str, eval_output: str | None) -> None:
    from run_evaluation import evaluate_longqa, load_jsonl as load_eval_jsonl

    golden = load_eval_jsonl(input_path)
    preds = load_eval_jsonl(output_path)
    if len(golden) != len(preds):
        logger.warning(
            "Golden (%d) and predictions (%d) have different lengths; "
            "evaluating against the first %d golden rows.",
            len(golden),
            len(preds),
            len(preds),
        )
        golden = golden[: len(preds)]
    results = evaluate_longqa(golden, preds)
    if eval_output is None:
        eval_output = os.path.join(os.path.dirname(output_path), "results.json")
    from run_evaluation import write_results

    summary_path = write_results(eval_output, results)
    print(
        f"LongQA Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    print(f"Results written to {eval_output}")
    print(f"Summary written to {summary_path}")


def parse_args() -> argparse.Namespace:
    from model import MODEL_TYPES

    parser = argparse.ArgumentParser(
        description="Generate LongQA predictions with timeline-summary scaffolding."
    )
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--output", default="output/egolongqa_timeline/predictions.jsonl")
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--timeline-output", default=None)
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-eval", action="store_true")

    parser.add_argument("--timeline-frames", type=int, default=64)
    parser.add_argument("--answer-frames", type=int, default=32)
    parser.add_argument("--max-timeline-bullets", type=int, default=12)
    parser.add_argument("--timeline-max-new-tokens", type=int, default=192)
    parser.add_argument("--answer-max-new-tokens", type=int, default=16)

    parser.add_argument("--model-type", default="qwen", choices=MODEL_TYPES)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    import time

    from model import (
        create_model,
        extract_frames,
        reset_prompt_token_stats,
        summarize_prompt_token_stats,
    )
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    video_folder = _resolve_path(args.video_folder)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    timeline_output = (
        _resolve_path(args.timeline_output)
        if args.timeline_output
        else os.path.splitext(output_path)[0] + "_timeline.jsonl"
    )

    rows = load_jsonl(input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(timeline_output) or ".", exist_ok=True)

    print(
        "Timeline LongQA config: "
        f"timeline_frames={args.timeline_frames}, "
        f"answer_frames={args.answer_frames}, "
        f"max_timeline_bullets={args.max_timeline_bullets}, "
        f"backend={args.backend}, concurrency={args.concurrency}"
    )

    model = create_model(
        args.model_type,
        args.llm_model,
        backend=args.backend,
        tp_size=args.tp,
        concurrency=args.concurrency,
        max_frames=max(args.timeline_frames, args.answer_frames),
    )
    reset_prompt_token_stats()

    start_time = time.time()
    with model, open(output_path, "w") as pred_f, open(timeline_output, "w") as meta_f:
        for row_idx, row in enumerate(rows):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            timeline_frames = extract_frames(
                video_path,
                frames_per_interval=args.timeline_frames,
                max_frames=args.timeline_frames,
            )
            timeline_messages = [
                {
                    "role": "user",
                    "content": TIMELINE_PROMPT_TEMPLATE.format(
                        question=row["question"],
                        mcq_options=row["mcq_options"],
                        max_bullets=args.max_timeline_bullets,
                    ),
                }
            ]
            timeline = model.generate(
                timeline_frames,
                timeline_messages,
                max_new_tokens=args.timeline_max_new_tokens,
            )

            answer_frames = extract_frames(
                video_path,
                frames_per_interval=args.answer_frames,
                max_frames=args.answer_frames,
            )
            answer_messages = [
                {
                    "role": "user",
                    "content": ANSWER_PROMPT_TEMPLATE.format(
                        timeline=timeline,
                        question=row["question"],
                        mcq_options=row["mcq_options"],
                    ),
                }
            ]
            response = model.generate(
                answer_frames,
                answer_messages,
                max_new_tokens=args.answer_max_new_tokens,
            )

            pred = dict(row)
            pred["mcq_answer"] = response
            pred_f.write(json.dumps(pred) + "\n")
            pred_f.flush()

            meta = {
                "index": row_idx,
                "video_path": row.get("video_path", ""),
                "timeline_frames": len(timeline_frames),
                "answer_frames": len(answer_frames),
                "timeline": timeline,
            }
            meta_f.write(json.dumps(meta) + "\n")
            meta_f.flush()
            print(f"  Progress: {row_idx + 1}/{len(rows)}")

    elapsed = time.time() - start_time
    print(f"Predictions written to {output_path}")
    print(f"Timeline metadata written to {timeline_output}")
    print(f"Runtime seconds: {elapsed:.0f}")
    _print_context_summary(summarize_prompt_token_stats())

    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
