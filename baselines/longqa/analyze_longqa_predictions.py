#!/usr/bin/env python3
"""Analyze complete or partial LongQA prediction files.

Useful for long-running experiments that are cancelled before evaluation. The
script reports prefix accuracy, rolling-window accuracy, and category accuracy
for the rows that have already been generated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_answer(raw: object) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    upper = text.upper()
    if len(upper) == 1 and upper in "ABCD":
        return upper
    m = re.search(r"\b(?:is|answer)\s*[:.]?\s*([A-Da-d])\s*\.?\s*$", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return upper[0] if upper[:1] in "ABCD" else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LongQA predictions.")
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
        help="Golden LongQA JSONL.",
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--prefix",
        type=int,
        nargs="*",
        default=[50, 100, 200, 300, 400, 500, 600, 700],
    )
    return parser.parse_args()


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(script_dir, path)


def resolve_output(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def main() -> None:
    args = parse_args()
    golden = load_jsonl(resolve(args.input))
    preds = load_jsonl(resolve(args.predictions))
    n = min(len(golden), len(preds))
    golden = golden[:n]
    preds = preds[:n]

    correct: list[bool] = []
    rows: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for idx, (gold, pred) in enumerate(zip(golden, preds)):
        gold_answer = normalize_answer(gold.get("mcq_answer", ""))
        pred_answer = normalize_answer(pred.get("mcq_answer", ""))
        ok = pred_answer == gold_answer
        correct.append(ok)
        category = str(gold.get("category", ""))
        category_stats[category]["total"] += 1
        if ok:
            category_stats[category]["correct"] += 1
        rows.append(
            {
                "index": idx,
                "video_path": gold.get("video_path", ""),
                "category": category,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "correct": ok,
            }
        )

    total_correct = sum(correct)
    prefix = []
    for k in args.prefix:
        if 0 < k <= n:
            prefix.append(
                {
                    "n": k,
                    "correct": sum(correct[:k]),
                    "accuracy": round(sum(correct[:k]) / k, 4),
                }
            )

    rolling = []
    if args.window > 0:
        for end in range(args.window, n + 1, args.window):
            start = end - args.window
            wins = sum(correct[start:end])
            rolling.append(
                {
                    "start": start,
                    "end": end,
                    "correct": wins,
                    "accuracy": round(wins / args.window, 4),
                }
            )
        if n and (not rolling or rolling[-1]["end"] != n):
            start = max(0, n - args.window)
            wins = sum(correct[start:n])
            rolling.append(
                {
                    "start": start,
                    "end": n,
                    "correct": wins,
                    "accuracy": round(wins / (n - start), 4),
                }
            )

    category_accuracy = {
        cat: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(stats["correct"] / stats["total"], 4),
        }
        for cat, stats in sorted(category_stats.items())
    }

    result = {
        "total_predictions": len(preds),
        "gold_rows": len(load_jsonl(resolve(args.input))),
        "evaluated": n,
        "correct": total_correct,
        "accuracy": round(total_correct / n, 4) if n else 0.0,
        "prefix": prefix,
        "rolling_window": args.window,
        "rolling": rolling,
        "category_accuracy": category_accuracy,
    }

    print(
        f"LongQA partial accuracy: {result['accuracy']:.4f} "
        f"({result['correct']}/{result['evaluated']})"
    )
    for item in prefix:
        print(f"  prefix {item['n']}: {item['accuracy']:.4f} ({item['correct']}/{item['n']})")
    if rolling:
        last = rolling[-1]
        denom = last["end"] - last["start"]
        print(
            f"  last window {last['start']}:{last['end']}: "
            f"{last['accuracy']:.4f} ({last['correct']}/{denom})"
        )

    if args.output:
        output = resolve_output(args.output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Analysis written to {output}")


if __name__ == "__main__":
    main()
