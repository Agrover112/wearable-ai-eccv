#!/usr/bin/env python3
"""Create stable LongQA evaluation folds with balanced difficulty and semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import load_jsonl, normalize_answer, sample_key
from run_generate_longqa_proofpack import compile_temporal_program_v2


def _index_predictions(path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in load_jsonl(path):
        key = sample_key(row)
        if key in result:
            raise RuntimeError(f"duplicate prediction key in {path}: {key}")
        result[key] = normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
    return result


def _video_durations(rows: list[dict[str, Any]], video_folder: str | None) -> list[float]:
    if not video_folder:
        return [0.0] * len(rows)
    import cv2

    durations: list[float] = []
    for index, row in enumerate(rows, 1):
        path = str(Path(video_folder) / str(row["video_path"]))
        capture = cv2.VideoCapture(path)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if frames <= 0 or fps <= 0:
            raise RuntimeError(f"could not read video duration: {path}")
        durations.append(frames / fps)
        if index % 100 == 0:
            print(f"Read duration metadata: {index}/{len(rows)}")
    return durations


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, len(ordered) * i // bins)] for i in range(1, bins)]


def _bin(value: float, edges: list[float]) -> int:
    return sum(value >= edge for edge in edges)


def _labels(item: dict[str, Any]) -> list[tuple[str, float]]:
    category = str(item["category"])
    operator = str(item["operator"])
    return [
        (f"category={category}", 1.0),
        (f"operator={operator}", 1.4),
        (f"answer={item['answer']}", 1.0),
        (f"duration={item['duration_bin']}", 0.8),
        (f"endpoint_correct={item['endpoint_correct']}", 2.0),
        (f"endpoint_option_agree={item['endpoint_option_agree']}", 1.5),
        (f"category_operator={category}|{operator}", 0.35),
    ]


def _assignment(items: list[dict[str, Any]], folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    totals = Counter(label for item in items for label, _ in _labels(item))
    shuffled = list(range(len(items)))
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda index: sum(1.0 / totals[label] for label, _ in _labels(items[index])),
        reverse=True,
    )
    target_size = math.ceil(len(items) / folds)
    fold_rows: list[list[int]] = [[] for _ in range(folds)]
    counts = [Counter() for _ in range(folds)]
    for index in shuffled:
        candidates = [fold for fold in range(folds) if len(fold_rows[fold]) < target_size]
        rng.shuffle(candidates)

        def cost(fold: int) -> float:
            value = 4.0 * len(fold_rows[fold]) / target_size
            for label, weight in _labels(items[index]):
                target = totals[label] / folds
                before = counts[fold][label] - target
                after = counts[fold][label] + 1 - target
                value += weight * (after * after - before * before) / max(target, 1.0)
            return value

        chosen = min(candidates, key=lambda fold: (cost(fold), len(fold_rows[fold]), fold))
        fold_rows[chosen].append(index)
        counts[chosen].update(label for label, _ in _labels(items[index]))
    return fold_rows


def _quality(items: list[dict[str, Any]], fold_rows: list[list[int]]) -> float:
    folds = len(fold_rows)
    totals = Counter(label for item in items for label, _ in _labels(item))
    score = 0.0
    for indices in fold_rows:
        counts = Counter(label for index in indices for label, _ in _labels(items[index]))
        for label, total in totals.items():
            target = total / folds
            score += ((counts[label] - target) / max(target, 1.0)) ** 2
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--endpoint-predictions", required=True)
    parser.add_argument("--option-predictions", required=True)
    parser.add_argument("--video-folder")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--output-pattern", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    if "{fold}" not in args.output_pattern:
        raise ValueError("--output-pattern must contain {fold}")

    rows = load_jsonl(args.annotations)
    if len(rows) % args.folds:
        raise ValueError("row count must be divisible by number of folds")
    truth = {sample_key(row): normalize_answer(row.get("mcq_answer")) for row in rows}
    endpoint = _index_predictions(args.endpoint_predictions)
    option = _index_predictions(args.option_predictions)
    missing = set(truth) - set(endpoint) | (set(truth) - set(option))
    if missing:
        raise RuntimeError(f"predictions are missing {len(missing)} annotation rows")

    durations = _video_durations(rows, args.video_folder)
    duration_edges = _quantile_edges(durations, 4) if args.video_folder else []
    items: list[dict[str, Any]] = []
    for index, (row, duration) in enumerate(zip(rows, durations)):
        key = sample_key(row)
        items.append(
            {
                "index": index,
                "key": key,
                "video_path": row.get("video_path", ""),
                "question": row.get("question", ""),
                "category": row.get("category", ""),
                "answer": truth[key],
                "operator": compile_temporal_program_v2(row.get("question", "")).operator,
                "duration_seconds": round(duration, 3),
                "duration_bin": _bin(duration, duration_edges),
                "endpoint_correct": endpoint[key] == truth[key],
                "endpoint_option_agree": endpoint[key] == option[key],
            }
        )

    candidates = [
        _assignment(items, args.folds, args.seed + restart)
        for restart in range(args.restarts)
    ]
    fold_rows = min(candidates, key=lambda assignment: _quality(items, assignment))
    all_indices: list[int] = []
    summary: dict[str, Any] = {
        "source": str(Path(args.annotations).resolve()),
        "seed": args.seed,
        "folds": args.folds,
        "rows_per_fold": len(rows) // args.folds,
        "duration_edges_seconds": duration_edges,
        "balance_features": [
            "category",
            "temporal_operator_v2",
            "gold_answer_position",
            "video_duration_quartile",
            "frozen_endpoint_correctness",
            "endpoint_option_agreement",
        ],
        "fold_summaries": [],
    }
    for fold, indices in enumerate(fold_rows):
        indices = sorted(indices)
        all_indices.extend(indices)
        samples = [items[index] for index in indices]
        output = Path(args.output_pattern.format(fold=fold))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "source": str(Path(args.annotations).resolve()),
                    "n": len(samples),
                    "seed": args.seed,
                    "fold": fold,
                    "num_folds": args.folds,
                    "samples": samples,
                },
                indent=2,
            )
            + "\n"
        )
        fold_summary = {
            "fold": fold,
            "rows": len(samples),
            "endpoint_correct": sum(item["endpoint_correct"] for item in samples),
            "endpoint_option_disagreements": sum(
                not item["endpoint_option_agree"] for item in samples
            ),
            "operators": dict(Counter(item["operator"] for item in samples)),
            "duration_bins": dict(Counter(str(item["duration_bin"]) for item in samples)),
        }
        summary["fold_summaries"].append(fold_summary)
        print(f"Fold {fold}: {fold_summary} -> {output}")
    if sorted(all_indices) != list(range(len(rows))):
        raise RuntimeError("folds do not form an exact partition")
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
