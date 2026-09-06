#!/usr/bin/env python3
"""Decompose rotation-pivot errors into candidate recall and selection failures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import (
    classify_question_types,
    has_temporal_cue,
    load_jsonl,
    normalize_answer,
    primary_question_type,
    sample_key,
)
from run_generate_longqa_proofpack import compile_temporal_program


def parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--rotation-predictions", required=True)
    parser.add_argument(
        "--candidate-run",
        action="append",
        type=parse_named_path,
        required=True,
        help="One of the three verifier candidates as NAME=PATH.",
    )
    parser.add_argument(
        "--extra-run",
        action="append",
        type=parse_named_path,
        default=[],
        help="Additional model prediction file as NAME=PATH.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def answer(row: dict[str, Any]) -> str:
    return normalize_answer(
        row.get("mcq_answer_parsed") or row.get("mcq_answer")
    )


def indexed(path: str) -> dict[str, dict[str, Any]]:
    return {sample_key(row): row for row in load_jsonl(path)}


def group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, dict]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result = {}
    for value, items in grouped.items():
        total = len(items)
        correct = sum(item["rotation_correct"] for item in items)
        candidate_recall = sum(item["candidate_recall_three"] for item in items)
        available_errors = sum(
            item["error_bucket"] == "candidate_available_not_selected"
            for item in items
        )
        result[value] = {
            "rows": total,
            "correct": correct,
            "accuracy": round(correct / total, 6),
            "errors": total - correct,
            "candidate_recall_three": candidate_recall,
            "candidate_recall_rate": round(candidate_recall / total, 6),
            "candidate_available_selection_errors": available_errors,
        }
    return dict(
        sorted(
            result.items(),
            key=lambda item: (-item[1]["errors"], item[0]),
        )
    )


def main() -> None:
    args = parse_args()
    annotations = load_jsonl(args.annotations)
    rotation = indexed(args.rotation_predictions)
    candidate_runs = {
        name: indexed(path) for name, path in args.candidate_run
    }
    if len(candidate_runs) != 3:
        raise RuntimeError(
            f"Expected exactly three candidate runs, got {len(candidate_runs)}"
        )
    extra_runs = {name: indexed(path) for name, path in args.extra_run}
    all_runs = {**candidate_runs, **extra_runs}
    rows = []
    for annotation in annotations:
        key = sample_key(annotation)
        gold = normalize_answer(
            annotation.get("mcq_answer") or annotation.get("answer")
        )
        selected = answer(rotation[key])
        candidate_answers = {
            name: answer(predictions[key])
            for name, predictions in candidate_runs.items()
        }
        all_answers = {
            name: answer(predictions[key])
            for name, predictions in all_runs.items()
        }
        candidate_set = set(candidate_answers.values())
        all_set = set(all_answers.values())
        is_correct = selected == gold
        candidate_recall = gold in candidate_set
        all_recall = gold in all_set
        if is_correct:
            bucket = "correct"
        elif candidate_recall:
            bucket = "candidate_available_not_selected"
        elif all_recall:
            bucket = "missing_three_available_extra"
        else:
            bucket = "missing_all_runs"
        q35_pivot = candidate_answers.get("q35_pivot")
        q35_uniform = candidate_answers.get("q35_uniform")
        if q35_pivot == gold and q35_uniform == gold:
            q35_view_pattern = "both_correct"
        elif q35_pivot == gold:
            q35_view_pattern = "pivot_only_correct"
        elif q35_uniform == gold:
            q35_view_pattern = "uniform_only_correct"
        else:
            q35_view_pattern = "neither_correct"
        question = str(annotation.get("question", ""))
        rows.append(
            {
                "sample_key": key,
                "video_path": annotation.get("video_path"),
                "question": question,
                "category": annotation.get("category", ""),
                "gold_answer": gold,
                "rotation_answer": selected,
                "rotation_correct": is_correct,
                "candidate_answers": candidate_answers,
                "extra_answers": {
                    name: all_answers[name] for name in extra_runs
                },
                "candidate_recall_three": candidate_recall,
                "candidate_recall_all_runs": all_recall,
                "error_bucket": bucket,
                "q35_view_pattern": q35_view_pattern,
                "primary_question_type": primary_question_type(question),
                "question_type_tags": classify_question_types(question),
                "temporal_operator": compile_temporal_program(question).operator,
                "has_temporal_cue": has_temporal_cue(question),
            }
        )

    errors = [row for row in rows if not row["rotation_correct"]]
    buckets = Counter(row["error_bucket"] for row in errors)
    q35_patterns_all = Counter(row["q35_view_pattern"] for row in rows)
    q35_patterns_errors = Counter(row["q35_view_pattern"] for row in errors)
    triple_oracle = sum(row["candidate_recall_three"] for row in rows)
    all_oracle = sum(row["candidate_recall_all_runs"] for row in rows)
    summary = {
        "rows": len(rows),
        "correct": len(rows) - len(errors),
        "accuracy": round((len(rows) - len(errors)) / len(rows), 6),
        "errors": len(errors),
        "error_buckets": dict(buckets),
        "three_candidate_oracle_correct": triple_oracle,
        "three_candidate_oracle_accuracy": round(triple_oracle / len(rows), 6),
        "all_run_oracle_correct": all_oracle,
        "all_run_oracle_accuracy": round(all_oracle / len(rows), 6),
        "candidate_available_rows": triple_oracle,
        "selection_correct_when_candidate_available": len(rows) - len(errors),
        "selection_accuracy_when_candidate_available": round(
            (len(rows) - len(errors)) / max(triple_oracle, 1), 6
        ),
        "q35_view_patterns_all": dict(q35_patterns_all),
        "q35_view_patterns_errors": dict(q35_patterns_errors),
        "by_category": group_summary(rows, "category"),
        "by_primary_question_type": group_summary(
            rows, "primary_question_type"
        ),
        "by_temporal_operator": group_summary(rows, "temporal_operator"),
        "by_gold_answer": group_summary(rows, "gold_answer"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
