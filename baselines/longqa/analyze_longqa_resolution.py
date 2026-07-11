#!/usr/bin/env python3
"""Compare LongQA prediction files by category and question type."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import (  # noqa: E402
    classify_question_types,
    load_jsonl,
    normalize_answer,
    primary_question_type,
    sample_key,
)


def parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must use non-empty LABEL=PATH")
    return label, path


def summarize_group(
    rows: list[dict[str, Any]],
    correctness: dict[str, dict[str, bool]],
    labels: list[str],
) -> dict[str, Any]:
    keys = [sample_key(row) for row in rows]
    accuracy = {
        label: round(sum(correctness[label][key] for key in keys) / len(keys), 4)
        if keys
        else 0.0
        for label in labels
    }
    best = max(accuracy.values(), default=0.0)
    return {
        "n": len(keys),
        "accuracy": accuracy,
        "best_runs": [label for label in labels if accuracy[label] == best],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LongQA image resolution runs.")
    parser.add_argument(
        "--annotations",
        default=str(
            REPO_ROOT
            / "data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
        ),
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("provide at least two --run LABEL=PATH values")

    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("run labels must be unique")
    golden = load_jsonl(args.annotations)
    golden_by_key = {sample_key(row): row for row in golden}
    predictions = {
        label: {sample_key(row): row for row in load_jsonl(path)}
        for label, path in args.run
    }
    common_keys = set(golden_by_key)
    for run_rows in predictions.values():
        common_keys &= set(run_rows)
    aligned = [row for row in golden if sample_key(row) in common_keys]
    correctness = {
        label: {
            key: normalize_answer(run_rows[key].get("mcq_answer_parsed") or run_rows[key].get("mcq_answer"))
            == normalize_answer(golden_by_key[key].get("mcq_answer"))
            for key in common_keys
        }
        for label, run_rows in predictions.items()
    }

    overlapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aligned:
        for tag in classify_question_types(row.get("question", "")):
            overlapping[tag].append(row)
        exclusive[primary_question_type(row.get("question", ""))].append(row)
        categories[str(row.get("category", ""))].append(row)

    result: dict[str, Any] = {
        "runs": {label: path for label, path in args.run},
        "common_samples": len(aligned),
        "overall": summarize_group(aligned, correctness, labels),
        "overlapping_question_types": {
            name: summarize_group(rows, correctness, labels)
            for name, rows in sorted(overlapping.items())
        },
        "exclusive_question_types": {
            name: summarize_group(rows, correctness, labels)
            for name, rows in sorted(exclusive.items())
        },
        "categories": {
            name: summarize_group(rows, correctness, labels)
            for name, rows in sorted(categories.items())
        },
        "pairwise": {},
    }
    for left_idx, left in enumerate(labels):
        for right in labels[left_idx + 1 :]:
            left_wins = sum(
                correctness[left][key] and not correctness[right][key]
                for key in common_keys
            )
            right_wins = sum(
                correctness[right][key] and not correctness[left][key]
                for key in common_keys
            )
            result["pairwise"][f"{left}_vs_{right}"] = {
                f"{left}_only_correct": left_wins,
                f"{right}_only_correct": right_wins,
                "net_for_right": right_wins - left_wins,
            }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(result, f, indent=2)

    csv_path = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    csv_rows = []
    for scope, groups in (
        ("overall", {"all": result["overall"]}),
        ("overlapping", result["overlapping_question_types"]),
        ("exclusive", result["exclusive_question_types"]),
        ("category", result["categories"]),
    ):
        for group, summary in groups.items():
            row = {"scope": scope, "group": group, "n": summary["n"]}
            row.update({f"accuracy_{label}": summary["accuracy"][label] for label in labels})
            row["best_runs"] = ",".join(summary["best_runs"])
            csv_rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Compared {len(aligned)} common samples across {', '.join(labels)}")
    for label in labels:
        print(f"  {label}: {result['overall']['accuracy'][label]:.4f}")
    for name, pair in result["pairwise"].items():
        print(f"  {name}: {pair}")
    print(f"JSON written to {output_json}")
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    main()
