#!/usr/bin/env python3
"""Audit the contribution of Candidate C and completed replacement candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--current-c", required=True)
    parser.add_argument("--final-predictions", default=None)
    parser.add_argument(
        "--alternative",
        action="append",
        type=named_path,
        default=[],
        help="Completed replacement candidate as NAME=PATH.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def index(path: str) -> dict[str, dict[str, Any]]:
    return {sample_key(row): row for row in load_jsonl(path)}


def answer(row: dict[str, Any]) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def majority(answers: list[str]) -> str:
    counts = Counter(answers)
    top = max(counts.values())
    tied = {value for value, count in counts.items() if count == top}
    return next(value for value in answers if value in tied)


def evaluate_candidate(
    annotations: list[dict[str, Any]],
    primary: dict[str, dict[str, Any]],
    secondary: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    for row in annotations:
        key = sample_key(row)
        truth = answer(row)
        a = answer(primary[key])
        b = answer(secondary[key])
        c = answer(candidate[key])
        base = majority([a, b])
        combined = majority([a, b, c])
        counts["rows"] += 1
        counts["a_b_disagreements"] += int(a != b)
        counts["three_way_disagreements"] += int(len({a, b, c}) > 1)
        counts["two_candidate_correct"] += int(base == truth)
        counts["three_candidate_correct"] += int(combined == truth)
        counts["two_candidate_oracle"] += int(truth in {a, b})
        counts["three_candidate_oracle"] += int(truth in {a, b, c})
        counts["candidate_unique_correct"] += int(c == truth and truth not in {a, b})
        counts["candidate_only_wrong"] += int(c != truth and a == b == truth)
        counts["majority_fixes"] += int(base != truth and combined == truth)
        counts["majority_regressions"] += int(base == truth and combined != truth)
    rows = counts["rows"]
    result = dict(counts)
    for field in (
        "two_candidate_correct",
        "three_candidate_correct",
        "two_candidate_oracle",
        "three_candidate_oracle",
    ):
        result[field + "_accuracy"] = round(counts[field] / max(rows, 1), 6)
    return result


def main() -> None:
    args = parse_args()
    annotations = apply_subset(load_jsonl(args.annotations), args.subset_file)
    primary = index(args.primary)
    secondary = index(args.secondary)
    candidates = {"current_c": index(args.current_c)}
    candidates.update({name: index(path) for name, path in args.alternative})
    required = {sample_key(row) for row in annotations}
    for name, predictions in {
        "primary": primary,
        "secondary": secondary,
        **candidates,
    }.items():
        missing = required - set(predictions)
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} selected rows")

    result: dict[str, Any] = {
        "rows": len(annotations),
        "candidate_results": {
            name: evaluate_candidate(annotations, primary, secondary, predictions)
            for name, predictions in candidates.items()
        },
    }
    if args.final_predictions:
        final = index(args.final_predictions)
        missing = required - set(final)
        if missing:
            raise RuntimeError(f"final predictions are missing {len(missing)} rows")
        selected_c = 0
        selected_c_unique_correct = 0
        final_correct = 0
        for row in annotations:
            key = sample_key(row)
            truth = answer(row)
            a = answer(primary[key])
            b = answer(secondary[key])
            c = answer(candidates["current_c"][key])
            chosen = answer(final[key])
            chose_only_c = chosen == c and chosen not in {a, b}
            selected_c += int(chose_only_c)
            selected_c_unique_correct += int(chose_only_c and chosen == truth)
            final_correct += int(chosen == truth)
        result["current_final"] = {
            "correct": final_correct,
            "accuracy": round(final_correct / max(len(annotations), 1), 6),
            "selected_answer_unique_to_current_c": selected_c,
            "selected_unique_c_answer_correct": selected_c_unique_correct,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
