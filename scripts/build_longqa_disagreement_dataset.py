#!/usr/bin/env python3
"""Build a row-level audit dataset from completed LongQA prediction files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import load_jsonl, normalize_answer, sample_key
from run_generate_longqa_proofpack import compile_temporal_program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a consolidated LongQA model-disagreement dataset."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Named prediction input, for example q35_uniform=predictions.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--only-disagreements",
        action="store_true",
        help="Omit rows on which every supplied system returns the same answer.",
    )
    return parser.parse_args()


def load_named_predictions(
    items: list[str],
) -> tuple[list[str], dict[str, dict[str, dict]]]:
    names: list[str] = []
    predictions: dict[str, dict[str, dict]] = {}
    for item in items:
        name, path = item.split("=", 1)
        if name in predictions:
            raise ValueError(f"Duplicate prediction name: {name}")
        names.append(name)
        predictions[name] = {
            sample_key(row): row for row in load_jsonl(path)
        }
    return names, predictions


def main() -> None:
    args = parse_args()
    annotations = load_jsonl(args.annotations)
    names, predictions = load_named_predictions(args.pred)
    required = {sample_key(row) for row in annotations}
    for name in names:
        missing = required - set(predictions[name])
        if missing:
            raise RuntimeError(
                f"{name} is missing {len(missing)} annotation rows"
            )

    output_rows = []
    pattern_counts: Counter[str] = Counter()
    pattern_correct: Counter[str] = Counter()
    pattern_oracle: Counter[str] = Counter()
    system_correct: Counter[str] = Counter()
    for index, row in enumerate(annotations):
        key = sample_key(row)
        gold = normalize_answer(row.get("mcq_answer") or row.get("answer"))
        answers = {
            name: normalize_answer(predictions[name][key].get("mcq_answer"))
            for name in names
        }
        vote_counts = Counter(answers.values())
        if len(vote_counts) == 1:
            pattern = "unanimous"
        elif max(vote_counts.values()) > 1:
            pattern = "split_vote"
        else:
            pattern = "all_different"
        if args.only_disagreements and pattern == "unanimous":
            continue

        top_count = max(vote_counts.values())
        tied = {answer for answer, count in vote_counts.items() if count == top_count}
        majority = next(answer for answer in answers.values() if answer in tied)
        correctness = {
            name: answer == gold for name, answer in answers.items()
        }
        correct_systems = [name for name in names if correctness[name]]
        record = {
            "index": index,
            "sample_key": key,
            "video_path": row.get("video_path"),
            "question": row.get("question"),
            "mcq_options": row.get("mcq_options"),
            "category": row.get("category"),
            "temporal_operator": compile_temporal_program(
                str(row.get("question", ""))
            ).operator,
            "gold_answer": gold,
            "candidate_answers": answers,
            "candidate_correct": correctness,
            "correct_systems": correct_systems,
            "agreement_pattern": pattern,
            "vote_counts": dict(vote_counts),
            "majority_answer": majority,
            "majority_tied": len(tied) > 1,
            "majority_correct": majority == gold,
            "oracle_correct": bool(correct_systems),
        }
        output_rows.append(record)
        pattern_counts[pattern] += 1
        pattern_correct[pattern] += int(record["majority_correct"])
        pattern_oracle[pattern] += int(record["oracle_correct"])
        for name, correct in correctness.items():
            system_correct[name] += int(correct)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "systems": names,
        "rows": len(output_rows),
        "only_disagreements": args.only_disagreements,
        "system_correct": dict(system_correct),
        "pattern_counts": dict(pattern_counts),
        "majority_correct_by_pattern": dict(pattern_correct),
        "oracle_correct_by_pattern": dict(pattern_oracle),
    }
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    with open(args.summary_output, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
