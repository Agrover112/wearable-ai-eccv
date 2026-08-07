#!/usr/bin/env python3
"""Apply a frozen temporal-operator route between two LongQA predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key
from run_generate_longqa_proofpack import compile_temporal_program_v2


def index(path: str) -> dict[str, dict]:
    return {sample_key(row): row for row in load_jsonl(path)}


def answer(row: dict) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file")
    parser.add_argument("--primary", required=True)
    parser.add_argument("--global-candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    primary = index(args.primary)
    global_candidate = index(args.global_candidate)
    required = {sample_key(row) for row in rows}
    for label, values in (("primary", primary), ("global candidate", global_candidate)):
        missing = required - set(values)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} rows")

    metrics = Counter()
    by_operator: dict[str, Counter] = {}
    output_rows = []
    for row in rows:
        key = sample_key(row)
        operator = compile_temporal_program_v2(row.get("question", "")).operator
        selected_source = "global_candidate" if operator == "GLOBAL" else "primary"
        source = global_candidate[key] if operator == "GLOBAL" else primary[key]
        selected = answer(source)
        truth = normalize_answer(row.get("mcq_answer") or row.get("answer"))
        primary_answer = answer(primary[key])
        correct = selected == truth
        metrics["correct"] += int(correct)
        metrics["primary_correct"] += int(primary_answer == truth)
        metrics["global_candidate_rows"] += int(operator == "GLOBAL")
        metrics["changed"] += int(selected != primary_answer)
        operator_metrics = by_operator.setdefault(operator, Counter())
        operator_metrics["rows"] += 1
        operator_metrics["correct"] += int(correct)
        prediction = dict(source)
        prediction.update(
            {
                "operator_route": "endpoint27_on_GLOBAL_primary_otherwise_v1",
                "operator_route_operator": operator,
                "operator_route_source": selected_source,
                "operator_route_primary": primary_answer,
                "mcq_answer": selected,
                "mcq_answer_raw": selected,
                "mcq_answer_parsed": selected,
            }
        )
        output_rows.append(prediction)

    summary = dict(metrics)
    summary.update(
        {
            "rows": len(rows),
            "accuracy": metrics["correct"] / len(rows),
            "primary_accuracy": metrics["primary_correct"] / len(rows),
            "policy": "endpoint27_on_GLOBAL_primary_otherwise_v1",
            "by_operator": {
                operator: dict(values) for operator, values in sorted(by_operator.items())
            },
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
