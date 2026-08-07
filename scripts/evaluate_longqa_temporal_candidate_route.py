#!/usr/bin/env python3
"""Route selected temporal operators to a secondary LongQA candidate."""

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
    rows = load_jsonl(path)
    indexed = {sample_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"Duplicate prediction keys in {path}")
    return indexed


def answer(row: dict) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file")
    parser.add_argument("--default-candidate", required=True)
    parser.add_argument("--temporal-candidate", required=True)
    parser.add_argument(
        "--operators", nargs="+", default=["BEFORE", "MULTI_TIME"]
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    default = index(args.default_candidate)
    temporal = index(args.temporal_candidate)
    required = {sample_key(row) for row in rows}
    for label, values in (("default", default), ("temporal", temporal)):
        missing = required - set(values)
        if missing:
            raise RuntimeError(f"{label} candidate is missing {len(missing)} rows")

    routed_operators = set(args.operators)
    metrics = Counter()
    output_rows = []
    by_operator: dict[str, Counter] = {}
    for row in rows:
        key = sample_key(row)
        operator = compile_temporal_program_v2(row["question"]).operator
        default_answer = answer(default[key])
        temporal_answer = answer(temporal[key])
        use_temporal = (
            operator in routed_operators and temporal_answer != default_answer
        )
        source = temporal[key] if use_temporal else default[key]
        selected = temporal_answer if use_temporal else default_answer
        truth = normalize_answer(row.get("mcq_answer") or row.get("answer"))
        metrics["correct"] += int(selected == truth)
        metrics["default_correct"] += int(default_answer == truth)
        metrics["changed"] += int(use_temporal)
        metrics["fixes"] += int(use_temporal and temporal_answer == truth)
        metrics["regressions"] += int(use_temporal and default_answer == truth)
        operator_metrics = by_operator.setdefault(operator, Counter())
        operator_metrics["rows"] += 1
        operator_metrics["correct"] += int(selected == truth)
        operator_metrics["changed"] += int(use_temporal)
        prediction = dict(source)
        prediction.update(
            {
                "temporal_route": "option_quota_on_before_multitime_v1",
                "temporal_route_operator": operator,
                "temporal_route_source": (
                    "option_quota27" if use_temporal else "endpoint27"
                ),
                "temporal_route_default_answer": default_answer,
                "temporal_route_candidate_answer": temporal_answer,
                "mcq_answer": selected,
                "mcq_answer_raw": selected,
                "mcq_answer_parsed": selected,
            }
        )
        output_rows.append(prediction)

    summary = {
        **dict(metrics),
        "rows": len(rows),
        "accuracy": metrics["correct"] / len(rows),
        "default_accuracy": metrics["default_correct"] / len(rows),
        "operators": sorted(routed_operators),
        "policy": "option_quota_on_before_multitime_v1",
        "uses_labels": False,
        "by_operator": {
            operator: dict(values)
            for operator, values in sorted(by_operator.items())
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
