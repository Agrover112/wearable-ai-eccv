#!/usr/bin/env python3
"""Evaluate a candidate on a LongQA rescue/guard diagnostic subset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import load_jsonl, normalize_answer, sample_key


def _answers(path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in load_jsonl(path):
        key = sample_key(row)
        if key in result:
            raise RuntimeError(f"duplicate prediction key in {path}: {key}")
        result[key] = normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    subset = json.loads(Path(args.subset).read_text())
    predictions = _answers(args.predictions)
    rows = subset["samples"]
    missing = [row["key"] for row in rows if row["key"] not in predictions]
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"predictions are missing {len(missing)}/{len(rows)} subset rows; "
            "pass --allow-partial only for throughput diagnostics"
        )

    evaluated = [row for row in rows if row["key"] in predictions]
    rescue = [row for row in evaluated if row["selection_role"] == "rescue"]
    guard = [row for row in evaluated if row["selection_role"] == "guard"]
    correct = {
        row["key"]: predictions[row["key"]] == normalize_answer(row["answer"])
        for row in evaluated
    }
    rescue_fixes = sum(correct[row["key"]] for row in rescue)
    guard_regressions = sum(not correct[row["key"]] for row in guard)

    by_bucket: dict[str, dict[str, int]] = {}
    for bucket in sorted({str(row.get("failure_bucket")) for row in rescue}):
        bucket_rows = [row for row in rescue if str(row.get("failure_bucket")) == bucket]
        by_bucket[bucket] = {
            "rows": len(bucket_rows),
            "fixed": sum(correct[row["key"]] for row in bucket_rows),
        }

    result = {
        "subset": str(Path(args.subset).resolve()),
        "predictions": str(Path(args.predictions).resolve()),
        "rows_requested": len(rows),
        "rows_evaluated": len(evaluated),
        "rows_missing": len(missing),
        "rescue_rows_evaluated": len(rescue),
        "rescue_fixes": rescue_fixes,
        "guard_rows_evaluated": len(guard),
        "guard_regressions": guard_regressions,
        "net_gain": rescue_fixes - guard_regressions,
        "diagnostic_accuracy": (
            sum(correct.values()) / len(evaluated) if evaluated else None
        ),
        "fixes_by_failure_bucket": by_bucket,
        "errors_by_operator": dict(
            Counter(row["operator"] for row in evaluated if not correct[row["key"]])
        ),
        "errors_by_category": dict(
            Counter(row["category"] for row in evaluated if not correct[row["key"]])
        ),
        "notes": [
            "Diagnostic accuracy is not a validation estimate because selection used labels.",
            "A useful candidate must repair rescue rows without regressing guard rows.",
            "Freeze the method before confirmation on an untouched balanced fold.",
        ],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
