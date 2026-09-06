#!/usr/bin/env python3
"""Evaluate complete LongQA prediction artifacts across fixed fold files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import load_jsonl, normalize_answer, sample_key


def _answers(path: str) -> dict[str, str]:
    return {
        sample_key(row): normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
        for row in load_jsonl(path)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.annotations)
    truth = {sample_key(row): normalize_answer(row.get("mcq_answer")) for row in rows}
    folds: list[set[str]] = []
    for path in args.fold:
        document = json.loads(Path(path).read_text())
        folds.append({str(row.get("key") or sample_key(row)) for row in document["samples"]})
    if set().union(*folds) != set(truth) or sum(map(len, folds)) != len(truth):
        raise RuntimeError("folds must be a disjoint exact partition of annotations")

    report = {"folds": args.fold, "candidates": {}}
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        predictions = _answers(path)
        if set(truth) - set(predictions):
            raise RuntimeError(f"candidate {name} is incomplete")
        scores = [sum(predictions[key] == truth[key] for key in fold) for fold in folds]
        report["candidates"][name] = {
            "fold_correct": scores,
            "fold_total": [len(fold) for fold in folds],
            "fold_accuracy": [round(score / len(fold), 6) for score, fold in zip(scores, folds)],
            "full_correct": sum(scores),
            "full_total": len(truth),
            "full_accuracy": round(sum(scores) / len(truth), 6),
            "fold_range": max(scores) - min(scores),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
