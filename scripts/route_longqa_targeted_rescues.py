#!/usr/bin/env python3
"""Apply fixed, label-free consensus policies to targeted LongQA specialists."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))


POLICIES = (
    "ocr_clause_confirmed",
    "occurrence_clause_confirmed",
    "any_family_clause_confirmed",
    "two_evidence_families",
)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> str:
    return f"{row.get('video_path', '')}||{row.get('question', '')}"


def answer(row: dict[str, Any]) -> str:
    value = str(row.get("mcq_answer_parsed") or row.get("mcq_answer") or "").strip().upper()
    return value if value in {"A", "B", "C", "D"} else ""


def index(path: str) -> dict[str, dict[str, Any]]:
    return {key(row): row for row in load_jsonl(path)}


def clone_prediction(source: dict[str, Any], selected: str, policy: str) -> dict[str, Any]:
    output = dict(source)
    output["mcq_answer"] = selected
    output["mcq_answer_raw"] = selected
    output["mcq_answer_parsed"] = selected
    output["prompt_variant"] = f"targeted_rescue_{policy}"
    output["targeted_rescue_policy"] = policy
    output["targeted_rescue_fallback"] = answer(source)
    output["targeted_rescue_changed"] = selected != answer(source)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--ocr-crops", required=True)
    parser.add_argument("--ocr-ledger", required=True)
    parser.add_argument("--occurrence", required=True)
    parser.add_argument("--clause-strict", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    from longqa_utils import apply_subset, compute_diagnostics

    args = parse_args()
    fallback_rows = apply_subset(load_jsonl(args.fallback), args.subset_file)
    annotations = apply_subset(load_jsonl(args.annotations), args.subset_file)
    sources = {
        "ocr_crops": index(args.ocr_crops),
        "ocr_ledger": index(args.ocr_ledger),
        "occurrence": index(args.occurrence),
        "clause": index(args.clause_strict),
    }
    required = {key(row) for row in fallback_rows}
    for label, records in sources.items():
        missing = required - set(records)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} router rows")

    predictions = {policy: [] for policy in POLICIES}
    audit = []
    for fallback in fallback_rows:
        row_key = key(fallback)
        base = answer(fallback)
        crop = sources["ocr_crops"][row_key]
        ledger = sources["ocr_ledger"][row_key]
        occurrence = sources["occurrence"][row_key]
        clause = sources["clause"][row_key]
        ocr_applied = bool(crop.get("specialist_applied")) and bool(
            ledger.get("specialist_applied")
        )
        ocr_answer = (
            answer(crop)
            if ocr_applied and answer(crop) == answer(ledger)
            else ""
        )
        occurrence_answer = (
            answer(occurrence) if occurrence.get("specialist_applied") else ""
        )
        clause_answer = (
            answer(clause) if clause.get("clause_support_applied") else ""
        )

        selected = {policy: base for policy in POLICIES}
        if ocr_answer and ocr_answer != base and clause_answer == ocr_answer:
            selected["ocr_clause_confirmed"] = ocr_answer
            selected["any_family_clause_confirmed"] = ocr_answer
        if (
            occurrence_answer
            and occurrence_answer != base
            and clause_answer == occurrence_answer
        ):
            selected["occurrence_clause_confirmed"] = occurrence_answer
            if selected["any_family_clause_confirmed"] == base:
                selected["any_family_clause_confirmed"] = occurrence_answer
        if (
            ocr_answer
            and occurrence_answer
            and ocr_answer == occurrence_answer
            and ocr_answer != base
        ):
            selected["two_evidence_families"] = ocr_answer

        for policy in POLICIES:
            predictions[policy].append(clone_prediction(fallback, selected[policy], policy))
        audit.append(
            {
                "sample_key": row_key,
                "fallback": base,
                "ocr_family": ocr_answer,
                "occurrence": occurrence_answer,
                "clause_strict": clause_answer,
                "selected": selected,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "audit.jsonl").open("w") as handle:
        for row in audit:
            handle.write(json.dumps(row) + "\n")
    summary = {}
    for policy, rows in predictions.items():
        with (output_dir / f"predictions_{policy}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        diagnostics = compute_diagnostics(
            annotations, rows, run_id=f"targeted_rescue_{policy}"
        )
        (output_dir / f"results_{policy}.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
        summary[policy] = {
            "correct": diagnostics["correct"],
            "total": diagnostics["total"],
            "accuracy": diagnostics["accuracy_raw"],
            "changes": sum(row["targeted_rescue_changed"] for row in rows),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
