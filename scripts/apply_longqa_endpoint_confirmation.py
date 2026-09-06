#!/usr/bin/env python3
"""Apply a frozen reranker-confirmed endpoint restoration policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key
from run_generate_longqa_grounded import _run_eval


def _index(path: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = sample_key(row)
        if key in result:
            raise RuntimeError(f"duplicate prediction key in {path}: {key}")
        result[key] = row
    return result


def _answer(row: dict[str, Any]) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--parent-subset", required=True)
    parser.add_argument("--disagreement-subset", required=True)
    parser.add_argument("--fusion-predictions", required=True)
    parser.add_argument("--endpoint-predictions", required=True)
    parser.add_argument("--reranker-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    parent_rows = apply_subset(annotations, args.parent_subset)
    disagreement_rows = apply_subset(annotations, args.disagreement_subset)
    parent_keys = [sample_key(row) for row in parent_rows]
    expected_disagreements = {sample_key(row) for row in disagreement_rows}

    fusion = _index(args.fusion_predictions)
    endpoint = _index(args.endpoint_predictions)
    reranker = _index(args.reranker_predictions)
    missing = set(parent_keys) - set(fusion) | (set(parent_keys) - set(endpoint))
    if missing:
        raise RuntimeError(f"base predictions are missing {len(missing)} parent rows")

    derived_disagreements = {
        key for key in parent_keys if _answer(fusion[key]) != _answer(endpoint[key])
    }
    if derived_disagreements != expected_disagreements:
        raise RuntimeError(
            "disagreement subset does not match current endpoint/fusion predictions: "
            f"expected={len(expected_disagreements)} derived={len(derived_disagreements)}"
        )
    if set(reranker) != expected_disagreements:
        raise RuntimeError(
            "reranker prediction keys do not exactly match the frozen disagreement set: "
            f"reranker={len(reranker)} expected={len(expected_disagreements)}"
        )

    output_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for key in parent_keys:
        result = dict(fusion[key])
        fusion_answer = _answer(fusion[key])
        endpoint_answer = _answer(endpoint[key])
        reranker_answer = _answer(reranker[key]) if key in reranker else None
        chosen = fusion_answer
        route = "fusion_default"
        if (
            key in expected_disagreements
            and reranker_answer == endpoint_answer
            and endpoint_answer != fusion_answer
        ):
            chosen = endpoint_answer
            route = "reranker_confirmed_endpoint"
            changes.append(
                {
                    "sample_key": key,
                    "fusion": fusion_answer,
                    "endpoint": endpoint_answer,
                    "reranker": reranker_answer,
                    "chosen": chosen,
                }
            )
        result["mcq_answer"] = chosen
        result["mcq_answer_raw"] = chosen
        result["mcq_answer_parsed"] = chosen
        result["prompt_variant"] = "reranker_confirmed_endpoint_restoration"
        result["endpoint_confirmation_route"] = route
        result["endpoint_confirmation_reranker"] = reranker_answer
        output_rows.append(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")

    truth = {sample_key(row): normalize_answer(row.get("mcq_answer")) for row in parent_rows}
    baseline_correct = sum(_answer(fusion[key]) == truth[key] for key in parent_keys)
    policy_correct = sum(_answer(row) == truth[sample_key(row)] for row in output_rows)
    fixes = sum(
        change["chosen"] == truth[change["sample_key"]]
        and change["fusion"] != truth[change["sample_key"]]
        for change in changes
    )
    regressions = sum(
        change["chosen"] != truth[change["sample_key"]]
        and change["fusion"] == truth[change["sample_key"]]
        for change in changes
    )
    audit = {
        "policy": (
            "Use fusion by default. On endpoint/fusion disagreements only, restore "
            "endpoint when the balanced Qwen-reranked branch agrees with endpoint."
        ),
        "parent_rows": len(parent_keys),
        "disagreement_rows": len(expected_disagreements),
        "changes": len(changes),
        "fixes": fixes,
        "regressions": regressions,
        "baseline_correct": baseline_correct,
        "policy_correct": policy_correct,
        "baseline_accuracy": baseline_correct / len(parent_keys),
        "policy_accuracy": policy_correct / len(parent_keys),
        "change_records": changes,
        "notes": [
            "The routing rule does not read gold labels.",
            "Fixes and regressions are computed only after writing frozen predictions.",
        ],
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    _run_eval(args.annotations, args.output, args.eval_output)


if __name__ == "__main__":
    main()
