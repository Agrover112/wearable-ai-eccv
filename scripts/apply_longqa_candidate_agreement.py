#!/usr/bin/env python3
"""Apply frozen agreement routers to independently generated LongQA candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key
from run_generate_longqa_grounded import _run_eval


def _index(path: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = sample_key(row)
        if key in indexed:
            raise RuntimeError(f"duplicate prediction key in {path}: {key}")
        indexed[key] = row
    return indexed


def _answer(row: dict[str, Any]) -> str:
    answer = normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))
    if answer not in {"A", "B", "C", "D"}:
        raise RuntimeError(f"invalid candidate answer {answer!r} for {sample_key(row)}")
    return answer


def _candidate_answer(row: dict[str, Any]) -> str | None:
    answer = normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))
    return answer if answer in {"A", "B", "C", "D"} else None


def _write_predictions(
    path: Path,
    parent_keys: list[str],
    fusion: dict[str, dict[str, Any]],
    endpoint: dict[str, dict[str, Any]],
    candidate_a: dict[str, dict[str, Any]],
    candidate_b: dict[str, dict[str, Any]],
    target_keys: set[str],
    policy_name: str,
    predicate: Callable[[str, str, str, str], bool],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for key in parent_keys:
        result = dict(fusion[key])
        fusion_answer = _answer(fusion[key])
        endpoint_answer = _answer(endpoint[key])
        answer_a = _candidate_answer(candidate_a[key]) if key in target_keys else None
        answer_b = _candidate_answer(candidate_b[key]) if key in target_keys else None
        chosen = fusion_answer
        route = "fusion_default"
        if (
            key in target_keys
            and answer_a is not None
            and answer_b is not None
            and predicate(fusion_answer, endpoint_answer, answer_a, answer_b)
        ):
            chosen = answer_a
            route = policy_name
            changes.append(
                {
                    "sample_key": key,
                    "fusion": fusion_answer,
                    "endpoint": endpoint_answer,
                    "candidate_a": answer_a,
                    "candidate_b": answer_b,
                    "chosen": chosen,
                }
            )
        result["mcq_answer"] = chosen
        result["mcq_answer_raw"] = chosen
        result["mcq_answer_parsed"] = chosen
        result["prompt_variant"] = policy_name
        result["candidate_agreement_route"] = route
        result["candidate_agreement_a"] = answer_a
        result["candidate_agreement_b"] = answer_b
        output_rows.append(result)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    return changes


def _audit(
    output: Path,
    policy: str,
    parent_rows: list[dict[str, Any]],
    fusion: dict[str, dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> dict[str, Any]:
    truth = {
        sample_key(row): normalize_answer(row.get("mcq_answer")) for row in parent_rows
    }
    predictions = {sample_key(row): row for row in prediction_rows}
    parent_keys = list(truth)
    baseline_correct = sum(_answer(fusion[key]) == truth[key] for key in parent_keys)
    policy_correct = sum(_answer(predictions[key]) == truth[key] for key in parent_keys)
    fixes = sum(
        change["fusion"] != truth[change["sample_key"]]
        and change["chosen"] == truth[change["sample_key"]]
        for change in changes
    )
    regressions = sum(
        change["fusion"] == truth[change["sample_key"]]
        and change["chosen"] != truth[change["sample_key"]]
        for change in changes
    )
    payload = {
        "policy": policy,
        "parent_rows": len(parent_keys),
        "changes": len(changes),
        "fixes": fixes,
        "regressions": regressions,
        "baseline_correct": baseline_correct,
        "policy_correct": policy_correct,
        "baseline_accuracy": baseline_correct / len(parent_keys),
        "policy_accuracy": policy_correct / len(parent_keys),
        "change_records": changes,
        "notes": [
            "The routing rules do not read gold labels.",
            "Predictions are written before fixes and regressions are evaluated.",
        ],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--parent-subset",
        help="Optional parent subset. Omit to produce complete-dataset predictions.",
    )
    parser.add_argument("--target-subset", required=True)
    parser.add_argument("--fusion-predictions", required=True)
    parser.add_argument("--endpoint-predictions", required=True)
    parser.add_argument("--candidate-a-predictions", required=True)
    parser.add_argument("--candidate-b-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    parent_rows = apply_subset(annotations, args.parent_subset)
    target_rows = apply_subset(annotations, args.target_subset)
    parent_keys = [sample_key(row) for row in parent_rows]
    target_keys = {sample_key(row) for row in target_rows}
    if len(parent_keys) != len(set(parent_keys)):
        raise RuntimeError("parent subset contains duplicate sample keys")

    fusion = _index(args.fusion_predictions)
    endpoint = _index(args.endpoint_predictions)
    candidate_a = _index(args.candidate_a_predictions)
    candidate_b = _index(args.candidate_b_predictions)
    parent_key_set = set(parent_keys)
    if not parent_key_set <= set(fusion) or not parent_key_set <= set(endpoint):
        raise RuntimeError("fusion or endpoint predictions do not cover the parent subset")
    derived_targets = {
        key for key in parent_keys if _answer(fusion[key]) != _answer(endpoint[key])
    }
    if target_keys != derived_targets:
        raise RuntimeError(
            "target subset is not the exact endpoint/fusion disagreement set: "
            f"target={len(target_keys)} derived={len(derived_targets)}"
        )
    if set(candidate_a) != target_keys or set(candidate_b) != target_keys:
        raise RuntimeError(
            "candidate prediction keys must exactly match the target subset: "
            f"target={len(target_keys)} a={len(candidate_a)} b={len(candidate_b)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = {
        "dual_agreement": (
            lambda fusion_answer, endpoint_answer, answer_a, answer_b: (
                answer_a == answer_b and answer_a != fusion_answer
            )
        ),
        "endpoint_confirmed_agreement": (
            lambda fusion_answer, endpoint_answer, answer_a, answer_b: (
                answer_a == answer_b == endpoint_answer
                and answer_a != fusion_answer
            )
        ),
    }
    summaries: dict[str, Any] = {
        "target_rows": len(target_keys),
        "target_definition": "endpoint/fusion disagreements within the parent subset",
        "candidate_a_abstentions": sum(
            _candidate_answer(candidate_a[key]) is None for key in target_keys
        ),
        "candidate_b_abstentions": sum(
            _candidate_answer(candidate_b[key]) is None for key in target_keys
        ),
        "policies": {},
    }
    for policy_name, predicate in policies.items():
        prediction_path = output_dir / f"predictions_{policy_name}.jsonl"
        changes = _write_predictions(
            prediction_path,
            parent_keys,
            fusion,
            endpoint,
            candidate_a,
            candidate_b,
            target_keys,
            policy_name,
            predicate,
        )
        prediction_rows = load_jsonl(str(prediction_path))
        audit = _audit(
            output_dir / f"audit_{policy_name}.json",
            policy_name,
            parent_rows,
            fusion,
            prediction_rows,
            changes,
        )
        _run_eval(
            args.annotations,
            str(prediction_path),
            str(output_dir / f"results_{policy_name}.json"),
        )
        summaries["policies"][policy_name] = audit

    (output_dir / "agreement_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
