#!/usr/bin/env python3
"""Export label-free margin-weighted fusion from cached evidence-rank features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import build_prediction_row, compute_diagnostics, load_jsonl, sample_key
from evaluate_longqa_evidence_rank_fusion import select_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.annotations)
    features: dict[str, dict[str, Any]] = {}
    for path in args.features:
        for feature in load_jsonl(path):
            key = str(feature["sample_key"])
            if key in features:
                raise RuntimeError(f"Duplicate feature row across inputs: {key}")
            features[key] = feature

    expected = {sample_key(row) for row in rows}
    missing = expected - features.keys()
    extra = features.keys() - expected
    if missing or extra:
        raise RuntimeError(f"Feature coverage mismatch: missing={len(missing)} extra={len(extra)}")

    predictions = []
    changes = 0
    for row in rows:
        feature = features[sample_key(row)]
        answer = select_answer(feature, "margin_weighted_probability")
        changes += int(answer != feature["baseline_answer"])
        prediction = build_prediction_row(
            row,
            answer,
            prompt_variant="evidence_rank_margin_weighted_probability",
        )
        prediction.update(
            {
                "rank_fusion_policy": "margin_weighted_probability",
                "rank_fusion_baseline": feature["baseline_answer"],
                "rank_fusion_challenger": feature["challenger_answer"],
            }
        )
        predictions.append(prediction)

    diagnostics = compute_diagnostics(rows, predictions, run_id="margin_weighted_probability")
    diagnostics.update(
        {
            "changes_from_endpoint": changes,
            "uses_labels_for_predictions": False,
            "feature_files": [str(Path(path).resolve()) for path in args.features],
            "note": (
                "This fixed rule weights each view by its top-versus-second option "
                "probability margin. Its selection followed inspection of validation results."
            ),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction) + "\n")
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(
        f"margin_weighted_probability: {diagnostics['correct']}/{diagnostics['total']} "
        f"({diagnostics['accuracy']:.4f}), changes={changes}"
    )


if __name__ == "__main__":
    main()
