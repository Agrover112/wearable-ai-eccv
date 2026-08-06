#!/usr/bin/env python3
"""Export and validate a minimal EgoLongQA leaderboard submission."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


VALID_ANSWERS = {"A", "B", "C", "D"}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fallback-key",
        default="multicandidate_judge_previous",
        help="Prediction field used only when mcq_answer is missing or invalid.",
    )
    parser.add_argument(
        "--preserve-fields",
        action="store_true",
        help="Keep the complete prediction row while repairing invalid answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_jsonl(args.predictions)
    annotations = load_jsonl(args.annotations)
    if len(predictions) != 700 or len(annotations) != 700:
        raise ValueError(
            f"EgoLongQA requires 700 rows; got {len(predictions)} predictions "
            f"and {len(annotations)} annotations"
        )

    exported: list[dict[str, object]] = []
    repaired: list[str] = []
    seen: set[str] = set()
    for index, (prediction, annotation) in enumerate(
        zip(predictions, annotations), 1
    ):
        video_path = str(prediction.get("video_path", "")).strip()
        expected_path = str(annotation.get("video_path", "")).strip()
        if not video_path or video_path != expected_path:
            raise ValueError(
                f"row {index}: prediction video_path {video_path!r} does not "
                f"match annotation order {expected_path!r}"
            )
        if video_path in seen:
            raise ValueError(f"row {index}: duplicate video_path {video_path!r}")
        seen.add(video_path)

        answer = str(prediction.get("mcq_answer", "")).strip().upper()
        if answer not in VALID_ANSWERS:
            fallback = str(prediction.get(args.fallback_key, "")).strip().upper()
            if fallback not in VALID_ANSWERS:
                raise ValueError(
                    f"row {index}: invalid mcq_answer {answer!r} and invalid "
                    f"fallback {fallback!r}"
                )
            answer = fallback
            repaired.append(video_path)
        if args.preserve_fields:
            exported_row = dict(prediction)
            exported_row["mcq_answer"] = answer
            exported_row["mcq_answer_raw"] = answer
            exported_row["mcq_answer_parsed"] = answer
            exported.append(exported_row)
        else:
            exported.append({"video_path": video_path, "mcq_answer": answer})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in exported:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    print(f"Wrote {len(exported)} predictions to {args.output}")
    print(f"Deterministic fallback repairs: {len(repaired)}")
    for video_path in repaired:
        print(f"  {video_path}")


if __name__ == "__main__":
    main()
