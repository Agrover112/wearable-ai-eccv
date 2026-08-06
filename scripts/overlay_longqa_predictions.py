#!/usr/bin/env python3
"""Overlay specialist predictions on a complete fallback prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, compute_diagnostics, load_jsonl, sample_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--override", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    args = parser.parse_args()

    annotations = apply_subset(load_jsonl(args.annotations), args.subset_file)
    fallback = {sample_key(row): row for row in load_jsonl(args.fallback)}
    override = {sample_key(row): row for row in load_jsonl(args.override)}
    missing = {sample_key(row) for row in annotations} - set(fallback)
    if missing:
        raise RuntimeError(f"Fallback predictions are missing {len(missing)} selected rows")
    output = []
    applied = 0
    for annotation in annotations:
        key = sample_key(annotation)
        row = dict(override.get(key, fallback[key]))
        row["specialist_overlay_applied"] = key in override
        applied += int(key in override)
        output.append(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output:
            handle.write(json.dumps(row) + "\n")
    result = compute_diagnostics(annotations, output, run_id="specialist_overlay")
    result["specialist_override_rows"] = applied
    Path(args.eval_output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Applied {applied} specialist predictions; accuracy="
        f"{result['accuracy_raw']:.4f} ({result['correct']}/{result['total']})"
    )


if __name__ == "__main__":
    main()
