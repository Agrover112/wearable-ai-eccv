"""Bounded orchestration for two isolated, role-conditioned specialists."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.protocol import SPECIALIST_SCHEMA, validate_specialist_report
from agents.temporal_agent import build_temporal_prompt
from agents.visual_agent import build_visual_prompt


def run_specialists_parallel(
    model: Any,
    frames: list[Any],
    row: dict[str, Any],
    options: dict[str, str],
    hypotheses: dict[str, Any],
    evidence: list[dict[str, Any]],
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run temporal and visual roles concurrently against the same evidence."""
    temporal_messages = [
        {
            "role": "user",
            "content": build_temporal_prompt(row, options, hypotheses, evidence),
        }
    ]
    visual_messages = [
        {
            "role": "user",
            "content": build_visual_prompt(row, options, hypotheses, evidence),
        }
    ]
    # PIL encoders can now operate independently in the request threads.
    visual_frames = [frame.copy() for frame in frames]
    with ThreadPoolExecutor(max_workers=2) as pool:
        temporal_future = pool.submit(
            model.generate_json,
            frames,
            temporal_messages,
            SPECIALIST_SCHEMA,
            "egolongqa_temporal_report",
            max_new_tokens,
            retry_max_new_tokens=max_new_tokens * 2,
        )
        visual_future = pool.submit(
            model.generate_json,
            visual_frames,
            visual_messages,
            SPECIALIST_SCHEMA,
            "egolongqa_visual_report",
            max_new_tokens,
            retry_max_new_tokens=max_new_tokens * 2,
        )
        temporal_report = temporal_future.result()
        visual_report = visual_future.result()
    return (
        validate_specialist_report(temporal_report, "temporal", len(evidence)),
        validate_specialist_report(visual_report, "visual", len(evidence)),
    )
