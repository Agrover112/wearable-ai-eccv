"""Timestamp-aware temporal evidence specialist."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "temporal_v2"


def build_temporal_prompt(
    row: dict[str, Any],
    options: dict[str, str],
    hypotheses: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in "ABCD")
    frame_text = "\n".join(
        (
            f"frame_id={item['frame_id']:02d} | t={item['timestamp']:.3f}s | "
            f"source={item['source']}"
        )
        for item in evidence
    )
    return f"""You are the temporal specialist in a long-video QA system.

The evidence images appear before this prompt in exactly the frame_id order
listed below. Evaluate every option independently. Focus on event order,
before/after, first/last, recurrence, and state transitions. A visible event at
one timestamp does not establish "first" or "last" unless the surrounding
evidence supports it. Sparse evidence means UNKNOWN, not CONTRADICTED. Cite
only supplied frame_ids. Do not select a final answer. Keep the summary under
40 words and each observation under 25 words. Cite at most two decisive frames
per option. Never enumerate frame IDs in the summary or repeat the same
observation across options.

Question:
{row["question"]}

Options:
{option_text}

Compiled hypotheses:
{json.dumps(hypotheses, sort_keys=True)}

Evidence index:
{frame_text}

Set role to "temporal" and return only the requested JSON object."""
