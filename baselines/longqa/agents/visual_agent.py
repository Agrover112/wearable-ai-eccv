"""Appearance, action, OCR, and counting evidence specialist."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "visual_v2"


def build_visual_prompt(
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
    return f"""You are the visual specialist in a long-video QA system.

The evidence images appear before this prompt in exactly the frame_id order
listed below. Evaluate every option independently using visible object
identity, actions, locations, attributes, counts, and readable text. Do not
infer unseen details from plausibility. Sparse, blurry, or absent evidence
means UNKNOWN, not CONTRADICTED. Cite only supplied frame_ids. Do not select a
final answer. Keep the summary under 40 words and each observation under 25
words. Cite at most two decisive frames per option. Never enumerate frame IDs
in the summary or repeat the same observation across options.

Question:
{row["question"]}

Options:
{option_text}

Compiled hypotheses:
{json.dumps(hypotheses, sort_keys=True)}

Evidence index:
{frame_text}

Set role to "visual" and return only the requested JSON object."""
