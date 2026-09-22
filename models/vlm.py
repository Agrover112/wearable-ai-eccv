"""Requests to a running vLLM server (see scripts/serve.sh) and answer parsing"""

import base64
import io
import json
import math
import re
import urllib.request
from typing import Any

from PIL import Image


def question_prompt(sample: dict[str, str]) -> str:
    """Final multiple-choice prompt, shared by every method"""
    return (
        "Watch the video and answer the following multiple-choice question.\n\n"
        f"Question: {sample['question']}\n\nOptions:\n{sample['mcq_options']}\n\n"
        "Answer with ONLY the single letter of the correct option (A, B, C, or D). "
        "Do not include any other text."
    )


def request(
    server: str,
    model: str,
    frames: list[Image.Image],
    prompt: str,
    max_tokens: int = 16,
    top_logprobs: int = 0,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One greedy chat completion with the frames placed before the prompt

    Returns the first choice; `top_logprobs` adds next-token probabilities and
    `response_format` constrains the output, e.g. to a JSON schema
    """
    # Frames go in as base64 JPEGs, in chronological order
    content = []
    for frame in frames:
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Qwen3.5 answers directly instead of reasoning first; Qwen3-VL ignores this
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if top_logprobs:
        payload.update(logprobs=True, top_logprobs=top_logprobs)
    if response_format is not None:
        payload["response_format"] = response_format
    http_request = urllib.request.Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=1800) as response:
        return json.load(response)["choices"][0]


def answer_letter(response: str) -> str:
    """Extract A-D from a response; the last explicit answer wins, otherwise ''"""
    text = response.strip()
    upper = text.upper()
    # A bare letter, the expected case
    if len(upper) == 1 and upper in "ABCD":
        return upper
    # Otherwise the last "answer: X", "option (X)", ... phrase
    matches = list(re.finditer(
        r"\b(?:final\s+answer|answer|option|choice)\s*[:.]?\s*\(?([A-D])\)?\b",
        text, re.IGNORECASE,
    ))
    if matches:
        return matches[-1].group(1).upper()
    # Otherwise the last standalone letter, then a leading letter
    matches = list(re.finditer(r"\b([A-Da-d])\b", text))
    if matches:
        return matches[-1].group(1).upper()
    return upper[:1] if upper[:1] in tuple("ABCD") else ""


def answer(server: str, model: str, frames: list[Image.Image], prompt: str) -> str:
    """The model's answer letter for the given frames, '' if unparsable"""
    result = request(server, model, frames, prompt)
    return answer_letter(result["message"]["content"])


def option_probabilities(
    server: str, model: str, frames: list[Image.Image], prompt: str,
) -> dict[str, float]:
    """Probability of each answer letter as the next token, normalized over A-D"""
    result = request(server, model, frames, prompt, max_tokens=1, top_logprobs=100)
    tokens = result["logprobs"]["content"][0]["top_logprobs"]
    # " a", "A", and "A " are the same letter; keep the most probable variant
    scores = {
        letter: max(t["logprob"] for t in tokens if t["token"].strip().upper() == letter)
        for letter in "ABCD"
    }
    # Softmax over the four letters only
    maximum = max(scores.values())
    weights = {letter: math.exp(score - maximum) for letter, score in scores.items()}
    total = sum(weights.values())
    return {letter: value / total for letter, value in weights.items()}
