import json
import math
from unittest.mock import patch

from model import VLLMModel
from run_generate_longqa_uncertainty import (
    _compact_score,
    _letter_probability,
    fill_with_uniform,
    select_segmented_positions,
    should_run_uncertainty_gate,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_token_uncertainty_includes_unobserved_tail():
    model = VLLMModel("test/model", max_frames=1, model_type="qwen")
    model._port = 1234
    payload = {
        "usage": {"prompt_tokens": 10},
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": math.log(0.6),
                            "top_logprobs": [
                                {"token": "A", "logprob": math.log(0.6)},
                                {"token": "B", "logprob": math.log(0.3)},
                            ],
                        }
                    ]
                }
            }
        ],
    }
    with patch("urllib.request.urlopen", return_value=_Response(payload)):
        score = model.score_token_uncertainty(
            [],
            [{"role": "user", "content": "Question"}],
            top_logprobs=2,
        )
    expected = -sum(value * math.log(value) for value in (0.6, 0.3, 0.1))
    assert math.isclose(score["entropy_lower_bound"], expected)
    assert math.isclose(score["tail_probability_mass"], 0.1)
    assert score["predicted_token"] == "A"


def test_compact_score_preserves_letter_probabilities():
    score = {
        "distribution": [
            {"token": "A", "logprob": math.log(0.7)},
            {"token": "B", "logprob": math.log(0.2)},
        ]
        + [
            {"token": f"x{index}", "logprob": -20.0}
            for index in range(30)
        ]
    }
    compact = _compact_score(score)
    assert len(compact["distribution"]) == 20
    assert math.isclose(_letter_probability(compact, "A"), 0.7)
    assert math.isclose(_letter_probability(compact, "B"), 0.2)


def test_segmented_selection_keeps_lowest_entropy_per_segment():
    scores = [{"entropy_lower_bound": float(15 - index)} for index in range(16)]
    assert select_segmented_positions(scores, segments=4, per_segment=2) == [
        2,
        3,
        6,
        7,
        10,
        11,
        14,
        15,
    ]


def test_uniform_fill_is_unique_and_reaches_budget():
    selected = fill_with_uniform([1, 3, 5], total_frames=100, uniform_quota=8, max_frames=16)
    assert len(selected) == 16
    assert len(set(selected)) == 16
    assert {1, 3, 5}.issubset(selected)


def test_uncertainty_gate_runs_only_on_disagreement():
    assert not should_run_uncertainty_gate(["B", "B", "B"])
    assert should_run_uncertainty_gate(["B", "C"])
    assert should_run_uncertainty_gate([])
