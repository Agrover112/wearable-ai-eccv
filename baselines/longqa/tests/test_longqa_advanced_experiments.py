import json
from unittest.mock import MagicMock, patch

from model import VLLMModel
from run_generate_longqa_likelihood import calibrated_choice
from run_generate_longqa_narrative_gate import parse_numbered_captions
from run_generate_longqa_event_ledger import (
    EVENT_SCHEMA,
    event_record_text,
    extract_concept_phrases,
    normalize_event_record,
)
from run_generate_longqa_event_ledger_detector import (
    detection_cache_path,
    normalize_detector_result,
)
from run_generate_longqa_semantic_likelihood import semantic_decision, winning_candidate


def test_blind_correction_can_change_visual_prior_choice():
    answer, scores = calibrated_choice(
        {"A": -0.5, "B": -0.6, "C": -2.0, "D": -3.0},
        {"A": -0.1, "B": -1.0, "C": -1.5, "D": -2.0},
        0.5,
    )
    assert answer == "B"
    assert scores["B"] > scores["A"]


def test_numbered_caption_parser_preserves_image_alignment():
    captions = parse_numbered_captions("1: pays at till\n3: exits shop\n2: takes receipt", 3)
    assert captions == ["pays at till", "takes receipt", "exits shop"]


def test_event_record_normalization_fills_schema_and_removes_empty_values():
    record = normalize_event_record(
        {"scene": " kitchen ", "entities": [" mug ", ""], "actions": "holding"}
    )
    assert record["scene"] == "kitchen"
    assert record["entities"] == ["mug"]
    assert record["actions"] == ["holding"]
    assert all(
        field in record
        for field in ("attributes", "relations", "ocr", "state_changes")
    )
    assert "entities: mug" in event_record_text(record)
    assert EVENT_SCHEMA["properties"]["scene"]["maxLength"] == 120
    for field in ("entities", "actions", "attributes", "relations", "ocr"):
        assert EVENT_SCHEMA["properties"][field]["maxItems"] == 6


def test_concepts_are_deduplicated_and_detection_thresholds_affect_cache_key():
    row = {
        "question": "What color was the ceramic mug?",
        "mcq_options": "A. red B. blue C. green D. white",
    }
    concepts = extract_concept_phrases(row)
    assert concepts
    assert len(concepts) == len(set(concepts))
    first = detection_cache_path("/tmp/x", "model", "video", 1, concepts, 0.25, 0.2)
    second = detection_cache_path("/tmp/x", "model", "video", 1, concepts, 0.30, 0.2)
    assert first != second


def test_detector_result_normalization_and_semantic_fallback_rule():
    detections = normalize_detector_result(
        {"text_labels": ["mug"], "scores": [0.75], "boxes": [[1, 2, 3, 4]]}
    )
    assert detections == [
        {"label": "mug", "score": 0.75, "box": [1.0, 2.0, 3.0, 4.0]}
    ]
    scores = {"A": {"mean_logprob": -1.2}, "B": {"mean_logprob": -0.5}}
    assert winning_candidate(scores) == "B"
    assert semantic_decision("A", "B", "B") == ("B", "context_consensus")
    assert semantic_decision("A", "B", "A") == (
        "A",
        "context_disagreement_primary_fallback",
    )


def test_complete_candidate_likelihood_uses_only_candidate_suffix_tokens():
    responses = [
        {
            "prompt_token_ids": [10, 20],
            "prompt_logprobs": [None, {"20": {"logprob": -0.1}}],
            "usage": {"prompt_tokens": 2},
        },
        {
            "prompt_token_ids": [10, 20, 31, 32],
            "prompt_logprobs": [
                None,
                {"20": {"logprob": -0.1}},
                {"31": {"logprob": -1.0}},
                {"32": {"logprob": -2.0}},
            ],
            "usage": {"prompt_tokens": 4},
        },
        {
            "prompt_token_ids": [10, 20, 41],
            "prompt_logprobs": [
                None,
                {"20": {"logprob": -0.1}},
                {"41": {"logprob": -0.5}},
            ],
            "usage": {"prompt_tokens": 3},
        },
    ]
    mocked_responses = []
    for payload in responses:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        mocked_responses.append(response)
    model = VLLMModel("model", max_frames=0)
    model._port = 1234
    model._context_window = 1024
    with patch("urllib.request.urlopen", side_effect=mocked_responses):
        scores = model.score_candidate_texts(
            [],
            [{"role": "user", "content": "Question"}],
            {"A": "two tokens", "B": "one"},
        )
    assert scores["A"]["token_count"] == 2
    assert scores["A"]["total_logprob"] == -3.0
    assert scores["A"]["mean_logprob"] == -1.5
    assert scores["B"]["token_count"] == 1
    assert scores["B"]["mean_logprob"] == -0.5
