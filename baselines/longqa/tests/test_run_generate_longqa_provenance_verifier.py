from types import SimpleNamespace

from PIL import Image

from run_generate_longqa_provenance_verifier import (
    DEFAULT_MODEL,
    DEFAULT_PRIMARY_FRAME_COUNT,
    DEFAULT_SECONDARY_FRAME_COUNT,
    DEFAULT_THINKING_MAX_NEW_TOKENS,
    build_provenance_content,
    build_provenance_payload,
    choose_candidate,
    parse_args,
    parse_verifier_choice,
    provenance_verifier_fingerprint,
    resume_position,
    select_provenance_evidence,
    should_call_verifier,
)


def _frame():
    return Image.new("RGB", (4, 4), "white")


def test_interleaved_payload_keeps_primary_and_secondary_provenance_separate():
    row = {
        "question": "What happened after payment?",
        "mcq_options": "A. Sat B. Left C. Ordered D. Ate",
    }
    primary = [
        (
            {"frame_index": 15, "timestamp": 1.0, "source": "pivot"},
            _frame(),
        )
    ]
    secondary = [
        (
            {"frame_index": 90, "timestamp": 6.0, "source": "uniform_secondary"},
            _frame(),
        )
    ]

    content = build_provenance_content(row, "B", "D", primary, secondary)
    text = [item["text"] for item in content if item["type"] == "text"]
    types = [item["type"] for item in content]

    assert "Candidate 1 (primary pass): option B: Left" in text[0]
    assert "Candidate 2 (secondary pass): option D: Ate" in text[0]
    assert "PRIMARY PROOFPACK EVIDENCE" in text[1]
    assert "PRIMARY frame 01 | t=1.000s | frame=15 | source=pivot" in text[2]
    assert "SECONDARY UNIFORM EVIDENCE" in text[3]
    assert "SECONDARY frame 01 | t=6.000s | frame=90 | source=uniform_secondary" in text[4]
    assert types == ["text", "text", "text", "image_url", "text", "text", "image_url", "text"]
    assert content[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    payload = build_provenance_payload(DEFAULT_MODEL, content, 16, thinking=False)
    assert payload["messages"][0]["content"] == content
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    thinking_payload = build_provenance_payload(
        DEFAULT_MODEL, content, 1024, thinking=True
    )
    assert thinking_payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert thinking_payload["temperature"] == 1.0
    assert thinking_payload["top_p"] == 0.95
    assert thinking_payload["top_k"] == 20


def test_candidate_parsing_and_fallback_are_restricted_to_two_candidates():
    assert parse_verifier_choice("CANDIDATE_1") == "candidate_1"
    assert parse_verifier_choice("Final decision: candidate 2") == "candidate_2"
    assert (
        parse_verifier_choice(
            "<think>CANDIDATE_1 has some support, but CANDIDATE_2 is stronger.</think>"
            "CANDIDATE_2"
        )
        == "candidate_2"
    )
    assert parse_verifier_choice("INSUFFICIENT") == "insufficient"
    assert parse_verifier_choice("I prefer option C") is None

    assert choose_candidate("candidate_1", "B", "D") == ("B", False)
    assert choose_candidate("candidate_2", "B", "D") == ("D", False)
    assert choose_candidate("insufficient", "B", "D") == ("B", True)
    assert choose_candidate(None, "B", "D") == ("B", True)


def test_evidence_selection_reserves_32_frames_per_source_without_cross_source_merge():
    proofpack = [
        {
            "frame_index": index * 10,
            "timestamp": float(index),
            "source": "pivot" if index < 2 else "anchor",
            "score": float(100 - index),
        }
        for index in range(40)
    ]
    primary, secondary = select_provenance_evidence(
        proofpack,
        total_frames=960,
        fps=15.0,
    )

    assert len(primary) == DEFAULT_PRIMARY_FRAME_COUNT
    assert len(secondary) == DEFAULT_SECONDARY_FRAME_COUNT
    assert len(primary) + len(secondary) == 64
    assert primary == sorted(primary, key=lambda item: (item["timestamp"], item["frame_index"]))
    assert all(item["source"] == "uniform_secondary" for item in secondary)
    assert primary[0]["frame_index"] == 0
    assert secondary[0]["frame_index"] == 0


def test_agreements_need_no_verifier_call():
    assert not should_call_verifier("B", "B")
    assert should_call_verifier("B", "D")


def test_defaults_and_fingerprint_cover_thinking_mode():
    args = parse_args(
        [
            "--primary-predictions",
            "primary.jsonl",
            "--secondary-predictions",
            "secondary.jsonl",
            "--primary-proofpack",
            "proofpack.jsonl",
            "--output",
            "predictions.jsonl",
        ]
    )
    assert args.llm_model == DEFAULT_MODEL
    assert args.primary_frame_count == DEFAULT_PRIMARY_FRAME_COUNT
    assert args.secondary_frame_count == DEFAULT_SECONDARY_FRAME_COUNT
    assert not args.thinking
    assert args.thinking_max_new_tokens == DEFAULT_THINKING_MAX_NEW_TOKENS

    base = SimpleNamespace(
        input="/scratch/input.jsonl",
        subset_file="/scratch/subset.json",
        video_folder="/scratch/videos",
        primary_predictions="/scratch/primary.jsonl",
        secondary_predictions="/scratch/secondary.jsonl",
        primary_proofpack="/scratch/proofpack.jsonl",
        primary_frame_count=32,
        secondary_frame_count=32,
        llm_model=DEFAULT_MODEL,
        tp=1,
        concurrency=1,
        thinking=False,
        max_new_tokens=16,
        thinking_max_new_tokens=1024,
    )
    thinking = SimpleNamespace(**{**vars(base), "thinking": True})
    assert provenance_verifier_fingerprint(base) != provenance_verifier_fingerprint(thinking)


def test_resume_requires_aligned_prediction_and_evidence_fingerprints():
    rows = [
        {"video_path": "first.mp4", "question": "First?"},
        {"video_path": "second.mp4", "question": "Second?"},
    ]
    predictions = [
        {
            "video_path": "first.mp4",
            "question": "First?",
            "mcq_answer": "A",
            "provenance_verifier_fingerprint": "abc123",
        },
        {
            "video_path": "second.mp4",
            "question": "Second?",
            "mcq_answer": "B",
            "provenance_verifier_fingerprint": "abc123",
        },
    ]
    evidence = [
        {
            "video_path": "first.mp4",
            "question": "First?",
            "provenance_verifier_fingerprint": "abc123",
        },
        {
            "video_path": "second.mp4",
            "question": "Second?",
            "provenance_verifier_fingerprint": "different",
        },
    ]

    assert resume_position(rows, predictions, evidence, "abc123") == 1
