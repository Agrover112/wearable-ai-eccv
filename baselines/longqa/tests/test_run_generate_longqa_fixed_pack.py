import json

import pytest

from run_generate_longqa_fixed_pack import (
    DEFAULT_BACKEND,
    DEFAULT_FRAME_COUNT,
    DEFAULT_LLM_MODEL,
    frame_pack_fingerprint,
    inference_fingerprint,
    load_proofpack_indices,
    parse_args,
    uniform_frame_indices,
)


def _rows():
    return [
        {"id": "first", "video_path": "one.mp4", "question": "First?"},
        {"id": "second", "video_path": "one.mp4", "question": "Second?"},
    ]


def test_uniform_indices_match_the_baseline_single_interval_formula():
    assert uniform_frame_indices(101, 4) == [0, 25, 50, 75]
    assert uniform_frame_indices(3, 8) == [0, 1]
    assert len(uniform_frame_indices(9000)) == DEFAULT_FRAME_COUNT


def test_load_proofpack_indices_keeps_row_alignment_for_repeated_videos(tmp_path):
    proofpack = tmp_path / "proofpack.jsonl"
    proofpack.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "index": 0,
                    "sample_key": "first",
                    "video_path": "one.mp4",
                    "selected": [{"frame_index": 10}, {"frame_index": 20}],
                },
                {
                    "index": 1,
                    "sample_key": "second",
                    "video_path": "one.mp4",
                    "selected": [{"frame_index": 30}, {"frame_index": 40}],
                },
            ]
        )
        + "\n"
    )

    assert load_proofpack_indices(str(proofpack), _rows()) == [[10, 20], [30, 40]]


def test_load_proofpack_indices_rejects_misaligned_rows(tmp_path):
    proofpack = tmp_path / "proofpack.jsonl"
    proofpack.write_text(
        json.dumps(
            {
                "index": 1,
                "sample_key": "first",
                "video_path": "one.mp4",
                "selected": [{"frame_index": 10}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "index": 0,
                "sample_key": "second",
                "video_path": "one.mp4",
                "selected": [{"frame_index": 20}],
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="index mismatch"):
        load_proofpack_indices(str(proofpack), _rows())


def test_defaults_and_fingerprints_track_model_and_frame_configuration(monkeypatch):
    args = parse_args(["--output", "predictions.jsonl"])
    assert args.frame_source == "uniform"
    assert args.frame_count == DEFAULT_FRAME_COUNT
    assert args.llm_model == DEFAULT_LLM_MODEL
    assert args.backend == DEFAULT_BACKEND

    monkeypatch.setenv("QWEN_MAX_PIXELS", "451584")
    pack_fingerprint = frame_pack_fingerprint(args, None)
    first = inference_fingerprint(args, pack_fingerprint)
    args.llm_model = "Qwen/Qwen3.5-27B"
    second = inference_fingerprint(args, pack_fingerprint)
    assert first != second
    args.frame_count = 32
    assert frame_pack_fingerprint(args, None) != pack_fingerprint
