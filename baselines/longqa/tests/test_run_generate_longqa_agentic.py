import json
from types import SimpleNamespace

import pytest
from PIL import Image

from agents.evidence import load_proofpack_map, select_evidence
from agents.protocol import (
    normalize_option_records,
    validate_specialist_report,
)
from run_generate_longqa_agentic import (
    _evaluate_subset,
    agentic_fingerprint,
    parse_args,
    resume_position,
)


def _row(key="sample"):
    return {
        "id": key,
        "video_path": "video.mp4",
        "question": "What happened after the door opened?",
        "mcq_options": "A. Sat B. Left C. Ate D. Slept",
        "mcq_answer": "B",
        "category": "Daily Activities",
    }


def test_proofpacks_are_indexed_by_sample_key_without_collapsing_videos(tmp_path):
    path = tmp_path / "proofpack.jsonl"
    records = [
        {
            "sample_key": "first",
            "video_path": "video.mp4",
            "selected": [{"frame_index": 10, "source": "anchor"}],
        },
        {
            "sample_key": "second",
            "video_path": "video.mp4",
            "selected": [{"frame_index": 20, "source": "pivot"}],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    indexed = load_proofpack_map(str(path))

    assert set(indexed) == {"first", "second"}
    assert indexed["second"]["selected"][0]["frame_index"] == 20


def test_evidence_selection_prioritizes_events_then_restores_time_order():
    row = _row()
    proofpack = {
        "sample_key": "sample",
        "video_path": "video.mp4",
        "selected": [
            {"frame_index": 90, "source": "anchor", "score": 0.9},
            {"frame_index": 70, "source": "pivot_context", "score": 0.1},
            {"frame_index": 50, "source": "pivot", "score": 0.2},
        ],
    }

    evidence = select_evidence(row, proofpack, fps=10.0, max_frames=2)

    assert [item["frame_index"] for item in evidence] == [50, 70]
    assert [item["frame_id"] for item in evidence] == [1, 2]
    assert [item["timestamp"] for item in evidence] == [5.0, 7.0]


def test_option_reports_require_every_letter_once_and_citations_in_range():
    report = {
        "role": "visual",
        "summary": "summary",
        "verdicts": [
            {
                "option": letter,
                "status": "UNKNOWN",
                "confidence": 10,
                "evidence": [],
                "missing_evidence": "not visible",
            }
            for letter in "DCBA"
        ],
    }
    normalized = validate_specialist_report(report, "visual", frame_count=2)
    assert [item["option"] for item in normalized["verdicts"]] == list("ABCD")

    normalized["verdicts"][0]["evidence"] = [
        {"frame_id": 3, "observation": "outside the supplied pack"}
    ]
    with pytest.raises(RuntimeError, match="only 2 frames"):
        validate_specialist_report(normalized, "visual", frame_count=2)

    duplicate = {"records": [{"option": "A"} for _ in range(4)]}
    with pytest.raises(RuntimeError, match="exactly once"):
        normalize_option_records(duplicate, "records")


def test_resume_requires_aligned_prediction_and_trace_fingerprints():
    rows = [_row("first"), _row("second")]
    predictions = [
        {
            **rows[0],
            "mcq_answer_parsed": "A",
            "agentic_fingerprint": "fp",
        },
        {
            **rows[1],
            "mcq_answer_parsed": "",
            "agentic_fingerprint": "fp",
        },
    ]
    traces = [
        {"sample_key": "first", "agentic_fingerprint": "fp"},
        {"sample_key": "second", "agentic_fingerprint": "fp"},
    ]

    assert resume_position(rows, predictions, traces, "fp") == 1


def test_fingerprint_tracks_input_subset_proofpack_and_prompt_budget(tmp_path):
    input_path = tmp_path / "input.jsonl"
    subset_path = tmp_path / "subset.json"
    input_path.write_text('{"id": "first"}\n')
    subset_path.write_text('{"samples": ["first"]}\n')
    args = SimpleNamespace(
        llm_model="Qwen/Qwen3.5-9B",
        llm_revision="revision-a",
        tp=1,
        concurrency=2,
        max_evidence_frames=64,
        hypothesis_max_new_tokens=768,
        specialist_max_new_tokens=1024,
        organizer_max_new_tokens=768,
        question_time_limit_seconds=300,
    )
    first = agentic_fingerprint(
        args,
        str(input_path),
        "/videos",
        str(subset_path),
        "proof-a",
    )
    second = agentic_fingerprint(
        args,
        str(input_path),
        "/videos",
        str(subset_path),
        "proof-b",
    )
    assert first != second
    args.max_evidence_frames = 32
    assert (
        agentic_fingerprint(
            args,
            str(input_path),
            "/videos",
            str(subset_path),
            "proof-a",
        )
        != first
    )
    args.max_evidence_frames = 64
    input_path.write_text('{"id": "first", "changed": true}\n')
    assert (
        agentic_fingerprint(
            args,
            str(input_path),
            "/videos",
            str(subset_path),
            "proof-a",
        )
        != first
    )


def test_subset_evaluation_uses_selected_rows_not_full_input_prefix(tmp_path):
    rows = [_row("sample-24"), _row("sample-62")]
    predictions = [
        {**rows[0], "mcq_answer": "B"},
        {**rows[1], "mcq_answer": "A"},
    ]
    traces = [
        {
            "sample_key": row["id"],
            "timing": {
                "proofpack_retrieval_seconds": 2.0,
                "evidence_preparation_seconds": 1.0,
                "orchestration_seconds": seconds - 2.0,
                "budgeted_inference_seconds": seconds,
                "full_pipeline_seconds": seconds + 1.0,
                "over_budget": False,
            },
        }
        for row, seconds in zip(rows, [10.0, 20.0])
    ]
    prediction_path = tmp_path / "predictions.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    result_path = tmp_path / "results.json"
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in predictions) + "\n"
    )
    trace_path.write_text("\n".join(json.dumps(row) for row in traces) + "\n")

    results = _evaluate_subset(
        rows,
        str(prediction_path),
        str(trace_path),
        str(result_path),
    )

    assert results["correct"] == 1
    assert results["total"] == 2
    assert results["timing"]["mean_budgeted_inference_seconds"] == 15.0
    assert results["timing"]["mean_proofpack_retrieval_seconds"] == 2.0
    assert results["timing"]["mean_full_pipeline_seconds"] == 16.0


def test_cli_defaults_to_four_agent_calls_and_parallel_specialists():
    args = parse_args(
        ["--proofpack", "proofpack.jsonl", "--output", "predictions.jsonl"]
    )
    assert args.llm_model == "Qwen/Qwen3.5-9B"
    assert args.llm_revision is None
    assert args.max_evidence_frames == 64
    assert args.concurrency == 2
    assert args.question_time_limit_seconds == 300

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--proofpack",
                "proofpack.jsonl",
                "--output",
                "predictions.jsonl",
                "--concurrency",
                "1",
            ]
        )


def test_agentic_launcher_requires_the_exact_dev20_proofpack():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    launcher = (
        project_root / "slurm_scripts/qwen35_agentic_dev20/run.sh"
    ).read_text()
    prerequisite = (
        project_root / "slurm_scripts/qwen35_parent_pivot_dev20/run.sh"
    ).read_text()

    assert "qwen35_parent_temporal_pivot_dev20/proofpack.jsonl" in launcher
    assert "qwen35_parent_temporal_pivot_dev140/proofpack.jsonl" not in launcher
    assert "configs/egolongqa_dev20_seed20260709.json" in prerequisite


def test_qwen35_structured_requests_disable_thinking(monkeypatch):
    from model import VLLMModel

    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": '{"value": "ok"}'}}],
                    "usage": {"prompt_tokens": 4},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = VLLMModel("Qwen/Qwen3.5-9B")
    model._port = 1234
    result = model.generate_json(
        [],
        [{"role": "user", "content": "Return JSON."}],
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    assert result == {"value": "ok"}
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_structured_request_retries_only_length_truncation(monkeypatch):
    from model import VLLMModel

    token_limits = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {"content": '{"value": "unterminated'},
                        "finish_reason": "length",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {"content": '{"value": "ok"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
    )

    class _Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        token_limits.append(json.loads(request.data)["max_tokens"])
        return _Response(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = VLLMModel("Qwen/Qwen3.5-9B")
    model._port = 1234

    result = model.generate_json(
        [],
        [{"role": "user", "content": "Return JSON."}],
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        max_new_tokens=128,
        retry_max_new_tokens=256,
    )

    assert result == {"value": "ok"}
    assert token_limits == [128, 256]


def test_structured_request_does_not_retry_other_invalid_json(monkeypatch):
    from model import VLLMModel

    calls = 0

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": "not json"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = VLLMModel("Qwen/Qwen3.5-9B")
    model._port = 1234

    with pytest.raises(RuntimeError, match="valid schema JSON"):
        model.generate_json(
            [],
            [{"role": "user", "content": "Return JSON."}],
            {"type": "object"},
            max_new_tokens=128,
            retry_max_new_tokens=256,
        )

    assert calls == 1


def test_vllm_server_command_pins_model_revision(monkeypatch, tmp_path):
    from model import VLLMModel

    captured = {}

    class _Process:
        pass

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _Process()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    model = VLLMModel(
        "Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    )
    model._port = 1234
    model._log = (tmp_path / "server.log").open("w")
    model._start_server()

    command = captured["command"]
    assert command[command.index("--revision") + 1] == (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    model._log.close()


class _FakeStructuredModel:
    def generate_json(
        self,
        frames,
        messages,
        schema,
        schema_name,
        max_new_tokens,
        retry_max_new_tokens=None,
    ):
        role = "temporal" if "temporal" in schema_name else "visual"
        return {
            "role": role,
            "summary": messages[0]["content"],
            "verdicts": [
                {
                    "option": letter,
                    "status": "UNKNOWN",
                    "confidence": 5,
                    "evidence": [],
                    "missing_evidence": "none",
                }
                for letter in "ABCD"
            ],
        }


def test_specialists_receive_independent_frame_objects_and_json_roles():
    from agents.orchestration import run_specialists_parallel

    frames = [Image.new("RGB", (4, 4), "white")]
    evidence = [
        {
            "frame_id": 1,
            "frame_index": 10,
            "timestamp": 1.0,
            "source": "pivot",
        }
    ]
    hypotheses = {
        "question_type": "TEMPORAL_ORDER",
        "hypotheses": [{"option": letter} for letter in "ABCD"],
    }
    temporal, visual = run_specialists_parallel(
        _FakeStructuredModel(),
        frames,
        _row(),
        {"A": "Sat", "B": "Left", "C": "Ate", "D": "Slept"},
        hypotheses,
        evidence,
        128,
    )

    assert temporal["role"] == "temporal"
    assert visual["role"] == "visual"
    assert "frame_id=01" in temporal["summary"]
