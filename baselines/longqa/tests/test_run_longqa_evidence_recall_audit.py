import json

from run_longqa_evidence_recall_audit import (
    ANNOTATION_SCHEMA,
    build_manifest_records,
    manifest_fingerprint,
    manifest_matches,
    score_frame_timestamps,
    select_rows_for_annotation,
    timestamp_covers_interval,
)


def test_interval_matching_expands_by_tolerance_seconds():
    assert timestamp_covers_interval(9.2, 10.0, 12.0, 1.0)
    assert timestamp_covers_interval(13.0, 10.0, 12.0, 1.0)
    assert not timestamp_covers_interval(8.99, 10.0, 12.0, 1.0)
    assert not timestamp_covers_interval(13.01, 10.0, 12.0, 1.0)


def test_multi_event_coverage_and_temporal_order_are_scored_separately():
    events = [
        {"event_id": "pivot", "start_seconds": 10.0, "end_seconds": 10.0, "required": True},
        {"event_id": "target", "start_seconds": 20.0, "end_seconds": 20.0, "required": True},
    ]
    complete = score_frame_timestamps([10.6, 19.2], events, ["pivot", "target"], 1.0)
    reversed_order = score_frame_timestamps([10.0, 20.0], events, ["target", "pivot"], 0.0)
    missing_target = score_frame_timestamps([10.0], events, ["pivot", "target"], 0.0)

    assert complete["covered_events"] == 2
    assert complete["all_events_covered"]
    assert complete["temporal_order_covered"]
    assert not reversed_order["temporal_order_covered"]
    assert missing_target["covered_events"] == 1
    assert not missing_target["all_events_covered"]
    assert not missing_target["temporal_order_covered"]


def test_selection_filters_disagreements_and_errors_in_input_order():
    rows = [
        {"id": "one", "video_path": "one.mp4", "question": "After lunch, what happened?", "mcq_answer": "A"},
        {"id": "two", "video_path": "two.mp4", "question": "What happened?", "mcq_answer": "B"},
        {"id": "three", "video_path": "three.mp4", "question": "What happened?", "mcq_answer": "C"},
    ]
    primary = {
        "one": {"sample_key": "one", "mcq_answer": "A"},
        "two": {"sample_key": "two", "mcq_answer": "A"},
    }
    secondary = {
        "one": {"sample_key": "one", "mcq_answer": "B"},
        "two": {"sample_key": "two", "mcq_answer": "B"},
    }

    selected = select_rows_for_annotation(
        rows, primary, secondary, "disagreements_or_errors"
    )

    assert [(index, row["id"]) for index, row, _ in selected] == [(0, "one"), (1, "two")]
    assert "prediction_disagreement" in selected[0][2]
    assert "primary_error" in selected[1][2]
    assert select_rows_for_annotation(rows, primary, secondary, "disagreements", 1)[0][1]["id"] == "one"


def test_manifest_schema_and_fingerprint_support_safe_resume(tmp_path):
    input_path = tmp_path / "input.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    input_path.write_text(json.dumps({"id": "sample", "video_path": "sample.mp4"}) + "\n")
    metadata_path.write_text(json.dumps({"sample_key": "sample"}) + "\n")
    first_fingerprint = manifest_fingerprint(
        str(input_path), str(metadata_path), None, None, None, None, "all", None
    )
    changed_fingerprint = manifest_fingerprint(
        str(input_path), str(metadata_path), None, None, None, None, "errors", None
    )
    selected = [
        (
            0,
            {
                "id": "sample",
                "video_path": "sample.mp4",
                "question": "Before entering, what happened?",
                "mcq_options": "A. One B. Two",
                "mcq_answer": "A",
            },
            [],
        )
    ]
    candidate = {
        "sample": {
            "sample_key": "sample",
            "duration_seconds": 60.0,
            "candidate_timestamps": [1.0, 20.0],
        }
    }
    final = {
        "sample": {
            "sample_key": "sample",
            "duration_seconds": 60.0,
            "selected": [{"timestamp": 20.0, "frame_index": 300}],
        }
    }
    manifest = build_manifest_records(
        selected, candidate, final, {}, {}, first_fingerprint, None
    )

    assert first_fingerprint != changed_fingerprint
    assert ANNOTATION_SCHEMA["record"]["decisive_events"][0]["event_id"]
    assert manifest[0]["operator"] == "BEFORE"
    assert [frame["timestamp"] for frame in manifest[0]["candidate_frames"]] == [1.0, 20.0]
    assert manifest_matches(manifest, manifest, first_fingerprint)
    assert not manifest_matches(manifest, manifest, changed_fingerprint)


def test_proofpack_candidate_count_reconstructs_uniform_grid_before_selected_fallback():
    selected = [
        (
            0,
            {
                "id": "sample",
                "video_path": "sample.mp4",
                "question": "What happened?",
                "mcq_options": "A. One B. Two",
                "mcq_answer": "A",
            },
            [],
        )
    ]
    proofpack = {
        "sample": {
            "sample_key": "sample",
            "video_fps": 10.0,
            "total_video_frames": 1_000,
            "candidate_frames": 128,
            "selected": [{"frame_index": 512, "timestamp": 51.2}],
        }
    }

    manifest = build_manifest_records(selected, proofpack, proofpack, {}, {}, "audit", None)

    candidate_frames = manifest[0]["candidate_frames"]
    assert manifest[0]["candidate_frame_source"] == "reconstructed_uniform_candidate_grid"
    assert len(candidate_frames) == 128
    assert candidate_frames[0] == {"frame_index": 0, "timestamp": 0.0}
    assert candidate_frames[-1] == {"frame_index": 999, "timestamp": 99.9}
    assert candidate_frames != manifest[0]["final_pack_frames"]
