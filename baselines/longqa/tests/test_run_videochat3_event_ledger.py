import pytest

from run_videochat3_event_ledger import (
    frame_indices_for_window,
    parse_event_record,
    window_bounds,
)


def test_window_bounds_cover_video_with_short_final_window():
    assert window_bounds(15.0, 7.0) == [(0, 7.0), (7, 14.0), (14, 15.0)]


def test_frame_indices_are_uniform_within_window():
    assert frame_indices_for_window(7.0, 14.0, 10.0, 200, 4) == [70, 93, 116, 139]


def test_parse_event_record_accepts_fenced_json():
    record = parse_event_record(
        "```json\n"
        '{"location": "kitchen", "entities": ["red mug"], '
        '"action": "rinses mug", "state_changes": ["dirty → rinsed"], '
        '"confidence": 0.91}\n```'
    )

    assert record == {
        "location": "kitchen",
        "entities": ["red mug"],
        "caption": "",
        "action": "rinses mug",
        "state_changes": ["dirty → rinsed"],
        "confidence": 0.91,
    }


def test_parse_event_record_surfaces_invalid_json():
    with pytest.raises(ValueError):
        parse_event_record("not json")
