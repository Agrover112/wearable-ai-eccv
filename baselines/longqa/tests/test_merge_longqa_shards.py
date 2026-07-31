from merge_longqa_shards import merge_rows


def test_merge_rows_restores_reference_order():
    reference = [
        {"video_path": "a.mp4", "question": "A?"},
        {"video_path": "b.mp4", "question": "B?"},
        {"video_path": "c.mp4", "question": "C?"},
    ]
    first = [
        {"video_path": "c.mp4", "question": "C?", "mcq_answer": "C"},
        {"video_path": "a.mp4", "question": "A?", "mcq_answer": "A"},
    ]
    second = [{"video_path": "b.mp4", "question": "B?", "mcq_answer": "B"}]

    merged = merge_rows(reference, [first, second])

    assert [row["mcq_answer"] for row in merged] == ["A", "B", "C"]


def test_merge_rows_uses_explicit_metadata_sample_keys():
    reference = [
        {"video_path": "same.mp4", "question": "First?"},
        {"video_path": "same.mp4", "question": "Second?"},
    ]
    metadata = [
        {"sample_key": "same.mp4||Second?", "index": 0, "selected": [2]},
        {"sample_key": "same.mp4||First?", "index": 1, "selected": [1]},
    ]

    merged = merge_rows(reference, [metadata])

    assert [row["selected"] for row in merged] == [[1], [2]]
    assert [row["index"] for row in merged] == [0, 1]
