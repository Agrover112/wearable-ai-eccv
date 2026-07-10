from run_generate_longqa_grounded import CandidateFrame, select_per_option_union_frames


def test_per_option_union_deduplicates_anchors_by_source_frame_index():
    candidates = [
        CandidateFrame(index=position * 10, timestamp=float(position), image=None)
        for position in range(12)
    ]
    query_scores = {
        "option_A": [float(position) for position in range(12)],
        "option_B": [float(12 - position) for position in range(12)],
        "option_C": [float(position % 3) for position in range(12)],
        "option_D": [float(position % 5) for position in range(12)],
    }

    selected, _ = select_per_option_union_frames(
        candidates,
        query_scores,
        top_k=8,
        top_k_per_option=2,
        anchor_k=8,
        window_radius=0,
        final_max_frames=8,
        temporal_nms_seconds=0.0,
        temporal_nms_candidates=None,
    )

    frame_indices = [frame.candidate.index for frame in selected]
    assert len(frame_indices) <= 8
    assert len(frame_indices) == len(set(frame_indices))
