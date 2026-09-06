from run_score_longqa_temporal_contrast import temporal_side_packs
from run_score_longqa_third_evidence_view import select_answer


def test_temporal_side_packs_share_pivot_context_but_separate_sides():
    requested, opposite = temporal_side_packs(
        total_frames=1000,
        pivots=[500],
        direction="forward",
        candidate_frames=128,
        max_frames=64,
        pivot_context=8,
    )

    assert len(requested) == len(opposite) == 64
    assert len(set(requested)) == len(set(opposite)) == 64
    assert max(requested) > 900
    assert min(opposite) < 100
    assert set(requested) & set(opposite)


def test_third_view_breaks_endpoint_option_tie_without_labels():
    record = {
        "baseline_answer": "A",
        "challenger_answer": "B",
        "rank_fusion_applied": True,
        "endpoint_view": {
            "top_answer": "A",
            "probabilities": {"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.1},
            "ranks": {"A": 1, "B": 2, "C": 3, "D": 4},
        },
        "option_view": {
            "top_answer": "B",
            "probabilities": {"A": 0.2, "B": 0.6, "C": 0.1, "D": 0.1},
            "ranks": {"A": 2, "B": 1, "C": 3, "D": 4},
        },
        "third_view": {
            "top_answer": "B",
            "probabilities": {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1},
            "ranks": {"A": 2, "B": 1, "C": 3, "D": 4},
        },
    }

    assert select_answer(record, "uncertainty_tiebreak") == "B"
    assert select_answer(record, "three_mean_probability") == "B"
