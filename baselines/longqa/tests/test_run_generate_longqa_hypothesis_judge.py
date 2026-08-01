from run_generate_longqa_hypothesis_judge import (
    decisive_answer,
    neutral_mapping,
    validate_report,
)


def evidence():
    return [
        {"display_id": "F001", "frame_index": 10, "timestamp": 1.0},
        {"display_id": "F002", "frame_index": 20, "timestamp": 2.0},
    ]


def candidate(verdict, support=None, contradiction=None):
    return {
        "verdict": verdict,
        "supporting_frame_ids": support or [],
        "contradicting_frame_ids": contradiction or [],
        "temporal_check": "PASS",
        "identity_check": "NOT_APPLICABLE",
        "coverage_check": "PASS",
        "brief_evidence": "visible evidence",
    }


def no_refinement():
    return {
        "needed": False,
        "reason": "NOT_NEEDED",
        "missing_visible_fact": "",
        "retrieval_queries": [],
        "anchor_frame_ids": [],
        "temporal_region": "NONE",
    }


def test_neutral_mapping_is_stable_and_complete():
    first = neutral_mapping("video||question", ["A", "C", "D"])
    second = neutral_mapping("video||question", ["D", "A", "C"])
    assert first == second
    assert set(first) == {"X", "Y", "Z"}
    assert set(first.values()) == {"A", "C", "D"}


def test_decisive_report_requires_support_and_all_other_contradictions():
    report = {
        "candidate_x": candidate("SUPPORTED", support=["F001"]),
        "candidate_y": candidate("CONTRADICTED", contradiction=["F002"]),
        "decisive_visible_fact": "fact",
        "refinement": no_refinement(),
    }
    valid, _ = validate_report(report, ["X", "Y"], evidence(), False)
    assert valid
    assert decisive_answer(report, {"X": "B", "Y": "D"}) == "B"
    report["candidate_y"] = candidate("INSUFFICIENT")
    assert decisive_answer(report, {"X": "B", "Y": "D"}) is None


def test_invalid_citation_and_final_round_refinement_are_rejected():
    report = {
        "candidate_x": candidate("SUPPORTED", support=["F999"]),
        "candidate_y": candidate("CONTRADICTED", contradiction=["F002"]),
        "decisive_visible_fact": "fact",
        "refinement": no_refinement(),
    }
    valid, reason = validate_report(report, ["X", "Y"], evidence(), False)
    assert not valid
    assert "citation" in reason
    report["candidate_x"] = candidate("INSUFFICIENT")
    report["candidate_y"] = candidate("INSUFFICIENT")
    report["refinement"] = {
        "needed": True,
        "reason": "TEMPORAL_GAP",
        "missing_visible_fact": "event after F001",
        "retrieval_queries": ["event after placing cup"],
        "anchor_frame_ids": ["F001"],
        "temporal_region": "AFTER",
    }
    valid, reason = validate_report(report, ["X", "Y"], evidence(), False)
    assert not valid
    assert "final round" in reason
