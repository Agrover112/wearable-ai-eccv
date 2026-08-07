from __future__ import annotations

import numpy as np
import pytest

from run_generate_longqa_proofpack import combine_rrf_retrieval_scores


def test_rrf_balances_target_and_option_rankings() -> None:
    scores = combine_rrf_retrieval_scores(
        {
            "target": [10.0, 5.0, 0.0],
            "option_A": [0.0, 5.0, 10.0],
            "option_B": [0.0, 5.0, 10.0],
        },
        rrf_constant=1.0,
    )

    assert scores[0] == pytest.approx(scores[2])
    assert scores[0] > scores[1]
    assert np.isfinite(scores).all()


def test_rrf_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        combine_rrf_retrieval_scores({"target": [1.0]}, rrf_constant=0.0)
    with pytest.raises(ValueError, match="equal length"):
        combine_rrf_retrieval_scores(
            {"target": [1.0, 0.0], "option_A": [1.0]}, rrf_constant=60.0
        )
