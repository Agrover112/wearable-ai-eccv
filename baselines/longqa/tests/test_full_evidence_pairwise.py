#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "data" / "wearable-ai" / "starter_kit"))

from evaluate_longqa_full_evidence_pairwise import pairwise_decision
from run_score_longqa_full_evidence_pairwise import (
    map_order_probabilities,
    normalize_logprobs,
)


class FullEvidencePairwiseTests(unittest.TestCase):
    def test_order_mapping(self):
        displayed = {"A": 0.2, "B": 0.7, "C": 0.1}
        self.assertEqual(
            map_order_probabilities(displayed, "D", "B"),
            {"D": 0.2, "B": 0.7, "INSUFFICIENT": 0.1},
        )

    def test_logprob_normalization(self):
        values = normalize_logprobs({"A": math.log(2), "B": math.log(1), "C": math.log(1)})
        self.assertAlmostEqual(values["A"], 0.5)
        self.assertAlmostEqual(sum(values.values()), 1.0)

    def test_conservative_switch_requires_both_orders(self):
        record = {
            "baseline_answer": "C",
            "challenger_answer": "B",
            "pairwise_applied": True,
            "forward_probabilities": {"C": 0.2, "B": 0.7, "INSUFFICIENT": 0.1},
            "reverse_probabilities": {"C": 0.2, "B": 0.65, "INSUFFICIENT": 0.15},
        }
        answer, decision = pairwise_decision(record, "conservative")
        self.assertEqual(answer, "B")
        self.assertTrue(decision["switched"])
        record["reverse_probabilities"] = {
            "C": 0.6,
            "B": 0.3,
            "INSUFFICIENT": 0.1,
        }
        answer, decision = pairwise_decision(record, "conservative")
        self.assertEqual(answer, "C")
        self.assertFalse(decision["switched"])

    def test_conservative_abstains_on_insufficient_evidence(self):
        record = {
            "baseline_answer": "A",
            "challenger_answer": "D",
            "pairwise_applied": True,
            "forward_probabilities": {"A": 0.1, "D": 0.45, "INSUFFICIENT": 0.45},
            "reverse_probabilities": {"A": 0.1, "D": 0.45, "INSUFFICIENT": 0.45},
        }
        answer, decision = pairwise_decision(record, "conservative")
        self.assertEqual(answer, "A")
        self.assertFalse(decision["switched"])


if __name__ == "__main__":
    unittest.main()
