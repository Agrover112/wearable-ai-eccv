import unittest

from run_generate_longqa_hieramamba import (
    add_global_frames,
    sample_interval_frames,
    select_query_proposals,
)


class HieraMambaFrameSelectionTests(unittest.TestCase):
    def test_relation_filter_keeps_target_after_anchor(self):
        record = {
            "duration_seconds": 100.0,
            "queries": [
                {
                    "id": "q1",
                    "relation": "none",
                    "retrieve": "top1",
                    "evidence_window": "inside",
                    "proposals": [
                        {"rank": 1, "segment_seconds": [40.0, 45.0], "score": 0.9}
                    ],
                },
                {
                    "id": "q2",
                    "relation": "after:q1",
                    "retrieve": "top5",
                    "evidence_window": "around",
                    "proposals": [
                        {"rank": 1, "segment_seconds": [10.0, 15.0], "score": 0.95},
                        {"rank": 2, "segment_seconds": [60.0, 65.0], "score": 0.8},
                    ],
                },
            ],
        }
        intervals, audit = select_query_proposals(
            record, top_k=3, padding_seconds=2.0
        )
        self.assertNotIn((8.0, 17.0), intervals)
        self.assertIn((58.0, 67.0), intervals)
        self.assertEqual([item["query_id"] for item in audit], ["q1", "q2"])


    def test_first_occurrence_uses_earliest_retrieved_proposal(self):
        record = {
            "duration_seconds": 100.0,
            "queries": [
                {
                    "id": "q1",
                    "relation": "first_occurrence",
                    "retrieve": "top5",
                    "evidence_window": "inside",
                    "proposals": [
                        {"rank": 1, "segment_seconds": [70.0, 75.0], "score": 0.9},
                        {"rank": 2, "segment_seconds": [20.0, 25.0], "score": 0.8},
                    ],
                }
            ],
        }
        intervals, _ = select_query_proposals(
            record, top_k=5, padding_seconds=2.0
        )
        self.assertEqual(intervals, [(20.0, 25.0)])


    def test_interval_sampling_and_global_fill_respect_budget(self):
        local = sample_interval_frames(
            [(10.0, 20.0), (50.0, 60.0)],
            fps=10.0,
            total_frames=1000,
            budget=12,
        )
        self.assertEqual(len(local), 12)
        self.assertEqual(local, sorted(set(local)))
        final, added = add_global_frames(
            local, 1000, global_quota=4, max_frames=16
        )
        self.assertEqual(len(final), 16)
        self.assertEqual(len(added), 4)
        self.assertEqual(final, sorted(set(final)))


if __name__ == "__main__":
    unittest.main()
