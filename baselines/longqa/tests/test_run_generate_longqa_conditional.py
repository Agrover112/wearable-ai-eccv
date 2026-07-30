import unittest

from run_generate_longqa_conditional import (
    build_candidate_prompt,
    majority_answer,
    parse_candidate_response,
    should_intervene,
)


class ConditionalLongQATests(unittest.TestCase):
    def test_majority_uses_input_order_for_three_way_tie(self):
        answer, counts, tied = majority_answer(["B", "C", "D"])
        self.assertEqual(answer, "B")
        self.assertEqual(counts, {"B": 1, "C": 1, "D": 1})
        self.assertTrue(tied)

    def test_disagreement_scopes(self):
        self.assertTrue(should_intervene(["A", "B", "C"], "all_different"))
        self.assertFalse(should_intervene(["A", "A", "B"], "all_different"))
        self.assertTrue(should_intervene(["A", "A", "B"], "any_disagreement"))
        self.assertFalse(should_intervene(["C", "C", "C"], "any_disagreement"))

    def test_candidate_prompt_omits_unproposed_options(self):
        row = {
            "question": "What happened next?",
            "mcq_options": "A. alpha B. beta C. gamma D. delta",
        }
        prompt = build_candidate_prompt(row, ["B", "D"])
        self.assertIn("B. beta", prompt)
        self.assertIn("D. delta", prompt)
        self.assertNotIn("A. alpha", prompt)
        self.assertNotIn("C. gamma", prompt)

    def test_candidate_parser_rejects_unlisted_answer(self):
        self.assertEqual(parse_candidate_response("B", ["B", "D"]), "B")
        self.assertIsNone(parse_candidate_response("A", ["B", "D"]))


if __name__ == "__main__":
    unittest.main()
