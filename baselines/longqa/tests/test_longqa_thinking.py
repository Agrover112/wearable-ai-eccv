import unittest
from unittest.mock import patch

from longqa_utils import build_longqa_prompt, normalize_answer
from model import _merge_reasoning_content, VLLMModel
from run_generate_longqa import (
    _complete_missing_final_answers,
    has_final_answer_marker,
)


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate_batch(self, frames, messages, max_new_tokens):
        self.calls.append((frames, messages, max_new_tokens))
        return ["Final Answer: C" for _ in messages]


class FakeThinkingModel(FakeModel):
    def __init__(self, retry_response="Final Answer: B"):
        super().__init__()
        self.final_calls = []
        self.retry_response = retry_response

    def generate_final_answer_batch(self, frames, messages, max_new_tokens):
        self.final_calls.append((frames, messages, max_new_tokens))
        return [self.retry_response for _ in messages]


class LongQAThinkingTests(unittest.TestCase):
    def test_thinking_prompt_requests_final_marker(self):
        prompt = build_longqa_prompt(
            "What happened first?", "A. One\nB. Two\nC. Three\nD. Four", "thinking"
        )
        self.assertIn("Final Answer: X", prompt)
        self.assertNotIn("ONLY the single letter", prompt)

    def test_marker_detection_and_normalization(self):
        response = "I compared the events.\nFinal Answer: B"
        self.assertTrue(has_final_answer_marker(response))
        self.assertEqual(normalize_answer(response), "B")

    def test_vllm_reasoning_field_survives_null_content(self):
        self.assertEqual(
            _merge_reasoning_content(
                {"content": None, "reasoning": "The relevant event is later."}
            ),
            "The relevant event is later.",
        )
        self.assertEqual(
            _merge_reasoning_content(
                {
                    "content": "Final Answer: A",
                    "reasoning_content": "I compared all options.",
                }
            ),
            "I compared all options.\n\nFinal Answer: A",
        )

    def test_retry_keeps_reasoning_as_assistant_context(self):
        model = FakeModel()
        frames = [["frame-a"], ["frame-b"]]
        messages = [
            [{"role": "user", "content": "question a"}],
            [{"role": "user", "content": "question b"}],
        ]
        responses = ["reasoning without a marker", "Final Answer: D"]
        completed = _complete_missing_final_answers(
            model, frames, messages, responses, required=True
        )
        self.assertEqual(len(model.calls), 1)
        retry_frames, retry_messages, retry_tokens = model.calls[0]
        self.assertEqual(retry_frames, [["frame-a"]])
        self.assertEqual(retry_messages[0][1]["role"], "assistant")
        self.assertEqual(
            retry_messages[0][1]["content"], "reasoning without a marker"
        )
        self.assertEqual(retry_tokens, 64)
        self.assertTrue(has_final_answer_marker(completed[0]))
        self.assertEqual(completed[1], "Final Answer: D")

    def test_thinking_retry_uses_answer_only_generation(self):
        model = FakeThinkingModel()
        completed = _complete_missing_final_answers(
            model,
            [["frame-a"]],
            [[{"role": "user", "content": "question a"}]],
            ["unfinished reasoning"],
            required=True,
        )
        self.assertEqual(len(model.final_calls), 1)
        self.assertEqual(model.final_calls[0][2], 64)
        self.assertEqual(normalize_answer(completed[0]), "B")

    def test_unfinished_retry_is_rejected(self):
        model = FakeThinkingModel(retry_response="still reasoning")
        with self.assertRaisesRegex(RuntimeError, "Refusing to write"):
            _complete_missing_final_answers(
                model,
                [["frame-a"]],
                [[{"role": "user", "content": "question a"}]],
                ["unfinished reasoning"],
                required=True,
            )

    def test_vllm_answer_retry_disables_additional_thinking(self):
        with patch.dict(
            "os.environ", {"VLLM_THINKING_TOKEN_BUDGET": "2048"}, clear=False
        ):
            model = VLLMModel("Qwen/Qwen3-VL-8B-Thinking")
        with patch.object(
            model, "generate_batch", return_value=["Final Answer: A"]
        ) as generate_batch:
            response = model.generate_final_answer_batch(
                [[]],
                [[{"role": "user", "content": "answer now"}]],
            )
        self.assertEqual(response, ["Final Answer: A"])
        self.assertEqual(
            generate_batch.call_args.kwargs["thinking_token_budget"],
            0,
        )

    def test_qwen35_disables_thinking_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            model = VLLMModel("Qwen/Qwen3.5-9B")
        request = model._apply_chat_template_options({"model": model.model_id})
        self.assertEqual(
            request["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(model._gdn_prefill_backend, "triton")

    def test_qwen35_thinking_can_be_explicitly_enabled(self):
        with patch.dict("os.environ", {"QWEN_ENABLE_THINKING": "true"}, clear=True):
            model = VLLMModel("Qwen/Qwen3.5-9B")
        request = model._apply_chat_template_options({"model": model.model_id})
        self.assertEqual(
            request["chat_template_kwargs"],
            {"enable_thinking": True},
        )

    def test_qwen3vl_request_behavior_is_unchanged(self):
        with patch.dict("os.environ", {}, clear=True):
            model = VLLMModel("Qwen/Qwen3-VL-8B-Instruct")
        request = model._apply_chat_template_options({"model": model.model_id})
        self.assertNotIn("chat_template_kwargs", request)
        self.assertIsNone(model._gdn_prefill_backend)

    def test_qwen35_gdn_backend_can_be_overridden(self):
        with patch.dict(
            "os.environ",
            {"VLLM_GDN_PREFILL_BACKEND": "flashinfer"},
            clear=True,
        ):
            model = VLLMModel("Qwen/Qwen3.5-9B")
        self.assertEqual(model._gdn_prefill_backend, "flashinfer")


if __name__ == "__main__":
    unittest.main()
