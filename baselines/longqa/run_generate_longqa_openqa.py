#!/usr/bin/env python3
"""Generate option-free LongQA answers and map them back to MCQ options."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from typing import Any

import numpy as np

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    parse_mcq_options,
    query_hash,
    sample_key,
)
from run_generate_longqa import _print_context_summary

logger = logging.getLogger(__name__)

OPENQA_PROMPT_TEMPLATE = (
    "Watch the video and answer the question using a concise factual phrase or "
    "sentence. Base the answer only on visible video evidence. Do not discuss "
    "multiple-choice options and do not add an explanation.\n\n"
    "Question: {question}\n\nAnswer:"
)


def build_openqa_prompt(question: object) -> str:
    return OPENQA_PROMPT_TEMPLATE.format(question=question)


def select_nearest_option(
    answer_embedding: np.ndarray,
    option_embeddings: np.ndarray,
    letters: list[str],
) -> tuple[str, dict[str, float], float]:
    """Return nearest option, per-option cosine scores, and top-1 margin."""
    if option_embeddings.shape[0] != len(letters):
        raise ValueError("option embedding count must match option letters")
    scores = option_embeddings @ answer_embedding
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    margin = float(scores[best] - scores[int(order[1])]) if len(order) > 1 else 0.0
    return (
        letters[best],
        {letter: round(float(score), 6) for letter, score in zip(letters, scores)},
        round(margin, 6),
    )


class MiniLMTextEncoder:
    """Minimal normalized mean-pooling encoder using existing Transformers."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        batch_size: int = 128,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        encoded: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = F.normalize(pooled, dim=-1)
                encoded.append(pooled.float().cpu().numpy())
        return np.concatenate(encoded, axis=0) if encoded else np.empty((0, 0))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _valid_answer_prefix(
    cached: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    config_fingerprint: str,
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for idx, record in enumerate(cached[: len(rows)]):
        if str(record.get("sample_key", "")) != sample_key(rows[idx]):
            break
        if not str(record.get("openqa_answer", "")).strip():
            break
        if str(record.get("openqa_config_fingerprint", "")) != config_fingerprint:
            break
        valid.append(record)
    return valid


def openqa_config_fingerprint(args: argparse.Namespace) -> str:
    from model import DEFAULT_MODEL_IDS

    payload = {
        "model_type": args.model_type,
        "llm_model": args.llm_model or DEFAULT_MODEL_IDS[args.model_type],
        "backend": args.backend,
        "max_frames": args.max_frames,
        "frames_per_interval": args.frames_per_interval,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": OPENQA_PROMPT_TEMPLATE,
    }
    return query_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _write_evaluation(input_path: str, output_path: str, eval_output: str) -> None:
    from run_evaluation import _filter_subset, evaluate_longqa, write_results

    golden = _load_jsonl(input_path)
    predictions = _load_jsonl(output_path)
    if len(golden) != len(predictions):
        golden, predictions = _filter_subset(golden, predictions, "longqa")
    results = evaluate_longqa(golden, predictions)
    summary = write_results(eval_output, results)
    print(
        f"LongQA Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    print(f"Results written to {eval_output}")
    print(f"Summary written to {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate open LongQA answers and match them to MCQ options."
    )
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--output", default="output/egolongqa_openqa/predictions.jsonl")
    parser.add_argument("--answers-output", default=None)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frames-per-interval", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-type", default="qwen", choices=["qwen", "llama4"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--text-encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--text-encoder-device", default="cpu")
    parser.add_argument("--text-encoder-batch-size", type=int, default=128)
    parser.add_argument("--no-resume-answers", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    from model import (
        DEFAULT_MODEL_IDS,
        create_model,
        extract_frames,
        reset_prompt_token_stats,
        summarize_prompt_token_stats,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    answers_path = _resolve_path(
        args.answers_output
        or os.path.join(os.path.dirname(args.output), "open_answers.jsonl")
    )
    eval_output = _resolve_path(
        args.eval_output or os.path.join(os.path.dirname(args.output), "results.json")
    )

    rows = apply_subset(_load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(answers_path) or ".", exist_ok=True)

    config_fingerprint = openqa_config_fingerprint(args)
    cached = [] if args.no_resume_answers else _load_jsonl(answers_path)
    answers = _valid_answer_prefix(cached, rows, config_fingerprint)
    if len(answers) != len(cached):
        with open(answers_path, "w") as f:
            for record in answers:
                f.write(json.dumps(record) + "\n")
    if answers:
        print(f"Resuming open answers from {len(answers)}/{len(rows)} cached rows")

    if len(answers) < len(rows):
        model = create_model(
            args.model_type,
            args.llm_model,
            backend=args.backend,
            tp_size=args.tp,
            concurrency=args.concurrency,
            max_frames=args.max_frames,
        )
        reset_prompt_token_stats()
        mode = "a" if answers else "w"
        with model, open(answers_path, mode) as answer_f:
            for start in range(len(answers), len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                frames = [
                    extract_frames(
                        os.path.join(video_folder, str(row["video_path"])),
                        frames_per_interval=args.frames_per_interval,
                        max_frames=args.max_frames,
                    )
                    for row in batch
                ]
                messages = [
                    [{"role": "user", "content": build_openqa_prompt(row["question"])}]
                    for row in batch
                ]
                responses = model.generate_batch(
                    frames,
                    messages,
                    max_new_tokens=args.max_new_tokens,
                )
                for row, response in zip(batch, responses):
                    record = {
                        "sample_key": sample_key(row),
                        "video_path": row.get("video_path", ""),
                        "question": row.get("question", ""),
                        "openqa_answer": str(response).strip(),
                        "frames": args.max_frames,
                        "frames_per_interval": args.frames_per_interval,
                        "llm_model": args.llm_model or DEFAULT_MODEL_IDS[args.model_type],
                        "openqa_config_fingerprint": config_fingerprint,
                    }
                    answers.append(record)
                    answer_f.write(json.dumps(record) + "\n")
                    answer_f.flush()
                print(f"  Open-QA progress: {min(start + len(batch), len(rows))}/{len(rows)}")
        _print_context_summary(summarize_prompt_token_stats())
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    else:
        print("Open-answer cache complete; skipping VLM generation")

    option_sets: list[tuple[list[str], list[str]]] = []
    flat_texts: list[str] = []
    for row, record in zip(rows, answers):
        options = parse_mcq_options(row.get("mcq_options", ""))
        if len(options) != 4:
            raise ValueError(f"Expected four options for {sample_key(row)}")
        letters = sorted(options)
        option_sets.append((letters, [options[letter] for letter in letters]))
        flat_texts.append(str(record["openqa_answer"]))
        flat_texts.extend(options[letter] for letter in letters)

    encoder = MiniLMTextEncoder(
        args.text_encoder,
        device=args.text_encoder_device,
        batch_size=args.text_encoder_batch_size,
    )
    embeddings = encoder.encode(flat_texts)
    with open(output_path, "w") as pred_f:
        offset = 0
        for row, record, (letters, _option_texts) in zip(rows, answers, option_sets):
            answer_embedding = embeddings[offset]
            option_embeddings = embeddings[offset + 1 : offset + 5]
            offset += 5
            letter, scores, margin = select_nearest_option(
                answer_embedding,
                option_embeddings,
                letters,
            )
            pred = build_prediction_row(row, letter, prompt_variant="open_qa_similarity")
            pred.update(
                {
                    "openqa_answer": record["openqa_answer"],
                    "text_encoder": args.text_encoder,
                    "option_similarity_scores": scores,
                    "similarity_margin": margin,
                    "frames": args.max_frames,
                    "frames_per_interval": args.frames_per_interval,
                    "openqa_config_fingerprint": config_fingerprint,
                }
            )
            pred_f.write(json.dumps(pred) + "\n")
    print(f"Open answers written to {answers_path}")
    print(f"Predictions written to {output_path}")
    if not args.no_eval:
        _write_evaluation(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
