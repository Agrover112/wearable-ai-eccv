#!/usr/bin/env python3
"""Convert EgoLongQA questions into structured HieraMamba query programs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt",
        default="src/mamba/wheels/HIERAMAMBA_QUERY_CONVERSION_PROMPT.md",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser.parse_args()


def prompt_text(path: Path) -> str:
    content = path.read_text()
    match = re.search(r"```text\s*(.*?)\s*```", content, re.DOTALL)
    if not match:
        raise ValueError(f"No ```text prompt block found in {path}")
    return match.group(1)


def first_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "queries" in value:
            return value
    raise ValueError(f"Model did not return a query-program JSON object: {text!r}")


def validate(program: dict) -> None:
    queries = program.get("queries")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 4:
        raise ValueError("query program must contain 1-4 queries")
    for index, query in enumerate(queries, 1):
        if query.get("id") != f"q{index}" or not query.get("text"):
            raise ValueError("queries must have sequential q1... IDs and non-empty text")


def main():
    args = parse_args()
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval()
    instruction = prompt_text(Path(args.prompt))
    document = json.loads(Path(args.manifest).read_text())
    for position, sample in enumerate(document["samples"], 1):
        messages = [{
            "role": "user",
            "content": instruction + "\n\nQUESTION:\n" + sample["question"] + "\n\nOUTPUT:",
        }]
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor.tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        answer = processor.tokenizer.decode(
            generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        program = first_json_object(answer)
        validate(program)
        sample["hieramamba_program"] = program
        Path(args.output).write_text(json.dumps(document, indent=2))
        print(f"[{position}/{len(document['samples'])}] {sample['video_path']}", flush=True)


if __name__ == "__main__":
    main()
