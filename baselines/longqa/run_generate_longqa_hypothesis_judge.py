#!/usr/bin/env python3
"""Run primary-aware, evidence-grounded arbitration on LongQA disagreements."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    normalize_answer,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_conditional import majority_answer
from run_generate_longqa_grounded import (
    CandidateFrame,
    TextImageGrounder,
    extract_candidate_frames,
    extract_frames_by_indices,
    load_jsonl,
    load_or_encode_grounder_features,
)
from run_generate_longqa_proofpack import (
    baseline_uniform_indices,
    compile_temporal_program,
    resize_frame_to_max_pixels,
)


MODES = ("hypothesis_existing", "hypothesis_fresh")
CANDIDATE_IDS = ("X", "Y", "Z")
VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}
CHECKS = {"PASS", "FAIL", "NOT_APPLICABLE", "INSUFFICIENT"}
REGIONS = {"BEFORE", "AFTER", "AROUND", "FULL_VIDEO", "NONE"}
RELATIONS = {
    "BEFORE", "AFTER", "FIRST", "LAST", "REVISIT", "STATE_CHANGE",
    "COUNT", "IDENTITY", "NONE",
}


def resolve(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def answer(row: dict[str, Any]) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def index_jsonl(path: str) -> dict[str, dict[str, Any]]:
    return {sample_key(row): row for row in load_jsonl(path)}


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def stage_cache(path: str, keys: list[str], fingerprint: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = load_jsonl(path)
    valid = []
    for index, record in enumerate(records[: len(keys)]):
        if record.get("sample_key") != keys[index] or record.get("fingerprint") != fingerprint:
            break
        valid.append(record)
    if len(valid) != len(records):
        write_jsonl(path, valid)
    return valid


def neutral_mapping(key: str, letters: list[str]) -> dict[str, str]:
    ordered = sorted(set(letters))
    digest = hashlib.sha1(key.encode()).digest()
    offset = int.from_bytes(digest[:4], "big") % len(ordered)
    ordered = ordered[offset:] + ordered[:offset]
    if digest[4] % 2:
        ordered.reverse()
    return dict(zip(CANDIDATE_IDS, ordered))


def candidate_payload(row: dict[str, Any], letters: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    options = parse_mcq_options(row["mcq_options"])
    mapping = neutral_mapping(sample_key(row), letters)
    texts = {candidate_id: options[letter] for candidate_id, letter in mapping.items()}
    return mapping, texts


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def planner_schema(candidate_ids: list[str]) -> dict[str, Any]:
    item = object_schema(
        {
            "hypothesis": {"type": "string"},
            "retrieval_queries": {
                "type": "array", "items": {"type": "string"},
                "minItems": 1, "maxItems": 2,
            },
        }
    )
    properties = {f"candidate_{cid.lower()}": item for cid in candidate_ids}
    properties.update(
        {
            "discriminative_clue": {"type": "string"},
            "relation": {"type": "string", "enum": sorted(RELATIONS)},
        }
    )
    return object_schema(properties)


def verifier_schema(candidate_ids: list[str]) -> dict[str, Any]:
    candidate = object_schema(
        {
            "verdict": {"type": "string", "enum": sorted(VERDICTS)},
            "supporting_frame_ids": {
                "type": "array", "items": {"type": "string"}, "maxItems": 3,
            },
            "contradicting_frame_ids": {
                "type": "array", "items": {"type": "string"}, "maxItems": 3,
            },
            "temporal_check": {"type": "string", "enum": sorted(CHECKS)},
            "identity_check": {"type": "string", "enum": sorted(CHECKS)},
            "coverage_check": {"type": "string", "enum": sorted(CHECKS)},
            "brief_evidence": {"type": "string"},
        }
    )
    refinement = object_schema(
        {
            "needed": {"type": "boolean"},
            "reason": {"type": "string"},
            "missing_visible_fact": {"type": "string"},
            "retrieval_queries": {
                "type": "array", "items": {"type": "string"}, "maxItems": 2,
            },
            "anchor_frame_ids": {
                "type": "array", "items": {"type": "string"}, "maxItems": 2,
            },
            "temporal_region": {"type": "string", "enum": sorted(REGIONS)},
        }
    )
    properties = {f"candidate_{cid.lower()}": candidate for cid in candidate_ids}
    properties["decisive_visible_fact"] = {"type": "string"}
    properties["refinement"] = refinement
    return object_schema(properties)


PLANNER_SYSTEM = """You create neutral visual-search queries for egocentric long-video question answering.

Do not decide which candidate is correct. Do not use vote counts, predictor identities, option letters, probabilities, or facts not stated in the question and candidate answers.

Rewrite every supplied candidate as a concrete, testable visual hypothesis. Identify the minimum observable fact that distinguishes them. Produce at most two short visual retrieval queries per candidate. For temporal questions, separately cover the reference event and the candidate-specific target event. Each query must fit comfortably within 64 tokens and describe something visible in a frame or short sequence. Return only schema-valid JSON."""


VERIFIER_SYSTEM = """You are an evidence verifier for egocentric long-video question answering.

Evaluate every candidate independently and symmetrically using only the supplied chronological images and frame IDs. SUPPORTED requires direct visible evidence for every decisive fact. CONTRADICTED requires direct visible incompatible evidence. Missing, unclear, blurred, occluded, or unsampled evidence is INSUFFICIENT, never CONTRADICTED.

For temporal questions, cite both the reference and target event and verify their timestamp order. FIRST and LAST require adequate global coverage. Identity claims require evidence that it is the same object, person, or place. Every decisive part of a multi-part answer must be supported. Every SUPPORTED or CONTRADICTED verdict must cite supplied frame IDs.

If refinement is available and evidence is not decisive, request exactly one missing visible fact with one or two short neutral retrieval queries. BEFORE, AFTER, and AROUND requests must cite supplied anchor frame IDs. Do not request refinement when exactly one candidate is supported and every alternative is contradicted. Reason internally and return only concise schema-valid JSON."""


def planner_messages(row: dict[str, Any], texts: dict[str, str]) -> list[dict[str, str]]:
    candidates = "\n".join(f"Candidate {cid}: {text}" for cid, text in texts.items())
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"Question: {row['question']}\n{candidates}"},
    ]


def fallback_plan(row: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    program = compile_temporal_program(row["question"])
    relation = program.operator if program.operator in RELATIONS else "NONE"
    result = {}
    for candidate_id, text in texts.items():
        target = program.target if program.target and program.target != row["question"] else ""
        queries = [text]
        if target:
            queries.append(str(target))
        result[f"candidate_{candidate_id.lower()}"] = {
            "hypothesis": text,
            "retrieval_queries": queries[:2],
        }
    result["discriminative_clue"] = str(program.target or row["question"])
    result["relation"] = relation
    return result


def verifier_messages(
    row: dict[str, Any], plan: dict[str, Any], evidence: list[dict[str, Any]],
    round_number: int, refinement_available: bool,
) -> list[dict[str, str]]:
    hypotheses = []
    for candidate_id in CANDIDATE_IDS:
        key = f"candidate_{candidate_id.lower()}"
        if key in plan:
            hypotheses.append(f"Candidate {candidate_id}: {plan[key]['hypothesis']}")
    index = "\n".join(
        f"{item['display_id']} = {float(item['timestamp']):.3f} s"
        for item in evidence
    )
    user = (
        f"Question: {row['question']}\n\n" + "\n".join(hypotheses)
        + f"\nDiscriminative clue: {plan['discriminative_clue']}"
        + f"\nVerification round: {round_number}"
        + f"\nRefinement available: {'true' if refinement_available else 'false'}"
        + f"\n\nEvidence index:\n{index}"
    )
    return [{"role": "system", "content": VERIFIER_SYSTEM}, {"role": "user", "content": user}]


def validate_report(
    report: dict[str, Any], candidate_ids: list[str], evidence: list[dict[str, Any]],
    refinement_available: bool,
) -> tuple[bool, str]:
    supplied = {item["display_id"] for item in evidence}
    for candidate_id in candidate_ids:
        record = report.get(f"candidate_{candidate_id.lower()}")
        if not isinstance(record, dict):
            return False, f"missing candidate {candidate_id}"
        verdict = record.get("verdict")
        if verdict not in VERDICTS:
            return False, f"invalid verdict for {candidate_id}"
        supporting = record.get("supporting_frame_ids", [])
        contradicting = record.get("contradicting_frame_ids", [])
        if not set(supporting + contradicting).issubset(supplied):
            return False, f"invalid citation for {candidate_id}"
        if verdict == "SUPPORTED" and not supporting:
            return False, f"unsupported SUPPORTED verdict for {candidate_id}"
        if verdict == "CONTRADICTED" and not contradicting:
            return False, f"unsupported CONTRADICTED verdict for {candidate_id}"
        if verdict == "SUPPORTED":
            checks = [record.get(name) for name in ("temporal_check", "identity_check", "coverage_check")]
            if any(value not in {"PASS", "NOT_APPLICABLE"} for value in checks):
                return False, f"failed check on SUPPORTED candidate {candidate_id}"
    refinement = report.get("refinement")
    if not isinstance(refinement, dict):
        return False, "missing refinement record"
    needed = bool(refinement.get("needed"))
    if needed and not refinement_available:
        return False, "refinement requested on final round"
    if needed:
        queries = refinement.get("retrieval_queries") or []
        region = refinement.get("temporal_region")
        anchors = refinement.get("anchor_frame_ids") or []
        if not refinement.get("missing_visible_fact") or not (1 <= len(queries) <= 2):
            return False, "incomplete refinement request"
        if region not in REGIONS - {"NONE"}:
            return False, "invalid refinement region"
        if not set(anchors).issubset(supplied):
            return False, "invalid refinement anchor"
        if region in {"BEFORE", "AFTER", "AROUND"} and not anchors:
            return False, "anchored refinement has no anchor"
    return True, "valid"


def decisive_answer(report: dict[str, Any], mapping: dict[str, str]) -> str | None:
    supported = []
    contradicted = []
    for candidate_id in mapping:
        verdict = report[f"candidate_{candidate_id.lower()}"]["verdict"]
        if verdict == "SUPPORTED":
            supported.append(candidate_id)
        elif verdict == "CONTRADICTED":
            contradicted.append(candidate_id)
    if len(supported) == 1 and len(contradicted) == len(mapping) - 1:
        return mapping[supported[0]]
    return None


def video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0.0, 0
        return float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def ranked_proof(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "pivot": 0, "directional_target": 0, "bridge": 1,
        "pivot_context": 1, "directional_target_context": 1,
        "event_center": 2, "event_center_context": 2, "uniform": 3,
        "anchor": 4, "coverage_fill": 5, "semantic_boundary": 6,
    }
    return sorted(
        selected,
        key=lambda item: (
            priority.get(str(item.get("source", "")), 7),
            -(float(item.get("score")) if item.get("score") is not None else -1e9),
        ),
    )


def nms_centers(candidates: list[CandidateFrame], scores: list[float], count: int, seconds: float) -> list[int]:
    chosen = []
    for position in sorted(range(len(scores)), key=lambda index: (-scores[index], candidates[index].timestamp)):
        if all(abs(candidates[position].timestamp - candidates[other].timestamp) >= seconds for other in chosen):
            chosen.append(position)
            if len(chosen) >= count:
                break
    return chosen


def clip_grounder_query(grounder: TextImageGrounder, text: str) -> str:
    tokenizer = grounder.processor.tokenizer
    config = getattr(grounder.model.config, "text_config", grounder.model.config)
    limit = int(getattr(config, "max_position_embeddings", 64))
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=limit)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def retrieve_by_candidate(
    grounder: TextImageGrounder, candidates: list[CandidateFrame], features: np.ndarray,
    plan: dict[str, Any], candidate_ids: list[str], centers_per_candidate: int,
    radius: int, nms_seconds: float,
) -> dict[str, list[dict[str, Any]]]:
    selected = {}
    for candidate_id in candidate_ids:
        queries = plan[f"candidate_{candidate_id.lower()}"]["retrieval_queries"]
        safe_queries = [clip_grounder_query(grounder, str(query)) for query in queries]
        score_sets = [grounder.score_embeddings(query, features) for query in safe_queries]
        scores = [max(values) for values in zip(*score_sets)]
        centers = nms_centers(candidates, scores, centers_per_candidate, nms_seconds)
        records = []
        seen = set()
        for center in centers:
            for position in range(max(0, center - radius), min(len(candidates), center + radius + 1)):
                frame = candidates[position]
                if frame.index in seen:
                    continue
                seen.add(frame.index)
                records.append(
                    {
                        "frame_index": frame.index,
                        "timestamp": frame.timestamp,
                        "score": float(scores[position]),
                        "source": "fresh_center" if position == center else "fresh_context",
                        "candidate_id": candidate_id,
                    }
                )
        selected[candidate_id] = records
    return selected


def add_unique(target: list[dict[str, Any]], seen: set[int], records: list[dict[str, Any]], limit: int | None = None) -> None:
    if limit is not None and limit <= 0:
        return
    added = 0
    for record in records:
        index = int(record["frame_index"])
        if index in seen:
            continue
        seen.add(index)
        target.append(record)
        added += 1
        if limit is not None and added >= limit:
            break


def build_evidence_pack(
    row: dict[str, Any], predictor_answers: list[str], mapping: dict[str, str],
    proof_selected: list[dict[str, Any]], fresh: dict[str, list[dict[str, Any]]],
    total_frames: int, fps: float, mode: str, max_frames: int,
) -> list[dict[str, Any]]:
    count = len(mapping)
    prior_quota = 8 if count == 2 else 5
    uniform_quota = 24 if count == 2 else 25
    fresh_quota = 12 if count == 2 else 8
    pivot_answer, uniform_answer, _tertiary_answer = predictor_answers
    proof = ranked_proof(proof_selected)
    uniform = baseline_uniform_indices(total_frames, max_frames)
    evidence: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate_id, letter in mapping.items():
        candidate_prior = []
        if letter == pivot_answer:
            candidate_prior.extend(
                {**item, "source": f"prior_pivot_{candidate_id}"} for item in proof
            )
        if letter == uniform_answer:
            candidate_prior.extend(
                {
                    "frame_index": index, "timestamp": index / fps,
                    "score": None, "source": f"prior_uniform_{candidate_id}",
                }
                for index in uniform
            )
        add_unique(evidence, seen, candidate_prior, prior_quota)
    if mode == "hypothesis_fresh":
        for candidate_id in mapping:
            add_unique(evidence, seen, fresh.get(candidate_id, []), fresh_quota)
    anchors = baseline_uniform_indices(total_frames, uniform_quota + (24 if mode == "hypothesis_existing" else 0))
    add_unique(
        evidence,
        seen,
        [{"frame_index": index, "timestamp": index / fps, "score": None, "source": "uniform_anchor"} for index in anchors],
    )
    fill = baseline_uniform_indices(total_frames, max_frames * 2)
    add_unique(
        evidence,
        seen,
        [{"frame_index": index, "timestamp": index / fps, "score": None, "source": "coverage_fill"} for index in fill],
        max_frames - len(evidence),
    )
    evidence = sorted(evidence[:max_frames], key=lambda item: (float(item["timestamp"]), int(item["frame_index"])))
    for index, item in enumerate(evidence, start=1):
        item["display_id"] = f"F{index:03d}"
    return evidence


def refinement_candidates(
    video_path: str, evidence: list[dict[str, Any]], refinement: dict[str, Any],
    local_count: int, full_count: int, around_seconds: float,
) -> list[CandidateFrame]:
    fps, total = video_metadata(video_path)
    by_id = {item["display_id"]: item for item in evidence}
    anchors = [int(by_id[value]["frame_index"]) for value in refinement.get("anchor_frame_ids", []) if value in by_id]
    region = refinement["temporal_region"]
    if region == "AROUND":
        low = max(0, min(anchors) - round(around_seconds * fps))
        high = min(total - 1, max(anchors) + round(around_seconds * fps))
        count = local_count
    elif region == "BEFORE":
        low, high, count = 0, max(anchors) - 1, local_count
    elif region == "AFTER":
        low, high, count = min(anchors) + 1, total - 1, local_count
    else:
        low, high, count = 0, total - 1, full_count
    if high < low:
        return []
    positions = [round(low + (index + 0.5) * (high - low + 1) / count) for index in range(count)]
    shown = {int(item["frame_index"]) for item in evidence}
    positions = sorted({min(high, max(low, index)) for index in positions} - shown)
    images = extract_frames_by_indices(video_path, positions)
    if len(images) != len(positions):
        return []
    return [CandidateFrame(index, index / fps, image) for index, image in zip(positions, images)]


def build_refined_pack(
    first: list[dict[str, Any]], report: dict[str, Any], new_frames: list[dict[str, Any]],
    total_frames: int, fps: float, max_frames: int,
) -> list[dict[str, Any]]:
    cited_ids = set()
    for key, value in report.items():
        if not key.startswith("candidate_") or not isinstance(value, dict):
            continue
        cited_ids.update(value.get("supporting_frame_ids", []))
        cited_ids.update(value.get("contradicting_frame_ids", []))
    cited = [item for item in first if item["display_id"] in cited_ids]
    remainder = [item for item in first if item["display_id"] not in cited_ids]
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    add_unique(result, seen, cited)
    add_unique(result, seen, new_frames)
    add_unique(result, seen, remainder, max_frames - len(result))
    fill = baseline_uniform_indices(total_frames, max_frames * 2)
    add_unique(
        result, seen,
        [{"frame_index": index, "timestamp": index / fps, "score": None, "source": "refinement_fill"} for index in fill],
        max_frames - len(result),
    )
    result = sorted(result[:max_frames], key=lambda item: (float(item["timestamp"]), int(item["frame_index"])))
    for index, item in enumerate(result, start=1):
        item["display_id"] = f"F{index:03d}"
    return result


def release_cuda() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def fingerprint(args: argparse.Namespace) -> str:
    ignored = {"output", "audit_output", "eval_output", "no_resume"}
    payload = {key: value for key, value in vars(args).items() if key not in ignored}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--secondary-predictions", required=True)
    parser.add_argument("--tertiary-predictions", required=True)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--grounder-model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--grounder-cache-dir", required=True)
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--centers-per-candidate", type=int, default=4)
    parser.add_argument("--temporal-nms-seconds", type=float, default=10.0)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--max-evidence-refinements", type=int, choices=[0, 1], default=0)
    parser.add_argument("--refinement-frame-budget", type=int, default=12)
    parser.add_argument("--refinement-local-candidate-frames", type=int, default=64)
    parser.add_argument("--refinement-full-candidate-frames", type=int, default=128)
    parser.add_argument("--refinement-around-seconds", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=451584)
    parser.add_argument("--planner-max-new-tokens", type=int, default=256)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=640)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "hypothesis_existing" and args.max_evidence_refinements:
        parser.error("hypothesis_existing requires --max-evidence-refinements 0")
    return args


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary
    from run_generate_longqa_grounded import _run_eval

    args = parse_args()
    for name in (
        "input", "video_folder", "subset_file", "primary_predictions",
        "secondary_predictions", "tertiary_predictions", "fallback_predictions",
        "proofpack", "proofpack_reference", "grounder_cache_dir", "output",
        "audit_output", "eval_output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    predictor_sets = [
        index_jsonl(args.primary_predictions), index_jsonl(args.secondary_predictions),
        index_jsonl(args.tertiary_predictions),
    ]
    fallback = index_jsonl(args.fallback_predictions)
    proof_reference = load_jsonl(args.proofpack_reference)
    proof_rows = load_jsonl(args.proofpack)
    if len(proof_reference) != len(proof_rows):
        raise RuntimeError(
            f"Proof-pack alignment mismatch: {len(proof_rows)} records for "
            f"{len(proof_reference)} references"
        )
    proof = {sample_key(reference): record for reference, record in zip(proof_reference, proof_rows)}
    required = {sample_key(row) for row in rows}
    for label, indexed in [("fallback", fallback), ("proofpack", proof)] + [
        (f"predictor {index + 1}", values) for index, values in enumerate(predictor_sets)
    ]:
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} required rows")

    cases = []
    for row in rows:
        key = sample_key(row)
        predictor_answers = [answer(values[key]) for values in predictor_sets]
        if len(set(predictor_answers)) <= 1:
            continue
        mapping, texts = candidate_payload(row, predictor_answers)
        cases.append({
            "row": row, "sample_key": key, "predictor_answers": predictor_answers,
            "mapping": mapping, "texts": texts, "fallback": answer(fallback[key]),
        })
    fp = fingerprint(args)
    stage_dir = os.path.join(os.path.dirname(args.output), "hypothesis_stages")
    os.makedirs(stage_dir, exist_ok=True)
    case_keys = [case["sample_key"] for case in cases]
    print(f"Hypothesis judge: mode={args.mode}, rows={len(rows)}, disagreements={len(cases)}, fingerprint={fp}")

    plan_path = os.path.join(stage_dir, "plans.jsonl")
    plans = [] if args.no_resume else stage_cache(plan_path, case_keys, fp)
    if len(plans) < len(cases):
        model = VLLMModel(args.llm_model, tp_size=1, concurrency=args.concurrency, max_frames=0, model_type="qwen")
        with model, open(plan_path, "a" if plans else "w") as handle:
            for case in cases[len(plans):]:
                candidate_ids = list(case["mapping"])
                try:
                    report = model.generate_json([], planner_messages(case["row"], case["texts"]), planner_schema(candidate_ids), "longqa_hypothesis_plan", args.planner_max_new_tokens)
                    failed = False
                    error = None
                except Exception as exception:
                    report = fallback_plan(case["row"], case["texts"])
                    failed = True
                    error = str(exception)
                record = {"sample_key": case["sample_key"], "fingerprint": fp, "plan": report, "planner_fallback": failed, "planner_error": error}
                plans.append(record)
                handle.write(json.dumps(record) + "\n"); handle.flush()
                print(f"  Planner: {len(plans)}/{len(cases)}")
        release_cuda()
    plans_by_key = {record["sample_key"]: record for record in plans}

    retrieval_path = os.path.join(stage_dir, "retrieval.jsonl")
    retrievals = [] if args.no_resume else stage_cache(retrieval_path, case_keys, fp)
    if len(retrievals) < len(cases):
        grounder = None
        if args.mode == "hypothesis_fresh":
            grounder = TextImageGrounder(args.grounder_model, device="cuda", batch_size=16, dtype="bfloat16")
        with open(retrieval_path, "a" if retrievals else "w") as handle:
            for case in cases[len(retrievals):]:
                key = case["sample_key"]
                video_path = os.path.join(args.video_folder, str(case["row"]["video_path"]))
                fps, total_frames = video_metadata(video_path)
                if args.mode == "hypothesis_fresh":
                    candidates, features, cache_hit = load_or_encode_grounder_features(video_path, args.candidate_frames, grounder, args.grounder_cache_dir)
                    fresh = retrieve_by_candidate(grounder, candidates, features, plans_by_key[key]["plan"], list(case["mapping"]), args.centers_per_candidate, args.neighborhood_radius, args.temporal_nms_seconds)
                else:
                    fresh, cache_hit = {}, False
                evidence = build_evidence_pack(case["row"], case["predictor_answers"], case["mapping"], proof[key]["selected"], fresh, total_frames, fps, args.mode, args.max_frames)
                record = {"sample_key": key, "fingerprint": fp, "evidence": evidence, "fresh_by_candidate": fresh, "feature_cache_hit": cache_hit}
                retrievals.append(record); handle.write(json.dumps(record) + "\n"); handle.flush()
                print(f"  Retrieval: {len(retrievals)}/{len(cases)}")
        del grounder
        release_cuda()
    retrieval_by_key = {record["sample_key"]: record for record in retrievals}

    verify1_path = os.path.join(stage_dir, "verify_round1.jsonl")
    verify1 = [] if args.no_resume else stage_cache(verify1_path, case_keys, fp)
    reset_prompt_token_stats()
    if len(verify1) < len(cases):
        model = VLLMModel(args.llm_model, tp_size=1, concurrency=args.concurrency, max_frames=args.max_frames, model_type="qwen")
        with model, open(verify1_path, "a" if verify1 else "w") as handle:
            for case in cases[len(verify1):]:
                key = case["sample_key"]
                evidence = retrieval_by_key[key]["evidence"]
                indices = [int(item["frame_index"]) for item in evidence]
                video_path = os.path.join(args.video_folder, str(case["row"]["video_path"]))
                frames = [resize_frame_to_max_pixels(frame, args.max_pixels) for frame in extract_frames_by_indices(video_path, indices)]
                try:
                    report = model.generate_json(frames, verifier_messages(case["row"], plans_by_key[key]["plan"], evidence, 1, args.max_evidence_refinements == 1), verifier_schema(list(case["mapping"])), "longqa_hypothesis_verify", args.verifier_max_new_tokens)
                    valid, validation = validate_report(report, list(case["mapping"]), evidence, args.max_evidence_refinements == 1)
                    error = None
                except Exception as exception:
                    report, valid, validation, error = {}, False, "generation failure", str(exception)
                record = {"sample_key": key, "fingerprint": fp, "report": report, "valid": valid, "validation": validation, "error": error}
                verify1.append(record); handle.write(json.dumps(record) + "\n"); handle.flush()
                print(f"  Verify round 1: {len(verify1)}/{len(cases)}")
        _print_context_summary(summarize_prompt_token_stats())
        release_cuda()
    verify1_by_key = {record["sample_key"]: record for record in verify1}

    refinement_by_key: dict[str, dict[str, Any]] = {}
    verify2_by_key: dict[str, dict[str, Any]] = {}
    refine_cases = []
    if args.max_evidence_refinements:
        for case in cases:
            first = verify1_by_key[case["sample_key"]]
            if not first["valid"] or decisive_answer(first["report"], case["mapping"]):
                continue
            if first["report"].get("refinement", {}).get("needed"):
                refine_cases.append(case)
        refine_keys = [case["sample_key"] for case in refine_cases]
        refinement_path = os.path.join(stage_dir, "refinement.jsonl")
        refinements = [] if args.no_resume else stage_cache(refinement_path, refine_keys, fp)
        if len(refinements) < len(refine_cases):
            grounder = TextImageGrounder(args.grounder_model, device="cuda", batch_size=16, dtype="bfloat16")
            with open(refinement_path, "a" if refinements else "w") as handle:
                for case in refine_cases[len(refinements):]:
                    key = case["sample_key"]
                    video_path = os.path.join(args.video_folder, str(case["row"]["video_path"]))
                    first_evidence = retrieval_by_key[key]["evidence"]
                    request = verify1_by_key[key]["report"]["refinement"]
                    candidates = refinement_candidates(video_path, first_evidence, request, args.refinement_local_candidate_frames, args.refinement_full_candidate_frames, args.refinement_around_seconds)
                    if candidates:
                        features = grounder.encode_images(candidates)
                        queries = [clip_grounder_query(grounder, query) for query in request["retrieval_queries"]]
                        score_sets = [grounder.score_embeddings(query, features) for query in queries]
                        scores = [max(values) for values in zip(*score_sets)]
                        centers = nms_centers(candidates, scores, 4, args.temporal_nms_seconds)
                        new_frames = []
                        seen = set()
                        for center in centers:
                            for position in range(max(0, center - 1), min(len(candidates), center + 2)):
                                frame = candidates[position]
                                if frame.index in seen: continue
                                seen.add(frame.index)
                                new_frames.append({"frame_index": frame.index, "timestamp": frame.timestamp, "score": float(scores[position]), "source": "refinement"})
                                if len(new_frames) >= args.refinement_frame_budget: break
                            if len(new_frames) >= args.refinement_frame_budget: break
                    else:
                        new_frames = []
                    fps, total_frames = video_metadata(video_path)
                    evidence = build_refined_pack(first_evidence, verify1_by_key[key]["report"], new_frames, total_frames, fps, args.max_frames) if new_frames else []
                    record = {"sample_key": key, "fingerprint": fp, "new_frames": new_frames, "evidence": evidence}
                    refinements.append(record); handle.write(json.dumps(record) + "\n"); handle.flush()
                    print(f"  Refinement retrieval: {len(refinements)}/{len(refine_cases)}")
            del grounder
            release_cuda()
        refinement_by_key = {record["sample_key"]: record for record in refinements}

        verify2_cases = [case for case in refine_cases if refinement_by_key[case["sample_key"]]["evidence"]]
        verify2_keys = [case["sample_key"] for case in verify2_cases]
        verify2_path = os.path.join(stage_dir, "verify_round2.jsonl")
        verify2 = [] if args.no_resume else stage_cache(verify2_path, verify2_keys, fp)
        if len(verify2) < len(verify2_cases):
            model = VLLMModel(args.llm_model, tp_size=1, concurrency=args.concurrency, max_frames=args.max_frames, model_type="qwen")
            with model, open(verify2_path, "a" if verify2 else "w") as handle:
                for case in verify2_cases[len(verify2):]:
                    key = case["sample_key"]
                    evidence = refinement_by_key[key]["evidence"]
                    video_path = os.path.join(args.video_folder, str(case["row"]["video_path"]))
                    frames = [resize_frame_to_max_pixels(frame, args.max_pixels) for frame in extract_frames_by_indices(video_path, [int(item["frame_index"]) for item in evidence])]
                    try:
                        report = model.generate_json(frames, verifier_messages(case["row"], plans_by_key[key]["plan"], evidence, 2, False), verifier_schema(list(case["mapping"])), "longqa_hypothesis_verify_final", args.verifier_max_new_tokens)
                        valid, validation = validate_report(report, list(case["mapping"]), evidence, False)
                        error = None
                    except Exception as exception:
                        report, valid, validation, error = {}, False, "generation failure", str(exception)
                    record = {"sample_key": key, "fingerprint": fp, "report": report, "valid": valid, "validation": validation, "error": error}
                    verify2.append(record); handle.write(json.dumps(record) + "\n"); handle.flush()
                    print(f"  Verify round 2: {len(verify2)}/{len(verify2_cases)}")
            release_cuda()
        verify2_by_key = {record["sample_key"]: record for record in verify2}

    case_by_key = {case["sample_key"]: case for case in cases}
    predictions = []
    audits = []
    for row in rows:
        key = sample_key(row)
        fallback_answer = answer(fallback[key])
        if key not in case_by_key:
            selected, decision_round, reason = fallback_answer, 0, "predictor_agreement"
        else:
            case = case_by_key[key]
            first = verify1_by_key[key]
            first_decision = decisive_answer(first["report"], case["mapping"]) if first["valid"] else None
            second = verify2_by_key.get(key)
            second_decision = decisive_answer(second["report"], case["mapping"]) if second and second["valid"] else None
            if first_decision:
                selected, decision_round, reason = first_decision, 1, "decisive_first_report"
            elif second_decision:
                selected, decision_round, reason = second_decision, 2, "decisive_refined_report"
            else:
                selected, decision_round, reason = fallback_answer, 0, "primary_fallback"
        prediction = build_prediction_row(row, selected, f"conditional_{args.mode}")
        prediction.update({"hypothesis_fingerprint": fp, "hypothesis_mode": args.mode, "hypothesis_applied": key in case_by_key, "primary_fallback": fallback_answer, "decision_round": decision_round, "decision_reason": reason})
        predictions.append(prediction)
        if key in case_by_key:
            case = case_by_key[key]
            audits.append({
                "schema": 1, "sample_key": key, "video_path": row.get("video_path"),
                "fingerprint": fp, "mode": args.mode,
                "predictor_answers": case["predictor_answers"],
                "vote_counts": dict(Counter(case["predictor_answers"])),
                "primary_fallback": fallback_answer, "candidate_mapping": case["mapping"],
                "planner": plans_by_key[key], "round_one_retrieval": retrieval_by_key[key],
                "round_one_verification": verify1_by_key[key],
                "refinement": refinement_by_key.get(key),
                "round_two_verification": verify2_by_key.get(key),
                "selected_answer": selected, "decision_round": decision_round,
                "decision_reason": reason, "override": selected != fallback_answer,
            })
    write_jsonl(args.output, predictions)
    write_jsonl(args.audit_output, audits)
    _run_eval(args.input, args.output, args.eval_output)
    print(f"Wrote {len(predictions)} predictions and {len(audits)} disagreement audits")


if __name__ == "__main__":
    main()
