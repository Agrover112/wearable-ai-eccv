#!/usr/bin/env python3
"""Generate LongQA predictions with CLIP/SigLIP temporal grounding.

This experimental entry point is intentionally separate from the official
starter-kit baselines. It keeps the final VLM call unchanged, but replaces
uniform final-frame sampling with:

  1. sample a larger uniform candidate pool from the full video,
  2. score each frame against the question/options with a CLIP-like model,
  3. keep top-K relevant frames plus optional uniform anchors/windows,
  4. feed the selected frames to the VLM and evaluate normally.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass

from longqa_utils import (
    PROMPT_VARIANTS,
    RETRIEVAL_QUERY_MODES,
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    parse_mcq_options,
    query_hash,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateFrame:
    index: int
    timestamp: float
    image: object


@dataclass(frozen=True)
class SelectedFrame:
    candidate: CandidateFrame
    score: float | None
    source: str


@dataclass(frozen=True)
class GroundingQuery:
    label: str
    text: str


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_if_exists(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        return []
    return load_jsonl(path)


def _uniform_positions(total: int, n: int) -> list[int]:
    if total <= 0 or n <= 0:
        return []
    n = min(n, total)
    if n == 1:
        return [total // 2]
    positions = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    deduped: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        pos = max(0, min(int(pos), total - 1))
        if pos not in seen:
            seen.add(pos)
            deduped.append(pos)
    return deduped


def extract_candidate_frames(video_path: str, num_candidates: int) -> list[CandidateFrame]:
    import cv2
    from PIL import Image

    if not os.path.exists(video_path):
        logger.warning("Video not found: %s", video_path)
        return []

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logger.warning("Could not open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            logger.warning("Invalid video metadata: %s", video_path)
            return []

        candidates: list[CandidateFrame] = []
        for frame_idx in _uniform_positions(total_frames, num_candidates):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            candidates.append(
                CandidateFrame(
                    index=frame_idx,
                    timestamp=frame_idx / fps,
                    image=Image.fromarray(frame_rgb),
                )
            )
        return candidates
    finally:
        cap.release()


def extract_frames_by_indices(video_path: str, frame_indices: list[int]) -> list[object]:
    import cv2
    from PIL import Image

    if not os.path.exists(video_path):
        logger.warning("Video not found: %s", video_path)
        return []

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logger.warning("Could not open video: %s", video_path)
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[object] = []
        for frame_idx in frame_indices:
            if frame_idx < 0 or frame_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        return frames
    finally:
        cap.release()


class TextImageGrounder:
    """CLIP/SigLIP text-image similarity scorer."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def score(self, text: str, frames: list[CandidateFrame]) -> list[float]:
        import torch
        import torch.nn.functional as F

        if not frames:
            return []

        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(frames), self.batch_size):
                batch = frames[start : start + self.batch_size]
                images = [frame.image for frame in batch]
                inputs = self.processor(
                    text=[text],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                outputs = self.model(**inputs)
                logits = getattr(outputs, "logits_per_image", None)
                if logits is not None:
                    batch_scores = logits[:, 0]
                elif hasattr(self.model, "get_image_features") and hasattr(
                    self.model, "get_text_features"
                ):
                    image_features = self.model.get_image_features(
                        pixel_values=inputs["pixel_values"]
                    )
                    text_kwargs = {
                        k: v
                        for k, v in inputs.items()
                        if k in ("input_ids", "attention_mask", "token_type_ids")
                    }
                    text_features = self.model.get_text_features(**text_kwargs)
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    batch_scores = image_features @ text_features[0]
                else:
                    raise RuntimeError(
                        "Grounding model does not expose logits_per_image or "
                        "get_image_features/get_text_features."
                    )
                scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())
        return scores


def build_grounding_queries(
    row: dict[str, object],
    retrieval_query_mode: str,
) -> list[GroundingQuery]:
    question = row["question"]
    options_text = row["mcq_options"]
    if retrieval_query_mode == "question":
        return [
            GroundingQuery(
                "question",
                "Find video frames relevant to answering this question.\n"
                f"Question: {question}",
            )
        ]
    if retrieval_query_mode == "question_options":
        return [
            GroundingQuery(
                "question_options",
                "Find video frames relevant to answering this multiple-choice question.\n"
                f"Question: {question}\n"
                f"Options:\n{options_text}",
            )
        ]
    if retrieval_query_mode == "question_temporal_boost":
        return [
            GroundingQuery(
                "question_temporal_boost",
                "Find frames relevant to answering this question. If the question "
                "asks about temporal order, prioritize the event before or after "
                "the anchor event.\n"
                f"Question: {question}\n"
                f"Options:\n{options_text}",
            )
        ]
    if retrieval_query_mode == "per_option_union":
        options = parse_mcq_options(options_text)
        if not options:
            return build_grounding_queries(row, "question_options")
        return [
            GroundingQuery(
                f"option_{letter}",
                f"Find frames relevant to this candidate answer.\n"
                f"Question: {question}\n"
                f"Candidate answer {letter}: {option}",
            )
            for letter, option in sorted(options.items())
        ]
    raise ValueError(f"Unknown retrieval_query_mode={retrieval_query_mode!r}")


def _temporal_nms(
    ranked: list[int],
    candidates: list[CandidateFrame],
    top_k: int,
    nms_seconds: float,
    candidate_limit: int,
) -> tuple[list[int], int]:
    if top_k <= 0:
        return [], 0
    if nms_seconds <= 0:
        return ranked[:top_k], 0
    pool = ranked[: max(candidate_limit, top_k)]
    accepted: list[int] = []
    suppressed = 0
    for idx in pool:
        ts = candidates[idx].timestamp
        if all(abs(ts - candidates[other].timestamp) >= nms_seconds for other in accepted):
            accepted.append(idx)
            if len(accepted) >= top_k:
                break
        else:
            suppressed += 1
    if len(accepted) < top_k:
        for idx in ranked:
            if idx not in accepted:
                accepted.append(idx)
                if len(accepted) >= top_k:
                    break
    return accepted[:top_k], suppressed


def select_grounded_frames(
    candidates: list[CandidateFrame],
    scores: list[float],
    top_k: int,
    anchor_k: int,
    window_radius: int,
    final_max_frames: int,
    temporal_nms_seconds: float = 0.0,
    temporal_nms_candidates: int | None = None,
    source_prefix: str = "retrieved",
) -> tuple[list[SelectedFrame], dict[str, object]]:
    if not candidates:
        return [], {"temporal_nms_suppressed": 0}
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")

    selected: dict[int, SelectedFrame] = {}
    ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    if temporal_nms_candidates is None:
        temporal_nms_candidates = max(top_k * 4, top_k)
    retrieved_indices, suppressed = _temporal_nms(
        ranked,
        candidates,
        top_k=max(top_k, 0),
        nms_seconds=temporal_nms_seconds,
        candidate_limit=temporal_nms_candidates,
    )

    def add(idx: int, source: str) -> None:
        if idx < 0 or idx >= len(candidates):
            return
        if idx not in selected:
            selected[idx] = SelectedFrame(candidates[idx], scores[idx], source)

    for idx in retrieved_indices:
        if window_radius > 0:
            for j in range(idx - window_radius, idx + window_radius + 1):
                add(j, f"{source_prefix}_window" if j != idx else source_prefix)
        else:
            add(idx, source_prefix)

    for idx in _uniform_positions(len(candidates), anchor_k):
        add(idx, "anchor")

    ordered = sorted(selected.values(), key=lambda sf: sf.candidate.timestamp)
    if len(ordered) <= final_max_frames:
        return ordered, {"temporal_nms_suppressed": suppressed}

    # Preserve temporal order but prioritize retrieved frames over anchors.
    priority = {
        source_prefix: 0,
        f"{source_prefix}_window": 1,
        "retrieved": 0,
        "retrieved_window": 1,
        "anchor": 2,
    }
    keep = sorted(
        ordered,
        key=lambda sf: (
            priority.get(sf.source, 3),
            -(sf.score if sf.score is not None else -1e9),
        ),
    )[:final_max_frames]
    keep_ids = {sf.candidate.index for sf in keep}
    return [sf for sf in ordered if sf.candidate.index in keep_ids], {
        "temporal_nms_suppressed": suppressed
    }


def score_queries(
    grounder: TextImageGrounder,
    queries: list[GroundingQuery],
    candidates: list[CandidateFrame],
) -> tuple[list[float], list[str], list[dict[str, object]]]:
    all_scores: list[list[float]] = []
    query_meta: list[dict[str, object]] = []
    for query in queries:
        scores = grounder.score(query.text, candidates)
        all_scores.append(scores)
        query_meta.append(
            {
                "label": query.label,
                "hash": query_hash(query.text),
            }
        )
    if not candidates:
        return [], [], query_meta
    merged_scores: list[float] = []
    merged_sources: list[str] = []
    for idx in range(len(candidates)):
        best_query = 0
        best_score = all_scores[0][idx]
        for q_idx, scores in enumerate(all_scores[1:], start=1):
            if scores[idx] > best_score:
                best_score = scores[idx]
                best_query = q_idx
        merged_scores.append(best_score)
        merged_sources.append(queries[best_query].label)
    return merged_scores, merged_sources, query_meta


def select_per_option_union_frames(
    candidates: list[CandidateFrame],
    query_scores: dict[str, list[float]],
    top_k: int,
    top_k_per_option: int | None,
    anchor_k: int,
    window_radius: int,
    final_max_frames: int,
    temporal_nms_seconds: float,
    temporal_nms_candidates: int | None,
) -> tuple[list[SelectedFrame], dict[str, object]]:
    selected_by_idx: dict[int, SelectedFrame] = {}
    suppressed_total = 0
    per_option_k = top_k_per_option or max(1, (top_k + max(len(query_scores), 1) - 1) // max(len(query_scores), 1))
    for label, scores in sorted(query_scores.items()):
        selected, meta = select_grounded_frames(
            candidates,
            scores,
            top_k=per_option_k,
            anchor_k=0,
            window_radius=window_radius,
            final_max_frames=max(final_max_frames, top_k),
            temporal_nms_seconds=temporal_nms_seconds,
            temporal_nms_candidates=temporal_nms_candidates,
            source_prefix=label,
        )
        suppressed_total += int(meta.get("temporal_nms_suppressed", 0))
        for sf in selected:
            prev = selected_by_idx.get(sf.candidate.index)
            if prev is None or (sf.score or -1e9) > (prev.score or -1e9):
                selected_by_idx[sf.candidate.index] = sf

    for idx in _uniform_positions(len(candidates), anchor_k):
        candidate = candidates[idx]
        selected_by_idx.setdefault(
            candidate.index,
            SelectedFrame(candidate, None, "anchor"),
        )

    ordered = sorted(selected_by_idx.values(), key=lambda sf: sf.candidate.timestamp)
    if len(ordered) > final_max_frames:
        priority = {"anchor": 2}
        keep = sorted(
            ordered,
            key=lambda sf: (
                priority.get(sf.source, 0),
                -(sf.score if sf.score is not None else -1e9),
            ),
        )[:final_max_frames]
        keep_ids = {sf.candidate.index for sf in keep}
        ordered = [sf for sf in ordered if sf.candidate.index in keep_ids]
    return ordered, {"temporal_nms_suppressed": suppressed_total}


def select_coarse_to_fine_frames(
    candidates: list[CandidateFrame],
    scores: list[float],
    num_windows: int,
    window_radius_candidates: int,
    frames_per_window: int,
    global_anchor_k: int,
    final_max_frames: int,
    temporal_nms_seconds: float,
    temporal_nms_candidates: int | None,
) -> tuple[list[SelectedFrame], dict[str, object]]:
    ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    centers, suppressed = _temporal_nms(
        ranked,
        candidates,
        top_k=num_windows,
        nms_seconds=temporal_nms_seconds,
        candidate_limit=temporal_nms_candidates or max(num_windows * 8, num_windows),
    )
    selected: dict[int, SelectedFrame] = {}
    half = max(window_radius_candidates, frames_per_window // 2)
    for center in centers:
        local = list(range(max(0, center - half), min(len(candidates), center + half + 1)))
        local = sorted(local, key=lambda idx: (abs(idx - center), idx))[:frames_per_window]
        for idx in local:
            source = "window_center" if idx == center else "window_local"
            prev = selected.get(idx)
            score = scores[idx]
            if prev is None or (score or -1e9) > (prev.score or -1e9):
                selected[idx] = SelectedFrame(candidates[idx], score, source)
    for idx in _uniform_positions(len(candidates), global_anchor_k):
        selected.setdefault(idx, SelectedFrame(candidates[idx], scores[idx], "anchor"))
    ordered = sorted(selected.values(), key=lambda sf: sf.candidate.timestamp)
    if len(ordered) > final_max_frames:
        priority = {"window_center": 0, "window_local": 1, "anchor": 2}
        keep = sorted(
            ordered,
            key=lambda sf: (
                priority.get(sf.source, 3),
                -(sf.score if sf.score is not None else -1e9),
            ),
        )[:final_max_frames]
        keep_ids = {sf.candidate.index for sf in keep}
        ordered = [sf for sf in ordered if sf.candidate.index in keep_ids]
    return ordered, {
        "temporal_nms_suppressed": suppressed,
        "window_centers": [
            {
                "candidate_index": idx,
                "frame_index": candidates[idx].index,
                "timestamp": round(candidates[idx].timestamp, 3),
                "score": round(scores[idx], 6),
            }
            for idx in centers
        ],
    }


def _run_eval(input_path: str, output_path: str, eval_output: str | None) -> None:
    from run_evaluation import (
        _filter_subset,
        evaluate_longqa,
        load_jsonl as load_eval_jsonl,
        write_results,
    )

    golden = load_eval_jsonl(input_path)
    preds = load_eval_jsonl(output_path)
    if len(golden) != len(preds):
        logger.warning(
            "Golden (%d) and predictions (%d) have different lengths; "
            "evaluating against the first %d golden rows.",
            len(golden),
            len(preds),
            len(preds),
        )
        golden, preds = _filter_subset(golden, preds, "longqa")
    results = evaluate_longqa(golden, preds)
    if eval_output is None:
        eval_output = os.path.join(os.path.dirname(output_path), "results.json")
    summary_path = write_results(eval_output, results)
    print(
        f"LongQA Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    print(f"Results written to {eval_output}")
    print(f"Summary written to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LongQA predictions with CLIP/SigLIP frame grounding."
    )
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--output", default="output/egolongqa_grounded/predictions.jsonl")
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--no-eval", action="store_true")

    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--top-k-per-option", type=int, default=None)
    parser.add_argument("--anchor-k", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=0)
    parser.add_argument("--final-max-frames", type=int, default=32)
    parser.add_argument(
        "--retrieval-query-mode",
        choices=RETRIEVAL_QUERY_MODES,
        default="question_options",
        help="Grounding query mode. Default preserves previous question+options behavior.",
    )
    parser.add_argument("--temporal-nms-seconds", type=float, default=0.0)
    parser.add_argument("--temporal-nms-candidates", type=int, default=None)
    parser.add_argument("--coarse-to-fine", action="store_true")
    parser.add_argument("--coarse-candidate-frames", type=int, default=None)
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--window-radius-candidates", type=int, default=2)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--global-anchor-k", type=int, default=32)
    parser.add_argument(
        "--grounder-model",
        default="google/siglip-base-patch16-224",
        help="CLIP/SigLIP-style model used only for frame selection.",
    )
    parser.add_argument("--grounder-device", default="cuda")
    parser.add_argument("--grounder-batch-size", type=int, default=32)
    parser.add_argument("--grounding-output", default=None)
    parser.add_argument(
        "--no-resume-grounding",
        action="store_true",
        help="Ignore any existing grounding-output file and recompute grounding.",
    )
    parser.add_argument(
        "--no-resume-predictions",
        action="store_true",
        help="Ignore any existing prediction file and regenerate from the start.",
    )

    parser.add_argument("--model-type", default="qwen", choices=["llama4", "qwen"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default="baseline",
    )
    return parser.parse_args()


def main() -> None:
    import gc
    import time

    from model import (
        create_model,
        reset_prompt_token_stats,
        summarize_prompt_token_stats,
    )
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    video_folder = _resolve_path(args.video_folder)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    grounding_output = (
        _resolve_path(args.grounding_output)
        if args.grounding_output
        else os.path.splitext(output_path)[0] + "_grounding.jsonl"
    )

    rows = load_jsonl(input_path)
    rows = apply_subset(rows, args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(grounding_output) or ".", exist_ok=True)

    print(
        "Grounded LongQA config: "
        f"candidate_frames={args.candidate_frames}, top_k={args.top_k}, "
        f"anchor_k={args.anchor_k}, window_radius={args.window_radius}, "
        f"final_max_frames={args.final_max_frames}, "
        f"retrieval_query_mode={args.retrieval_query_mode}, "
        f"temporal_nms_seconds={args.temporal_nms_seconds}, "
        f"coarse_to_fine={args.coarse_to_fine}, "
        f"prompt_variant={args.prompt_variant}, "
        f"grounder={args.grounder_model}, backend={args.backend}, "
        f"resume_grounding={not args.no_resume_grounding}, "
        f"resume_predictions={not args.no_resume_predictions}"
    )

    start_time = time.time()
    cached_records = (
        load_jsonl_if_exists(grounding_output)
        if not args.no_resume_grounding
        else []
    )
    valid_cached_records: list[dict[str, object]] = []
    for row_idx, meta in enumerate(cached_records[: len(rows)]):
        expected_video = rows[row_idx].get("video_path", "")
        if int(meta.get("index", -1)) != row_idx:
            break
        if str(meta.get("video_path", "")) != str(expected_video):
            break
        expected_candidate_count = (
            args.coarse_candidate_frames
            if args.coarse_to_fine and args.coarse_candidate_frames
            else args.candidate_frames
        )
        if int(meta.get("candidate_frames", -1)) != expected_candidate_count:
            break
        cached_mode = meta.get("retrieval_query_mode")
        if cached_mode not in (None, args.retrieval_query_mode):
            break
        if int(meta.get("selected_frames", -1)) > args.final_max_frames:
            break
        valid_cached_records.append(meta)
    selection_records = valid_cached_records
    if selection_records:
        print(
            f"Resuming grounding from {len(selection_records)}/{len(rows)} cached rows"
        )

    if len(selection_records) < len(rows):
        grounder = TextImageGrounder(
            args.grounder_model,
            device=args.grounder_device,
            batch_size=args.grounder_batch_size,
        )
        if len(selection_records) != len(cached_records):
            with open(grounding_output, "w") as meta_f:
                for meta in selection_records:
                    meta_f.write(json.dumps(meta) + "\n")
        mode = "a" if selection_records else "w"
        with open(grounding_output, mode) as meta_f:
            for row_idx, row in enumerate(
                rows[len(selection_records) :],
                start=len(selection_records),
            ):
                video_path = os.path.join(video_folder, str(row["video_path"]))
                candidate_count = (
                    args.coarse_candidate_frames
                    if args.coarse_to_fine and args.coarse_candidate_frames
                    else args.candidate_frames
                )
                candidates = extract_candidate_frames(video_path, candidate_count)
                queries = build_grounding_queries(row, args.retrieval_query_mode)
                query_meta: list[dict[str, object]]
                selection_meta: dict[str, object]
                if args.retrieval_query_mode == "per_option_union" and not args.coarse_to_fine:
                    query_scores: dict[str, list[float]] = {}
                    query_meta = []
                    for query in queries:
                        query_scores[query.label] = grounder.score(query.text, candidates)
                        query_meta.append(
                            {"label": query.label, "hash": query_hash(query.text)}
                        )
                    selected, selection_meta = select_per_option_union_frames(
                        candidates,
                        query_scores,
                        top_k=args.top_k,
                        top_k_per_option=args.top_k_per_option,
                        anchor_k=args.anchor_k,
                        window_radius=args.window_radius,
                        final_max_frames=args.final_max_frames,
                        temporal_nms_seconds=args.temporal_nms_seconds,
                        temporal_nms_candidates=args.temporal_nms_candidates,
                    )
                else:
                    scores, _best_sources, query_meta = score_queries(
                        grounder,
                        queries,
                        candidates,
                    )
                    if args.coarse_to_fine:
                        selected, selection_meta = select_coarse_to_fine_frames(
                            candidates,
                            scores,
                            num_windows=args.num_windows,
                            window_radius_candidates=args.window_radius_candidates,
                            frames_per_window=args.frames_per_window,
                            global_anchor_k=args.global_anchor_k,
                            final_max_frames=args.final_max_frames,
                            temporal_nms_seconds=args.temporal_nms_seconds,
                            temporal_nms_candidates=args.temporal_nms_candidates,
                        )
                    else:
                        selected, selection_meta = select_grounded_frames(
                            candidates,
                            scores,
                            top_k=args.top_k,
                            anchor_k=args.anchor_k,
                            window_radius=args.window_radius,
                            final_max_frames=args.final_max_frames,
                            temporal_nms_seconds=args.temporal_nms_seconds,
                            temporal_nms_candidates=args.temporal_nms_candidates,
                            source_prefix="retrieved",
                        )
                meta = {
                    "index": row_idx,
                    "video_path": row.get("video_path", ""),
                    "candidate_frames": len(candidates),
                    "selected_frames": len(selected),
                    "retrieval_query_mode": args.retrieval_query_mode,
                    "prompt_variant": args.prompt_variant,
                    "temporal_nms_seconds": args.temporal_nms_seconds,
                    "temporal_nms_candidates": args.temporal_nms_candidates,
                    "coarse_to_fine": args.coarse_to_fine,
                    "queries": query_meta,
                    "selection_meta": selection_meta,
                    "selected": [
                        {
                            "frame_index": sf.candidate.index,
                            "timestamp": round(sf.candidate.timestamp, 3),
                            "score": round(sf.score, 6) if sf.score is not None else None,
                            "source": sf.source,
                        }
                        for sf in selected
                    ],
                }
                selection_records.append(meta)
                meta_f.write(json.dumps(meta) + "\n")
                meta_f.flush()
                print(f"  Grounding progress: {row_idx + 1}/{len(rows)}")

        del grounder
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Grounding cache complete; skipping grounder model load")

    if len(selection_records) != len(rows):
        raise RuntimeError(
            f"Grounding incomplete: {len(selection_records)}/{len(rows)} records"
        )

    existing_predictions = (
        load_jsonl_if_exists(output_path) if not args.no_resume_predictions else []
    )
    pred_start = 0
    for row_idx, pred in enumerate(existing_predictions[: len(rows)]):
        if str(pred.get("video_path", "")) != str(rows[row_idx].get("video_path", "")):
            break
        if not str(pred.get("mcq_answer", "")).strip():
            break
        pred_start += 1
    if pred_start:
        print(f"Resuming generation from {pred_start}/{len(rows)} cached predictions")

    model = create_model(
        args.model_type,
        args.llm_model,
        backend=args.backend,
        tp_size=args.tp,
        concurrency=args.concurrency,
        max_frames=args.final_max_frames,
    )
    reset_prompt_token_stats()

    pred_mode = "a" if pred_start else "w"
    with model, open(output_path, pred_mode) as pred_f:
        for row_idx, (row, meta) in enumerate(
            zip(rows[pred_start:], selection_records[pred_start:]),
            start=pred_start,
        ):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            frame_indices = [
                int(item["frame_index"])
                for item in meta["selected"]  # type: ignore[index]
            ]
            frames = extract_frames_by_indices(video_path, frame_indices)
            messages = [
                {
                    "role": "user",
                    "content": build_longqa_prompt(
                        row["question"],
                        row["mcq_options"],
                        prompt_variant=args.prompt_variant,
                    ),
                }
            ]
            response = model.generate(frames, messages, max_new_tokens=16)
            pred = build_prediction_row(
                row,
                response,
                prompt_variant=args.prompt_variant,
            )
            pred["retrieval_query_mode"] = args.retrieval_query_mode
            pred["coarse_to_fine"] = args.coarse_to_fine
            pred_f.write(json.dumps(pred) + "\n")
            pred_f.flush()
            print(f"  Generation progress: {row_idx + 1}/{len(rows)}")

    elapsed = time.time() - start_time
    print(f"Predictions written to {output_path}")
    print(f"Grounding metadata written to {grounding_output}")
    print(f"Runtime seconds: {elapsed:.0f}")
    _print_context_summary(summarize_prompt_token_stats())

    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
