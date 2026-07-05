#!/usr/bin/env python3
"""Train a small offline baseline model from EarlyCare speech embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from research.speech_ml.evaluate_baseline import (
    DEFAULT_POSITIVE_LABELS,
    SpeakerExample,
    cosine_similarity,
    mean_vector,
    normalize_label,
    read_rows,
    speaker_examples,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_example(example: SpeakerExample, positive_centroid: list[float], negative_centroid: list[float]) -> float:
    return cosine_similarity(example.embedding, positive_centroid) - cosine_similarity(example.embedding, negative_centroid)


def balanced_accuracy_at_threshold(examples: list[SpeakerExample], scores: list[float], threshold: float) -> float:
    tp = sum(1 for example, score in zip(examples, scores) if example.is_positive and score >= threshold)
    tn = sum(1 for example, score in zip(examples, scores) if not example.is_positive and score < threshold)
    fp = sum(1 for example, score in zip(examples, scores) if not example.is_positive and score >= threshold)
    fn = sum(1 for example, score in zip(examples, scores) if example.is_positive and score < threshold)
    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    return round((sensitivity + specificity) / 2, 6)


def best_threshold(examples: list[SpeakerExample], scores: list[float]) -> tuple[float, float]:
    candidates = sorted(set(scores + [0.0]))
    if len(candidates) > 1:
        candidates.extend(round((left + right) / 2, 6) for left, right in zip(candidates, candidates[1:]))
    ranked = sorted(
        ((balanced_accuracy_at_threshold(examples, scores, threshold), threshold) for threshold in candidates),
        key=lambda item: (item[0], -abs(item[1])),
        reverse=True,
    )
    if not ranked:
        return 0.0, 0.0
    accuracy, threshold = ranked[0]
    return round(threshold, 6), accuracy


def dataset_counts(examples: list[SpeakerExample]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for example in examples:
        bucket = counts.setdefault(example.dataset, {"positive_speakers": 0, "negative_speakers": 0})
        key = "positive_speakers" if example.is_positive else "negative_speakers"
        bucket[key] += 1
    return counts


def train_model(rows: list[dict[str, object]], examples: list[SpeakerExample], args: argparse.Namespace) -> dict[str, object]:
    positive = [example for example in examples if example.is_positive]
    negative = [example for example in examples if not example.is_positive]
    if not positive or not negative:
        return {
            "status": "insufficient-data",
            "reason": "Need at least one positive and one negative speaker to train a baseline model.",
            "rows": len(rows),
            "speakers": len(examples),
            "positive_labels": sorted(args.positive_labels),
        }

    positive_centroid = mean_vector(example.embedding for example in positive)
    negative_centroid = mean_vector(example.embedding for example in negative)
    scores = [score_example(example, positive_centroid, negative_centroid) for example in examples]
    threshold, train_balanced_accuracy = best_threshold(examples, scores)
    dimensions = len(positive_centroid)

    return {
        "status": "ok",
        "model_type": "speaker-centroid-baseline",
        "trained_at": utc_now(),
        "input": str(args.input),
        "rows": len(rows),
        "speakers": len(examples),
        "embedding_dimensions": dimensions,
        "positive_labels": sorted(args.positive_labels),
        "threshold": threshold,
        "train_balanced_accuracy": train_balanced_accuracy,
        "centroids": {
            "positive": [round(value, 8) for value in positive_centroid],
            "negative": [round(value, 8) for value in negative_centroid],
        },
        "train_speakers": [
            {
                "speaker_id": example.speaker_id,
                "dataset": example.dataset,
                "label": example.label,
                "row_count": example.row_count,
                "score": round(score, 6),
            }
            for example, score in zip(examples, scores)
        ],
        "dataset_counts": dataset_counts(examples),
        "safety": {
            "intended_use": "offline speech-deviation research and Parkinson's watch validation only",
            "excluded_use": "diagnosis or emergency routing without model-card validation and human follow-up",
            "app_runtime": "do not load this artifact in FastAPI request paths until latency, validation, and safety gates are complete",
        },
    }


def write_model(model: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline EarlyCare speech baseline from extracted embeddings.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows from extract_embeddings.py")
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/speech_baseline_model.json"))
    parser.add_argument("--positive-labels", default=",".join(sorted(DEFAULT_POSITIVE_LABELS)))
    args = parser.parse_args(argv)
    args.positive_labels = {normalize_label(label) for label in args.positive_labels.split(",") if label.strip()}
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.input)
    examples = speaker_examples(rows, args.positive_labels)
    model = train_model(rows, examples, args)
    write_model(model, args.output)
    print(f"wrote baseline model to {args.output} ({model['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
