#!/usr/bin/env python3
"""Evaluate offline EarlyCare speech embeddings with speaker-level splits."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_POSITIVE_LABELS = {"pd", "parkinson", "parkinsons", "parkinsonian", "positive", "1", "case"}


@dataclass(frozen=True)
class SpeakerExample:
    speaker_id: str
    dataset: str
    label: str
    is_positive: bool
    embedding: list[float]
    row_count: int


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


def mean_vector(vectors: Iterable[Sequence[float]]) -> list[float]:
    vectors = [list(vector) for vector in vectors]
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("'", "").replace("-", "_").replace(" ", "_")


def speaker_examples(rows: list[dict[str, object]], positive_labels: set[str]) -> list[SpeakerExample]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        speaker_id = str(row.get("speaker_id") or "")
        dataset = str(row.get("dataset") or "unknown")
        embedding = row.get("embedding")
        if not speaker_id or not isinstance(embedding, list) or not embedding:
            continue
        grouped[(dataset, speaker_id)].append(row)

    examples: list[SpeakerExample] = []
    for (dataset, speaker_id), speaker_rows in grouped.items():
        labels = [str(row.get("label") or "unknown") for row in speaker_rows]
        label = max(set(labels), key=labels.count)
        embeddings = [row["embedding"] for row in speaker_rows if isinstance(row.get("embedding"), list)]
        if not embeddings:
            continue
        examples.append(
            SpeakerExample(
                speaker_id=speaker_id,
                dataset=dataset,
                label=label,
                is_positive=normalize_label(label) in positive_labels,
                embedding=mean_vector(embeddings),  # type: ignore[arg-type]
                row_count=len(speaker_rows),
            )
        )
    return sorted(examples, key=lambda item: (item.dataset, item.label, item.speaker_id))


def deterministic_split(examples: list[SpeakerExample], test_fraction: float) -> tuple[list[SpeakerExample], list[SpeakerExample], str | None]:
    by_class = {True: [example for example in examples if example.is_positive], False: [example for example in examples if not example.is_positive]}
    if len(by_class[True]) < 2 or len(by_class[False]) < 2:
        return [], [], "Need at least two positive and two negative speakers for a speaker-level train/test split."

    train: list[SpeakerExample] = []
    test: list[SpeakerExample] = []
    for label, group in by_class.items():
        _ = label
        group = sorted(group, key=lambda item: (item.dataset, item.speaker_id))
        test_count = max(1, min(len(group) - 1, round(len(group) * test_fraction)))
        test.extend(group[-test_count:])
        train.extend(group[:-test_count])
    return train, test, None


def dataset_split(
    examples: list[SpeakerExample],
    train_dataset: str,
    test_dataset: str,
) -> tuple[list[SpeakerExample], list[SpeakerExample], str | None]:
    train = [example for example in examples if example.dataset == train_dataset]
    test = [example for example in examples if example.dataset == test_dataset]
    if not train or not test:
        return [], [], "Train or test dataset has no usable speaker embeddings."
    if {example.speaker_id for example in train} & {example.speaker_id for example in test}:
        return [], [], "Speaker leakage detected between train and test datasets."
    if not any(example.is_positive for example in train) or all(example.is_positive for example in train):
        return [], [], "Train dataset needs both positive and negative speakers."
    if not any(example.is_positive for example in test) or all(example.is_positive for example in test):
        return [], [], "Test dataset needs both positive and negative speakers."
    return train, test, None


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1
            elif positive == negative:
                wins += 0.5
    return round(wins / total, 6)


def probability(score: float) -> float:
    return 1 / (1 + math.exp(-max(-20, min(20, score * 5))))


def evaluate(train: list[SpeakerExample], test: list[SpeakerExample]) -> dict[str, object]:
    positive_centroid = mean_vector(example.embedding for example in train if example.is_positive)
    negative_centroid = mean_vector(example.embedding for example in train if not example.is_positive)
    predictions = []
    for example in test:
        score = cosine_similarity(example.embedding, positive_centroid) - cosine_similarity(example.embedding, negative_centroid)
        predicted_positive = score >= 0
        predictions.append(
            {
                "speaker_id": example.speaker_id,
                "dataset": example.dataset,
                "label": example.label,
                "actual_positive": example.is_positive,
                "predicted_positive": predicted_positive,
                "score": round(score, 6),
                "probability": round(probability(score), 6),
                "row_count": example.row_count,
            }
        )

    tp = sum(1 for item in predictions if item["actual_positive"] and item["predicted_positive"])
    tn = sum(1 for item in predictions if not item["actual_positive"] and not item["predicted_positive"])
    fp = sum(1 for item in predictions if not item["actual_positive"] and item["predicted_positive"])
    fn = sum(1 for item in predictions if item["actual_positive"] and not item["predicted_positive"])
    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    labels = [bool(item["actual_positive"]) for item in predictions]
    scores = [float(item["score"]) for item in predictions]
    brier = sum((float(item["probability"]) - (1 if item["actual_positive"] else 0)) ** 2 for item in predictions) / max(len(predictions), 1)

    return {
        "sensitivity": round(sensitivity, 6),
        "specificity": round(specificity, 6),
        "balanced_accuracy": round((sensitivity + specificity) / 2, 6),
        "roc_auc": roc_auc(labels, scores),
        "calibration": {"brier_score": round(brier, 6)},
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "false_positives": [item for item in predictions if not item["actual_positive"] and item["predicted_positive"]],
        "false_negatives": [item for item in predictions if item["actual_positive"] and not item["predicted_positive"]],
        "predictions": predictions,
    }


def build_report(
    rows: list[dict[str, object]],
    examples: list[SpeakerExample],
    train: list[SpeakerExample],
    test: list[SpeakerExample],
    split_error: str | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    speaker_leakage = bool({example.speaker_id for example in train} & {example.speaker_id for example in test})
    base = {
        "status": "insufficient-data" if split_error else "ok",
        "input": str(args.input),
        "rows": len(rows),
        "speakers": len(examples),
        "positive_labels": sorted(args.positive_labels),
        "split": {
            "mode": "dataset" if args.train_dataset or args.test_dataset else "deterministic",
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "test_fraction": args.test_fraction,
            "train_speakers": [example.speaker_id for example in train],
            "test_speakers": [example.speaker_id for example in test],
            "speaker_leakage": speaker_leakage,
        },
        "class_counts": {
            "positive_speakers": sum(1 for example in examples if example.is_positive),
            "negative_speakers": sum(1 for example in examples if not example.is_positive),
        },
    }
    if split_error:
        return {**base, "reason": split_error}
    return {**base, "metrics": evaluate(train, test)}


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EarlyCare speech embeddings with speaker-level splits.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows from extract_embeddings.py")
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/speech_eval.json"))
    parser.add_argument("--positive-labels", default=",".join(sorted(DEFAULT_POSITIVE_LABELS)))
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--train-dataset", help="Train only on this dataset name.")
    parser.add_argument("--test-dataset", help="Test only on this dataset name.")
    args = parser.parse_args(argv)
    args.positive_labels = {normalize_label(label) for label in args.positive_labels.split(",") if label.strip()}
    if (args.train_dataset and not args.test_dataset) or (args.test_dataset and not args.train_dataset):
        parser.error("--train-dataset and --test-dataset must be provided together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.input)
    examples = speaker_examples(rows, args.positive_labels)
    if args.train_dataset and args.test_dataset:
        train, test, error = dataset_split(examples, args.train_dataset, args.test_dataset)
    else:
        train, test, error = deterministic_split(examples, args.test_fraction)
    report = build_report(rows, examples, train, test, error, args)
    write_report(report, args.output)
    print(f"wrote evaluation report to {args.output} ({report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
