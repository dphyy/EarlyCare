#!/usr/bin/env python3
"""Build offline personal speech baselines from repeated speaker embeddings."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from research.speech_ml.evaluate_baseline import cosine_similarity, mean_vector


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return round(max(0.0, 1.0 - cosine_similarity(a, b)), 8)


def embedding(row: dict[str, object]) -> list[float] | None:
    value = row.get("embedding")
    if not isinstance(value, list) or not value:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def provenance_value(row: dict[str, object], key: str, default: str = "") -> str:
    provenance = row.get("provenance")
    if isinstance(provenance, dict) and provenance.get(key):
        return str(provenance[key])
    if row.get(key):
        return str(row[key])
    return default


def source_id(row: dict[str, object], index: int) -> str:
    return provenance_value(row, "source_id", f"row-{index}")


def group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[tuple[int, dict[str, object], list[float]]]]:
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, object], list[float]]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        speaker_id = str(row.get("speaker_id") or "").strip()
        dataset = str(row.get("dataset") or "unknown").strip() or "unknown"
        vector = embedding(row)
        if not speaker_id or vector is None:
            continue
        grouped[(dataset, speaker_id)].append((index, row, vector))
    return grouped


def threshold(mean: float, stddev: float, multiplier: float, floor: float) -> float:
    return round(max(floor, mean + stddev * multiplier), 8)


def build_speaker_baseline(
    dataset: str,
    speaker_id: str,
    rows: list[tuple[int, dict[str, object], list[float]]],
    min_samples: int,
    watch_sigma: float,
    amber_sigma: float,
) -> dict[str, object]:
    label_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    vectors = [vector for _, _, vector in rows]
    for _, row, _ in rows:
        label = str(row.get("label") or "unknown")
        task = str(row.get("task") or "unknown")
        language = provenance_value(row, "language", str(row.get("language") or "unknown")) or "unknown"
        source_type = provenance_value(row, "source_type", "raw_audio")
        label_counts[label] = label_counts.get(label, 0) + 1
        task_counts[task] = task_counts.get(task, 0) + 1
        language_counts[language] = language_counts.get(language, 0) + 1
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1

    base = {
        "dataset": dataset,
        "speaker_id": speaker_id,
        "samples": len(rows),
        "labels": label_counts,
        "tasks": task_counts,
        "languages": language_counts,
        "source_types": source_type_counts,
    }
    if len(rows) < min_samples:
        return {
            **base,
            "status": "insufficient-data",
            "reason": f"Need at least {min_samples} samples to estimate a personal speech baseline.",
        }

    centroid = mean_vector(vectors)
    distances = [distance(vector, centroid) for vector in vectors]
    mean_distance = statistics.mean(distances)
    stddev_distance = statistics.pstdev(distances) if len(distances) > 1 else 0.0
    watch_threshold = threshold(mean_distance, stddev_distance, watch_sigma, 0.03)
    amber_threshold = max(watch_threshold, threshold(mean_distance, stddev_distance, amber_sigma, 0.08))
    return {
        **base,
        "status": "ok",
        "embedding_dimensions": len(centroid),
        "centroid": [round(value, 8) for value in centroid],
        "distance_stats": {
            "mean": round(mean_distance, 8),
            "stddev": round(stddev_distance, 8),
            "max": round(max(distances), 8),
        },
        "thresholds": {
            "watch": watch_threshold,
            "amber": amber_threshold,
            "method": f"mean + sigma * stddev, floors 0.03/0.08, watch_sigma={watch_sigma}, amber_sigma={amber_sigma}",
        },
        "sample_distances": [
            {
                "source_id": source_id(row, index),
                "distance": distances[offset],
                "task": str(row.get("task") or "unknown"),
                "label": str(row.get("label") or "unknown"),
            }
            for offset, (index, row, _) in enumerate(rows)
        ],
    }


def build_report(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    grouped = group_rows(rows)
    baselines = [
        build_speaker_baseline(dataset, speaker_id, speaker_rows, args.min_samples, args.watch_sigma, args.amber_sigma)
        for (dataset, speaker_id), speaker_rows in sorted(grouped.items())
    ]
    ok_count = sum(1 for item in baselines if item["status"] == "ok")
    return {
        "status": "ok" if ok_count else "insufficient-data",
        "generated_at": utc_now(),
        "input": str(args.input),
        "rows": len(rows),
        "usable_rows": sum(len(speaker_rows) for speaker_rows in grouped.values()),
        "speakers": len(baselines),
        "speakers_with_baselines": ok_count,
        "min_samples": args.min_samples,
        "baselines": baselines,
        "safety": {
            "intended_use": "offline personal speech baseline drift research only",
            "excluded_use": "diagnosis, emergency dispatch, or Parkinson's/concussion detection",
            "app_runtime": "review thresholds manually before any app integration",
        },
    }


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-speaker personal speech baselines from repeated embedding rows.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows from run_experiment.py or extract_embeddings.py")
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/personal_speech_baselines.json"))
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--watch-sigma", type=float, default=2.0)
    parser.add_argument("--amber-sigma", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.min_samples < 2:
        parser.error("--min-samples must be at least 2")
    if args.watch_sigma < 0 or args.amber_sigma < 0:
        parser.error("sigma values must be non-negative")
    if args.amber_sigma < args.watch_sigma:
        parser.error("--amber-sigma must be greater than or equal to --watch-sigma")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.input)
    report = build_report(rows, args)
    write_report(report, args.output)
    print(f"wrote personal baseline report to {args.output} ({report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
