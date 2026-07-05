#!/usr/bin/env python3
"""Convert feature-only speech datasets into EarlyCare research JSONL rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_EXCLUDED_COLUMNS = {
    "id",
    "subject",
    "subject_id",
    "speaker",
    "speaker_id",
    "name",
    "class",
    "label",
    "status",
    "target",
    "updrs",
    "motor_updrs",
    "total_updrs",
    "test_time",
}

POSITIVE_VALUES = {"1", "true", "yes", "pd", "pwp", "parkinson", "parkinsons", "parkinsonian", "patient", "case"}
NEGATIVE_VALUES = {"0", "false", "no", "control", "healthy", "hc", "normal", "comparison"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def label_from_value(value: str) -> str:
    normalized = normalize_name(value)
    if normalized in POSITIVE_VALUES:
        return "pd"
    if normalized in NEGATIVE_VALUES:
        return "control"
    return value.strip() or "unknown"


def find_column(fieldnames: Sequence[str], requested: str | None, candidates: Sequence[str]) -> str | None:
    if requested:
        for fieldname in fieldnames:
            if normalize_name(fieldname) == normalize_name(requested):
                return fieldname
        raise ValueError(f"Column not found: {requested}")
    normalized = {normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        if normalize_name(candidate) in normalized:
            return normalized[normalize_name(candidate)]
    return None


def numeric_columns(rows: list[dict[str, str]], excluded_columns: set[str]) -> list[str]:
    if not rows:
        return []
    columns: list[str] = []
    for column in rows[0].keys():
        if normalize_name(column) in excluded_columns:
            continue
        values = [parse_float(row.get(column)) for row in rows]
        numeric_count = sum(1 for value in values if value is not None)
        if numeric_count == len(rows):
            columns.append(column)
    return columns


def zscore_vectors(rows: list[dict[str, str]], columns: list[str]) -> list[list[float]]:
    stats: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = [parse_float(row.get(column)) or 0.0 for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stddev = math.sqrt(variance) or 1.0
        stats[column] = (mean, stddev)

    vectors: list[list[float]] = []
    for row in rows:
        vector = []
        for column in columns:
            mean, stddev = stats[column]
            value = parse_float(row.get(column)) or 0.0
            vector.append(round((value - mean) / stddev, 6))
        vectors.append(vector)
    return vectors


def read_csv(path: Path) -> list[dict[str, str]]:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, dialect=dialect))


def build_rows(args: argparse.Namespace, source_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    if not source_rows:
        raise ValueError("Feature table has no rows")

    fieldnames = list(source_rows[0].keys())
    speaker_column = find_column(fieldnames, args.speaker_column, ["speaker_id", "subject_id", "subject id", "subject", "id"])
    label_column = find_column(fieldnames, args.label_column, ["class information", "class", "status", "label", "target"])
    task_column = find_column(fieldnames, args.task_column, ["task", "sample", "recording"])
    updrs_column = find_column(fieldnames, args.updrs_column, ["updrs", "motor_updrs", "total_updrs"])
    if speaker_column is None:
        raise ValueError("Could not find speaker column. Pass --speaker-column.")
    if label_column is None:
        raise ValueError("Could not find label column. Pass --label-column.")

    excluded = {normalize_name(column) for column in args.exclude_columns.split(",") if column.strip()}
    excluded.update({normalize_name(speaker_column), normalize_name(label_column)})
    if task_column:
        excluded.add(normalize_name(task_column))
    if updrs_column:
        excluded.add(normalize_name(updrs_column))

    feature_columns = numeric_columns(source_rows, excluded)
    if not feature_columns:
        raise ValueError("No fully numeric feature columns found after exclusions.")

    extracted_at = utc_now()
    vectors = zscore_vectors(source_rows, feature_columns)
    rows: list[dict[str, object]] = []
    for index, (source_row, vector) in enumerate(zip(source_rows, vectors), start=1):
        source_id = source_row.get(args.source_id_column or "") or f"{Path(args.input).stem}-{index}"
        speaker_id = str(source_row[speaker_column]).strip() or f"speaker-{index}"
        label = label_from_value(str(source_row[label_column]))
        task = str(source_row[task_column]).strip() if task_column else args.task
        metrics = {
            "speechRate": 0,
            "avgPauseMs": 0,
            "responseLatencyMs": 0,
            "pitchVariability": 0,
            "phraseAccuracy": 0,
            "embedding": vector,
            "updatedAt": extracted_at,
        }
        provenance: dict[str, object] = {
            "source_id": source_id,
            "source_table": str(args.input),
            "source_type": "feature_table",
            "feature_columns": feature_columns,
            "model": "feature-table-zscore",
            "model_name": "feature-table-zscore",
            "language": args.language,
            "extracted_at": extracted_at,
        }
        if updrs_column and parse_float(source_row.get(updrs_column)) is not None:
            provenance["updrs"] = parse_float(source_row.get(updrs_column))
        rows.append(
            {
                "dataset": args.dataset,
                "speaker_id": speaker_id,
                "label": label,
                "task": task,
                "embedding": vector,
                "speech_metrics": metrics,
                "provenance": provenance,
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a feature-only speech CSV into EarlyCare research JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Feature table CSV, for example the UCI Parkinson speech training file.")
    parser.add_argument("--output", type=Path, required=True, help="JSONL output under research/artifacts.")
    parser.add_argument("--dataset", default="UCI Parkinson Speech")
    parser.add_argument("--language", default="", help="Language stored in provenance.")
    parser.add_argument("--speaker-column", help="Speaker/subject column. Auto-detected when possible.")
    parser.add_argument("--label-column", help="Class/label column. Auto-detected when possible.")
    parser.add_argument("--task-column", help="Task/sample column. Auto-detected when possible.")
    parser.add_argument("--updrs-column", help="Optional UPDRS column stored in provenance.")
    parser.add_argument("--source-id-column", help="Optional source row id column.")
    parser.add_argument("--task", default="feature_table", help="Task value when no task column exists.")
    parser.add_argument("--exclude-columns", default=",".join(sorted(DEFAULT_EXCLUDED_COLUMNS)), help="Comma-separated columns excluded from feature vector.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_rows = read_csv(args.input)
    rows = build_rows(args, source_rows)
    write_jsonl(rows, args.output)
    speakers = {(row["dataset"], row["speaker_id"]) for row in rows}
    print(f"wrote {len(rows)} feature rows for {len(speakers)} speakers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
