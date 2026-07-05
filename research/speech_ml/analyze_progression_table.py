#!/usr/bin/env python3
"""Analyze progression-only feature tables such as UCI Parkinsons Telemonitoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SPEAKER_CANDIDATES = ["speaker_id", "subject_id", "subject id", "subject#", "subject", "id"]
TIME_CANDIDATES = ["test_time", "time", "days", "day", "visit", "recording_time"]
TARGET_CANDIDATES = ["motor_updrs", "total_updrs", "updrs"]
EXCLUDED_COLUMNS = {
    "id",
    "subject",
    "subject_id",
    "subject#",
    "speaker",
    "speaker_id",
    "name",
    "age",
    "sex",
    "test_time",
    "time",
    "days",
    "day",
    "visit",
    "recording_time",
    "class",
    "label",
    "status",
    "target",
    "updrs",
    "motor_updrs",
    "total_updrs",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value.strip())
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def read_table(path: Path) -> list[dict[str, str]]:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, dialect=dialect))


def find_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    normalized = {normalize_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        if normalize_name(candidate) in normalized:
            return normalized[normalize_name(candidate)]
    return None


def target_columns(fieldnames: Sequence[str], requested: str | None) -> list[str]:
    if requested:
        requested_names = {normalize_name(value) for value in requested.split(",") if value.strip()}
        matched = [fieldname for fieldname in fieldnames if normalize_name(fieldname) in requested_names]
        missing = requested_names - {normalize_name(fieldname) for fieldname in matched}
        if missing:
            raise SystemExit(f"Target columns not found: {', '.join(sorted(missing))}")
        return matched
    return [fieldname for fieldname in fieldnames if normalize_name(fieldname) in TARGET_CANDIDATES]


def numeric_columns(rows: list[dict[str, str]], excluded: set[str]) -> list[str]:
    if not rows:
        return []
    columns: list[str] = []
    for column in rows[0].keys():
        if normalize_name(column) in excluded:
            continue
        values = [parse_float(row.get(column)) for row in rows]
        if values and all(value is not None for value in values):
            columns.append(column)
    return columns


def linear_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def select_fetch_manifest_table(manifest_path: Path, requested_table: str | None = None) -> tuple[Path, dict[str, object]]:
    payload = json.loads(manifest_path.read_text())
    raw_summaries = payload.get("table_summaries")
    summaries = [summary for summary in raw_summaries if isinstance(summary, dict)] if isinstance(raw_summaries, list) else []
    if requested_table:
        summaries = [summary for summary in summaries if str(summary.get("path", "")).strip() == requested_table.strip()]
        if not summaries:
            raise SystemExit(f"Dataset fetch manifest has no table summary for {requested_table}")

    ready = [summary for summary in summaries if summary.get("progression_ready") is True]
    if not ready:
        classifier_ready = [str(summary.get("path", "")).strip() for summary in summaries if summary.get("classification_ready") is True]
        detail = f" Classifier-ready tables: {', '.join(path for path in classifier_ready if path)}." if classifier_ready else ""
        raise SystemExit(f"No progression_ready table found in {manifest_path}.{detail}")
    selected = ready[0]
    table_path = str(selected.get("path", "")).strip()
    if not table_path:
        raise SystemExit(f"Selected table summary in {manifest_path} is missing path")
    path = manifest_path.parent / table_path
    if not path.exists():
        raise SystemExit(f"Selected table from dataset fetch manifest does not exist: {path}")
    return path, selected


def speaker_trends(
    rows: list[dict[str, str]],
    speaker_column: str,
    time_column: str | None,
    targets: list[str],
    min_samples: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        speaker = row.get(speaker_column, "").strip()
        if speaker:
            grouped[speaker].append(row)

    trends: list[dict[str, object]] = []
    for speaker_id, speaker_rows in sorted(grouped.items()):
        usable_rows = []
        for index, row in enumerate(speaker_rows):
            time_value = parse_float(row.get(time_column)) if time_column else float(index)
            if time_value is None:
                time_value = float(index)
            usable_rows.append((time_value, row))
        usable_rows.sort(key=lambda item: item[0])
        trend: dict[str, object] = {
            "speaker_id": speaker_id,
            "samples": len(usable_rows),
            "status": "ok" if len(usable_rows) >= min_samples else "insufficient-data",
            "targets": {},
        }
        target_payload: dict[str, object] = {}
        for target in targets:
            points = [(time_value, value) for time_value, row in usable_rows if (value := parse_float(row.get(target))) is not None]
            values = [point[1] for point in points]
            target_payload[target] = {
                "first": round_or_none(values[0]) if values else None,
                "last": round_or_none(values[-1]) if values else None,
                "delta": round_or_none(values[-1] - values[0]) if len(values) >= 2 else None,
                "slope_per_time_unit": round_or_none(linear_slope(points)),
                "samples": len(points),
            }
        trend["targets"] = target_payload
        trends.append(trend)
    return trends


def centered_pairs(rows: list[dict[str, str]], speaker_column: str, left: str, right: str) -> tuple[list[float], list[float]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        speaker_id = row.get(speaker_column, "").strip()
        left_value = parse_float(row.get(left))
        right_value = parse_float(row.get(right))
        if speaker_id and left_value is not None and right_value is not None:
            grouped[speaker_id].append((left_value, right_value))

    centered_left: list[float] = []
    centered_right: list[float] = []
    for pairs in grouped.values():
        if len(pairs) < 2:
            continue
        mean_left = sum(pair[0] for pair in pairs) / len(pairs)
        mean_right = sum(pair[1] for pair in pairs) / len(pairs)
        centered_left.extend(pair[0] - mean_left for pair in pairs)
        centered_right.extend(pair[1] - mean_right for pair in pairs)
    return centered_left, centered_right


def feature_associations(rows: list[dict[str, str]], speaker_column: str, features: list[str], targets: list[str], top_k: int) -> dict[str, list[dict[str, object]]]:
    associations: dict[str, list[dict[str, object]]] = {}
    for target in targets:
        target_associations: list[dict[str, object]] = []
        for feature in features:
            feature_values = []
            target_values = []
            for row in rows:
                feature_value = parse_float(row.get(feature))
                target_value = parse_float(row.get(target))
                if feature_value is not None and target_value is not None:
                    feature_values.append(feature_value)
                    target_values.append(target_value)
            centered_feature, centered_target = centered_pairs(rows, speaker_column, feature, target)
            row_corr = pearson(feature_values, target_values)
            centered_corr = pearson(centered_feature, centered_target)
            if row_corr is None and centered_corr is None:
                continue
            target_associations.append(
                {
                    "feature": feature,
                    "row_level_pearson": round_or_none(row_corr),
                    "subject_centered_pearson": round_or_none(centered_corr),
                    "rows": len(feature_values),
                    "centered_rows": len(centered_feature),
                }
            )
        associations[target] = sorted(
            target_associations,
            key=lambda item: abs(float(item["subject_centered_pearson"] or item["row_level_pearson"] or 0)),
            reverse=True,
        )[:top_k]
    return associations


def build_report(args: argparse.Namespace, input_path: Path, selected_summary: dict[str, object] | None = None) -> dict[str, object]:
    rows = read_table(input_path)
    if not rows:
        return {"status": "insufficient-data", "reason": "Table has no rows", "input": str(input_path)}

    fieldnames = list(rows[0].keys())
    speaker_column = args.speaker_column or find_column(fieldnames, SPEAKER_CANDIDATES)
    time_column = args.time_column or find_column(fieldnames, TIME_CANDIDATES)
    targets = target_columns(fieldnames, args.target_columns)
    if speaker_column is None:
        raise SystemExit("No speaker/subject column detected. Pass --speaker-column.")
    if not targets:
        raise SystemExit("No UPDRS target columns detected. Pass --target-columns.")

    excluded = set(EXCLUDED_COLUMNS)
    excluded.add(normalize_name(speaker_column))
    if time_column:
        excluded.add(normalize_name(time_column))
    excluded.update(normalize_name(target) for target in targets)
    features = numeric_columns(rows, excluded)
    if not features:
        raise SystemExit("No fully numeric feature columns found after excluding metadata and targets.")

    trends = speaker_trends(rows, speaker_column, time_column, targets, args.min_samples)
    usable_trends = [trend for trend in trends if trend["status"] == "ok"]
    associations = feature_associations(rows, speaker_column, features, targets, args.top_k)
    return {
        "status": "ok" if usable_trends else "insufficient-data",
        "generated_at": utc_now(),
        "input": str(input_path),
        "dataset_fetch_manifest": str(args.dataset_fetch_manifest) if args.dataset_fetch_manifest else None,
        "selected_table_summary": selected_summary,
        "rows": len(rows),
        "speakers": len({row.get(speaker_column, "").strip() for row in rows if row.get(speaker_column, "").strip()}),
        "speaker_column": speaker_column,
        "time_column": time_column,
        "target_columns": targets,
        "numeric_feature_count": len(features),
        "numeric_feature_columns": features,
        "min_samples_per_speaker": args.min_samples,
        "speakers_with_trends": len(usable_trends),
        "speaker_trends": trends,
        "feature_associations": associations,
        "safety": {
            "intended_use": "offline longitudinal speech-feature progression research only",
            "excluded_use": "PD/control diagnosis, concussion detection, emergency dispatch, or app runtime routing",
            "note": "Associations are exploratory and may be confounded by subject, device, medication state, and recording conditions.",
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze progression-only voice feature tables such as UCI Parkinsons Telemonitoring.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Feature table CSV, TSV, or .data file.")
    input_group.add_argument("--dataset-fetch-manifest", type=Path, help="dataset_fetch_manifest.json with a progression_ready table.")
    parser.add_argument("--fetch-table", help="Table path inside --dataset-fetch-manifest. Defaults to the first progression-ready table.")
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/progression_analysis.json"))
    parser.add_argument("--speaker-column", help="Speaker/subject column. Auto-detected when possible.")
    parser.add_argument("--time-column", help="Time column. Auto-detected when possible.")
    parser.add_argument("--target-columns", help="Comma-separated UPDRS target columns. Auto-detected when possible.")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum rows per speaker for trend summaries.")
    parser.add_argument("--top-k", type=int, default=8, help="Top feature associations per target.")
    args = parser.parse_args(argv)
    if args.min_samples < 2:
        parser.error("--min-samples must be at least 2")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected_summary = None
    input_path = args.input
    if args.dataset_fetch_manifest:
        input_path, selected_summary = select_fetch_manifest_table(args.dataset_fetch_manifest, args.fetch_table)
        args.speaker_column = args.speaker_column or str(selected_summary.get("speaker_column") or "")
        args.target_columns = args.target_columns or ",".join(str(value) for value in selected_summary.get("updrs_columns", []) if value)
    if input_path is None:
        raise SystemExit("No input table selected")
    report = build_report(args, input_path, selected_summary)
    write_json(args.output, report)
    print(f"wrote progression analysis to {args.output} ({report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
