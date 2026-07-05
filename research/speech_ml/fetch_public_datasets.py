#!/usr/bin/env python3
"""Fetch public feature-only speech datasets into ignored local folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen


USER_AGENT = "EarlyCare speech ML research fetcher"
TABLE_SUFFIXES = {".csv", ".data", ".tsv"}
NESTED_ARCHIVE_SUFFIXES = {".rar", ".7z"}
SPEAKER_CANDIDATES = ["speaker_id", "subject_id", "subject id", "subject#", "subject", "id"]
LABEL_CANDIDATES = ["class information", "class", "status", "label", "target"]
EXCLUDED_FEATURE_COLUMNS = {
    "id",
    "subject",
    "subject_id",
    "subject#",
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


@dataclass(frozen=True)
class PublicDataset:
    dataset_id: str
    name: str
    url: str
    output_subdir: str
    registry_status: str
    use_note: str
    citation_url: str


DATASETS = {
    "uci-parkinson-speech": PublicDataset(
        dataset_id="uci-parkinson-speech",
        name="UCI Parkinson's Speech with Multiple Types of Sound Recordings",
        url="https://archive.ics.uci.edu/static/public/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings.zip",
        output_subdir="uci-parkinson-speech",
        registry_status="feature-only",
        use_note="Feature-level Parkinson's/control sanity check; not raw-audio embedding validation.",
        citation_url="https://archive.ics.uci.edu/dataset/301/parkinson%2Bspeech%2Bdataset%2Bwith%2Bmultiple%2Btypes%2Bof%2BAudio%2Brecordings",
    ),
    "uci-parkinsons-telemonitoring": PublicDataset(
        dataset_id="uci-parkinsons-telemonitoring",
        name="UCI Parkinsons Telemonitoring",
        url="https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip",
        output_subdir="uci-parkinsons-telemonitoring",
        registry_status="feature-only",
        use_note="Longitudinal UPDRS/progression feature table; no healthy controls and not app model training.",
        citation_url="https://archive.ics.uci.edu/dataset/189/parkinsons%2Btelemonitoring",
    ),
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


def numeric_feature_columns(rows: list[dict[str, str]], excluded_columns: set[str]) -> list[str]:
    if not rows:
        return []
    columns: list[str] = []
    for column in rows[0].keys():
        if normalize_name(column) in excluded_columns:
            continue
        values = [parse_float(row.get(column)) for row in rows]
        if values and all(value is not None for value in values):
            columns.append(column)
    return columns


def label_bucket(value: str) -> str:
    normalized = normalize_name(value)
    if normalized in POSITIVE_VALUES:
        return "positive"
    if normalized in NEGATIVE_VALUES:
        return "negative"
    return "unknown"


def summarize_table(path: Path, root: Path) -> dict[str, object]:
    rows = read_table(path)
    fieldnames = list(rows[0].keys()) if rows else []
    speaker_column = find_column(fieldnames, SPEAKER_CANDIDATES)
    label_column = find_column(fieldnames, LABEL_CANDIDATES)
    updrs_columns = [fieldname for fieldname in fieldnames if "updrs" in normalize_name(fieldname)]
    excluded = set(EXCLUDED_FEATURE_COLUMNS)
    if speaker_column:
        excluded.add(normalize_name(speaker_column))
    if label_column:
        excluded.add(normalize_name(label_column))
    excluded.update(normalize_name(column) for column in updrs_columns)
    feature_columns = numeric_feature_columns(rows, excluded)
    speakers = {row.get(speaker_column, "").strip() for row in rows} if speaker_column else set()
    speakers.discard("")
    label_counts = Counter(row.get(label_column, "").strip() for row in rows) if label_column else Counter()
    label_buckets = Counter(label_bucket(label) for label in label_counts)
    classification_ready = bool(speaker_column and label_column and feature_columns and label_buckets["positive"] and label_buckets["negative"])
    progression_ready = bool(speaker_column and updrs_columns and feature_columns)
    notes: list[str] = []
    if not speaker_column:
        notes.append("No speaker/subject column detected; speaker-level splits are not possible yet.")
    if not label_column:
        notes.append("No class/label column detected; current classifier training should not run on this table.")
    elif not (label_buckets["positive"] and label_buckets["negative"]):
        notes.append("Class labels do not include both positive and negative groups.")
    if not feature_columns:
        notes.append("No fully numeric feature columns detected after exclusions.")
    if progression_ready and not classification_ready:
        notes.append("Progression analysis candidate only; do not use as PD/control classifier input.")
    return {
        "path": str(path.relative_to(root)),
        "rows": len(rows),
        "columns": len(fieldnames),
        "speaker_column": speaker_column,
        "speakers": len(speakers),
        "label_column": label_column,
        "label_counts": dict(label_counts),
        "updrs_columns": updrs_columns,
        "numeric_feature_count": len(feature_columns),
        "numeric_feature_columns": feature_columns[:20],
        "classification_ready": classification_ready,
        "progression_ready": progression_ready,
        "notes": notes,
    }


def safe_extract_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = output_dir / member.filename
            resolved = target.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(f"Unsafe archive path: {member.filename}") from error
            archive.extract(member, output_dir)
            extracted.append(target)
    return extracted


def download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        output_path.write_bytes(response.read())


def table_candidates(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES and not path.name.lower().endswith(".names")
    )


def nested_archives(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in NESTED_ARCHIVE_SUFFIXES)


def available_extractor() -> tuple[str, list[str]] | None:
    candidates = [
        ("unar", ["unar", "-quiet", "-o"]),
        ("7z", ["7z", "x", "-y", "-o"]),
        ("unrar", ["unrar", "x", "-o+"]),
    ]
    for name, command in candidates:
        if shutil.which(name):
            return name, command
    return None


def extract_nested_archive(archive_path: Path, output_dir: Path, extractor: tuple[str, list[str]]) -> None:
    name, command = extractor
    if name == "unar":
        subprocess.run([*command, str(output_dir), str(archive_path)], check=True)
    elif name == "7z":
        subprocess.run([f"{command[0]}", command[1], command[2], f"{command[3]}{output_dir}", str(archive_path)], check=True)
    else:
        subprocess.run([*command, str(archive_path), str(output_dir)], check=True)


def write_extraction_note(output_dir: Path, archives: list[Path]) -> Path:
    note_path = output_dir / "EXTRACTION_REQUIRED.md"
    archive_lines = "\n".join(f"- `{path.relative_to(output_dir)}`" for path in archives)
    note_path.write_text(
        "\n".join(
            [
                "# Manual Extraction Required",
                "",
                "The downloaded UCI package contains nested archives that Python's standard library cannot extract.",
                "Install `unar`, `unrar`, or `7z`, then re-run with `--allow-external-extractors`, or extract these files manually:",
                "",
                archive_lines,
                "",
                "Keep extracted dataset files under `research/datasets/`; do not commit raw or derived data.",
            ]
        )
        + "\n"
    )
    return note_path


def write_manifest(
    manifest_path: Path,
    dataset: PublicDataset,
    source_url: str,
    archive_path: Path,
    extracted_files: list[Path],
    tables: list[Path],
    table_summaries: list[dict[str, object]],
    archives: list[Path],
    notes: list[str],
) -> None:
    root = manifest_path.parent
    payload = {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "registry_status": dataset.registry_status,
        "use_note": dataset.use_note,
        "citation_url": dataset.citation_url,
        "source_url": source_url,
        "downloaded_at": utc_now(),
        "archive": str(archive_path.relative_to(root)),
        "extracted_files": [str(path.relative_to(root)) for path in extracted_files if path.exists()],
        "table_candidates": [str(path.relative_to(root)) for path in tables],
        "table_summaries": table_summaries,
        "nested_archives": [str(path.relative_to(root)) for path in archives],
        "notes": notes,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def fetch_dataset(dataset: PublicDataset, args: argparse.Namespace, source_url: str | None = None) -> Path:
    url = source_url or dataset.url
    output_dir = args.output_root / dataset.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "download.zip"
    download(url, archive_path)
    extracted_files = safe_extract_zip(archive_path, output_dir)
    archives = nested_archives(output_dir)
    notes: list[str] = []

    extractor = available_extractor() if args.allow_external_extractors else None
    if archives and extractor:
        for archive in archives:
            extract_nested_archive(archive, output_dir, extractor)
        notes.append(f"Nested archives extracted with {extractor[0]}.")
    elif archives:
        note_path = write_extraction_note(output_dir, archives)
        notes.append(f"Nested archive extraction required; see {note_path.name}.")

    tables = table_candidates(output_dir)
    table_summaries = [summarize_table(table, output_dir) for table in tables]
    if not tables:
        notes.append("No feature table was found after extraction.")
    manifest_path = output_dir / "dataset_fetch_manifest.json"
    write_manifest(manifest_path, dataset, url, archive_path, extracted_files, tables, table_summaries, archives, notes)
    print(f"wrote dataset fetch manifest to {manifest_path}")
    for table in tables:
        print(f"table candidate: {table}")
    for summary in table_summaries:
        status = "classification-ready" if summary["classification_ready"] else "not classifier-ready"
        if summary["progression_ready"] and not summary["classification_ready"]:
            status = "progression-only"
        print(f"table readiness: {summary['path']} ({status})")
    for note in notes:
        print(f"note: {note}")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public feature-only datasets into ignored research/datasets folders.")
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], required=True)
    parser.add_argument("--output-root", type=Path, default=Path("research/datasets"))
    parser.add_argument("--source-url", help="Override source URL for a single dataset, mainly for tests.")
    parser.add_argument("--allow-external-extractors", action="store_true", help="Use installed unar, unrar, or 7z for nested archives.")
    args = parser.parse_args(argv)
    if args.dataset == "all" and args.source_url:
        parser.error("--source-url can only be used with one dataset")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_ids = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for dataset_id in dataset_ids:
        fetch_dataset(DATASETS[dataset_id], args, args.source_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
