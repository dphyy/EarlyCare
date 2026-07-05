#!/usr/bin/env python3
"""Prepare EarlyCare speech ML manifests from local audio folders."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
DEFAULT_POSITIVE_TOKENS = {"pd", "pwp", "parkinson", "parkinsonian", "patient", "patients"}
DEFAULT_NEGATIVE_TOKENS = {"control", "controls", "healthy", "hc", "normal"}
DEFAULT_TASK_TOKENS = {
    "ddk",
    "pataka",
    "pa-ta-ka",
    "vowel",
    "vowels",
    "phonation",
    "repeat",
    "sentence",
    "sentences",
    "monologue",
    "reading",
}


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    speaker_id: str
    label: str
    task: str
    audio_path: str
    language: str
    transcript: str
    source_id: str
    review_status: str


def split_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def parse_token_set(value: str | None, defaults: set[str]) -> set[str]:
    if not value:
        return defaults
    return {token.strip().lower() for token in value.split(",") if token.strip()}


def token_matches(tokens: list[str], expected: set[str]) -> bool:
    for token in tokens:
        if token in expected:
            return True
        if any(token.startswith(prefix) and any(char.isdigit() for char in token) for prefix in expected if len(prefix) >= 2):
            return True
    return False


def infer_label(relative_path: Path, positive_tokens: set[str], negative_tokens: set[str]) -> tuple[str, str]:
    tokens_by_part = [split_tokens(part) for part in relative_path.parts]
    for tokens in tokens_by_part:
        if token_matches(tokens, positive_tokens):
            return "pd", "inferred"
        if token_matches(tokens, negative_tokens):
            return "control", "inferred"
    return "unknown", "needs-review"


def infer_task(relative_path: Path) -> str:
    for part in reversed(relative_path.parts[:-1]):
        tokens = split_tokens(part)
        if token_matches(tokens, DEFAULT_TASK_TOKENS):
            return part
    return relative_path.parent.name if relative_path.parent != Path(".") else "unknown"


def infer_speaker_id(relative_path: Path, label: str, speaker_regex: str | None = None) -> str:
    relative_text = str(relative_path)
    if speaker_regex:
        match = re.search(speaker_regex, relative_text)
        if match:
            return match.group(1) if match.groups() else match.group(0)

    ignored = {"pd", "control", "controls", "healthy", "hc", "normal", "patient", "patients"} | DEFAULT_TASK_TOKENS
    for part in relative_path.parts[:-1]:
        tokens = split_tokens(part)
        if not tokens:
            continue
        if any(char.isdigit() for char in part):
            return part
        if any(token in ignored for token in tokens):
            continue

    stem_tokens = split_tokens(relative_path.stem)
    for token in stem_tokens:
        if token not in ignored and any(char.isdigit() for char in token):
            return token
    return f"{label}-{relative_path.stem}"


def manifest_rows(
    audio_root: Path,
    dataset: str,
    language: str,
    positive_tokens: set[str],
    negative_tokens: set[str],
    speaker_regex: str | None,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for path in sorted(audio_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        relative_path = path.relative_to(audio_root)
        label, review_status = infer_label(relative_path, positive_tokens, negative_tokens)
        speaker_id = infer_speaker_id(relative_path, label, speaker_regex)
        rows.append(
            ManifestRow(
                dataset=dataset,
                speaker_id=speaker_id,
                label=label,
                task=infer_task(relative_path),
                audio_path=str(relative_path),
                language=language,
                transcript="",
                source_id=path.stem,
                review_status=review_status,
            )
        )
    return rows


def write_manifest(rows: list[ManifestRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "speaker_id", "label", "task", "audio_path", "language", "transcript", "source_id", "review_status"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an EarlyCare speech ML manifest from a local audio dataset.")
    parser.add_argument("--audio-root", type=Path, required=True, help="Local ignored dataset folder to scan.")
    parser.add_argument("--output", type=Path, required=True, help="CSV manifest path to write under research/datasets.")
    parser.add_argument("--dataset", default="unknown", help="Dataset name stored in the manifest.")
    parser.add_argument("--language", default="", help="Language stored for every row when metadata is not available.")
    parser.add_argument("--positive-tokens", help="Comma-separated path tokens that imply a positive/Parkinsonian speaker.")
    parser.add_argument("--negative-tokens", help="Comma-separated path tokens that imply a control speaker.")
    parser.add_argument("--speaker-regex", help="Optional regex used to extract speaker_id from the relative path. First capture group wins.")
    args = parser.parse_args(argv)
    if not args.audio_root.exists():
        parser.error(f"--audio-root does not exist: {args.audio_root}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = manifest_rows(
        audio_root=args.audio_root,
        dataset=args.dataset,
        language=args.language,
        positive_tokens=parse_token_set(args.positive_tokens, DEFAULT_POSITIVE_TOKENS),
        negative_tokens=parse_token_set(args.negative_tokens, DEFAULT_NEGATIVE_TOKENS),
        speaker_regex=args.speaker_regex,
    )
    write_manifest(rows, args.output)
    needs_review = sum(1 for row in rows if row.review_status == "needs-review")
    print(f"wrote {len(rows)} rows to {args.output} ({needs_review} need review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
