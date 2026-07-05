#!/usr/bin/env python3
"""Inspect EarlyCare speech ML dataset readiness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("dataset_registry.json")
DEFAULT_DATASETS_ROOT = Path("research/datasets")
REQUIRED_DATASET_FIELDS = {
    "id",
    "name",
    "status",
    "target",
    "source_urls",
    "labels",
    "language",
    "tasks",
    "participants",
    "raw_audio",
    "training_mode",
    "earlycare_use",
    "required_before_training",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def validate_registry(payload: dict[str, object]) -> None:
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset registry must contain a non-empty datasets list")
    seen: set[str] = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"dataset entry {index} is not an object")
        missing = sorted(REQUIRED_DATASET_FIELDS - set(dataset))
        if missing:
            raise ValueError(f"dataset entry {dataset.get('id', index)} is missing fields: {', '.join(missing)}")
        dataset_id = str(dataset["id"])
        if dataset_id in seen:
            raise ValueError(f"duplicate dataset id: {dataset_id}")
        seen.add(dataset_id)
        if not isinstance(dataset.get("source_urls"), list) or not dataset["source_urls"]:
            raise ValueError(f"dataset {dataset_id} must include at least one source URL")
        if not isinstance(dataset.get("required_before_training"), list):
            raise ValueError(f"dataset {dataset_id} required_before_training must be a list")


def command_for_fetch(fetcher_dataset_id: str) -> str:
    return f"python3 research/speech_ml/fetch_public_datasets.py --dataset {fetcher_dataset_id}"


def command_for_classifier(manifest_path: Path) -> str:
    return (
        "python3 research/speech_ml/run_experiment.py "
        f"--dataset-fetch-manifest {manifest_path} "
        "--output-dir research/artifacts"
    )


def command_for_progression(manifest_path: Path) -> str:
    return (
        "python3 research/speech_ml/analyze_progression_table.py "
        f"--dataset-fetch-manifest {manifest_path} "
        "--output research/artifacts/uci-telemonitoring_progression.json"
    )


def inspect_fetch_manifest(dataset: dict[str, object], datasets_root: Path) -> dict[str, object]:
    fetcher_dataset_id = dataset.get("fetcher_dataset_id")
    if not fetcher_dataset_id:
        return {
            "local_status": "not-applicable",
            "manifest_path": None,
            "ready_for": [],
            "next_action": "; ".join(str(item) for item in dataset.get("required_before_training", [])),
            "notes": ["No automated public fetcher is configured for this dataset."],
        }

    fetcher_id = str(fetcher_dataset_id)
    manifest_path = datasets_root / fetcher_id / "dataset_fetch_manifest.json"
    if not manifest_path.exists():
        return {
            "local_status": "not-fetched",
            "manifest_path": str(manifest_path),
            "ready_for": [],
            "next_action": command_for_fetch(fetcher_id),
            "notes": ["No local fetch manifest found."],
        }

    manifest = read_json(manifest_path)
    summaries = manifest.get("table_summaries") if isinstance(manifest.get("table_summaries"), list) else []
    classification_tables = [
        summary
        for summary in summaries
        if isinstance(summary, dict) and summary.get("classification_ready") is True
    ]
    progression_tables = [
        summary
        for summary in summaries
        if isinstance(summary, dict) and summary.get("progression_ready") is True
    ]
    nested_archives = manifest.get("nested_archives") if isinstance(manifest.get("nested_archives"), list) else []
    notes = list(manifest.get("notes", [])) if isinstance(manifest.get("notes"), list) else []

    if classification_tables:
        return {
            "local_status": "classification-ready",
            "manifest_path": str(manifest_path),
            "ready_for": ["feature_baseline_training"],
            "next_action": command_for_classifier(manifest_path),
            "notes": notes,
            "selected_table": classification_tables[0].get("path"),
        }
    if progression_tables:
        return {
            "local_status": "progression-ready",
            "manifest_path": str(manifest_path),
            "ready_for": ["progression_analysis"],
            "next_action": command_for_progression(manifest_path),
            "notes": notes,
            "selected_table": progression_tables[0].get("path"),
        }
    if nested_archives:
        return {
            "local_status": "extraction-required",
            "manifest_path": str(manifest_path),
            "ready_for": [],
            "next_action": f"{command_for_fetch(fetcher_id)} --allow-external-extractors",
            "notes": notes or ["Nested archive extraction is required before table readiness can be assessed."],
        }
    return {
        "local_status": "fetched-not-ready",
        "manifest_path": str(manifest_path),
        "ready_for": [],
        "next_action": "Review table summaries and dataset extraction notes.",
        "notes": notes or ["No trainable or progression-ready table was found."],
    }


def build_dataset_entry(dataset: dict[str, object], datasets_root: Path) -> dict[str, object]:
    local = inspect_fetch_manifest(dataset, datasets_root)
    ready_for = list(local["ready_for"]) if isinstance(local.get("ready_for"), list) else []
    training_mode = str(dataset["training_mode"])
    trainable_now = "feature_baseline_training" in ready_for or training_mode == "raw_audio_embedding_benchmark" and "raw_audio_embedding_extraction" in ready_for
    analysis_ready = "progression_analysis" in ready_for
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "status": dataset["status"],
        "target": dataset["target"],
        "training_mode": training_mode,
        "raw_audio": dataset["raw_audio"],
        "local_status": local["local_status"],
        "ready_for": ready_for,
        "trainable_now": trainable_now,
        "analysis_ready": analysis_ready,
        "app_model_allowed": False,
        "next_action": local["next_action"],
        "manifest_path": local["manifest_path"],
        "selected_table": local.get("selected_table"),
        "blockers": dataset.get("required_before_training", []),
        "notes": local["notes"],
        "source_urls": dataset["source_urls"],
        "earlycare_use": dataset["earlycare_use"],
    }


def build_readiness_report(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    target: str | None = None,
    status: str | None = None,
    ready_only: bool = False,
) -> dict[str, object]:
    registry = read_json(registry_path)
    validate_registry(registry)
    datasets = registry["datasets"]
    entries = [build_dataset_entry(dataset, datasets_root) for dataset in datasets if isinstance(dataset, dict)]
    if target:
        entries = [entry for entry in entries if entry["target"] == target]
    if status:
        entries = [entry for entry in entries if entry["status"] == status]
    if ready_only:
        entries = [entry for entry in entries if entry["trainable_now"] or entry["analysis_ready"]]
    return {
        "generated_at": utc_now(),
        "registry_path": str(registry_path),
        "datasets_root": str(datasets_root),
        "filters": {"target": target, "status": status, "ready_only": ready_only},
        "counts": {
            "datasets": len(entries),
            "trainable_now": sum(1 for entry in entries if entry["trainable_now"]),
            "analysis_ready": sum(1 for entry in entries if entry["analysis_ready"]),
            "app_model_allowed": sum(1 for entry in entries if entry["app_model_allowed"]),
        },
        "use_rules": registry.get("use_rules", []),
        "datasets": entries,
    }


def format_list(values: object) -> str:
    if not isinstance(values, list) or not values:
        return ""
    return ", ".join(str(value) for value in values)


def write_markdown(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    lines = [
        "# EarlyCare Dataset Readiness",
        "",
        "This is an offline research planning artifact. It is not a validated app model and must not be used for diagnosis or routing.",
        "",
        "## Summary",
        "",
        f"- Registry: `{report.get('registry_path')}`",
        f"- Datasets root: `{report.get('datasets_root')}`",
        f"- Datasets shown: {counts.get('datasets', 0)}",
        f"- Trainable now: {counts.get('trainable_now', 0)}",
        f"- Analysis ready: {counts.get('analysis_ready', 0)}",
        f"- App models allowed: {counts.get('app_model_allowed', 0)}",
        "",
        "## Dataset Table",
        "",
        "| Dataset | Status | Target | Local Status | Ready For | Next Action | App Model |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report.get("datasets", []):
        if not isinstance(entry, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry.get("name", "")),
                    str(entry.get("status", "")),
                    str(entry.get("target", "")),
                    str(entry.get("local_status", "")),
                    format_list(entry.get("ready_for")) or "blocked",
                    str(entry.get("next_action", "")).replace("|", "/"),
                    "blocked",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Use Rules", ""])
    for rule in report.get("use_rules", []):
        lines.append(f"- {rule}")
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect EarlyCare speech ML dataset readiness.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--target", help="Filter by exact target value, such as parkinsons-watch.")
    parser.add_argument("--status", help="Filter by exact registry status, such as feature-only.")
    parser.add_argument("--ready-only", action="store_true", help="Show only locally trainable or analysis-ready datasets.")
    parser.add_argument("--output", type=Path, help="Write a markdown readiness report.")
    parser.add_argument("--json-output", type=Path, help="Write a JSON readiness report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_readiness_report(
        registry_path=args.registry,
        datasets_root=args.datasets_root,
        target=args.target,
        status=args.status,
        ready_only=args.ready_only,
    )
    if args.output:
        write_markdown(args.output, report)
        print(f"wrote dataset readiness report to {args.output}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote dataset readiness json to {args.json_output}")
    if not args.output and not args.json_output:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
