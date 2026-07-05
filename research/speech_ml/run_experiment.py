#!/usr/bin/env python3
"""Run a complete offline EarlyCare speech ML research experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from research.speech_ml import convert_feature_table, evaluate_baseline, extract_embeddings, train_baseline


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "speech-experiment"


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in sorted(counter.items())]


def manifest_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    datasets = Counter((row.get("dataset") or "unknown").strip() or "unknown" for row in rows)
    labels = Counter((row.get("label") or "unknown").strip() or "unknown" for row in rows)
    tasks = Counter((row.get("task") or "unknown").strip() or "unknown" for row in rows)
    speakers = {
        (
            (row.get("dataset") or "unknown").strip() or "unknown",
            (row.get("speaker_id") or row.get("speaker") or "").strip(),
        )
        for row in rows
        if (row.get("speaker_id") or row.get("speaker") or "").strip()
    }
    review_rows = [row for row in rows if (row.get("review_status") or "").strip().lower() == "needs-review"]
    return {
        "rows": len(rows),
        "speakers": len(speakers),
        "datasets": datasets,
        "labels": labels,
        "tasks": tasks,
        "needs_review": len(review_rows),
    }


def review_row_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if (row.get("review_status") or "").strip().lower() == "needs-review")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def artifact_paths(output_dir: Path, slug: str) -> dict[str, Path]:
    return {
        "embeddings": output_dir / f"{slug}_embeddings.jsonl",
        "evaluation": output_dir / f"{slug}_eval.json",
        "model": output_dir / f"{slug}_baseline_model.json",
        "report": output_dir / f"{slug}_experiment.md",
    }


def extract_args(args: argparse.Namespace, embeddings_path: Path) -> list[str]:
    values = [
        "--manifest",
        str(args.manifest),
        "--audio-root",
        str(args.audio_root),
        "--output",
        str(embeddings_path),
        "--model",
        args.model,
        "--dimensions",
        str(args.dimensions),
        "--device",
        args.device,
    ]
    if args.trust_remote_code:
        values.append("--trust-remote-code")
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    return values


def feature_table_args(args: argparse.Namespace, embeddings_path: Path) -> list[str]:
    values = [
        "--input",
        str(args.feature_table),
        "--output",
        str(embeddings_path),
        "--dataset",
        args.dataset,
        "--language",
        args.language,
        "--task",
        args.task,
    ]
    optional_pairs = [
        ("--speaker-column", args.speaker_column),
        ("--label-column", args.label_column),
        ("--task-column", args.task_column),
        ("--updrs-column", args.updrs_column),
        ("--source-id-column", args.source_id_column),
        ("--exclude-columns", args.exclude_columns),
    ]
    for flag, value in optional_pairs:
        if value:
            values.extend([flag, value])
    return values


def evaluation_args(args: argparse.Namespace, embeddings_path: Path, evaluation_path: Path) -> list[str]:
    values = [
        "--input",
        str(embeddings_path),
        "--output",
        str(evaluation_path),
        "--positive-labels",
        args.positive_labels,
        "--test-fraction",
        str(args.test_fraction),
    ]
    if args.train_dataset and args.test_dataset:
        values.extend(["--train-dataset", args.train_dataset, "--test-dataset", args.test_dataset])
    return values


def training_args(args: argparse.Namespace, embeddings_path: Path, model_path: Path) -> list[str]:
    return [
        "--input",
        str(embeddings_path),
        "--output",
        str(model_path),
        "--positive-labels",
        args.positive_labels,
    ]


def subgroup_report_lines(metrics: dict[str, object]) -> list[str]:
    subgroups = metrics.get("subgroups")
    if not isinstance(subgroups, dict):
        return ["- Not available"]

    lines: list[str] = []
    for field in ["dataset", "task", "language", "source_type"]:
        groups = subgroups.get(field)
        if not isinstance(groups, dict) or not groups:
            continue
        lines.append(f"### {field.replace('_', ' ').title()}")
        lines.append("")
        for name, values in sorted(groups.items()):
            if not isinstance(values, dict):
                continue
            lines.append(
                f"- {name}: speakers={values.get('speakers')}, "
                f"balanced_accuracy={values.get('balanced_accuracy')}, "
                f"sensitivity={values.get('sensitivity')}, "
                f"specificity={values.get('specificity')}"
            )
        lines.append("")
    return lines or ["- Not available"]


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
    summary: dict[str, object],
    evaluation: dict[str, object],
    model: dict[str, object],
) -> None:
    metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
    split = evaluation.get("split") if isinstance(evaluation.get("split"), dict) else {}
    input_label = args.manifest if args.manifest else args.feature_table
    input_mode = "raw-audio manifest" if args.manifest else "feature table"
    model_label = args.model if args.manifest else "feature-table-zscore"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# EarlyCare Speech ML Experiment: {args.experiment_name}",
        "",
        f"- Generated at: {utc_now()}",
        f"- Input mode: {input_mode}",
        f"- Input: `{input_label}`",
        f"- Audio root: `{args.audio_root}`" if args.manifest else f"- Source type: feature table",
        f"- Model: `{model_label}`",
        "- Scope: offline research only; not a diagnosis model and not app runtime routing.",
        "",
        "## Artifacts",
        "",
        f"- Embeddings: `{paths['embeddings']}`",
        f"- Evaluation: `{paths['evaluation']}`",
        f"- Baseline model: `{paths['model']}`",
        "",
        "## Manifest Summary",
        "",
        f"- Rows: {summary['rows']}",
        f"- Speakers: {summary['speakers']}",
        f"- Rows needing review: {summary['needs_review']}",
        "",
        "### Datasets",
        "",
        *counter_lines(summary["datasets"]),  # type: ignore[arg-type]
        "",
        "### Labels",
        "",
        *counter_lines(summary["labels"]),  # type: ignore[arg-type]
        "",
        "### Tasks",
        "",
        *counter_lines(summary["tasks"]),  # type: ignore[arg-type]
        "",
        "## Evaluation",
        "",
        f"- Status: {evaluation.get('status')}",
        f"- Split mode: {split.get('mode')}",
        f"- Speaker leakage: {split.get('speaker_leakage')}",
    ]
    if metrics:
        lines.extend(
            [
                f"- Balanced accuracy: {metrics.get('balanced_accuracy')}",
                f"- ROC-AUC: {metrics.get('roc_auc')}",
                f"- Sensitivity: {metrics.get('sensitivity')}",
                f"- Specificity: {metrics.get('specificity')}",
                "",
                "## Subgroup Checks",
                "",
                *subgroup_report_lines(metrics),
            ]
        )
    else:
        lines.append(f"- Reason: {evaluation.get('reason')}")

    lines.extend(
        [
            "",
            "## Baseline Model",
            "",
            f"- Status: {model.get('status')}",
            f"- Type: {model.get('model_type')}",
            f"- Embedding dimensions: {model.get('embedding_dimensions')}",
            f"- Train balanced accuracy: {model.get('train_balanced_accuracy')}",
            "",
            "## Safety Notes",
            "",
            "- This artifact can support speech-deviation research only.",
            "- Do not present it as Parkinson's, concussion, TBI, stroke, or emergency diagnosis.",
            "- Do not load it in FastAPI request paths without a completed model card, subgroup checks, latency checks, and human follow-up workflow.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EarlyCare offline speech extraction, evaluation, training, and reporting.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path, help="Reviewed CSV or JSONL manifest.")
    input_group.add_argument("--feature-table", type=Path, help="Feature-only CSV or TSV dataset.")
    parser.add_argument("--audio-root", type=Path, default=Path("research/datasets"), help="Folder that manifest audio_path values are relative to.")
    parser.add_argument("--output-dir", type=Path, default=Path("research/artifacts"))
    parser.add_argument("--experiment-name", help="Human-readable experiment name. Defaults to manifest stem plus model.")
    parser.add_argument("--model", choices=["demo", "meralion", "wavlm", "wav2vec2"], default="demo")
    parser.add_argument("--dimensions", type=int, default=16, help="Demo embedding dimensions.")
    parser.add_argument("--device", default="cpu", help="Device for optional heavy encoders.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Hugging Face model loading.")
    parser.add_argument("--positive-labels", default=",".join(sorted(evaluate_baseline.DEFAULT_POSITIVE_LABELS)))
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--train-dataset", help="Train only on this dataset name.")
    parser.add_argument("--test-dataset", help="Test only on this dataset name.")
    parser.add_argument("--limit", type=int, help="Process only the first N manifest rows.")
    parser.add_argument("--allow-review-rows", action="store_true", help="Allow rows marked needs-review to proceed.")
    parser.add_argument("--dataset", default="UCI Parkinson Speech", help="Dataset name for --feature-table mode.")
    parser.add_argument("--language", default="", help="Language stored in feature-table provenance.")
    parser.add_argument("--speaker-column", help="Feature-table speaker/subject column.")
    parser.add_argument("--label-column", help="Feature-table class/label column.")
    parser.add_argument("--task-column", help="Feature-table task/sample column.")
    parser.add_argument("--updrs-column", help="Feature-table UPDRS column stored in provenance.")
    parser.add_argument("--source-id-column", help="Feature-table source row id column.")
    parser.add_argument("--task", default="feature_table", help="Feature-table task value when no task column exists.")
    parser.add_argument("--exclude-columns", help="Comma-separated feature-table columns excluded from the vector.")
    args = parser.parse_args(argv)
    if (args.train_dataset and not args.test_dataset) or (args.test_dataset and not args.train_dataset):
        parser.error("--train-dataset and --test-dataset must be provided together")
    if not args.experiment_name:
        input_stem = args.manifest.stem if args.manifest else args.feature_table.stem
        model_name = args.model if args.manifest else "feature-table"
        args.experiment_name = f"{input_stem}-{model_name}"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    slug = slugify(args.experiment_name)
    paths = artifact_paths(args.output_dir, slug)

    if args.manifest:
        manifest_rows = read_manifest_rows(args.manifest)
        unresolved = review_row_count(manifest_rows)
        if unresolved and not args.allow_review_rows:
            raise SystemExit(f"Manifest has {unresolved} rows marked needs-review. Review them or pass --allow-review-rows.")
        extract_embeddings.main(extract_args(args, paths["embeddings"]))
        summary = manifest_summary(manifest_rows)
    else:
        convert_feature_table.main(feature_table_args(args, paths["embeddings"]))
        summary = manifest_summary(
            [
                {
                    "dataset": str(row.get("dataset") or "unknown"),
                    "speaker_id": str(row.get("speaker_id") or ""),
                    "label": str(row.get("label") or "unknown"),
                    "task": str(row.get("task") or "feature_table"),
                    "review_status": "",
                }
                for row in read_jsonl(paths["embeddings"])
            ]
        )

    evaluate_baseline.main(evaluation_args(args, paths["embeddings"], paths["evaluation"]))
    train_baseline.main(training_args(args, paths["embeddings"], paths["model"]))

    evaluation = read_json(paths["evaluation"])
    model = read_json(paths["model"])
    write_report(paths["report"], args, paths, summary, evaluation, model)
    print(f"wrote experiment report to {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
