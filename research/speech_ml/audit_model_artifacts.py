#!/usr/bin/env python3
"""Audit EarlyCare speech ML artifacts before app handoff."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.speech_ml.make_enrichment_payload import BLOCKED_MODEL_COPY_PHRASES, REQUIRED_MODEL_CARD_FLAGS


ARTIFACT_SUFFIXES = {
    "embeddings": "_embeddings.jsonl",
    "evaluation": "_eval.json",
    "model": "_baseline_model.json",
    "personal_baselines": "_personal_baselines.json",
    "experiment_report": "_experiment.md",
    "model_card": "_model_card.md",
    "model_card_gate": "_model_card_gate.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def row_count(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def command_safe_action(gate: dict[str, object]) -> bool:
    action = gate.get("humanFollowUpAction")
    if not isinstance(action, str) or not action.strip():
        return False
    lowered = action.lower()
    return not any(phrase in lowered for phrase in BLOCKED_MODEL_COPY_PHRASES)


def discover_experiments(artifacts_dir: Path, run_report: Path | None, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(set(requested))

    experiments: set[str] = set()
    report_path = run_report or artifacts_dir / "ready_experiments_run.json"
    if report_path.exists():
        payload = read_json(report_path)
        actions = payload.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("kind") == "feature_baseline_training":
                    prefix = action.get("output_prefix")
                    if isinstance(prefix, str) and prefix:
                        experiments.add(prefix)

    for gate_path in artifacts_dir.glob("*_model_card_gate.json"):
        experiments.add(gate_path.name[: -len("_model_card_gate.json")])
    return sorted(experiments)


def artifact_paths(artifacts_dir: Path, experiment: str) -> dict[str, Path]:
    return {name: artifacts_dir / f"{experiment}{suffix}" for name, suffix in ARTIFACT_SUFFIXES.items()}


def gate_missing_flags(gate: dict[str, object]) -> list[str]:
    missing = [field for field in REQUIRED_MODEL_CARD_FLAGS if gate.get(field) is not True]
    if not command_safe_action(gate):
        missing.append("humanFollowUpAction")
    return sorted(set(missing))


def audit_experiment(artifacts_dir: Path, experiment: str) -> dict[str, object]:
    paths = artifact_paths(artifacts_dir, experiment)
    missing_files = [name for name, path in paths.items() if not path.exists()]
    evaluation = read_json(paths["evaluation"]) if paths["evaluation"].exists() else {}
    model = read_json(paths["model"]) if paths["model"].exists() else {}
    gate = read_json(paths["model_card_gate"]) if paths["model_card_gate"].exists() else {}
    metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
    split = evaluation.get("split") if isinstance(evaluation.get("split"), dict) else {}
    gate_missing = gate_missing_flags(gate) if gate else list(REQUIRED_MODEL_CARD_FLAGS)

    embedding_rows = row_count(paths["embeddings"]) if paths["embeddings"].exists() else 0
    evaluation_ok = evaluation.get("status") == "ok" and split.get("speaker_leakage") is False
    model_ok = model.get("status") == "ok"
    validated_model_allowed = not missing_files and evaluation_ok and model_ok and not gate_missing
    offline_embedding_allowed = embedding_rows > 0

    if validated_model_allowed:
        release_status = "validated-ready"
    elif evaluation_ok and model_ok:
        release_status = "research-only"
    else:
        release_status = "incomplete"

    blockers: list[str] = []
    if missing_files:
        blockers.append(f"missing files: {', '.join(missing_files)}")
    if not evaluation_ok:
        blockers.append("speaker-level evaluation is not ok")
    if not model_ok:
        blockers.append("baseline model is not ok")
    if gate_missing:
        blockers.append(f"release-gate checks missing: {', '.join(gate_missing)}")

    return {
        "experiment": experiment,
        "release_status": release_status,
        "validated_model_allowed": validated_model_allowed,
        "offline_embedding_allowed": offline_embedding_allowed,
        "embedding_rows": embedding_rows,
        "missing_files": missing_files,
        "blockers": blockers,
        "paths": {name: str(path) for name, path in paths.items()},
        "evaluation": {
            "status": evaluation.get("status"),
            "speaker_leakage": split.get("speaker_leakage"),
            "train_speakers": len(split.get("train_speakers", [])) if isinstance(split.get("train_speakers"), list) else 0,
            "test_speakers": len(split.get("test_speakers", [])) if isinstance(split.get("test_speakers"), list) else 0,
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "roc_auc": metrics.get("roc_auc"),
        },
        "model": {
            "status": model.get("status"),
            "model_type": model.get("model_type"),
            "train_balanced_accuracy": model.get("train_balanced_accuracy"),
            "embedding_dimensions": model.get("embedding_dimensions"),
        },
        "gate": {
            "missing": gate_missing,
            "human_follow_up_action": gate.get("humanFollowUpAction") if gate else None,
        },
    }


def build_report(artifacts_dir: Path, run_report: Path | None, experiments: list[str] | None) -> dict[str, object]:
    selected = discover_experiments(artifacts_dir, run_report, experiments)
    audits = [audit_experiment(artifacts_dir, experiment) for experiment in selected]
    return {
        "generated_at": utc_now(),
        "artifacts_dir": str(artifacts_dir),
        "run_report": str(run_report) if run_report else None,
        "experiments": audits,
        "counts": {
            "experiments": len(audits),
            "validated_ready": sum(1 for item in audits if item["validated_model_allowed"]),
            "research_only": sum(1 for item in audits if item["release_status"] == "research-only"),
            "incomplete": sum(1 for item in audits if item["release_status"] == "incomplete"),
        },
        "safety": {
            "intended_use": "Offline artifact review before app handoff.",
            "excluded_use": "Do not use research-only or incomplete artifacts as validated app models.",
        },
    }


def markdown_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "/")


def write_markdown(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    lines = [
        "# EarlyCare Speech Model Artifact Audit",
        "",
        "This is an offline release-gate review. Research-only artifacts must not be used as validated app models.",
        "",
        "## Summary",
        "",
        f"- Artifacts directory: `{report.get('artifacts_dir')}`",
        f"- Experiments: {counts.get('experiments', 0)}",
        f"- Validated-ready: {counts.get('validated_ready', 0)}",
        f"- Research-only: {counts.get('research_only', 0)}",
        f"- Incomplete: {counts.get('incomplete', 0)}",
        "",
        "## Experiments",
        "",
        "| Experiment | Status | Embedding Rows | Eval Balanced Accuracy | Sensitivity | Specificity | ROC-AUC | Gate Blockers |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report.get("experiments", []):
        if not isinstance(item, dict):
            continue
        evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), dict) else {}
        gate = item.get("gate") if isinstance(item.get("gate"), dict) else {}
        missing = gate.get("missing") if isinstance(gate.get("missing"), list) else []
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_value(item.get("experiment")),
                    markdown_value(item.get("release_status")),
                    markdown_value(item.get("embedding_rows")),
                    markdown_value(evaluation.get("balanced_accuracy")),
                    markdown_value(evaluation.get("sensitivity")),
                    markdown_value(evaluation.get("specificity")),
                    markdown_value(evaluation.get("roc_auc")),
                    markdown_value(", ".join(str(value) for value in missing)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Safety", "", f"- {report['safety']['excluded_use']}"])
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit EarlyCare speech ML research artifacts before app handoff.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("research/artifacts"))
    parser.add_argument("--run-report", type=Path, help="Optional ready_experiments_run.json used for experiment discovery.")
    parser.add_argument("--experiment", action="append", help="Audit only this experiment prefix. Can be repeated.")
    parser.add_argument("--output", type=Path, help="Markdown audit report path. Defaults to <artifacts-dir>/model_artifact_audit.md.")
    parser.add_argument("--json-output", type=Path, help="JSON audit report path. Defaults to <artifacts-dir>/model_artifact_audit.json.")
    parser.add_argument("--require-validated", action="store_true", help="Exit non-zero unless every selected experiment is validated-ready.")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = args.artifacts_dir / "model_artifact_audit.md"
    if args.json_output is None:
        args.json_output = args.artifacts_dir / "model_artifact_audit.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.artifacts_dir, args.run_report, args.experiment)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(args.output, report)
    print(f"wrote model artifact audit to {args.output}")
    print(f"wrote model artifact audit json to {args.json_output}")
    if args.require_validated and report["counts"]["validated_ready"] != report["counts"]["experiments"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
