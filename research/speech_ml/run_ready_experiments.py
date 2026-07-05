#!/usr/bin/env python3
"""Run locally ready EarlyCare speech ML research experiments."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.speech_ml import audit_model_artifacts, dataset_registry


SCRIPT_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "dataset"


def script_path(name: str) -> str:
    return str((SCRIPT_DIR / name).resolve())


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def build_actions(args: argparse.Namespace, readiness: dict[str, object]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for entry in readiness.get("datasets", []):
        if not isinstance(entry, dict):
            continue
        dataset_id = str(entry.get("id") or "")
        if args.only and dataset_id not in args.only:
            continue
        manifest_path = entry.get("manifest_path")
        if not manifest_path:
            continue
        ready_for = entry.get("ready_for") if isinstance(entry.get("ready_for"), list) else []
        if "feature_baseline_training" in ready_for:
            experiment_name = f"{args.experiment_prefix}-{dataset_id}" if args.experiment_prefix else dataset_id
            command = [
                sys.executable,
                script_path("run_experiment.py"),
                "--dataset-fetch-manifest",
                str(manifest_path),
                "--output-dir",
                str(args.output_dir),
                "--experiment-name",
                experiment_name,
            ]
            actions.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": entry.get("name"),
                    "kind": "feature_baseline_training",
                    "manifest_path": manifest_path,
                    "output_prefix": slugify(experiment_name),
                    "command": command,
                }
            )
        if args.include_progression and "progression_analysis" in ready_for:
            output_path = args.output_dir / f"{slugify(dataset_id)}_progression.json"
            command = [
                sys.executable,
                script_path("analyze_progression_table.py"),
                "--dataset-fetch-manifest",
                str(manifest_path),
                "--output",
                str(output_path),
            ]
            actions.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": entry.get("name"),
                    "kind": "progression_analysis",
                    "manifest_path": manifest_path,
                    "output_path": str(output_path),
                    "command": command,
                }
            )
    return actions


def run_actions(actions: list[dict[str, object]], dry_run: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for action in actions:
        command = action.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            results.append({**action, "status": "error", "returncode": 1, "error": "invalid command"})
            continue
        if dry_run:
            print(f"planned: {command_text(command)}")
            results.append({**action, "status": "planned", "returncode": None})
            continue
        print(f"running: {command_text(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        results.append(
            {
                **action,
                "status": "ok" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode != 0:
            break
    return results


def write_run_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    if args.dry_run:
        return {"status": "skipped", "reason": "dry-run"}
    report = audit_model_artifacts.build_report(args.output_dir, args.run_report, None)
    args.audit_json_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json_output.write_text(json.dumps(report, indent=2) + "\n")
    audit_model_artifacts.write_markdown(args.audit_output, report)
    print(f"wrote model artifact audit to {args.audit_output}")
    print(f"wrote model artifact audit json to {args.audit_json_output}")
    all_validated = report["counts"]["validated_ready"] == report["counts"]["experiments"]
    status = "ok" if not args.require_validated or all_validated else "failed"
    return {
        "status": status,
        "require_validated": args.require_validated,
        "output": str(args.audit_output),
        "json_output": str(args.audit_json_output),
        "counts": report["counts"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locally ready EarlyCare speech ML research experiments.")
    parser.add_argument("--registry", type=Path, default=dataset_registry.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--datasets-root", type=Path, default=dataset_registry.DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("research/artifacts"))
    parser.add_argument("--run-report", type=Path, help="JSON report path. Defaults to <output-dir>/ready_experiments_run.json.")
    parser.add_argument("--audit", action="store_true", help="Audit generated model artifacts after experiments run.")
    parser.add_argument("--audit-output", type=Path, help="Markdown audit report path. Defaults to <output-dir>/model_artifact_audit.md.")
    parser.add_argument("--audit-json-output", type=Path, help="JSON audit report path. Defaults to <output-dir>/model_artifact_audit.json.")
    parser.add_argument("--require-validated", action="store_true", help="With --audit, exit non-zero unless selected experiments are validated-ready.")
    parser.add_argument("--only", action="append", help="Run only this registry dataset id. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running experiments.")
    parser.add_argument("--include-progression", action="store_true", help="Also run progression-only analyses.")
    parser.add_argument("--experiment-prefix", default="ready", help="Prefix for generated experiment names.")
    args = parser.parse_args(argv)
    if args.run_report is None:
        args.run_report = args.output_dir / "ready_experiments_run.json"
    if args.audit_output is None:
        args.audit_output = args.output_dir / "model_artifact_audit.md"
    if args.audit_json_output is None:
        args.audit_json_output = args.output_dir / "model_artifact_audit.json"
    if args.require_validated and not args.audit:
        parser.error("--require-validated requires --audit")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    readiness = dataset_registry.build_readiness_report(registry_path=args.registry, datasets_root=args.datasets_root)
    actions = build_actions(args, readiness)
    if not actions:
        print("no locally ready experiments found")
    results = run_actions(actions, args.dry_run)
    failed = [result for result in results if result.get("status") == "failed"]
    report: dict[str, object] = {
        "generated_at": utc_now(),
        "dry_run": args.dry_run,
        "include_progression": args.include_progression,
        "audit_requested": args.audit,
        "registry": str(args.registry),
        "datasets_root": str(args.datasets_root),
        "output_dir": str(args.output_dir),
        "actions_planned": len(actions),
        "actions_succeeded": sum(1 for result in results if result.get("status") == "ok"),
        "actions_failed": len(failed),
        "actions": results or actions,
        "safety": {
            "intended_use": "Offline speech ML research orchestration.",
            "excluded_use": "Do not treat generated artifacts as a validated app model or diagnostic output.",
        },
    }
    write_run_report(args.run_report, report)
    print(f"wrote ready experiment run report to {args.run_report}")
    audit_status = {"status": "not-requested"}
    if args.audit:
        audit_status = run_audit(args)
        report["audit"] = audit_status
        write_run_report(args.run_report, report)
    return 1 if failed or audit_status.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
