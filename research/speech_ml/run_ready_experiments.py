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

from research.speech_ml import dataset_registry


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locally ready EarlyCare speech ML research experiments.")
    parser.add_argument("--registry", type=Path, default=dataset_registry.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--datasets-root", type=Path, default=dataset_registry.DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("research/artifacts"))
    parser.add_argument("--run-report", type=Path, help="JSON report path. Defaults to <output-dir>/ready_experiments_run.json.")
    parser.add_argument("--only", action="append", help="Run only this registry dataset id. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running experiments.")
    parser.add_argument("--include-progression", action="store_true", help="Also run progression-only analyses.")
    parser.add_argument("--experiment-prefix", default="ready", help="Prefix for generated experiment names.")
    args = parser.parse_args(argv)
    if args.run_report is None:
        args.run_report = args.output_dir / "ready_experiments_run.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    readiness = dataset_registry.build_readiness_report(registry_path=args.registry, datasets_root=args.datasets_root)
    actions = build_actions(args, readiness)
    if not actions:
        print("no locally ready experiments found")
    results = run_actions(actions, args.dry_run)
    failed = [result for result in results if result.get("status") == "failed"]
    report = {
        "generated_at": utc_now(),
        "dry_run": args.dry_run,
        "include_progression": args.include_progression,
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
