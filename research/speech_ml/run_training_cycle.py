#!/usr/bin/env python3
"""Orchestrate the local EarlyCare speech ML training cycle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.speech_ml import audit_model_artifacts, dataset_registry, fetch_public_datasets, run_ready_experiments


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_registry(path: Path) -> dict[str, object]:
    payload = dataset_registry.read_json(path)
    dataset_registry.validate_registry(payload)
    return payload


def readiness_by_id(readiness: dict[str, object]) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for entry in readiness.get("datasets", []):
        if isinstance(entry, dict) and entry.get("id"):
            entries[str(entry["id"])] = entry
    return entries


def selected_dataset(dataset_id: str, only: list[str] | None) -> bool:
    return not only or dataset_id in only


def build_fetch_actions(
    registry: dict[str, object],
    readiness: dict[str, object],
    datasets_root: Path,
    only: list[str] | None,
    force_fetch: bool,
    allow_external_extractors: bool,
) -> list[dict[str, object]]:
    readiness_entries = readiness_by_id(readiness)
    actions: list[dict[str, object]] = []
    for dataset in registry.get("datasets", []):
        if not isinstance(dataset, dict):
            continue
        dataset_id = str(dataset.get("id") or "")
        if not dataset_id or not selected_dataset(dataset_id, only):
            continue
        fetcher_id = dataset.get("fetcher_dataset_id")
        if not isinstance(fetcher_id, str) or fetcher_id not in fetch_public_datasets.DATASETS:
            continue

        local_status = str(readiness_entries.get(dataset_id, {}).get("local_status") or "unknown")
        manifest_path = datasets_root / fetcher_id / "dataset_fetch_manifest.json"
        already_ready = local_status in {"classification-ready", "progression-ready"}
        status = "planned"
        reason = "fetch requested"
        if already_ready and not force_fetch:
            status = "skipped-existing"
            reason = "local manifest is already ready"

        command = [
            sys.executable,
            str(Path(__file__).with_name("fetch_public_datasets.py")),
            "--dataset",
            fetcher_id,
            "--output-root",
            str(datasets_root),
        ]
        if allow_external_extractors:
            command.append("--allow-external-extractors")

        actions.append(
            {
                "dataset_id": dataset_id,
                "fetcher_dataset_id": fetcher_id,
                "name": dataset.get("name"),
                "local_status_before": local_status,
                "manifest_path": str(manifest_path),
                "status": status,
                "reason": reason,
                "command": command,
            }
        )
    return actions


def run_fetch_actions(
    actions: list[dict[str, object]],
    datasets_root: Path,
    dry_run: bool,
    allow_external_extractors: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    fetch_args = argparse.Namespace(output_root=datasets_root, allow_external_extractors=allow_external_extractors)
    for action in actions:
        if action.get("status") == "skipped-existing":
            results.append(action)
            continue
        fetcher_id = str(action["fetcher_dataset_id"])
        if dry_run:
            print("planned fetch: " + " ".join(str(part) for part in action["command"]))
            results.append({**action, "status": "planned"})
            continue
        try:
            manifest_path = fetch_public_datasets.fetch_dataset(fetch_public_datasets.DATASETS[fetcher_id], fetch_args)
        except Exception as error:  # pragma: no cover - defensive reporting for external downloads/extractors.
            results.append({**action, "status": "failed", "error": str(error)})
            break
        results.append({**action, "status": "ok", "manifest_path": str(manifest_path)})
    return results


def run_ready_args(args: argparse.Namespace, dry_run: bool) -> list[str]:
    values = [
        "--registry",
        str(args.registry),
        "--datasets-root",
        str(args.datasets_root),
        "--output-dir",
        str(args.output_dir),
        "--run-report",
        str(args.ready_run_report),
        "--experiment-prefix",
        args.experiment_prefix,
    ]
    for dataset_id in args.only or []:
        values.extend(["--only", dataset_id])
    if dry_run:
        values.append("--dry-run")
    if args.include_progression:
        values.append("--include-progression")
    if args.audit:
        values.append("--audit")
        values.extend(["--audit-output", str(args.audit_output), "--audit-json-output", str(args.audit_json_output)])
    if args.require_validated:
        values.append("--require-validated")
    return values


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_existing_audit(args: argparse.Namespace) -> int:
    values = [
        "--artifacts-dir",
        str(args.output_dir),
        "--output",
        str(args.audit_output),
        "--json-output",
        str(args.audit_json_output),
    ]
    if args.require_validated:
        values.append("--require-validated")
    return audit_model_artifacts.main(values)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def count_value(report: dict[str, object] | None, key: str) -> object:
    if not report:
        return 0
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    return counts.get(key, 0)


def write_markdown(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = report.get("readiness_after") if isinstance(report.get("readiness_after"), dict) else {}
    ready_run = report.get("ready_run") if isinstance(report.get("ready_run"), dict) else None
    audit = report.get("audit") if isinstance(report.get("audit"), dict) else None
    lines = [
        "# EarlyCare Speech ML Training Cycle",
        "",
        "This is an offline research report. It is not a validated app model and must not be used for diagnosis.",
        "",
        "## Summary",
        "",
        f"- Dry run: {report.get('dry_run')}",
        f"- Fetch requested: {report.get('fetch_requested')}",
        f"- Ready experiments requested: {report.get('run_ready_requested')}",
        f"- Audit requested: {report.get('audit_requested')}",
        f"- Trainable now after cycle: {count_value(readiness, 'trainable_now')}",
        f"- Analysis ready after cycle: {count_value(readiness, 'analysis_ready')}",
        f"- Ready actions succeeded: {ready_run.get('actions_succeeded', 0) if ready_run else 0}",
        f"- Ready actions failed: {ready_run.get('actions_failed', 0) if ready_run else 0}",
        f"- Validated-ready artifacts: {count_value(audit, 'validated_ready')}",
        f"- Research-only artifacts: {count_value(audit, 'research_only')}",
        "",
        "## Fetch Actions",
        "",
        "| Dataset | Status | Reason |",
        "| --- | --- | --- |",
    ]
    fetch_results = report.get("fetch_results") if isinstance(report.get("fetch_results"), list) else []
    if fetch_results:
        for action in fetch_results:
            if isinstance(action, dict):
                lines.append(f"| {action.get('dataset_id')} | {action.get('status')} | {str(action.get('reason', '')).replace('|', '/')} |")
    else:
        lines.append("| none | skipped | fetch not requested |")
    lines.extend(["", "## Safety", "", f"- {report['safety']['excluded_use']}"])
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or plan the EarlyCare offline speech ML training cycle.")
    parser.add_argument("--registry", type=Path, default=dataset_registry.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--datasets-root", type=Path, default=dataset_registry.DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("research/artifacts"))
    parser.add_argument("--report", type=Path, help="JSON cycle report. Defaults to <output-dir>/training_cycle_report.json.")
    parser.add_argument("--markdown-report", type=Path, help="Markdown cycle report. Defaults to <output-dir>/training_cycle_report.md.")
    parser.add_argument("--readiness-output", type=Path, help="Markdown readiness report. Defaults to <output-dir>/dataset_readiness.md.")
    parser.add_argument("--readiness-json-output", type=Path, help="JSON readiness report. Defaults to <output-dir>/dataset_readiness.json.")
    parser.add_argument("--ready-run-report", type=Path, help="Ready experiment report. Defaults to <output-dir>/ready_experiments_run.json.")
    parser.add_argument("--audit-output", type=Path, help="Markdown artifact audit. Defaults to <output-dir>/model_artifact_audit.md.")
    parser.add_argument("--audit-json-output", type=Path, help="JSON artifact audit. Defaults to <output-dir>/model_artifact_audit.json.")
    parser.add_argument("--only", action="append", help="Limit to this registry dataset id. Can be repeated.")
    parser.add_argument("--fetch-supported", action="store_true", help="Fetch supported public datasets before checking readiness.")
    parser.add_argument("--force-fetch", action="store_true", help="Re-fetch supported datasets even when a ready local manifest exists.")
    parser.add_argument("--allow-external-extractors", action="store_true", help="Allow unar, unrar, or 7z for nested dataset archives.")
    parser.add_argument("--run-ready", action="store_true", help="Run locally ready feature-baseline experiments.")
    parser.add_argument("--include-progression", action="store_true", help="With --run-ready, also run progression-only analyses.")
    parser.add_argument("--audit", action="store_true", help="Audit generated or existing artifacts.")
    parser.add_argument("--require-validated", action="store_true", help="With --audit, exit non-zero unless audited artifacts are validated-ready.")
    parser.add_argument("--dry-run", action="store_true", help="Plan fetch and training commands without downloads or model training.")
    parser.add_argument("--experiment-prefix", default="ready", help="Prefix for generated experiment names.")
    args = parser.parse_args(argv)
    if args.report is None:
        args.report = args.output_dir / "training_cycle_report.json"
    if args.markdown_report is None:
        args.markdown_report = args.output_dir / "training_cycle_report.md"
    if args.readiness_output is None:
        args.readiness_output = args.output_dir / "dataset_readiness.md"
    if args.readiness_json_output is None:
        args.readiness_json_output = args.output_dir / "dataset_readiness.json"
    if args.ready_run_report is None:
        args.ready_run_report = args.output_dir / "ready_experiments_run.json"
    if args.audit_output is None:
        args.audit_output = args.output_dir / "model_artifact_audit.md"
    if args.audit_json_output is None:
        args.audit_json_output = args.output_dir / "model_artifact_audit.json"
    if args.require_validated and not args.audit:
        parser.error("--require-validated requires --audit")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = read_registry(args.registry)
    readiness_before = dataset_registry.build_readiness_report(args.registry, args.datasets_root)
    fetch_actions = build_fetch_actions(
        registry,
        readiness_before,
        args.datasets_root,
        args.only,
        args.force_fetch,
        args.allow_external_extractors,
    )
    fetch_results = run_fetch_actions(fetch_actions, args.datasets_root, args.dry_run, args.allow_external_extractors) if args.fetch_supported else []
    fetch_failed = any(action.get("status") == "failed" for action in fetch_results)

    readiness_after = dataset_registry.build_readiness_report(args.registry, args.datasets_root)
    dataset_registry.write_markdown(args.readiness_output, readiness_after)
    write_json(args.readiness_json_output, readiness_after)
    print(f"wrote dataset readiness report to {args.readiness_output}")
    print(f"wrote dataset readiness json to {args.readiness_json_output}")

    ready_code = 0
    if args.run_ready and not fetch_failed:
        ready_code = run_ready_experiments.main(run_ready_args(args, args.dry_run))

    audit_code = 0
    if args.audit and not args.run_ready and not args.dry_run and not fetch_failed:
        audit_code = run_existing_audit(args)

    ready_run = read_json_if_exists(args.ready_run_report)
    audit_report = read_json_if_exists(args.audit_json_output)
    report = {
        "generated_at": utc_now(),
        "dry_run": args.dry_run,
        "fetch_requested": args.fetch_supported,
        "run_ready_requested": args.run_ready,
        "audit_requested": args.audit,
        "registry": str(args.registry),
        "datasets_root": str(args.datasets_root),
        "output_dir": str(args.output_dir),
        "fetch_results": fetch_results,
        "readiness_before": readiness_before,
        "readiness_after": readiness_after,
        "ready_run": ready_run,
        "audit": audit_report,
        "safety": {
            "intended_use": "Offline speech ML dataset acquisition, training orchestration, and artifact audit.",
            "excluded_use": "Do not use generated artifacts as diagnostic output or validated app models unless the audit marks them validated-ready.",
        },
    }
    write_json(args.report, report)
    write_markdown(args.markdown_report, report)
    print(f"wrote training cycle report to {args.report}")
    print(f"wrote training cycle markdown to {args.markdown_report}")
    return 1 if fetch_failed or ready_code or audit_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
