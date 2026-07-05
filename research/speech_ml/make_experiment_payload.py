#!/usr/bin/env python3
"""Build app enrichment payloads from audited experiment artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.speech_ml import make_enrichment_payload


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def find_audit_entry(audit_report: Path, experiment: str) -> dict[str, object]:
    if not audit_report.exists():
        raise SystemExit(f"Audit report not found: {audit_report}. Run audit_model_artifacts.py first.")
    report = read_json(audit_report)
    experiments = report.get("experiments")
    if not isinstance(experiments, list):
        raise SystemExit(f"Audit report has no experiments list: {audit_report}")
    for entry in experiments:
        if isinstance(entry, dict) and entry.get("experiment") == experiment:
            return entry
    raise SystemExit(f"Experiment {experiment!r} was not found in audit report {audit_report}")


def validate_audit(entry: dict[str, object], runtime_mode: str) -> None:
    if runtime_mode == "validated model":
        if entry.get("validated_model_allowed") is not True:
            blockers = entry.get("blockers")
            detail = f": {', '.join(str(item) for item in blockers)}" if isinstance(blockers, list) and blockers else ""
            raise SystemExit(f"Audit does not allow validated model handoff for {entry.get('experiment')}{detail}")
        return
    if entry.get("offline_embedding_allowed") is not True:
        raise SystemExit(f"Audit does not allow offline embedding handoff for {entry.get('experiment')}")


def build_payload_args(args: argparse.Namespace) -> argparse.Namespace:
    input_path = args.artifacts_dir / f"{args.experiment}_embeddings.jsonl"
    model_card_gate = args.model_card_gate
    artifact_uri = args.artifact_uri
    if args.runtime_mode == "validated model":
        model_card_gate = model_card_gate or args.artifacts_dir / f"{args.experiment}_model_card_gate.json"
        artifact_uri = artifact_uri or str(args.artifacts_dir / f"{args.experiment}_baseline_model.json")
    return argparse.Namespace(
        input=input_path,
        output=args.output,
        row_index=args.row_index,
        dataset=args.dataset,
        speaker_id=args.speaker_id,
        source_id=args.source_id,
        task=args.task,
        runtime_mode=args.runtime_mode,
        feature_extractor=args.feature_extractor,
        model_name=args.model_name,
        model_version=args.model_version,
        artifact_uri=artifact_uri,
        model_card_gate=model_card_gate,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a speech-enrichment payload from an audited experiment artifact.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("research/artifacts"))
    parser.add_argument("--experiment", required=True, help="Experiment prefix, for example ready-uci-parkinson-speech.")
    parser.add_argument("--audit-report", type=Path, help="Audit JSON path. Defaults to <artifacts-dir>/model_artifact_audit.json.")
    parser.add_argument("--output", type=Path, required=True, help="Payload JSON output path.")
    parser.add_argument("--row-index", type=int, default=1, help="One-based row index after optional filters.")
    parser.add_argument("--dataset", help="Optional exact dataset filter.")
    parser.add_argument("--speaker-id", help="Optional exact speaker_id filter.")
    parser.add_argument("--source-id", help="Optional exact provenance.source_id filter.")
    parser.add_argument("--task", help="Optional exact task filter.")
    parser.add_argument("--runtime-mode", choices=["offline embedding", "validated model"], default="offline embedding")
    parser.add_argument("--feature-extractor", help="Override featureExtractor in the API payload.")
    parser.add_argument("--model-name", help="Override modelName in the API payload.")
    parser.add_argument("--model-version", help="Model version required for validated model payloads.")
    parser.add_argument("--artifact-uri", help="Stable artifact URI or model/run path.")
    parser.add_argument("--model-card-gate", type=Path, help="JSON release-gate evidence. Defaults to the experiment gate for validated model mode.")
    args = parser.parse_args(argv)
    if args.audit_report is None:
        args.audit_report = args.artifacts_dir / "model_artifact_audit.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_entry = find_audit_entry(args.audit_report, args.experiment)
    validate_audit(audit_entry, args.runtime_mode)
    payload_args = build_payload_args(args)
    rows = make_enrichment_payload.read_jsonl(payload_args.input)
    row, source_row_index = make_enrichment_payload.select_row(rows, payload_args)
    payload = make_enrichment_payload.build_payload(payload_args, row, source_row_index)
    provenance = payload.setdefault("provenance", {})
    if isinstance(provenance, dict):
        provenance.setdefault("artifact_audit", str(args.audit_report))
        provenance.setdefault("artifact_release_status", audit_entry.get("release_status"))
    make_enrichment_payload.write_json(args.output, payload)
    print(f"wrote audited speech enrichment payload to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
