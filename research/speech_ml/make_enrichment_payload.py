#!/usr/bin/env python3
"""Build backend speech-enrichment payloads from offline research rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


REQUIRED_MODEL_CARD_FLAGS = [
    "datasetAccessReviewed",
    "speakerSplitVerified",
    "evaluationMetricsRecorded",
    "subgroupChecksReviewed",
    "failureModesDocumented",
    "uiCopyReviewed",
    "humanFollowUpActionDefined",
    "rollbackPathDocumented",
]

BLOCKED_MODEL_COPY_PHRASES = (
    "parkinson's detected",
    "parkinson detected",
    "detected parkinson",
    "concussion detected",
    "detected concussion",
    "disease diagnosis",
    "medical certainty",
    "emergency confirmed",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def dict_field(row: dict[str, object], key: str) -> dict[str, object]:
    value = row.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Row field {key} must be an object")
    return dict(value)


def string_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def row_source_id(row: dict[str, object]) -> str | None:
    provenance = dict_field(row, "provenance")
    return string_value(provenance.get("source_id") or row.get("source_id"))


def row_matches(row: dict[str, object], args: argparse.Namespace) -> bool:
    provenance = dict_field(row, "provenance")
    checks = {
        "dataset": args.dataset,
        "speaker_id": args.speaker_id,
        "task": args.task,
        "source_id": args.source_id,
    }
    values = {
        "dataset": row.get("dataset") or provenance.get("dataset"),
        "speaker_id": row.get("speaker_id") or provenance.get("speaker_id"),
        "task": row.get("task") or provenance.get("task"),
        "source_id": row_source_id(row),
    }
    return all(expected is None or string_value(values[key]) == expected for key, expected in checks.items())


def select_row(rows: list[dict[str, object]], args: argparse.Namespace) -> tuple[dict[str, object], int]:
    filtered = [row for row in rows if row_matches(row, args)]
    if not filtered:
        raise SystemExit("No speech rows matched the requested filters")
    if args.row_index < 1 or args.row_index > len(filtered):
        raise SystemExit(f"--row-index must be between 1 and {len(filtered)} after filtering")
    return filtered[args.row_index - 1], rows.index(filtered[args.row_index - 1]) + 1


def payload_embedding(row: dict[str, object]) -> list[float]:
    metrics = dict_field(row, "speech_metrics")
    embedding = row.get("embedding") or metrics.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise SystemExit("Selected row needs a non-empty embedding")
    return [float(value) for value in embedding]


def speech_metrics(row: dict[str, object], embedding: list[float]) -> dict[str, object]:
    metrics = dict_field(row, "speech_metrics")
    provenance = dict_field(row, "provenance")
    metrics.setdefault("speechRate", 0)
    metrics.setdefault("avgPauseMs", 0)
    metrics.setdefault("responseLatencyMs", 0)
    metrics.setdefault("pitchVariability", 0)
    metrics.setdefault("phraseAccuracy", 0)
    metrics["embedding"] = embedding
    generated_at = provenance.get("extracted_at") or provenance.get("generated_at")
    if generated_at and not metrics.get("updatedAt"):
        metrics["updatedAt"] = generated_at
    return metrics


def load_model_card_gate(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit("--model-card-gate must point to a JSON object")
    return payload


def validate_model_card_gate(gate: dict[str, object] | None) -> None:
    if gate is None:
        raise SystemExit("validated model payloads require --model-card-gate")
    missing = [field for field in REQUIRED_MODEL_CARD_FLAGS if gate.get(field) is not True]
    if missing:
        raise SystemExit(f"validated model is missing release-gate checks: {', '.join(missing)}")
    action = string_value(gate.get("humanFollowUpAction"))
    if action is None:
        raise SystemExit("validated model requires humanFollowUpAction")
    lowered = action.lower()
    blocked = [phrase for phrase in BLOCKED_MODEL_COPY_PHRASES if phrase in lowered]
    if blocked:
        raise SystemExit(f"humanFollowUpAction uses blocked diagnosis language: {', '.join(blocked)}")


def build_payload(args: argparse.Namespace, row: dict[str, object], source_row_index: int) -> dict[str, object]:
    provenance = dict_field(row, "provenance")
    embedding = payload_embedding(row)
    model_name = args.model_name or string_value(provenance.get("model_name") or provenance.get("model")) or "offline speech embedding"
    feature_extractor = args.feature_extractor or string_value(provenance.get("feature_extractor") or provenance.get("model_name") or provenance.get("model")) or model_name
    model_version = args.model_version or string_value(provenance.get("model_version"))
    artifact_uri = args.artifact_uri or string_value(provenance.get("artifact_uri") or provenance.get("artifact")) or f"{args.input}#row={source_row_index}"
    model_card = load_model_card_gate(args.model_card_gate)

    provenance.setdefault("dataset", row.get("dataset"))
    provenance.setdefault("speaker_id", row.get("speaker_id"))
    provenance.setdefault("label", row.get("label"))
    provenance.setdefault("task", row.get("task"))
    provenance.setdefault("source_jsonl", str(args.input))
    provenance.setdefault("source_row_index", source_row_index)
    provenance.setdefault("artifact_uri", artifact_uri)

    if args.runtime_mode == "validated model":
        missing = [
            field
            for field, value in {
                "modelName": model_name,
                "modelVersion": model_version,
                "featureExtractor": feature_extractor,
                "artifactUri": artifact_uri,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit(f"validated model is missing required provenance: {', '.join(missing)}")
        validate_model_card_gate(model_card)

    payload: dict[str, object] = {
        "runtimeMode": args.runtime_mode,
        "featureExtractor": feature_extractor,
        "modelName": model_name,
        "artifactUri": artifact_uri,
        "embedding": embedding,
        "speech_metrics": speech_metrics(row, embedding),
        "provenance": provenance,
    }
    if model_version:
        payload["modelVersion"] = model_version
    if model_card is not None:
        payload["modelCard"] = model_card
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one offline speech JSONL row into a FastAPI speech-enrichment payload.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows from run_experiment.py, extract_embeddings.py, or convert_feature_table.py.")
    parser.add_argument("--output", type=Path, required=True, help="Payload JSON output under research/artifacts.")
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
    parser.add_argument("--model-card-gate", type=Path, help="JSON release-gate evidence from the model-card review.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.input)
    row, source_row_index = select_row(rows, args)
    payload = build_payload(args, row, source_row_index)
    write_json(args.output, payload)
    print(f"wrote speech enrichment payload to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
