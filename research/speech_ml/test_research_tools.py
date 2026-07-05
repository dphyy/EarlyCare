import csv
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

from research.speech_ml import (
    analyze_progression_table,
    audit_model_artifacts,
    build_personal_baselines,
    convert_feature_table,
    dataset_registry,
    evaluate_baseline,
    extract_embeddings,
    fetch_public_datasets,
    make_enrichment_payload,
    make_experiment_payload,
    prepare_manifest,
    run_ready_experiments,
    run_experiment,
    train_baseline,
)


def write_wav(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    duration_seconds = 0.25
    total_samples = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        for index in range(total_samples):
            value = int(12000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            audio.writeframes(struct.pack("<h", value))


class ResearchToolTests(unittest.TestCase):
    def test_run_experiment_script_help_works_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "research/speech_ml/run_experiment.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--dataset-fetch-manifest", result.stdout)

    def test_analyze_progression_script_help_works_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "research/speech_ml/analyze_progression_table.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--dataset-fetch-manifest", result.stdout)

    def test_dataset_registry_script_help_works_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "research/speech_ml/dataset_registry.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--ready-only", result.stdout)

    def test_run_ready_experiments_script_help_works_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "research/speech_ml/run_ready_experiments.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--include-progression", result.stdout)

    def test_audit_model_artifacts_script_help_works_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "research/speech_ml/audit_model_artifacts.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--require-validated", result.stdout)

    def test_make_experiment_payload_script_help_works_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "research/speech_ml/make_experiment_payload.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--audit-report", result.stdout)

    def test_dataset_registry_reports_fetch_manifest_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            datasets_root = root / "datasets"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "use_rules": ["no diagnosis"],
                        "datasets": [
                            {
                                "id": "uci",
                                "name": "UCI Parkinson Speech",
                                "status": "feature-only",
                                "target": "parkinsons-watch",
                                "source_urls": ["https://example.com/uci"],
                                "labels": "pd/control",
                                "language": "Turkish",
                                "tasks": ["vowels"],
                                "participants": "sample",
                                "raw_audio": "no-feature-table",
                                "fetcher_dataset_id": "uci-parkinson-speech",
                                "training_mode": "feature_classification_smoke",
                                "earlycare_use": "feature sanity check",
                                "required_before_training": ["fetch"],
                            },
                            {
                                "id": "tele",
                                "name": "UCI Telemonitoring",
                                "status": "feature-only",
                                "target": "parkinsons-progression",
                                "source_urls": ["https://example.com/tele"],
                                "labels": "updrs",
                                "language": "English",
                                "tasks": ["voice measures"],
                                "participants": "sample",
                                "raw_audio": "no-feature-table",
                                "fetcher_dataset_id": "uci-parkinsons-telemonitoring",
                                "training_mode": "progression_analysis",
                                "earlycare_use": "progression only",
                                "required_before_training": ["fetch"],
                            },
                            {
                                "id": "concussion",
                                "name": "Concussion pilots",
                                "status": "literature-only",
                                "target": "acute-concussion-research",
                                "source_urls": ["https://example.com/concussion"],
                                "labels": "study dependent",
                                "language": "study dependent",
                                "tasks": ["pataka"],
                                "participants": "sample",
                                "raw_audio": "not-verified-public",
                                "fetcher_dataset_id": None,
                                "training_mode": "literature_only_no_training",
                                "earlycare_use": "evidence framing",
                                "required_before_training": ["find approved acute concussion speech data"],
                            },
                        ],
                    }
                )
                + "\n"
            )
            uci_root = datasets_root / "uci-parkinson-speech"
            uci_root.mkdir(parents=True)
            (uci_root / "dataset_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "table_summaries": [
                            {
                                "path": "training_data.csv",
                                "classification_ready": True,
                                "progression_ready": False,
                            }
                        ]
                    }
                )
                + "\n"
            )
            tele_root = datasets_root / "uci-parkinsons-telemonitoring"
            tele_root.mkdir(parents=True)
            (tele_root / "dataset_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "table_summaries": [
                            {
                                "path": "parkinsons_updrs.data",
                                "classification_ready": False,
                                "progression_ready": True,
                            }
                        ]
                    }
                )
                + "\n"
            )

            report = dataset_registry.build_readiness_report(registry_path=registry_path, datasets_root=datasets_root)
            entries = {entry["id"]: entry for entry in report["datasets"]}

            self.assertEqual(report["counts"]["trainable_now"], 1)
            self.assertEqual(report["counts"]["analysis_ready"], 1)
            self.assertEqual(entries["uci"]["local_status"], "classification-ready")
            self.assertEqual(entries["uci"]["ready_for"], ["feature_baseline_training"])
            self.assertIn("run_experiment.py", entries["uci"]["next_action"])
            self.assertFalse(entries["uci"]["app_model_allowed"])
            self.assertEqual(entries["tele"]["local_status"], "progression-ready")
            self.assertIn("analyze_progression_table.py", entries["tele"]["next_action"])
            self.assertEqual(entries["concussion"]["local_status"], "not-applicable")
            self.assertFalse(entries["concussion"]["trainable_now"])

    def test_dataset_registry_writes_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "readiness.md"
            json_output_path = root / "readiness.json"

            exit_code = dataset_registry.main(
                [
                    "--output",
                    str(output_path),
                    "--json-output",
                    str(json_output_path),
                    "--target",
                    "parkinsons-watch",
                ]
            )

            self.assertEqual(exit_code, 0)
            markdown = output_path.read_text()
            payload = json.loads(json_output_path.read_text())
            self.assertIn("EarlyCare Dataset Readiness", markdown)
            self.assertIn("not a validated app model", markdown)
            self.assertTrue(payload["datasets"])
            self.assertTrue(all(entry["target"] == "parkinsons-watch" for entry in payload["datasets"]))

    def test_run_ready_experiments_dry_run_plans_ready_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            datasets_root = root / "datasets"
            output_dir = root / "artifacts"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "use_rules": ["no diagnosis"],
                        "datasets": [
                            {
                                "id": "uci",
                                "name": "UCI Parkinson Speech",
                                "status": "feature-only",
                                "target": "parkinsons-watch",
                                "source_urls": ["https://example.com/uci"],
                                "labels": "pd/control",
                                "language": "Turkish",
                                "tasks": ["vowels"],
                                "participants": "sample",
                                "raw_audio": "no-feature-table",
                                "fetcher_dataset_id": "uci-parkinson-speech",
                                "training_mode": "feature_classification_smoke",
                                "earlycare_use": "feature sanity check",
                                "required_before_training": ["fetch"],
                            },
                            {
                                "id": "tele",
                                "name": "Telemonitoring",
                                "status": "feature-only",
                                "target": "parkinsons-progression",
                                "source_urls": ["https://example.com/tele"],
                                "labels": "updrs",
                                "language": "English",
                                "tasks": ["voice measures"],
                                "participants": "sample",
                                "raw_audio": "no-feature-table",
                                "fetcher_dataset_id": "uci-parkinsons-telemonitoring",
                                "training_mode": "progression_analysis",
                                "earlycare_use": "progression only",
                                "required_before_training": ["fetch"],
                            },
                        ],
                    }
                )
                + "\n"
            )
            uci_root = datasets_root / "uci-parkinson-speech"
            uci_root.mkdir(parents=True)
            (uci_root / "dataset_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "table_summaries": [
                            {
                                "path": "training_data.csv",
                                "classification_ready": True,
                                "progression_ready": False,
                            }
                        ]
                    }
                )
                + "\n"
            )
            tele_root = datasets_root / "uci-parkinsons-telemonitoring"
            tele_root.mkdir(parents=True)
            (tele_root / "dataset_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "table_summaries": [
                            {
                                "path": "parkinsons_updrs.data",
                                "classification_ready": False,
                                "progression_ready": True,
                            }
                        ]
                    }
                )
                + "\n"
            )

            exit_code = run_ready_experiments.main(
                [
                    "--registry",
                    str(registry_path),
                    "--datasets-root",
                    str(datasets_root),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    "--include-progression",
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads((output_dir / "ready_experiments_run.json").read_text())
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["actions_planned"], 2)
            kinds = {action["kind"] for action in report["actions"]}
            self.assertEqual(kinds, {"feature_baseline_training", "progression_analysis"})
            self.assertIn("Do not treat", report["safety"]["excluded_use"])

    def test_run_ready_experiments_executes_feature_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            datasets_root = root / "datasets"
            output_dir = root / "artifacts"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "use_rules": ["no diagnosis"],
                        "datasets": [
                            {
                                "id": "uci-parkinson-speech",
                                "name": "UCI Parkinson Speech",
                                "status": "feature-only",
                                "target": "parkinsons-watch",
                                "source_urls": ["https://example.com/uci"],
                                "labels": "pd/control",
                                "language": "Turkish",
                                "tasks": ["vowels"],
                                "participants": "sample",
                                "raw_audio": "no-feature-table",
                                "fetcher_dataset_id": "uci-parkinson-speech",
                                "training_mode": "feature_classification_smoke",
                                "earlycare_use": "feature sanity check",
                                "required_before_training": ["fetch"],
                            }
                        ],
                    }
                )
                + "\n"
            )
            dataset_root = datasets_root / "uci-parkinson-speech"
            dataset_root.mkdir(parents=True)
            input_path = dataset_root / "training_data.csv"
            with input_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Subject id", "Jitter", "Shimmer", "class information"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"Subject id": "pd-001", "Jitter": "10", "Shimmer": "12", "class information": "1"},
                        {"Subject id": "pd-001", "Jitter": "11", "Shimmer": "13", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "12", "Shimmer": "14", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "13", "Shimmer": "15", "class information": "1"},
                        {"Subject id": "ctl-001", "Jitter": "0", "Shimmer": "1", "class information": "0"},
                        {"Subject id": "ctl-001", "Jitter": "1", "Shimmer": "2", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "2", "Shimmer": "3", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "3", "Shimmer": "4", "class information": "0"},
                    ]
                )
            (dataset_root / "dataset_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "uci-parkinson-speech",
                        "name": "UCI Parkinson Speech",
                        "table_summaries": [
                            {
                                "path": "training_data.csv",
                                "classification_ready": True,
                                "progression_ready": False,
                            }
                        ],
                    }
                )
                + "\n"
            )

            exit_code = run_ready_experiments.main(
                [
                    "--registry",
                    str(registry_path),
                    "--datasets-root",
                    str(datasets_root),
                    "--output-dir",
                    str(output_dir),
                    "--only",
                    "uci-parkinson-speech",
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads((output_dir / "ready_experiments_run.json").read_text())
            self.assertEqual(report["actions_succeeded"], 1)
            self.assertEqual(report["actions_failed"], 0)
            self.assertEqual(json.loads((output_dir / "ready-uci-parkinson-speech_eval.json").read_text())["status"], "ok")
            self.assertEqual(json.loads((output_dir / "ready-uci-parkinson-speech_baseline_model.json").read_text())["status"], "ok")

    def test_run_ready_experiments_can_audit_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            datasets_root = root / "datasets"
            output_dir = root / "artifacts"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "use_rules": ["no diagnosis"],
                        "datasets": [
                            {
                                "id": "uci-parkinson-speech",
                                "name": "UCI Parkinson Speech",
                                "status": "feature-only",
                                "target": "parkinsons-watch",
                                "source_urls": ["https://example.com/uci"],
                                "labels": "pd/control",
                                "language": "Turkish",
                                "tasks": ["vowels"],
                                "participants": "sample",
                                "raw_audio": "no-feature-table",
                                "fetcher_dataset_id": "uci-parkinson-speech",
                                "training_mode": "feature_classification_smoke",
                                "earlycare_use": "feature sanity check",
                                "required_before_training": ["fetch"],
                            }
                        ],
                    }
                )
                + "\n"
            )
            dataset_root = datasets_root / "uci-parkinson-speech"
            dataset_root.mkdir(parents=True)
            input_path = dataset_root / "training_data.csv"
            with input_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Subject id", "Jitter", "Shimmer", "class information"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"Subject id": "pd-001", "Jitter": "10", "Shimmer": "12", "class information": "1"},
                        {"Subject id": "pd-001", "Jitter": "11", "Shimmer": "13", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "12", "Shimmer": "14", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "13", "Shimmer": "15", "class information": "1"},
                        {"Subject id": "ctl-001", "Jitter": "0", "Shimmer": "1", "class information": "0"},
                        {"Subject id": "ctl-001", "Jitter": "1", "Shimmer": "2", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "2", "Shimmer": "3", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "3", "Shimmer": "4", "class information": "0"},
                    ]
                )
            (dataset_root / "dataset_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "uci-parkinson-speech",
                        "name": "UCI Parkinson Speech",
                        "table_summaries": [
                            {
                                "path": "training_data.csv",
                                "classification_ready": True,
                                "progression_ready": False,
                            }
                        ],
                    }
                )
                + "\n"
            )

            exit_code = run_ready_experiments.main(
                [
                    "--registry",
                    str(registry_path),
                    "--datasets-root",
                    str(datasets_root),
                    "--output-dir",
                    str(output_dir),
                    "--only",
                    "uci-parkinson-speech",
                    "--audit",
                ]
            )

            self.assertEqual(exit_code, 0)
            run_report = json.loads((output_dir / "ready_experiments_run.json").read_text())
            self.assertEqual(run_report["audit"]["status"], "ok")
            audit = json.loads((output_dir / "model_artifact_audit.json").read_text())
            self.assertEqual(audit["counts"]["research_only"], 1)
            self.assertEqual(audit["counts"]["validated_ready"], 0)
            self.assertIn("research-only", (output_dir / "model_artifact_audit.md").read_text())

            blocked_code = run_ready_experiments.main(
                [
                    "--registry",
                    str(registry_path),
                    "--datasets-root",
                    str(datasets_root),
                    "--output-dir",
                    str(output_dir),
                    "--only",
                    "uci-parkinson-speech",
                    "--audit",
                    "--require-validated",
                ]
            )
            self.assertEqual(blocked_code, 1)

    def test_audit_model_artifacts_reports_research_only_and_validated_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            prefix = "sample-run"
            (artifacts / f"{prefix}_embeddings.jsonl").write_text(json.dumps({"speaker_id": "s-1", "embedding": [0.1]}) + "\n")
            (artifacts / f"{prefix}_eval.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "split": {"speaker_leakage": False, "train_speakers": ["p1", "c1"], "test_speakers": ["p2", "c2"]},
                        "metrics": {
                            "balanced_accuracy": 0.75,
                            "sensitivity": 1.0,
                            "specificity": 0.5,
                            "roc_auc": 0.5,
                        },
                    }
                )
                + "\n"
            )
            (artifacts / f"{prefix}_baseline_model.json").write_text(
                json.dumps({"status": "ok", "model_type": "speaker-centroid-baseline", "train_balanced_accuracy": 1.0}) + "\n"
            )
            (artifacts / f"{prefix}_personal_baselines.json").write_text(json.dumps({"status": "insufficient-data"}) + "\n")
            (artifacts / f"{prefix}_experiment.md").write_text("# report\n")
            (artifacts / f"{prefix}_model_card.md").write_text("# card\n")
            (artifacts / f"{prefix}_model_card_gate.json").write_text(
                json.dumps(
                    {
                        "datasetAccessReviewed": False,
                        "speakerSplitVerified": True,
                        "evaluationMetricsRecorded": True,
                        "subgroupChecksReviewed": False,
                        "failureModesDocumented": False,
                        "uiCopyReviewed": False,
                        "humanFollowUpActionDefined": False,
                        "rollbackPathDocumented": False,
                        "humanFollowUpAction": None,
                    }
                )
                + "\n"
            )

            exit_code = audit_model_artifacts.main(["--artifacts-dir", str(artifacts), "--experiment", prefix])

            self.assertEqual(exit_code, 0)
            report = json.loads((artifacts / "model_artifact_audit.json").read_text())
            audit = report["experiments"][0]
            self.assertEqual(audit["release_status"], "research-only")
            self.assertFalse(audit["validated_model_allowed"])
            self.assertTrue(audit["offline_embedding_allowed"])
            self.assertIn("datasetAccessReviewed", audit["gate"]["missing"])
            markdown = (artifacts / "model_artifact_audit.md").read_text()
            self.assertIn("Speech Model Artifact Audit", markdown)
            self.assertIn("Research-only", markdown)

            blocked_code = audit_model_artifacts.main(
                ["--artifacts-dir", str(artifacts), "--experiment", prefix, "--require-validated"]
            )
            self.assertEqual(blocked_code, 1)

    def test_audit_model_artifacts_allows_completed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            prefix = "validated-run"
            (artifacts / f"{prefix}_embeddings.jsonl").write_text(json.dumps({"speaker_id": "s-1", "embedding": [0.1]}) + "\n")
            (artifacts / f"{prefix}_eval.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "split": {"speaker_leakage": False, "train_speakers": ["p1", "c1"], "test_speakers": ["p2", "c2"]},
                        "metrics": {
                            "balanced_accuracy": 0.75,
                            "sensitivity": 1.0,
                            "specificity": 0.5,
                            "roc_auc": 0.5,
                        },
                    }
                )
                + "\n"
            )
            (artifacts / f"{prefix}_baseline_model.json").write_text(
                json.dumps({"status": "ok", "model_type": "speaker-centroid-baseline", "train_balanced_accuracy": 1.0}) + "\n"
            )
            (artifacts / f"{prefix}_personal_baselines.json").write_text(json.dumps({"status": "ok"}) + "\n")
            (artifacts / f"{prefix}_experiment.md").write_text("# report\n")
            (artifacts / f"{prefix}_model_card.md").write_text("# card\n")
            (artifacts / f"{prefix}_model_card_gate.json").write_text(
                json.dumps(
                    {
                        "datasetAccessReviewed": True,
                        "speakerSplitVerified": True,
                        "evaluationMetricsRecorded": True,
                        "subgroupChecksReviewed": True,
                        "failureModesDocumented": True,
                        "uiCopyReviewed": True,
                        "humanFollowUpActionDefined": True,
                        "rollbackPathDocumented": True,
                        "humanFollowUpAction": "Review the speech deviation with the call transcript and arrange human follow-up.",
                    }
                )
                + "\n"
            )

            exit_code = audit_model_artifacts.main(
                ["--artifacts-dir", str(artifacts), "--experiment", prefix, "--require-validated"]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads((artifacts / "model_artifact_audit.json").read_text())
            self.assertEqual(report["counts"]["validated_ready"], 1)
            self.assertTrue(report["experiments"][0]["validated_model_allowed"])

    def test_make_experiment_payload_requires_audit_and_exports_offline_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            prefix = "sample-run"
            output_path = root / "payload.json"
            (artifacts / f"{prefix}_embeddings.jsonl").write_text(
                json.dumps(
                    {
                        "dataset": "sample",
                        "speaker_id": "s-001",
                        "label": "control",
                        "task": "repeat_phrase",
                        "embedding": [0.1, 0.2, 0.3],
                        "speech_metrics": {"speechRate": 120},
                        "provenance": {"model_name": "demo-standard-library"},
                    }
                )
                + "\n"
            )
            with self.assertRaises(SystemExit) as raised:
                make_experiment_payload.main(["--artifacts-dir", str(artifacts), "--experiment", prefix, "--output", str(output_path)])
            self.assertIn("Audit report not found", str(raised.exception))

            (artifacts / "model_artifact_audit.json").write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "experiment": prefix,
                                "release_status": "research-only",
                                "validated_model_allowed": False,
                                "offline_embedding_allowed": True,
                                "blockers": ["release-gate checks missing"],
                            }
                        ]
                    }
                )
                + "\n"
            )

            exit_code = make_experiment_payload.main(
                ["--artifacts-dir", str(artifacts), "--experiment", prefix, "--output", str(output_path), "--speaker-id", "s-001"]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["runtimeMode"], "offline embedding")
            self.assertEqual(payload["modelName"], "demo-standard-library")
            self.assertEqual(payload["provenance"]["artifact_release_status"], "research-only")
            self.assertEqual(payload["provenance"]["artifact_audit"], str(artifacts / "model_artifact_audit.json"))

    def test_make_experiment_payload_blocks_validated_mode_without_validated_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            prefix = "sample-run"
            output_path = root / "payload.json"
            (artifacts / f"{prefix}_embeddings.jsonl").write_text(
                json.dumps(
                    {
                        "dataset": "sample",
                        "speaker_id": "s-001",
                        "label": "control",
                        "task": "repeat_phrase",
                        "embedding": [0.1, 0.2],
                        "speech_metrics": {"speechRate": 120, "embedding": [0.1, 0.2]},
                        "provenance": {
                            "model_name": "speech-watch-baseline",
                            "model_version": "2026-07-05",
                            "feature_extractor": "wavlm-base-plus",
                        },
                    }
                )
                + "\n"
            )
            (artifacts / f"{prefix}_model_card_gate.json").write_text(
                json.dumps(
                    {
                        "datasetAccessReviewed": True,
                        "speakerSplitVerified": True,
                        "evaluationMetricsRecorded": True,
                        "subgroupChecksReviewed": True,
                        "failureModesDocumented": True,
                        "uiCopyReviewed": True,
                        "humanFollowUpActionDefined": True,
                        "rollbackPathDocumented": True,
                        "humanFollowUpAction": "Review the speech deviation with the call transcript and arrange human follow-up.",
                    }
                )
                + "\n"
            )
            audit_path = artifacts / "model_artifact_audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "experiment": prefix,
                                "release_status": "research-only",
                                "validated_model_allowed": False,
                                "offline_embedding_allowed": True,
                                "blockers": ["release-gate checks missing"],
                            }
                        ]
                    }
                )
                + "\n"
            )

            with self.assertRaises(SystemExit) as raised:
                make_experiment_payload.main(
                    [
                        "--artifacts-dir",
                        str(artifacts),
                        "--experiment",
                        prefix,
                        "--output",
                        str(output_path),
                        "--runtime-mode",
                        "validated model",
                        "--model-version",
                        "2026-07-05",
                    ]
                )
            self.assertIn("Audit does not allow validated model", str(raised.exception))

            audit_path.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "experiment": prefix,
                                "release_status": "validated-ready",
                                "validated_model_allowed": True,
                                "offline_embedding_allowed": True,
                                "blockers": [],
                            }
                        ]
                    }
                )
                + "\n"
            )
            exit_code = make_experiment_payload.main(
                [
                    "--artifacts-dir",
                    str(artifacts),
                    "--experiment",
                    prefix,
                    "--output",
                    str(output_path),
                    "--runtime-mode",
                    "validated model",
                    "--model-version",
                    "2026-07-05",
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["runtimeMode"], "validated model")
            self.assertEqual(payload["artifactUri"], str(artifacts / f"{prefix}_baseline_model.json"))
            self.assertEqual(payload["provenance"]["artifact_release_status"], "validated-ready")

    def test_prepare_manifest_infers_rows_from_audio_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_root = root / "datasets" / "NeuroVoz"
            output_path = root / "datasets" / "neurovoz_manifest.csv"
            write_wav(audio_root / "PD" / "speaker-001" / "ddk" / "a.wav", 220)
            write_wav(audio_root / "HC" / "speaker-002" / "vowels" / "e.wav", 240)
            write_wav(audio_root / "unknown" / "speaker-003" / "reading" / "x.wav", 260)

            exit_code = prepare_manifest.main(
                [
                    "--audio-root",
                    str(audio_root),
                    "--output",
                    str(output_path),
                    "--dataset",
                    "NeuroVoz",
                    "--language",
                    "Spanish",
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)
            rows_by_speaker = {row["speaker_id"]: row for row in rows}
            self.assertEqual(rows_by_speaker["speaker-001"]["label"], "pd")
            self.assertEqual(rows_by_speaker["speaker-001"]["task"], "ddk")
            self.assertEqual(rows_by_speaker["speaker-001"]["review_status"], "inferred")
            self.assertEqual(rows_by_speaker["speaker-002"]["label"], "control")
            self.assertEqual(rows_by_speaker["speaker-002"]["task"], "vowels")
            self.assertEqual(rows_by_speaker["speaker-003"]["label"], "unknown")
            self.assertEqual(rows_by_speaker["speaker-003"]["review_status"], "needs-review")
            self.assertEqual(rows_by_speaker["speaker-003"]["dataset"], "NeuroVoz")
            self.assertEqual(rows_by_speaker["speaker-003"]["language"], "Spanish")

    def test_prepare_manifest_accepts_custom_tokens_and_speaker_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_root = root / "custom"
            output_path = root / "manifest.csv"
            write_wav(audio_root / "case" / "participant_77" / "phonation" / "sample.wav", 220)
            write_wav(audio_root / "comparison" / "participant_88" / "phonation" / "sample.wav", 240)

            prepare_manifest.main(
                [
                    "--audio-root",
                    str(audio_root),
                    "--output",
                    str(output_path),
                    "--dataset",
                    "CustomSet",
                    "--positive-tokens",
                    "case",
                    "--negative-tokens",
                    "comparison",
                    "--speaker-regex",
                    r"participant_(\d+)",
                ]
            )

            with output_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            rows_by_speaker = {row["speaker_id"]: row for row in rows}
            self.assertEqual(rows_by_speaker["77"]["label"], "pd")
            self.assertEqual(rows_by_speaker["88"]["label"], "control")

    def test_fetch_public_dataset_extracts_table_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_zip = root / "source.zip"
            output_root = root / "datasets"
            with zipfile.ZipFile(source_zip, "w") as archive:
                archive.writestr("training_data.csv", "Subject id,Jitter,class information\n1,0.2,1\n2,0.1,0\n")
                archive.writestr("notes/readme.txt", "sample")

            exit_code = fetch_public_datasets.main(
                [
                    "--dataset",
                    "uci-parkinson-speech",
                    "--source-url",
                    source_zip.resolve().as_uri(),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest_path = output_root / "uci-parkinson-speech" / "dataset_fetch_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["dataset_id"], "uci-parkinson-speech")
            self.assertEqual(manifest["table_candidates"], ["training_data.csv"])
            self.assertEqual(manifest["nested_archives"], [])
            self.assertEqual(len(manifest["table_summaries"]), 1)
            self.assertTrue(manifest["table_summaries"][0]["classification_ready"])
            self.assertFalse(manifest["table_summaries"][0]["progression_ready"])
            self.assertEqual(manifest["table_summaries"][0]["speaker_column"], "Subject id")
            self.assertEqual(manifest["table_summaries"][0]["label_column"], "class information")
            self.assertTrue((output_root / "uci-parkinson-speech" / "training_data.csv").exists())

    def test_fetch_public_dataset_marks_progression_only_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_zip = root / "source.zip"
            output_root = root / "datasets"
            with zipfile.ZipFile(source_zip, "w") as archive:
                archive.writestr(
                    "parkinsons_updrs.data",
                    "subject#,motor_UPDRS,total_UPDRS,Jitter(%),Shimmer\n"
                    "1,20,30,0.1,0.2\n"
                    "1,21,31,0.2,0.3\n"
                    "2,25,35,0.3,0.4\n",
                )

            exit_code = fetch_public_datasets.main(
                [
                    "--dataset",
                    "uci-parkinsons-telemonitoring",
                    "--source-url",
                    source_zip.resolve().as_uri(),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads((output_root / "uci-parkinsons-telemonitoring" / "dataset_fetch_manifest.json").read_text())
            summary = manifest["table_summaries"][0]
            self.assertEqual(summary["speaker_column"], "subject#")
            self.assertIsNone(summary["label_column"])
            self.assertEqual(summary["updrs_columns"], ["motor_UPDRS", "total_UPDRS"])
            self.assertFalse(summary["classification_ready"])
            self.assertTrue(summary["progression_ready"])
            self.assertIn("Progression analysis candidate only", summary["notes"][-1])

    def test_fetch_public_dataset_flags_nested_rar_without_external_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_zip = root / "source.zip"
            output_root = root / "datasets"
            with zipfile.ZipFile(source_zip, "w") as archive:
                archive.writestr("Parkinson_Multiple_Sound_Recording.rar", b"not-a-real-rar")

            exit_code = fetch_public_datasets.main(
                [
                    "--dataset",
                    "uci-parkinson-speech",
                    "--source-url",
                    source_zip.resolve().as_uri(),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(exit_code, 0)
            dataset_root = output_root / "uci-parkinson-speech"
            manifest = json.loads((dataset_root / "dataset_fetch_manifest.json").read_text())
            self.assertEqual(manifest["table_candidates"], [])
            self.assertEqual(manifest["table_summaries"], [])
            self.assertEqual(manifest["nested_archives"], ["Parkinson_Multiple_Sound_Recording.rar"])
            self.assertTrue((dataset_root / "EXTRACTION_REQUIRED.md").exists())
            self.assertIn("Nested archive extraction required", manifest["notes"][0])

    def test_fetch_public_dataset_rejects_unsafe_zip_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_zip = root / "unsafe.zip"
            with zipfile.ZipFile(source_zip, "w") as archive:
                archive.writestr("../escape.csv", "x,y\n1,2\n")

            with self.assertRaises(ValueError) as raised:
                fetch_public_datasets.safe_extract_zip(source_zip, root / "datasets")

            self.assertIn("Unsafe archive path", str(raised.exception))

    def test_analyze_progression_table_writes_trends_and_associations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "parkinsons_updrs.data"
            output_path = root / "progression.json"
            input_path.write_text(
                "subject#,test_time,motor_UPDRS,total_UPDRS,Jitter(%),Shimmer\n"
                "1,0,20,30,0.10,0.20\n"
                "1,1,22,32,0.20,0.30\n"
                "1,2,24,34,0.30,0.40\n"
                "2,0,10,15,0.05,0.10\n"
                "2,1,11,16,0.06,0.11\n"
                "2,2,12,17,0.07,0.12\n"
            )

            exit_code = analyze_progression_table.main(["--input", str(input_path), "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["speaker_column"], "subject#")
            self.assertEqual(report["time_column"], "test_time")
            self.assertEqual(report["target_columns"], ["motor_UPDRS", "total_UPDRS"])
            self.assertEqual(report["speakers_with_trends"], 2)
            self.assertEqual(report["target_trend_summary"]["motor_UPDRS"]["mean_delta"], 3.0)
            trends = {item["speaker_id"]: item for item in report["speaker_trends"]}
            self.assertEqual(trends["1"]["targets"]["motor_UPDRS"]["slope_per_time_unit"], 2.0)
            self.assertEqual(trends["2"]["targets"]["total_UPDRS"]["delta"], 2.0)
            self.assertIn("Jitter(%)", [item["feature"] for item in report["feature_associations"]["motor_UPDRS"]])
            self.assertIn("diagnosis", report["safety"]["excluded_use"])
            markdown = output_path.with_suffix(".md").read_text()
            self.assertIn("EarlyCare Progression Analysis", markdown)
            self.assertIn("Target Trends", markdown)
            self.assertIn("not a diagnosis model", markdown)

    def test_analyze_progression_table_uses_fetch_manifest_and_rejects_classifier_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "uci-parkinsons-telemonitoring"
            dataset_root.mkdir(parents=True)
            table_path = dataset_root / "parkinsons_updrs.data"
            output_path = root / "progression.json"
            table_path.write_text(
                "subject#,test_time,motor_UPDRS,total_UPDRS,Jitter(%)\n"
                "1,0,20,30,0.10\n"
                "1,1,22,32,0.20\n"
                "2,0,10,15,0.05\n"
                "2,1,12,17,0.08\n"
            )
            fetch_manifest_path = dataset_root / "dataset_fetch_manifest.json"
            fetch_manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "uci-parkinsons-telemonitoring",
                        "name": "UCI Parkinsons Telemonitoring",
                        "table_summaries": [
                            {
                                "path": "parkinsons_updrs.data",
                                "speaker_column": "subject#",
                                "updrs_columns": ["motor_UPDRS", "total_UPDRS"],
                                "classification_ready": False,
                                "progression_ready": True,
                            }
                        ],
                    }
                )
                + "\n"
            )

            exit_code = analyze_progression_table.main(
                ["--dataset-fetch-manifest", str(fetch_manifest_path), "--output", str(output_path)]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["dataset_fetch_manifest"], str(fetch_manifest_path))
            self.assertEqual(report["selected_table_summary"]["path"], "parkinsons_updrs.data")
            self.assertTrue(output_path.with_suffix(".md").exists())

            classifier_manifest_path = root / "classifier_manifest.json"
            classifier_manifest_path.write_text(
                json.dumps(
                    {
                        "table_summaries": [
                            {
                                "path": "training_data.csv",
                                "classification_ready": True,
                                "progression_ready": False,
                            }
                        ]
                    }
                )
                + "\n"
            )
            with self.assertRaises(SystemExit) as raised:
                analyze_progression_table.main(["--dataset-fetch-manifest", str(classifier_manifest_path), "--output", str(output_path)])
            self.assertIn("No progression_ready table", str(raised.exception))

    def test_run_experiment_writes_artifacts_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_root = root / "datasets" / "sample"
            manifest_path = root / "datasets" / "sample_manifest.csv"
            output_dir = root / "artifacts"
            write_wav(audio_root / "pd" / "pd-001" / "ddk" / "a.wav", 220)
            write_wav(audio_root / "pd" / "pd-001" / "vowels" / "e.wav", 225)
            write_wav(audio_root / "pd" / "pd-002" / "ddk" / "a.wav", 240)
            write_wav(audio_root / "pd" / "pd-002" / "vowels" / "e.wav", 245)
            write_wav(audio_root / "control" / "control-001" / "ddk" / "a.wav", 260)
            write_wav(audio_root / "control" / "control-001" / "vowels" / "e.wav", 265)
            write_wav(audio_root / "control" / "control-002" / "ddk" / "a.wav", 280)
            write_wav(audio_root / "control" / "control-002" / "vowels" / "e.wav", 285)
            prepare_manifest.main(
                [
                    "--audio-root",
                    str(audio_root),
                    "--output",
                    str(manifest_path),
                    "--dataset",
                    "SampleSet",
                    "--language",
                    "English",
                ]
            )

            exit_code = run_experiment.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--audio-root",
                    str(audio_root),
                    "--output-dir",
                    str(output_dir),
                    "--experiment-name",
                    "demo-run",
                    "--test-fraction",
                    "0.5",
                    "--personal-min-samples",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            embeddings_path = output_dir / "demo-run_embeddings.jsonl"
            evaluation_path = output_dir / "demo-run_eval.json"
            model_path = output_dir / "demo-run_baseline_model.json"
            personal_baseline_path = output_dir / "demo-run_personal_baselines.json"
            report_path = output_dir / "demo-run_experiment.md"
            model_card_path = output_dir / "demo-run_model_card.md"
            model_card_gate_path = output_dir / "demo-run_model_card_gate.json"
            self.assertTrue(embeddings_path.exists())
            self.assertEqual(len(embeddings_path.read_text().splitlines()), 8)
            self.assertEqual(json.loads(evaluation_path.read_text())["status"], "ok")
            self.assertEqual(json.loads(model_path.read_text())["status"], "ok")
            personal_baselines = json.loads(personal_baseline_path.read_text())
            self.assertEqual(personal_baselines["status"], "ok")
            self.assertEqual(personal_baselines["speakers_with_baselines"], 4)
            gate = json.loads(model_card_gate_path.read_text())
            self.assertTrue(gate["speakerSplitVerified"])
            self.assertTrue(gate["evaluationMetricsRecorded"])
            self.assertFalse(gate["datasetAccessReviewed"])
            self.assertFalse(gate["humanFollowUpActionDefined"])
            self.assertIn("Release Gate", model_card_path.read_text())
            report = report_path.read_text()
            self.assertIn("offline research only", report)
            self.assertIn("Rows needing review: 0", report)
            self.assertIn("Subgroup Checks", report)
            self.assertIn("Personal baselines", report)
            self.assertIn("Model card draft", report)

    def test_run_experiment_refuses_unreviewed_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset", "speaker_id", "label", "task", "audio_path", "review_status"])
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "sample",
                        "speaker_id": "unknown-001",
                        "label": "unknown",
                        "task": "ddk",
                        "audio_path": "unknown.wav",
                        "review_status": "needs-review",
                    }
                )

            with self.assertRaises(SystemExit) as raised:
                run_experiment.main(["--manifest", str(manifest_path), "--audio-root", str(root)])

            self.assertIn("needs-review", str(raised.exception))

    def test_run_experiment_supports_feature_table_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "uci_like.csv"
            output_dir = root / "artifacts"
            with input_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Subject id", "Jitter", "Shimmer", "class information"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"Subject id": "pd-001", "Jitter": "10", "Shimmer": "12", "class information": "1"},
                        {"Subject id": "pd-001", "Jitter": "11", "Shimmer": "13", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "12", "Shimmer": "14", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "13", "Shimmer": "15", "class information": "1"},
                        {"Subject id": "ctl-001", "Jitter": "0", "Shimmer": "1", "class information": "0"},
                        {"Subject id": "ctl-001", "Jitter": "1", "Shimmer": "2", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "2", "Shimmer": "3", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "3", "Shimmer": "4", "class information": "0"},
                    ]
                )

            exit_code = run_experiment.main(
                [
                    "--feature-table",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--experiment-name",
                    "feature-run",
                    "--dataset",
                    "UCI Parkinson Speech",
                    "--language",
                    "Turkish",
                    "--test-fraction",
                    "0.5",
                    "--personal-min-samples",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            rows_path = output_dir / "feature-run_embeddings.jsonl"
            report_path = output_dir / "feature-run_experiment.md"
            evaluation_path = output_dir / "feature-run_eval.json"
            model_path = output_dir / "feature-run_baseline_model.json"
            personal_baseline_path = output_dir / "feature-run_personal_baselines.json"
            model_card_path = output_dir / "feature-run_model_card.md"
            model_card_gate_path = output_dir / "feature-run_model_card_gate.json"
            self.assertEqual(json.loads(evaluation_path.read_text())["status"], "ok")
            self.assertEqual(json.loads(model_path.read_text())["status"], "ok")
            self.assertEqual(json.loads(personal_baseline_path.read_text())["speakers_with_baselines"], 4)
            self.assertFalse(json.loads(model_card_gate_path.read_text())["subgroupChecksReviewed"])
            first_row = json.loads(rows_path.read_text().splitlines()[0])
            self.assertEqual(first_row["provenance"]["source_type"], "feature_table")
            report = report_path.read_text()
            self.assertIn("Input mode: feature table", report)
            self.assertIn("Model: `feature-table-zscore`", report)
            self.assertIn("Source Type", report)
            self.assertIn("feature table", model_card_path.read_text())

    def test_run_experiment_supports_dataset_fetch_manifest_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "uci-parkinson-speech"
            dataset_root.mkdir(parents=True)
            input_path = dataset_root / "training_data.csv"
            output_dir = root / "artifacts"
            with input_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Subject id", "Jitter", "Shimmer", "class information"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"Subject id": "pd-001", "Jitter": "10", "Shimmer": "12", "class information": "1"},
                        {"Subject id": "pd-001", "Jitter": "11", "Shimmer": "13", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "12", "Shimmer": "14", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "13", "Shimmer": "15", "class information": "1"},
                        {"Subject id": "ctl-001", "Jitter": "0", "Shimmer": "1", "class information": "0"},
                        {"Subject id": "ctl-001", "Jitter": "1", "Shimmer": "2", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "2", "Shimmer": "3", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "3", "Shimmer": "4", "class information": "0"},
                    ]
                )
            fetch_manifest_path = dataset_root / "dataset_fetch_manifest.json"
            fetch_manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "uci-parkinson-speech",
                        "name": "UCI Parkinson's Speech with Multiple Types of Sound Recordings",
                        "table_summaries": [
                            {
                                "path": "training_data.csv",
                                "speaker_column": "Subject id",
                                "label_column": "class information",
                                "classification_ready": True,
                                "progression_ready": False,
                            }
                        ],
                    }
                )
                + "\n"
            )

            exit_code = run_experiment.main(
                [
                    "--dataset-fetch-manifest",
                    str(fetch_manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--experiment-name",
                    "fetch-run",
                    "--test-fraction",
                    "0.5",
                    "--personal-min-samples",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            rows_path = output_dir / "fetch-run_embeddings.jsonl"
            first_row = json.loads(rows_path.read_text().splitlines()[0])
            self.assertEqual(first_row["dataset"], "UCI Parkinson's Speech with Multiple Types of Sound Recordings")
            self.assertEqual(first_row["provenance"]["language"], "Turkish")
            self.assertEqual(json.loads((output_dir / "fetch-run_eval.json").read_text())["status"], "ok")
            report = (output_dir / "fetch-run_experiment.md").read_text()
            self.assertIn("dataset_fetch_manifest.json", report)
            self.assertIn("training_data.csv", report)

    def test_run_experiment_refuses_non_classification_fetch_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "uci-parkinsons-telemonitoring"
            dataset_root.mkdir(parents=True)
            table_path = dataset_root / "parkinsons_updrs.data"
            table_path.write_text("subject#,motor_UPDRS,total_UPDRS,Jitter(%)\n1,20,30,0.1\n")
            fetch_manifest_path = dataset_root / "dataset_fetch_manifest.json"
            fetch_manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "uci-parkinsons-telemonitoring",
                        "name": "UCI Parkinsons Telemonitoring",
                        "table_summaries": [
                            {
                                "path": "parkinsons_updrs.data",
                                "speaker_column": "subject#",
                                "label_column": None,
                                "classification_ready": False,
                                "progression_ready": True,
                            }
                        ],
                    }
                )
                + "\n"
            )

            with self.assertRaises(SystemExit) as raised:
                run_experiment.main(["--dataset-fetch-manifest", str(fetch_manifest_path), "--output-dir", str(root / "artifacts")])

            self.assertIn("No classification_ready table", str(raised.exception))
            self.assertIn("Progression-only", str(raised.exception))

    def test_build_personal_baselines_writes_thresholds_and_insufficient_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "personal.json"
            rows = [
                {"dataset": "sample", "speaker_id": "s-1", "label": "control", "task": "repeat", "embedding": [1.0, 0.0]},
                {"dataset": "sample", "speaker_id": "s-1", "label": "control", "task": "repeat", "embedding": [0.98, 0.02]},
                {"dataset": "sample", "speaker_id": "s-1", "label": "control", "task": "repeat", "embedding": [0.97, 0.03]},
                {"dataset": "sample", "speaker_id": "s-2", "label": "control", "task": "repeat", "embedding": [0.0, 1.0]},
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            exit_code = build_personal_baselines.main(["--input", str(input_path), "--output", str(output_path), "--min-samples", "3"])

            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["status"], "ok")
            baselines = {item["speaker_id"]: item for item in report["baselines"]}
            self.assertEqual(baselines["s-1"]["status"], "ok")
            self.assertIn("watch", baselines["s-1"]["thresholds"])
            self.assertEqual(baselines["s-2"]["status"], "insufficient-data")
            self.assertIn("diagnosis", report["safety"]["excluded_use"])

    def test_convert_feature_table_writes_evaluable_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "uci_like.csv"
            output_path = root / "feature_rows.jsonl"
            eval_path = root / "feature_eval.json"
            with input_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Subject id", "Jitter", "Shimmer", "UPDRS", "class information"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"Subject id": "pd-001", "Jitter": "10", "Shimmer": "12", "UPDRS": "22", "class information": "1"},
                        {"Subject id": "pd-001", "Jitter": "11", "Shimmer": "13", "UPDRS": "22", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "12", "Shimmer": "14", "UPDRS": "24", "class information": "1"},
                        {"Subject id": "pd-002", "Jitter": "13", "Shimmer": "15", "UPDRS": "24", "class information": "1"},
                        {"Subject id": "ctl-001", "Jitter": "0", "Shimmer": "1", "UPDRS": "", "class information": "0"},
                        {"Subject id": "ctl-001", "Jitter": "1", "Shimmer": "2", "UPDRS": "", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "2", "Shimmer": "3", "UPDRS": "", "class information": "0"},
                        {"Subject id": "ctl-002", "Jitter": "3", "Shimmer": "4", "UPDRS": "", "class information": "0"},
                    ]
                )

            exit_code = convert_feature_table.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--dataset",
                    "UCI Parkinson Speech",
                    "--language",
                    "Turkish",
                ]
            )

            self.assertEqual(exit_code, 0)
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]["label"], "pd")
            self.assertEqual(rows[-1]["label"], "control")
            self.assertEqual(rows[0]["provenance"]["source_type"], "feature_table")
            self.assertEqual(rows[0]["provenance"]["model"], "feature-table-zscore")
            self.assertEqual(rows[0]["provenance"]["feature_columns"], ["Jitter", "Shimmer"])

            evaluate_baseline.main(["--input", str(output_path), "--output", str(eval_path), "--test-fraction", "0.5"])
            report = json.loads(eval_path.read_text())
            self.assertEqual(report["status"], "ok")
            self.assertFalse(report["split"]["speaker_leakage"])

    def test_extract_embeddings_writes_backend_compatible_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "sample" / "control" / "s-001.wav"
            manifest_path = root / "manifest.csv"
            output_path = root / "embeddings.jsonl"
            write_wav(audio_path, 220)

            with manifest_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset", "speaker_id", "label", "task", "audio_path", "language", "transcript"])
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "sample",
                        "speaker_id": "s-001",
                        "label": "control",
                        "task": "repeat_phrase",
                        "audio_path": "sample/control/s-001.wav",
                        "language": "English",
                        "transcript": "today i am safe at home and i can ask for help",
                    }
                )

            exit_code = extract_embeddings.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--audio-root",
                    str(root),
                    "--output",
                    str(output_path),
                    "--model",
                    "demo",
                    "--dimensions",
                    "8",
                ]
            )

            self.assertEqual(exit_code, 0)
            row = json.loads(output_path.read_text().strip())
            self.assertEqual(row["dataset"], "sample")
            self.assertEqual(row["speaker_id"], "s-001")
            self.assertEqual(len(row["embedding"]), 8)
            self.assertEqual(row["speech_metrics"]["embedding"], row["embedding"])
            self.assertEqual(row["speech_metrics"]["phraseAccuracy"], 0.96)
            self.assertEqual(row["provenance"]["model"], "demo")
            self.assertGreater(row["provenance"]["duration_seconds"], 0)

    def test_make_enrichment_payload_writes_offline_api_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "payload.json"
            rows = [
                {
                    "dataset": "sample",
                    "speaker_id": "s-001",
                    "label": "control",
                    "task": "repeat_phrase",
                    "embedding": [0.1, 0.2, 0.3],
                    "speech_metrics": {
                        "speechRate": 120,
                        "avgPauseMs": 450,
                        "responseLatencyMs": 900,
                        "pitchVariability": 0.4,
                        "phraseAccuracy": 0.96,
                    },
                    "provenance": {
                        "source_id": "recording-001",
                        "model": "demo",
                        "model_name": "demo-standard-library",
                        "extracted_at": "2026-07-05T00:00:00+00:00",
                    },
                }
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            exit_code = make_enrichment_payload.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--speaker-id",
                    "s-001",
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["runtimeMode"], "offline embedding")
            self.assertEqual(payload["modelName"], "demo-standard-library")
            self.assertEqual(payload["featureExtractor"], "demo-standard-library")
            self.assertEqual(payload["embedding"], [0.1, 0.2, 0.3])
            self.assertEqual(payload["speech_metrics"]["embedding"], [0.1, 0.2, 0.3])
            self.assertEqual(payload["speech_metrics"]["updatedAt"], "2026-07-05T00:00:00+00:00")
            self.assertEqual(payload["provenance"]["dataset"], "sample")
            self.assertEqual(payload["provenance"]["speaker_id"], "s-001")
            self.assertEqual(payload["provenance"]["source_row_index"], 1)
            self.assertNotIn("modelCard", payload)

    def test_make_enrichment_payload_requires_complete_gate_for_validated_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "payload.json"
            incomplete_gate_path = root / "incomplete_gate.json"
            complete_gate_path = root / "complete_gate.json"
            row = {
                "dataset": "sample",
                "speaker_id": "s-001",
                "label": "control",
                "task": "repeat_phrase",
                "embedding": [0.1, 0.2],
                "speech_metrics": {
                    "speechRate": 120,
                    "avgPauseMs": 450,
                    "responseLatencyMs": 900,
                    "pitchVariability": 0.4,
                    "phraseAccuracy": 0.96,
                    "embedding": [0.1, 0.2],
                },
                "provenance": {
                    "source_id": "recording-001",
                    "model_name": "speech-watch-baseline",
                    "model_version": "2026-07-05",
                    "feature_extractor": "wavlm-base-plus",
                    "extracted_at": "2026-07-05T00:00:00+00:00",
                },
            }
            input_path.write_text(json.dumps(row) + "\n")
            incomplete_gate_path.write_text(json.dumps({"speakerSplitVerified": True}) + "\n")
            complete_gate = {
                "datasetAccessReviewed": True,
                "speakerSplitVerified": True,
                "evaluationMetricsRecorded": True,
                "subgroupChecksReviewed": True,
                "failureModesDocumented": True,
                "uiCopyReviewed": True,
                "humanFollowUpActionDefined": True,
                "rollbackPathDocumented": True,
                "humanFollowUpAction": "Review the speech deviation with the call transcript and arrange human follow-up.",
            }
            complete_gate_path.write_text(json.dumps(complete_gate) + "\n")

            with self.assertRaises(SystemExit) as raised:
                make_enrichment_payload.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--runtime-mode",
                        "validated model",
                        "--model-card-gate",
                        str(incomplete_gate_path),
                    ]
                )
            self.assertIn("release-gate checks", str(raised.exception))

            exit_code = make_enrichment_payload.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--runtime-mode",
                    "validated model",
                    "--model-card-gate",
                    str(complete_gate_path),
                    "--artifact-uri",
                    "research/artifacts/sample_baseline_model.json",
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["runtimeMode"], "validated model")
            self.assertEqual(payload["modelName"], "speech-watch-baseline")
            self.assertEqual(payload["modelVersion"], "2026-07-05")
            self.assertEqual(payload["featureExtractor"], "wavlm-base-plus")
            self.assertEqual(payload["artifactUri"], "research/artifacts/sample_baseline_model.json")
            self.assertEqual(payload["modelCard"], complete_gate)

    def test_evaluate_baseline_uses_speaker_level_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "eval.json"
            rows = [
                {
                    "dataset": "sample",
                    "speaker_id": "pd-train",
                    "label": "pd",
                    "task": "ddk",
                    "embedding": [1.0, 0.0],
                    "provenance": {"language": "English", "source_type": "feature_table"},
                },
                {
                    "dataset": "sample",
                    "speaker_id": "pd-test",
                    "label": "pd",
                    "task": "ddk",
                    "embedding": [0.9, 0.1],
                    "provenance": {"language": "English", "source_type": "feature_table"},
                },
                {
                    "dataset": "sample",
                    "speaker_id": "ctl-train",
                    "label": "control",
                    "task": "ddk",
                    "embedding": [0.0, 1.0],
                    "provenance": {"language": "English", "source_type": "feature_table"},
                },
                {
                    "dataset": "sample",
                    "speaker_id": "ctl-test",
                    "label": "control",
                    "task": "ddk",
                    "embedding": [0.1, 0.9],
                    "provenance": {"language": "English", "source_type": "feature_table"},
                },
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            exit_code = evaluate_baseline.main(["--input", str(input_path), "--output", str(output_path), "--test-fraction", "0.5"])

            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["status"], "ok")
            self.assertFalse(report["split"]["speaker_leakage"])
            self.assertEqual(report["metrics"]["balanced_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["confusion"], {"tp": 1, "tn": 1, "fp": 0, "fn": 0})
            self.assertEqual(report["metrics"]["subgroups"]["task"]["ddk"]["balanced_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["subgroups"]["language"]["English"]["speakers"], 2)
            self.assertEqual(report["metrics"]["subgroups"]["source_type"]["feature_table"]["speakers"], 2)

    def test_evaluate_baseline_reports_insufficient_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "eval.json"
            input_path.write_text(json.dumps({"dataset": "sample", "speaker_id": "only-one", "label": "pd", "embedding": [1.0, 0.0]}) + "\n")

            evaluate_baseline.main(["--input", str(input_path), "--output", str(output_path)])

            report = json.loads(output_path.read_text())
            self.assertEqual(report["status"], "insufficient-data")
            self.assertIn("Need at least two positive", report["reason"])

    def test_train_baseline_writes_research_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "model.json"
            rows = [
                {"dataset": "sample", "speaker_id": "pd-1", "label": "pd", "embedding": [1.0, 0.0]},
                {"dataset": "sample", "speaker_id": "pd-2", "label": "pd", "embedding": [0.9, 0.1]},
                {"dataset": "sample", "speaker_id": "ctl-1", "label": "control", "embedding": [0.0, 1.0]},
                {"dataset": "sample", "speaker_id": "ctl-2", "label": "control", "embedding": [0.1, 0.9]},
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            exit_code = train_baseline.main(["--input", str(input_path), "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            model = json.loads(output_path.read_text())
            self.assertEqual(model["status"], "ok")
            self.assertEqual(model["model_type"], "speaker-centroid-baseline")
            self.assertEqual(model["embedding_dimensions"], 2)
            self.assertEqual(model["dataset_counts"]["sample"], {"positive_speakers": 2, "negative_speakers": 2})
            self.assertIn("diagnosis", model["safety"]["excluded_use"])

    def test_train_baseline_reports_insufficient_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "model.json"
            input_path.write_text(json.dumps({"dataset": "sample", "speaker_id": "pd-1", "label": "pd", "embedding": [1.0, 0.0]}) + "\n")

            train_baseline.main(["--input", str(input_path), "--output", str(output_path)])

            model = json.loads(output_path.read_text())
            self.assertEqual(model["status"], "insufficient-data")
            self.assertIn("positive and one negative", model["reason"])


if __name__ == "__main__":
    unittest.main()
