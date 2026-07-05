import csv
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from research.speech_ml import evaluate_baseline, extract_embeddings, train_baseline


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

    def test_evaluate_baseline_uses_speaker_level_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "embeddings.jsonl"
            output_path = root / "eval.json"
            rows = [
                {"dataset": "sample", "speaker_id": "pd-train", "label": "pd", "embedding": [1.0, 0.0]},
                {"dataset": "sample", "speaker_id": "pd-test", "label": "pd", "embedding": [0.9, 0.1]},
                {"dataset": "sample", "speaker_id": "ctl-train", "label": "control", "embedding": [0.0, 1.0]},
                {"dataset": "sample", "speaker_id": "ctl-test", "label": "control", "embedding": [0.1, 0.9]},
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            exit_code = evaluate_baseline.main(["--input", str(input_path), "--output", str(output_path), "--test-fraction", "0.5"])

            self.assertEqual(exit_code, 0)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["status"], "ok")
            self.assertFalse(report["split"]["speaker_leakage"])
            self.assertEqual(report["metrics"]["balanced_accuracy"], 1.0)
            self.assertEqual(report["metrics"]["confusion"], {"tp": 1, "tn": 1, "fp": 0, "fn": 0})

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
