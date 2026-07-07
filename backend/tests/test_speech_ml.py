import importlib.util
import json
import math
import subprocess
import sys
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from app.parkinsons_speech_model.evaluation import group_folds
from app.parkinsons_speech_model.inference import predict_speech_marker
from app.parkinsons_speech_model.parkinsons_features import CONVERSATIONAL_PARKINSONS_FEATURE_NAMES, extract_conversational_parkinsons_features
from app.parkinsons_speech_model.preprocessing import preprocess_audio
from app.parkinsons_speech_model.tabular_training import select_best_candidate, subject_id_from_name, CandidateResult


def write_test_wav(path: Path, sample_rate: int = 8_000, seconds: float = 1.0) -> None:
    sample_count = int(sample_rate * seconds)
    samples = [
        int(0.25 * math.sin(2 * math.pi * 220 * index / sample_rate) * 32767)
        for index in range(sample_count)
    ]
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.asarray(samples, dtype="<i2").tobytes())


class DummyProbabilityModel:
    def predict_proba(self, features):
        return np.asarray([[0.2, 0.8] for _ in range(len(features))])


class SpeechMlTests(unittest.TestCase):
    def test_preprocess_audio_resamples_to_16khz_and_chunks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "control_001_vowel.wav"
            write_test_wav(audio_path, sample_rate=8_000, seconds=1.0)

            chunks, quality = preprocess_audio(audio_path, max_duration_seconds=0.5)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertLessEqual(max(len(chunk) for chunk in chunks), 8_000)
        self.assertEqual(quality.sample_rate, 16_000)

    def test_group_folds_keep_speakers_out_of_both_train_and_test(self) -> None:
        labels = [0, 0, 1, 1, 0, 1]
        groups = ["a", "a", "b", "b", "c", "c"]

        folds = group_folds(labels, groups, n_splits=3)

        for train_indices, test_indices in folds:
            train_groups = {groups[index] for index in train_indices}
            test_groups = {groups[index] for index in test_indices}
            self.assertFalse(train_groups & test_groups)

    def test_predict_parkinsons_speech_model_returns_warning_without_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "patient.wav"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            write_test_wav(audio_path)

            result = predict_speech_marker(audio_path, artifacts)

        self.assertIsNone(result.probability)
        self.assertTrue(any("No trained speech model artifacts" in warning for warning in result.warnings))

    def test_conversational_parkinsons_features_are_complete_and_finite(self) -> None:
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "patient.wav"
            write_test_wav(audio_path, sample_rate=16_000, seconds=4.0)

            features, warnings = extract_conversational_parkinsons_features(audio_path)

        self.assertEqual(list(features.keys()), CONVERSATIONAL_PARKINSONS_FEATURE_NAMES)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertTrue(any("conversational Parkinson speech marker" in warning for warning in warnings))

    def test_conversational_schema_excludes_legacy_nontransferable_features(self) -> None:
        excluded = {
            "MDVP:Shimmer",
            "MDVP:Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "MDVP:APQ",
            "Shimmer:DDA",
            "RPDE",
            "DFA",
            "spread1",
            "spread2",
            "D2",
            "PPE",
        }
        self.assertEqual(len(CONVERSATIONAL_PARKINSONS_FEATURE_NAMES), 10)
        self.assertFalse(excluded & set(CONVERSATIONAL_PARKINSONS_FEATURE_NAMES))

    def test_legacy_feature_helpers_are_removed_from_runtime_extractor(self) -> None:
        source = Path("backend/app/parkinsons_speech_model/parkinsons_features.py").read_text()
        for legacy_name in ["_shimmer_from_audio", "_nonlinear_pitch_features", "_dfa", "_correlation_dimension"]:
            self.assertNotIn(legacy_name, source)

    def test_short_silent_audio_returns_quality_warning(self) -> None:
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "silent.wav"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)
                wav_file.writeframes(np.zeros(16_000, dtype="<i2").tobytes())

            _, warnings = extract_conversational_parkinsons_features(audio_path)

        self.assertTrue(any("shorter than 3 seconds" in warning for warning in warnings))
        self.assertTrue(any("mostly silence" in warning for warning in warnings))

    def test_subject_id_from_uci_name_removes_recording_suffix(self) -> None:
        self.assertEqual(subject_id_from_name("phon_R01_S01_1"), "phon_R01_S01")

    def test_select_best_candidate_prefers_roc_auc_then_balanced_accuracy(self) -> None:
        winner = select_best_candidate(
            [
                CandidateResult(name="balanced", metrics={"roc_auc": None, "balanced_accuracy": 0.95}),
                CandidateResult(name="auc", metrics={"roc_auc": 0.8, "balanced_accuracy": 0.6}),
            ]
        )

        self.assertEqual(winner.name, "auc")

    @unittest.skipUnless(importlib.util.find_spec("joblib"), "joblib is optional")
    def test_predict_parkinsons_speech_model_uses_saved_tabular_artifacts(self) -> None:
        import joblib  # type: ignore

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "patient.wav"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            write_test_wav(audio_path, sample_rate=16_000, seconds=4.0)
            joblib.dump(DummyProbabilityModel(), artifacts / "parkinsons_tabular_model.joblib")
            (artifacts / "feature_schema.json").write_text(json.dumps(CONVERSATIONAL_PARKINSONS_FEATURE_NAMES))
            (artifacts / "model_card.json").write_text(json.dumps({"model_version": "test-tabular-v0"}))

            result = predict_speech_marker(audio_path, artifacts)

        self.assertEqual(result.model_version, "test-tabular-v0")
        self.assertEqual(result.probability, 0.8)
        self.assertIsNotNone(result.features_summary)

    @unittest.skipUnless(importlib.util.find_spec("joblib"), "joblib is optional")
    def test_predict_parkinsons_speech_model_blocks_out_of_training_range_features(self) -> None:
        import joblib  # type: ignore

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "patient.wav"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            write_test_wav(audio_path, sample_rate=16_000, seconds=4.0)
            joblib.dump(DummyProbabilityModel(), artifacts / "parkinsons_tabular_model.joblib")
            (artifacts / "feature_schema.json").write_text(json.dumps(CONVERSATIONAL_PARKINSONS_FEATURE_NAMES))
            (artifacts / "model_card.json").write_text(json.dumps({"model_version": "test-tabular-v0"}))
            (artifacts / "feature_reference_ranges.json").write_text(
                json.dumps({name: {"min": -0.001, "max": 0.001} for name in CONVERSATIONAL_PARKINSONS_FEATURE_NAMES})
            )

            result = predict_speech_marker(audio_path, artifacts)

        self.assertIsNone(result.probability)
        self.assertTrue(any("outside the UCI/Kaggle training range" in warning for warning in result.warnings))
        self.assertGreater(result.features_summary["speechModelFeatureOutOfRangeCount"], 0)

    @unittest.skipUnless(importlib.util.find_spec("joblib"), "joblib is optional")
    def test_long_patient_speech_audio_is_scored_in_chunks(self) -> None:
        import joblib  # type: ignore

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "conversation.wav"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            write_test_wav(audio_path, sample_rate=16_000, seconds=35.0)
            joblib.dump(DummyProbabilityModel(), artifacts / "parkinsons_tabular_model.joblib")
            (artifacts / "feature_schema.json").write_text(json.dumps(CONVERSATIONAL_PARKINSONS_FEATURE_NAMES))
            (artifacts / "model_card.json").write_text(json.dumps({"model_version": "test-tabular-v0"}))

            result = predict_speech_marker(audio_path, artifacts)

        self.assertEqual(result.probability, 0.8)
        self.assertFalse(any("long conversational audio" in warning.lower() for warning in result.warnings))
        self.assertTrue(any("scored" in warning.lower() and "chunks" in warning.lower() for warning in result.warnings))
        self.assertEqual(result.features_summary["speechModelUsable"], "true")
        self.assertGreater(result.features_summary["scoredChunkCount"], 1)

    @unittest.skipUnless(
        all(importlib.util.find_spec(package) for package in ["pandas", "sklearn", "joblib"]),
        "tabular training dependencies are optional",
    )
    def test_train_parkinsons_tabular_script_writes_artifacts(self) -> None:
        rows = []
        for index in range(8):
            label = 1 if index >= 4 else 0
            subject = f"phon_R01_S{index // 2 + 1:02d}_{index % 2 + 1}"
            base = 1.0 + label * 2.0 + index * 0.01
            rows.append([subject, *[base + feature_index * 0.001 for feature_index in range(len(CONVERSATIONAL_PARKINSONS_FEATURE_NAMES))], label])

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "parkinsons.data"
            output_dir = root / "artifacts"
            csv_path.write_text(
                ",".join(["name", *CONVERSATIONAL_PARKINSONS_FEATURE_NAMES, "status"])
                + "\n"
                + "\n".join(",".join(str(value) for value in row) for row in rows)
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "backend/scripts/train_parkinsons_tabular_model.py",
                    str(csv_path),
                    "--output-dir",
                    str(output_dir),
                    "--splits",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "parkinsons_tabular_model.joblib").exists())
            self.assertTrue((output_dir / "feature_schema.json").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "model_card.json").exists())
            self.assertTrue((output_dir / "feature_reference_ranges.json").exists())
            schema = json.loads((output_dir / "feature_schema.json").read_text())
            model_card = json.loads((output_dir / "model_card.json").read_text())
            self.assertEqual(schema, CONVERSATIONAL_PARKINSONS_FEATURE_NAMES)
            self.assertEqual(model_card["feature_schema"], CONVERSATIONAL_PARKINSONS_FEATURE_NAMES)
            self.assertIn("excluded_source_features", model_card)
