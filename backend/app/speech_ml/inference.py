from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.speech_ml import TARGET_SAMPLE_RATE
from app.speech_ml.parkinsons_features import UCI_PARKINSONS_FEATURE_NAMES, extract_uci_parkinsons_features, ordered_feature_vector, pitch_track, voiced_window_count
from app.speech_ml.preprocessing import load_audio, preprocess_audio


@dataclass
class SpeechModelResult:
    model_version: str | None
    probability: float | None
    warnings: list[str]
    features_summary: dict[str, float | int | str | None] | None


def _quality_gate(audio_path: Path, quality, features: dict[str, float], warnings: list[str]) -> tuple[bool, dict[str, float | int | str | None]]:
    audio, sample_rate = load_audio(audio_path, target_sample_rate=TARGET_SAMPLE_RATE)
    pitches = pitch_track(audio, sample_rate)
    voiced_frames = int(len(pitches))
    voiced_windows = voiced_window_count(audio, sample_rate)
    f0_std = float(np.std(pitches)) if len(pitches) else 0.0
    f0_range = float(np.max(pitches) - np.min(pitches)) if len(pitches) else 0.0
    duration = quality.duration_seconds
    silence_ratio = quality.silence_ratio
    clipping_ratio = quality.clipping_ratio
    usable = True

    if duration < 3:
        warnings.append("Speech marker unavailable: patient-only audio is too short for the tabular voice model.")
        usable = False
    if silence_ratio > 0.6:
        warnings.append("Speech marker unavailable: patient-only audio is too silence-heavy for reliable UCI-style scoring.")
        usable = False
    if clipping_ratio > 0.01:
        warnings.append("Speech marker unavailable: patient-only audio is clipped.")
        usable = False
    if voiced_frames < 20:
        warnings.append("Speech marker unavailable: not enough stable voiced frames were detected.")
        usable = False
    if duration > 30 and voiced_windows < 2:
        warnings.append("Speech marker unavailable: long conversational audio did not contain enough stable voiced windows.")
        usable = False
    if duration > 30:
        warnings.append("Speech marker low confidence: this is long conversational audio, while the model was trained on controlled phonation features.")
        usable = False
    if features.get("MDVP:Fhi(Hz)", 0) > 500 or f0_range > 350:
        warnings.append("Speech marker unavailable: pitch extraction appears unstable for this recording.")
        usable = False

    return usable, {
        "speechModelUsable": "true" if usable else "false",
        "voicedFrameCount": voiced_frames,
        "voicedWindowCount": voiced_windows,
        "f0Std": round(f0_std, 4),
        "f0Range": round(f0_range, 4),
    }


def predict_speech_marker(audio_path: Path, artifact_dir: Path) -> SpeechModelResult:
    warnings: list[str] = []
    chunks, quality = preprocess_audio(audio_path)
    warnings.extend(quality.warnings)
    features, feature_warnings = extract_uci_parkinsons_features(audio_path)
    warnings.extend(feature_warnings)
    usable, quality_summary = _quality_gate(audio_path, quality, features, warnings)

    model_path = artifact_dir / "parkinsons_tabular_model.joblib"
    schema_path = artifact_dir / "feature_schema.json"
    config_path = artifact_dir / "model_card.json"
    if not model_path.exists() or not config_path.exists():
        warnings.append("No trained speech model artifacts were found; probability is unavailable.")
        return SpeechModelResult(
            model_version=None,
            probability=None,
            warnings=warnings,
            features_summary={
                "duration_seconds": quality.duration_seconds,
                "silence_ratio": round(quality.silence_ratio, 4),
                "clipping_ratio": round(quality.clipping_ratio, 4),
                "chunk_count": len(chunks),
                **quality_summary,
                **{key: round(value, 4) for key, value in features.items()},
            },
        )

    with config_path.open() as config_file:
        model_card = json.load(config_file)
    feature_schema = UCI_PARKINSONS_FEATURE_NAMES
    if schema_path.exists():
        with schema_path.open() as schema_file:
            loaded_schema = json.load(schema_file)
        if isinstance(loaded_schema, list) and all(isinstance(item, str) for item in loaded_schema):
            feature_schema = loaded_schema
    try:
        import joblib  # type: ignore

        if usable:
            model = joblib.load(model_path)
            vector = np.asarray([ordered_feature_vector(features, feature_schema)], dtype=np.float32)
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(vector)[0, 1])
            else:
                decision = float(model.decision_function(vector)[0])
                probability = float(1 / (1 + np.exp(-decision)))
        else:
            probability = None
    except Exception as exc:
        warnings.append(f"Trained speech model could not be loaded or executed: {exc}")
        probability = None
    return SpeechModelResult(
        model_version=str(model_card.get("model_version") or model_card.get("model_id") or "speech-marker-research"),
        probability=probability,
        warnings=warnings,
        features_summary={
            "duration_seconds": quality.duration_seconds,
            "silence_ratio": round(quality.silence_ratio, 4),
            "clipping_ratio": round(quality.clipping_ratio, 4),
            "chunk_count": len(chunks),
            **quality_summary,
            **{key: round(value, 4) for key, value in features.items()},
        },
    )
