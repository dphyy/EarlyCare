from __future__ import annotations

import json
import tempfile
import wave
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


def _public_warnings(warnings: list[str]) -> list[str]:
    hidden_fragments = [
        "controlled sustained phonation",
        "conversational earlycare audio is an approximate screening input",
    ]
    public: list[str] = []
    for warning in warnings:
        lowered = warning.lower()
        if any(fragment in lowered for fragment in hidden_fragments):
            continue
        if warning not in public:
            public.append(warning)
    return public


def _write_pcm_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    samples = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


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
        warnings.append("Speech marker unavailable: long patient-speech audio did not contain enough stable voiced windows.")
        usable = False
    if features.get("MDVP:Fhi(Hz)", 0) > 800 or f0_range > 500:
        warnings.append("Speech marker unavailable: pitch extraction appears unstable for this recording.")
        usable = False
    elif features.get("MDVP:Fhi(Hz)", 0) > 500 or f0_range > 350:
        warnings.append("Speech marker low confidence: pitch varied widely across this patient-speech sample.")

    return usable, {
        "speechModelUsable": "true" if usable else "false",
        "voicedFrameCount": voiced_frames,
        "voicedWindowCount": voiced_windows,
        "f0Std": round(f0_std, 4),
        "f0Range": round(f0_range, 4),
    }


def _predict_probability(model, features: dict[str, float], feature_schema: list[str]) -> float:
    vector = np.asarray([ordered_feature_vector(features, feature_schema)], dtype=np.float32)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(vector)[0, 1])
    decision = float(model.decision_function(vector)[0])
    return float(1 / (1 + np.exp(-decision)))


def _score_single_audio(audio_path: Path, model, feature_schema: list[str]) -> tuple[float | None, list[str], dict[str, float | int | str | None]]:
    chunks, quality = preprocess_audio(audio_path)
    warnings = list(quality.warnings)
    features, feature_warnings = extract_uci_parkinsons_features(audio_path)
    warnings.extend(feature_warnings)
    usable, quality_summary = _quality_gate(audio_path, quality, features, warnings)
    probability = _predict_probability(model, features, feature_schema) if usable else None
    return probability, _public_warnings(warnings), {
        "duration_seconds": quality.duration_seconds,
        "silence_ratio": round(quality.silence_ratio, 4),
        "clipping_ratio": round(quality.clipping_ratio, 4),
        "chunk_count": len(chunks),
        **quality_summary,
        **{key: round(value, 4) for key, value in features.items()},
    }


def _score_chunked_audio(audio_path: Path, model, feature_schema: list[str]) -> tuple[float | None, list[str], dict[str, float | int | str | None]]:
    chunks, quality = preprocess_audio(audio_path, max_duration_seconds=12.0)
    if quality.duration_seconds <= 24 or len(chunks) <= 1:
        return _score_single_audio(audio_path, model, feature_schema)

    probabilities: list[float] = []
    warnings = _public_warnings(quality.warnings)
    scored_summaries: list[dict[str, float | int | str | None]] = []
    skipped_chunks = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for index, chunk in enumerate(chunks):
            if len(chunk) / TARGET_SAMPLE_RATE < 3:
                skipped_chunks += 1
                continue
            chunk_path = root / f"patient-speech-chunk-{index}.wav"
            _write_pcm_wav(chunk_path, chunk, TARGET_SAMPLE_RATE)
            probability, chunk_warnings, summary = _score_single_audio(chunk_path, model, feature_schema)
            warnings.extend(chunk_warnings)
            if probability is None:
                skipped_chunks += 1
                continue
            probabilities.append(probability)
            scored_summaries.append(summary)

    if probabilities:
        median_probability = float(np.median(probabilities))
        warnings.append(f"Speech marker scored {len(probabilities)} patient-speech chunks and aggregated the median probability.")
        summary = dict(scored_summaries[0])
        summary.update(
            {
                "duration_seconds": quality.duration_seconds,
                "chunk_count": len(chunks),
                "scoredChunkCount": len(probabilities),
                "skippedChunkCount": skipped_chunks,
                "speechModelUsable": "true",
            }
        )
        return median_probability, _public_warnings(warnings), summary

    warnings.append("Speech marker unavailable: no patient-speech chunk passed quality checks.")
    return None, _public_warnings(warnings), {
        "duration_seconds": quality.duration_seconds,
        "silence_ratio": round(quality.silence_ratio, 4),
        "clipping_ratio": round(quality.clipping_ratio, 4),
        "chunk_count": len(chunks),
        "scoredChunkCount": 0,
        "skippedChunkCount": skipped_chunks,
        "speechModelUsable": "false",
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
            warnings=_public_warnings(warnings),
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

        model = joblib.load(model_path)
        probability, warnings, features_summary = _score_chunked_audio(audio_path, model, feature_schema)
    except Exception as exc:
        warnings.append(f"Trained speech model could not be loaded or executed: {exc}")
        probability = None
        features_summary = {
            "duration_seconds": quality.duration_seconds,
            "silence_ratio": round(quality.silence_ratio, 4),
            "clipping_ratio": round(quality.clipping_ratio, 4),
            "chunk_count": len(chunks),
            **quality_summary,
            **{key: round(value, 4) for key, value in features.items()},
        }
    return SpeechModelResult(
        model_version=str(model_card.get("model_version") or model_card.get("model_id") or "speech-marker-research"),
        probability=probability,
        warnings=_public_warnings(warnings),
        features_summary=features_summary,
    )
