from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from app.speech_ml.preprocessing import TARGET_SAMPLE_RATE, load_audio


CONVERSATIONAL_PARKINSONS_FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "NHR",
    "HNR",
]


def pitch_track(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return _autocorrelation_pitch(audio, sample_rate)


def voiced_window_count(audio: np.ndarray, sample_rate: int, window_seconds: float = 4.0, min_voiced_frames: int = 20) -> int:
    window_size = max(1, int(sample_rate * window_seconds))
    count = 0
    for start in range(0, max(1, len(audio) - window_size + 1), window_size):
        window = audio[start : start + window_size]
        if len(window) < window_size:
            continue
        pitches = _autocorrelation_pitch(window, sample_rate)
        if len(pitches) >= min_voiced_frames:
            count += 1
    return count


def _safe(value: float | int | np.floating | None, default: float = 0.0) -> float:
    try:
        next_value = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return next_value if math.isfinite(next_value) else default


def _base_features() -> dict[str, float]:
    return {name: 0.0 for name in CONVERSATIONAL_PARKINSONS_FEATURE_NAMES}


def _autocorrelation_pitch(audio: np.ndarray, sample_rate: int, frame_seconds: float = 0.04, hop_seconds: float = 0.02) -> np.ndarray:
    frame_size = max(1, int(sample_rate * frame_seconds))
    hop = max(1, int(sample_rate * hop_seconds))
    min_lag = max(1, int(sample_rate / 500))
    max_lag = max(min_lag + 1, int(sample_rate / 75))
    pitches: list[float] = []
    for start in range(0, max(0, len(audio) - frame_size), hop):
        frame = audio[start : start + frame_size]
        if len(frame) < frame_size or float(np.sqrt(np.mean(np.square(frame)))) < 0.01:
            continue
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if len(corr) <= max_lag or corr[0] <= 0:
            continue
        search = corr[min_lag:max_lag]
        lag = int(np.argmax(search) + min_lag)
        if corr[lag] / corr[0] > 0.25:
            pitches.append(sample_rate / lag)
    return np.asarray(pitches, dtype=np.float32)


def _pitch_periods(pitches: np.ndarray) -> np.ndarray:
    voiced = pitches[pitches > 0]
    return 1.0 / voiced if len(voiced) else np.asarray([], dtype=np.float32)


def _jitter_from_pitch(pitches: np.ndarray) -> tuple[float, float, float, float, float]:
    periods = _pitch_periods(pitches)
    if len(periods) < 6:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = np.abs(np.diff(periods))
    mean_period = float(np.mean(periods))
    local = float(np.mean(diffs) / max(mean_period, 1e-9))
    local_abs = float(np.mean(diffs))
    rap = float(np.mean(np.abs(periods[1:-1] - np.convolve(periods, np.ones(3) / 3, mode="valid"))) / max(mean_period, 1e-9))
    ppq = float(np.mean(np.abs(periods[2:-2] - np.convolve(periods, np.ones(5) / 5, mode="valid"))) / max(mean_period, 1e-9))
    ddp = 3 * rap
    return local, local_abs, rap, ppq, ddp


def _spectral_noise_ratio(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    if len(audio) < sample_rate // 2:
        return 0.0, 0.0
    windowed = audio[: min(len(audio), sample_rate * 3)] * np.hanning(min(len(audio), sample_rate * 3))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(windowed), 1 / sample_rate)
    tonal = float(np.sum(spectrum[(freqs >= 75) & (freqs <= 500)]))
    noise = float(np.sum(spectrum[(freqs > 500) & (freqs <= 4000)]))
    nhr = noise / max(tonal, 1e-9)
    hnr = 10 * math.log10(max(tonal, 1e-9) / max(noise, 1e-9))
    return nhr, hnr


def _praat_features(audio_path: Path, fallback_pitches: np.ndarray, warnings: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    try:
        import parselmouth  # type: ignore
        from parselmouth.praat import call  # type: ignore
    except ImportError:
        warnings.append("Parselmouth is not installed; using approximate NumPy voice features.")
        return features

    def safe_call(label: str, func) -> None:
        try:
            features[label] = _safe(func())
        except Exception:
            warnings.append(f"Praat feature {label} could not be extracted; using approximate/default value.")

    try:
        sound = parselmouth.Sound(str(audio_path))
        pitch = sound.to_pitch()
        values = pitch.selected_array["frequency"]
        voiced = values[values > 0] if len(values) else fallback_pitches
        if len(voiced):
            features["MDVP:Fo(Hz)"] = _safe(np.mean(voiced))
            features["MDVP:Fhi(Hz)"] = _safe(np.max(voiced))
            features["MDVP:Flo(Hz)"] = _safe(np.min(voiced))
        harmonicity = sound.to_harmonicity_cc()
        safe_call("HNR", lambda: call(harmonicity, "Get mean", 0, 0))
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        safe_call("MDVP:Jitter(%)", lambda: call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        safe_call("MDVP:Jitter(Abs)", lambda: call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3))
        safe_call("MDVP:RAP", lambda: call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
        safe_call("MDVP:PPQ", lambda: call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
        safe_call("Jitter:DDP", lambda: call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3))
    except Exception as exc:
        warnings.append(f"Praat feature extraction failed; using approximate NumPy features. Details: {exc}")
    return features


def extract_conversational_parkinsons_features(audio_path: Path) -> tuple[dict[str, float], list[str]]:
    audio, sample_rate = load_audio(audio_path, target_sample_rate=TARGET_SAMPLE_RATE)
    warnings: list[str] = [
        "The conversational Parkinson speech marker uses pitch, jitter, and harmonic/noise features derived from patient speech."
    ]
    duration = len(audio) / sample_rate if sample_rate else 0.0
    if duration < 3:
        warnings.append("Audio is shorter than 3 seconds; high-risk speech marker may be unreliable.")
    if len(audio) and float(np.mean(np.abs(audio) < 0.01)) > 0.85:
        warnings.append("Audio is mostly silence; ask for clearer patient speech before relying on this score.")
    if len(audio) and float(np.mean(np.abs(audio) > 0.98)) > 0.01:
        warnings.append("Audio appears clipped; reduce microphone gain before relying on this score.")

    features = _base_features()
    pitches = _autocorrelation_pitch(audio, sample_rate)
    if len(pitches) < 6:
        warnings.append("Stable voiced segments were weak or missing; pitch and jitter features are approximate.")
    if len(pitches):
        features["MDVP:Fo(Hz)"] = _safe(np.mean(pitches))
        features["MDVP:Fhi(Hz)"] = _safe(np.max(pitches))
        features["MDVP:Flo(Hz)"] = _safe(np.min(pitches))

    jitter, jitter_abs, rap, ppq, ddp = _jitter_from_pitch(pitches)
    features.update(
        {
            "MDVP:Jitter(%)": jitter,
            "MDVP:Jitter(Abs)": jitter_abs,
            "MDVP:RAP": rap,
            "MDVP:PPQ": ppq,
            "Jitter:DDP": ddp,
        }
    )
    nhr, hnr = _spectral_noise_ratio(audio, sample_rate)
    features["NHR"] = nhr
    features["HNR"] = hnr
    praat_features = _praat_features(audio_path, pitches, warnings)
    features.update({key: value for key, value in praat_features.items() if key in features})
    if "HNR" in praat_features and praat_features["HNR"]:
        features["NHR"] = 1 / max(10 ** (features["HNR"] / 10), 1e-9)

    return {name: _safe(features.get(name)) for name in CONVERSATIONAL_PARKINSONS_FEATURE_NAMES}, warnings


def ordered_feature_vector(features: dict[str, float], feature_schema: list[str] | None = None) -> list[float]:
    schema = feature_schema or CONVERSATIONAL_PARKINSONS_FEATURE_NAMES
    return [_safe(features.get(name)) for name in schema]
