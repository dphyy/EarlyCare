from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from app.speech_ml.preprocessing import TARGET_SAMPLE_RATE, load_audio


UCI_PARKINSONS_FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
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
    return {name: 0.0 for name in UCI_PARKINSONS_FEATURE_NAMES}


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


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_seconds: float = 0.04, hop_seconds: float = 0.02) -> np.ndarray:
    frame_size = max(1, int(sample_rate * frame_seconds))
    hop = max(1, int(sample_rate * hop_seconds))
    values = [
        float(np.sqrt(np.mean(np.square(audio[start : start + frame_size]))))
        for start in range(0, max(0, len(audio) - frame_size), hop)
        if len(audio[start : start + frame_size]) == frame_size
    ]
    return np.asarray(values, dtype=np.float32)


def _shimmer_from_audio(audio: np.ndarray, sample_rate: int) -> tuple[float, float, float, float, float, float]:
    amplitudes = _frame_rms(audio, sample_rate)
    amplitudes = amplitudes[amplitudes > 1e-5]
    if len(amplitudes) < 6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    mean_amp = float(np.mean(amplitudes))
    diffs = np.abs(np.diff(amplitudes))
    shimmer = float(np.mean(diffs) / max(mean_amp, 1e-9))
    shimmer_db = float(np.mean(np.abs(20 * np.log10(np.maximum(amplitudes[1:], 1e-9) / np.maximum(amplitudes[:-1], 1e-9)))))
    apq3 = float(np.mean(np.abs(amplitudes[1:-1] - np.convolve(amplitudes, np.ones(3) / 3, mode="valid"))) / max(mean_amp, 1e-9))
    apq5 = float(np.mean(np.abs(amplitudes[2:-2] - np.convolve(amplitudes, np.ones(5) / 5, mode="valid"))) / max(mean_amp, 1e-9))
    window = min(11, len(amplitudes))
    apq = float(np.mean(np.abs(amplitudes[window // 2 : -(window // 2)] - np.convolve(amplitudes, np.ones(window) / window, mode="valid"))) / max(mean_amp, 1e-9)) if len(amplitudes) > window else apq5
    dda = 3 * apq3
    return shimmer, shimmer_db, apq3, apq5, apq, dda


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


def _entropy(values: np.ndarray, bins: int = 20) -> float:
    if len(values) < 3:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=False)
    probs = hist[hist > 0] / max(1, np.sum(hist))
    return float(-np.sum(probs * np.log(probs)))


def _dfa(values: np.ndarray) -> float:
    if len(values) < 16:
        return 0.0
    series = np.cumsum(values - np.mean(values))
    scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(5, len(series) // 4)), 8)).astype(int))
    fluctuations: list[float] = []
    valid_scales: list[int] = []
    for scale in scales:
        if scale < 4:
            continue
        segments = len(series) // scale
        if segments < 2:
            continue
        rms_values: list[float] = []
        for index in range(segments):
            chunk = series[index * scale : (index + 1) * scale]
            x = np.arange(scale)
            coeffs = np.polyfit(x, chunk, 1)
            trend = np.polyval(coeffs, x)
            rms_values.append(float(np.sqrt(np.mean(np.square(chunk - trend)))))
        fluctuations.append(float(np.mean(rms_values)))
        valid_scales.append(int(scale))
    if len(fluctuations) < 2 or any(value <= 0 for value in fluctuations):
        return 0.0
    slope, _ = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)
    return float(slope)


def _correlation_dimension(values: np.ndarray, embedding_dim: int = 3) -> float:
    if len(values) < embedding_dim + 20:
        return 0.0
    embedded = np.asarray([values[index : index + embedding_dim] for index in range(len(values) - embedding_dim + 1)])
    if len(embedded) > 150:
        embedded = embedded[np.linspace(0, len(embedded) - 1, 150).astype(int)]
    distances = np.linalg.norm(embedded[:, None, :] - embedded[None, :, :], axis=2)
    distances = distances[np.triu_indices_from(distances, k=1)]
    distances = distances[distances > 0]
    if len(distances) < 10:
        return 0.0
    radii = np.percentile(distances, [10, 20, 30, 40, 50])
    counts = np.asarray([np.mean(distances < radius) for radius in radii if radius > 0])
    radii = radii[: len(counts)]
    valid = counts > 0
    if np.sum(valid) < 2 or len(np.unique(radii[valid])) < 2 or len(np.unique(counts[valid])) < 2:
        return 0.0
    slope, _ = np.polyfit(np.log(radii[valid]), np.log(counts[valid]), 1)
    return float(slope)


def _nonlinear_pitch_features(pitches: np.ndarray, audio: np.ndarray) -> tuple[float, float, float, float, float, float]:
    voiced = pitches[pitches > 0]
    if len(voiced) < 6:
        return 0.0, _dfa(audio[:: max(1, len(audio) // 1000)]), 0.0, 0.0, 0.0, 0.0
    log_pitch = np.log(voiced)
    spread1 = float(np.mean(log_pitch))
    spread2 = float(np.std(log_pitch))
    ppe = _entropy(log_pitch - np.mean(log_pitch))
    rpde = _entropy(_pitch_periods(voiced))
    d2 = _correlation_dimension(log_pitch)
    return rpde, _dfa(log_pitch), spread1, spread2, d2 if d2 else ppe, ppe


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
        safe_call("MDVP:Shimmer", lambda: call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        safe_call("MDVP:Shimmer(dB)", lambda: call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        safe_call("Shimmer:APQ3", lambda: call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        safe_call("Shimmer:APQ5", lambda: call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        safe_call("MDVP:APQ", lambda: call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        safe_call("Shimmer:DDA", lambda: call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception as exc:
        warnings.append(f"Praat feature extraction failed; using approximate NumPy features. Details: {exc}")
    return features


def extract_uci_parkinsons_features(audio_path: Path) -> tuple[dict[str, float], list[str]]:
    audio, sample_rate = load_audio(audio_path, target_sample_rate=TARGET_SAMPLE_RATE)
    warnings: list[str] = [
        "UCI/Kaggle Parkinson features were designed for controlled sustained phonation; conversational EarlyCare audio is an approximate screening input."
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
        warnings.append("Stable voiced segments were weak or missing; many dysphonia features are approximate.")
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
    shimmer, shimmer_db, apq3, apq5, apq, dda = _shimmer_from_audio(audio, sample_rate)
    features.update(
        {
            "MDVP:Shimmer": shimmer,
            "MDVP:Shimmer(dB)": shimmer_db,
            "Shimmer:APQ3": apq3,
            "Shimmer:APQ5": apq5,
            "MDVP:APQ": apq,
            "Shimmer:DDA": dda,
        }
    )
    nhr, hnr = _spectral_noise_ratio(audio, sample_rate)
    features["NHR"] = nhr
    features["HNR"] = hnr
    rpde, dfa, spread1, spread2, d2, ppe = _nonlinear_pitch_features(pitches, audio)
    features.update({"RPDE": rpde, "DFA": dfa, "spread1": spread1, "spread2": spread2, "D2": d2, "PPE": ppe})
    praat_features = _praat_features(audio_path, pitches, warnings)
    features.update({key: value for key, value in praat_features.items() if key in features})
    if "HNR" in praat_features and praat_features["HNR"]:
        features["NHR"] = 1 / max(10 ** (features["HNR"] / 10), 1e-9)

    return {name: _safe(features.get(name)) for name in UCI_PARKINSONS_FEATURE_NAMES}, warnings


def ordered_feature_vector(features: dict[str, float], feature_schema: list[str] | None = None) -> list[float]:
    schema = feature_schema or UCI_PARKINSONS_FEATURE_NAMES
    return [_safe(features.get(name)) for name in schema]
