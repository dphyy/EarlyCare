"""Audio loading, normalization, and quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


@dataclass(frozen=True)
class AudioQuality:
    ok: bool
    reason: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 0
    rms: float = 0.0
    clipping_fraction: float = 0.0


@dataclass(frozen=True)
class ProcessedAudio:
    waveform: np.ndarray
    sample_rate: int
    quality: AudioQuality


def _read_wave_or_text(path: Path) -> tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix in {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}:
        audio, sample_rate = sf.read(str(path), always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    if suffix == ".nsp":
        data = path.read_bytes()
        sample_rate = int.from_bytes(data[40:44], "little", signed=False)
        marker = data.index(b"SDA_")
        n_bytes = int.from_bytes(data[marker + 4 : marker + 8], "little", signed=False)
        payload = data[marker + 8 : marker + 8 + n_bytes]
        audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        return audio, sample_rate

    if suffix == ".txt":
        values = np.loadtxt(path, dtype=np.float32)
        # VOICED text files contain samples from 8 kHz recordings.
        return np.asarray(values, dtype=np.float32), 8000

    if suffix == ".hea":
        try:
            import wfdb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Reading WFDB .hea/.dat records requires installing the 'voiced' extra: "
                "pip install -e '.[voiced]'"
            ) from exc
        record = wfdb.rdrecord(str(path.with_suffix("")))
        audio = np.asarray(record.p_signal, dtype=np.float32)
        return audio, int(record.fs)

    raise ValueError(f"Unsupported audio format: {path}")


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio_path = Path(path)
    audio, sample_rate = _read_wave_or_text(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}")
    return audio.astype(np.float32, copy=False), sample_rate


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def assess_quality(
    audio: np.ndarray,
    sample_rate: int,
    min_seconds: float = 0.5,
    silence_rms_threshold: float = 0.001,
    clipping_threshold: float = 0.99,
    clipping_fraction_threshold: float = 0.01,
) -> AudioQuality:
    duration_sec = float(audio.shape[0] / sample_rate) if sample_rate else 0.0
    if audio.size == 0 or sample_rate <= 0:
        return AudioQuality(False, "empty_or_invalid_audio", duration_sec, sample_rate)
    if duration_sec < min_seconds:
        return AudioQuality(False, "too_short", duration_sec, sample_rate)

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    clipping_fraction = float(np.mean(np.abs(audio) >= clipping_threshold))
    if rms < silence_rms_threshold:
        return AudioQuality(False, "silent_or_near_silent", duration_sec, sample_rate, rms, clipping_fraction)
    if clipping_fraction > clipping_fraction_threshold:
        return AudioQuality(False, "clipped", duration_sec, sample_rate, rms, clipping_fraction)
    return AudioQuality(True, "", duration_sec, sample_rate, rms, clipping_fraction)


def preprocess_audio(
    path: str | Path,
    target_rate: int = 16000,
    max_seconds: float = 30.0,
    min_seconds: float = 0.5,
    silence_rms_threshold: float = 0.001,
    clipping_threshold: float = 0.99,
    clipping_fraction_threshold: float = 0.01,
) -> ProcessedAudio:
    try:
        audio, source_rate = load_audio(path)
    except Exception as exc:  # noqa: BLE001 - the reason is surfaced in quality metadata.
        quality = AudioQuality(False, f"unreadable:{type(exc).__name__}", 0.0, 0)
        return ProcessedAudio(np.zeros(0, dtype=np.float32), target_rate, quality)

    audio = normalize_audio(audio)
    audio = resample_audio(audio, source_rate, target_rate)
    max_samples = int(max_seconds * target_rate)
    if max_samples > 0 and audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    audio = normalize_audio(audio)
    quality = assess_quality(
        audio,
        target_rate,
        min_seconds=min_seconds,
        silence_rms_threshold=silence_rms_threshold,
        clipping_threshold=clipping_threshold,
        clipping_fraction_threshold=clipping_fraction_threshold,
    )
    return ProcessedAudio(audio, target_rate, quality)
