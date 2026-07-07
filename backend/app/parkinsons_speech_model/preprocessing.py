from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.parkinsons_speech_model import TARGET_SAMPLE_RATE


@dataclass
class AudioQuality:
    sample_rate: int
    duration_seconds: float
    silence_ratio: float
    clipping_ratio: float
    warnings: list[str]


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise RuntimeError("Only 16-bit PCM WAV can be read without optional audio dependencies")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def load_audio(path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    try:
        import librosa  # type: ignore

        audio, sample_rate = librosa.load(path, sr=target_sample_rate, mono=True)
        return audio.astype(np.float32), sample_rate
    except ImportError:
        audio, sample_rate = _read_pcm_wav(path)
        if sample_rate != target_sample_rate:
            audio = resample_linear(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        return audio.astype(np.float32), sample_rate


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or not len(audio):
        return audio.astype(np.float32)
    source_times = np.arange(len(audio)) / source_rate
    target_length = max(1, round(len(audio) * target_rate / source_rate))
    target_times = np.arange(target_length) / target_rate
    return np.interp(target_times, source_times, audio).astype(np.float32)


def trim_long_silence(audio: np.ndarray, threshold: float = 0.015, max_silent_samples: int = TARGET_SAMPLE_RATE // 2) -> np.ndarray:
    if not len(audio):
        return audio
    keep = np.abs(audio) >= threshold
    if keep.any():
        first = int(np.argmax(keep))
        last = int(len(keep) - np.argmax(keep[::-1]))
        audio = audio[first:last]
    chunks: list[np.ndarray] = []
    silent_run = 0
    for sample in audio:
        if abs(float(sample)) < threshold:
            silent_run += 1
            if silent_run <= max_silent_samples:
                chunks.append(np.array([sample], dtype=np.float32))
        else:
            silent_run = 0
            chunks.append(np.array([sample], dtype=np.float32))
    return np.concatenate(chunks) if chunks else audio[:0]


def normalize_amplitude(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if not len(audio):
        return audio.astype(np.float32)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs <= 0:
        return audio.astype(np.float32)
    return (audio * min(peak / max_abs, 10.0)).astype(np.float32)


def chunk_audio(audio: np.ndarray, sample_rate: int, max_duration_seconds: float = 20.0) -> list[np.ndarray]:
    max_samples = max(1, int(sample_rate * max_duration_seconds))
    return [audio[index : index + max_samples] for index in range(0, len(audio), max_samples)] or [audio]


def assess_quality(audio: np.ndarray, sample_rate: int) -> AudioQuality:
    duration = len(audio) / sample_rate if sample_rate else 0
    silence_ratio = float(np.mean(np.abs(audio) < 0.01)) if len(audio) else 1.0
    clipping_ratio = float(np.mean(np.abs(audio) > 0.98)) if len(audio) else 0.0
    warnings: list[str] = []
    if duration < 3:
        warnings.append("Audio is shorter than 3 seconds; model confidence may be unreliable.")
    if silence_ratio > 0.85:
        warnings.append("Audio is mostly silence; ask for a clearer patient recording.")
    if clipping_ratio > 0.01:
        warnings.append("Audio appears clipped; reduce input gain before relying on this score.")
    if sample_rate != TARGET_SAMPLE_RATE:
        warnings.append(f"Audio was resampled from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz.")
    return AudioQuality(sample_rate=sample_rate, duration_seconds=round(duration, 3), silence_ratio=silence_ratio, clipping_ratio=clipping_ratio, warnings=warnings)


def preprocess_audio(path: Path, max_duration_seconds: float = 20.0) -> tuple[list[np.ndarray], AudioQuality]:
    original_audio, original_rate = load_audio(path, target_sample_rate=TARGET_SAMPLE_RATE)
    quality = assess_quality(original_audio, original_rate)
    audio = trim_long_silence(original_audio)
    audio = normalize_amplitude(audio)
    return chunk_audio(audio, TARGET_SAMPLE_RATE, max_duration_seconds), quality
