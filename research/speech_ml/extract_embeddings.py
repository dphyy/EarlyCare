#!/usr/bin/env python3
"""Extract offline speech rows for EarlyCare research.

This script writes backend-compatible JSONL rows without changing the FastAPI
runtime. The default `demo` model is deterministic and standard-library only,
so it can smoke-test the data path before heavyweight encoders are installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPEAT_PHRASE = "today i am safe at home and i can ask for help"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
HEAVY_MODEL_NAMES = {
    "meralion": "MERaLiON/MERaLiON-SpeechEncoder-2",
    "wavlm": "microsoft/wavlm-base-plus",
    "wav2vec2": "facebook/wav2vec2-base",
}


@dataclass(frozen=True)
class AudioSample:
    dataset: str
    speaker_id: str
    label: str
    task: str
    audio_path: Path
    language: str = ""
    transcript: str = ""
    source_id: str = ""


@dataclass(frozen=True)
class WavStats:
    duration_seconds: float
    sample_rate: int
    channels: int
    rms: float
    peak: float
    silence_ratio: float
    avg_silence_ms: float
    zero_crossing_rate: float
    zero_crossing_variability: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return round(value * 2 - 1, 6)


def read_manifest(path: Path, audio_root: Path) -> list[AudioSample]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

    samples: list[AudioSample] = []
    for index, row in enumerate(rows, start=1):
        audio_value = row.get("audio_path") or row.get("path") or row.get("file")
        if not audio_value:
            raise ValueError(f"Manifest row {index} is missing audio_path")
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = audio_root / audio_path
        samples.append(
            AudioSample(
                dataset=(row.get("dataset") or "unknown").strip(),
                speaker_id=(row.get("speaker_id") or row.get("speaker") or f"speaker-{index}").strip(),
                label=(row.get("label") or "unknown").strip(),
                task=(row.get("task") or "unknown").strip(),
                audio_path=audio_path,
                language=(row.get("language") or "").strip(),
                transcript=(row.get("transcript") or row.get("text") or "").strip(),
                source_id=(row.get("source_id") or row.get("recording_id") or audio_path.stem).strip(),
            )
        )
    return samples


def scan_audio_root(audio_root: Path, dataset: str) -> list[AudioSample]:
    samples: list[AudioSample] = []
    for path in sorted(audio_root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        relative_parts = path.relative_to(audio_root).parts
        label = relative_parts[0] if len(relative_parts) > 1 else "unknown"
        speaker_id = relative_parts[1] if len(relative_parts) > 2 else path.stem.split("_")[0]
        task = relative_parts[2] if len(relative_parts) > 3 else path.parent.name
        samples.append(
            AudioSample(
                dataset=dataset,
                speaker_id=speaker_id,
                label=label,
                task=task,
                audio_path=path,
                source_id=path.stem,
            )
        )
    return samples


def wav_samples(path: Path) -> tuple[list[float], int, int]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()
        frames = audio.readframes(frame_count)

    if sample_width not in {1, 2, 4}:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}")

    if sample_width == 1:
        values = [(byte - 128) / 128 for byte in frames]
    else:
        fmt = {2: "<h", 4: "<i"}[sample_width]
        max_value = float(2 ** (sample_width * 8 - 1))
        values = [struct.unpack_from(fmt, frames, offset)[0] / max_value for offset in range(0, len(frames), sample_width)]

    if channels > 1:
        mono = []
        for offset in range(0, len(values), channels):
            mono.append(sum(values[offset : offset + channels]) / channels)
        values = mono

    return values, sample_rate, channels


def analyze_wav(path: Path) -> WavStats:
    if path.suffix.lower() != ".wav":
        raise ValueError(f"The demo extractor supports WAV files only: {path}")
    values, sample_rate, channels = wav_samples(path)
    if not values or sample_rate <= 0:
        raise ValueError(f"Empty or invalid WAV file: {path}")

    duration_seconds = len(values) / sample_rate
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    peak = max(abs(value) for value in values)
    window_size = max(1, int(sample_rate * 0.02))
    windows = [values[index : index + window_size] for index in range(0, len(values), window_size)]
    window_rms = [math.sqrt(sum(value * value for value in window) / max(len(window), 1)) for window in windows]
    threshold = max(0.01, rms * 0.35)
    silent_flags = [value < threshold for value in window_rms]
    silence_ratio = sum(1 for flag in silent_flags if flag) / max(len(silent_flags), 1)

    silent_runs: list[int] = []
    current_run = 0
    for flag in silent_flags:
        if flag:
            current_run += 1
        elif current_run:
            silent_runs.append(current_run)
            current_run = 0
    if current_run:
        silent_runs.append(current_run)
    avg_silence_ms = statistics.mean(silent_runs) * 20 if silent_runs else 0

    crossing_counts: list[float] = []
    for window in windows:
        if len(window) < 2:
            continue
        crossings = sum(1 for index in range(1, len(window)) if (window[index - 1] < 0) != (window[index] < 0))
        crossing_counts.append(crossings / len(window))
    zero_crossing_rate = statistics.mean(crossing_counts) if crossing_counts else 0
    zero_crossing_variability = statistics.pstdev(crossing_counts) if len(crossing_counts) > 1 else 0

    return WavStats(
        duration_seconds=round(duration_seconds, 6),
        sample_rate=sample_rate,
        channels=channels,
        rms=round(rms, 6),
        peak=round(peak, 6),
        silence_ratio=round(silence_ratio, 6),
        avg_silence_ms=round(avg_silence_ms, 3),
        zero_crossing_rate=round(zero_crossing_rate, 6),
        zero_crossing_variability=round(zero_crossing_variability, 6),
    )


def demo_embedding(sample: AudioSample, stats: WavStats, dimensions: int) -> list[float]:
    base_values = [
        stats.duration_seconds / 30,
        stats.rms,
        stats.peak,
        stats.silence_ratio,
        stats.avg_silence_ms / 2000,
        stats.zero_crossing_rate,
        stats.zero_crossing_variability,
    ]
    embedding: list[float] = []
    for index in range(dimensions):
        metric = base_values[index % len(base_values)]
        jitter = stable_float(f"{sample.dataset}:{sample.speaker_id}:{sample.source_id}:{index}") * 0.05
        embedding.append(round(max(-1, min(1, metric + jitter)), 6))
    return embedding


def speech_metrics(sample: AudioSample, stats: WavStats, embedding: list[float], extracted_at: str) -> dict[str, object]:
    word_count = len(sample.transcript.split())
    speech_rate = round(word_count / max(stats.duration_seconds / 60, 0.25), 1) if sample.transcript else 0
    phrase_accuracy = 0.96 if REPEAT_PHRASE in sample.transcript.lower() else 0
    return {
        "speechRate": speech_rate,
        "avgPauseMs": stats.avg_silence_ms,
        "responseLatencyMs": 0,
        "pitchVariability": round(min(1, stats.zero_crossing_variability * 20), 2),
        "phraseAccuracy": phrase_accuracy,
        "embedding": embedding,
        "updatedAt": extracted_at,
    }


class HeavySpeechEncoder:
    def __init__(self, model_key: str, device: str, trust_remote_code: bool) -> None:
        try:
            import soundfile as sf  # type: ignore
            import torch  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "Heavy encoders require optional research dependencies: torch, transformers, and soundfile. "
                "Install them in a separate research environment, not the app runtime."
            ) from error

        self.sf = sf
        self.torch = torch
        self.model_name = HEAVY_MODEL_NAMES[model_key]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=trust_remote_code).to(device)
        self.model.eval()

    def extract(self, audio_path: Path) -> tuple[list[float], dict[str, object]]:
        audio, sample_rate = self.sf.read(str(audio_path), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().tolist()
        return [round(float(value), 6) for value in hidden], {"sample_rate": int(sample_rate)}


def build_row(sample: AudioSample, model: str, dimensions: int, encoder: HeavySpeechEncoder | None, extracted_at: str) -> dict[str, object]:
    stats = analyze_wav(sample.audio_path) if sample.audio_path.suffix.lower() == ".wav" else None
    if model == "demo":
        if stats is None:
            raise ValueError(f"The demo extractor supports WAV files only: {sample.audio_path}")
        embedding = demo_embedding(sample, stats, dimensions)
        audio_extra: dict[str, object] = {}
    else:
        if encoder is None:
            raise RuntimeError(f"Encoder for {model} was not initialized")
        embedding, audio_extra = encoder.extract(sample.audio_path)
        if stats is None:
            stats = WavStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

    metrics = speech_metrics(sample, stats, embedding, extracted_at)
    return {
        "dataset": sample.dataset,
        "speaker_id": sample.speaker_id,
        "label": sample.label,
        "task": sample.task,
        "embedding": embedding,
        "speech_metrics": metrics,
        "provenance": {
            "source_id": sample.source_id,
            "audio_path": str(sample.audio_path),
            "language": sample.language,
            "model": model,
            "model_name": "demo-standard-library" if model == "demo" else HEAVY_MODEL_NAMES[model],
            "duration_seconds": stats.duration_seconds,
            "sample_rate": stats.sample_rate,
            "channels": stats.channels,
            "rms": stats.rms,
            "peak": stats.peak,
            "silence_ratio": stats.silence_ratio,
            "zero_crossing_rate": stats.zero_crossing_rate,
            "zero_crossing_variability": stats.zero_crossing_variability,
            "extracted_at": extracted_at,
            **audio_extra,
        },
    }


def write_jsonl(rows: Iterable[dict[str, object]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract EarlyCare offline speech embeddings into JSONL.")
    parser.add_argument("--manifest", type=Path, help="CSV or JSONL manifest with audio_path, dataset, speaker_id, label, and task columns.")
    parser.add_argument("--audio-root", type=Path, default=Path("research/datasets"), help="Ignored local folder containing dataset audio.")
    parser.add_argument("--dataset", default="local", help="Dataset name when scanning --audio-root without a manifest.")
    parser.add_argument("--output", type=Path, default=Path("research/artifacts/speech_embeddings.jsonl"))
    parser.add_argument("--model", choices=["demo", "meralion", "wavlm", "wav2vec2"], default="demo")
    parser.add_argument("--dimensions", type=int, default=16, help="Demo embedding dimensions.")
    parser.add_argument("--device", default="cpu", help="Device for optional heavy encoders.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Hugging Face model loading.")
    parser.add_argument("--limit", type=int, help="Process only the first N samples.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = read_manifest(args.manifest, args.audio_root) if args.manifest else scan_audio_root(args.audio_root, args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No audio samples found. Add a manifest or files under research/datasets/.")

    encoder = None if args.model == "demo" else HeavySpeechEncoder(args.model, args.device, args.trust_remote_code)
    extracted_at = utc_now()
    rows = (build_row(sample, args.model, args.dimensions, encoder, extracted_at) for sample in samples)
    count = write_jsonl(rows, args.output)
    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
