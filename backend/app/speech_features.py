import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from app.models import SpeechProfile, TranscriptMessage, TranscriptSegment


REPEAT_PHRASE = "today i am safe at home and i can ask for help"
HAN_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\W\d_\u4e00-\u9fff]+", re.UNICODE)


def parse_iso(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        normalized = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def word_count(text: str) -> int:
    han_count = len(HAN_CHARACTER_PATTERN.findall(text))
    text_without_han = HAN_CHARACTER_PATTERN.sub(" ", text)
    return max(1, han_count + len(WORD_PATTERN.findall(text_without_han)))


def estimated_utterance_seconds(text: str) -> float:
    words = word_count(re.sub(r"^(Agent|Patient):\s*", "", text))
    return round(min(12, max(0.9, words / 2.4)), 3)


@dataclass(frozen=True)
class SpeechFeatureInput:
    audio_path: Path | None
    messages: list[TranscriptMessage]
    started_at: str
    completed_at: str
    segments: list[TranscriptSegment] | None = None


class EmbeddingExtractor(Protocol):
    provider: str

    def extract_embedding(self, payload: SpeechFeatureInput) -> list[float] | None:
        ...


class NoopEmbeddingExtractor:
    provider = "none"

    def extract_embedding(self, payload: SpeechFeatureInput) -> list[float] | None:
        _ = payload
        return None


class DemoSpeechFeatureExtractor:
    provider = "demo metrics"

    def __init__(self, embedding_extractor: EmbeddingExtractor | None = None) -> None:
        self.embedding_extractor = embedding_extractor or NoopEmbeddingExtractor()

    def extract(self, payload: SpeechFeatureInput) -> SpeechProfile | None:
        senior_messages = [message for message in payload.messages if message.role == "Senior" and message.text.strip()]
        timed_segments = [
            segment
            for segment in payload.segments or []
            if segment.startTimeSeconds is not None
            and segment.endTimeSeconds is not None
            and (segment.originalText or segment.text).strip()
            and (
                segment.role in (None, "Patient")
                or segment.speaker in (None, "Patient")
                or (segment.originalText or segment.text).startswith("Patient:")
            )
        ]
        if not senior_messages and not timed_segments:
            return None

        call_start = parse_iso(payload.started_at)
        call_end = parse_iso(payload.completed_at)
        duration_seconds = (call_end - call_start).total_seconds() if call_start and call_end else 0

        if timed_segments:
            segment_words = sum(word_count(segment.originalText or segment.text) for segment in timed_segments)
            spoken_seconds = sum(max(0.1, (segment.endTimeSeconds or 0) - (segment.startTimeSeconds or 0)) for segment in timed_segments)
            speech_rate = round(segment_words / max(spoken_seconds / 60, 0.25), 1)
            segment_starts = sorted(segment.startTimeSeconds for segment in timed_segments if segment.startTimeSeconds is not None)
            pause_values = [
                max(0, (timed_segments[index].startTimeSeconds or 0) - (timed_segments[index - 1].endTimeSeconds or 0)) * 1000
                for index in range(1, len(timed_segments))
            ]
        else:
            words = sum(word_count(message.text) for message in senior_messages)
            speech_rate = round(words / max(duration_seconds / 60, 0.25), 1) if duration_seconds > 0 else 0
            segment_starts = []
            pause_values = []

        latency_values: list[float] = []
        previous_senior_at: datetime | None = None
        previous_agent_at: datetime | None = None
        for message in payload.messages:
            timestamp = parse_iso(message.timestamp)
            if not timestamp:
                continue
            if message.role == "Agent":
                previous_agent_at = timestamp
            elif message.role == "Senior":
                if previous_senior_at and not timed_segments:
                    pause_values.append(max(0, (timestamp - previous_senior_at).total_seconds() * 1000))
                if previous_agent_at:
                    latency_values.append(max(0, (timestamp - previous_agent_at).total_seconds() * 1000))
                    previous_agent_at = None
                previous_senior_at = timestamp

        avg_pause_ms = round(sum(pause_values) / len(pause_values), 1) if pause_values else 0
        response_latency_ms = round(sum(latency_values) / len(latency_values), 1) if latency_values else 0
        combined = " ".join(message.text.lower() for message in senior_messages)
        phrase_accuracy = 0.96 if REPEAT_PHRASE in combined else 0
        if not phrase_accuracy and timed_segments:
            segment_text = " ".join((segment.englishText or segment.originalText or segment.text).lower() for segment in timed_segments)
            phrase_accuracy = 0.96 if REPEAT_PHRASE in segment_text else 0

        return SpeechProfile(
            speechRate=speech_rate,
            avgPauseMs=avg_pause_ms,
            responseLatencyMs=response_latency_ms,
            pitchVariability=round(min(1, len(set(segment_starts)) / 20), 2) if segment_starts else 0,
            phraseAccuracy=phrase_accuracy,
            embedding=self.embedding_extractor.extract_embedding(payload),
            updatedAt=payload.completed_at,
        )
