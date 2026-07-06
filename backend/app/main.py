import os
import json
import mimetypes
import re
import wave
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.data import CHECKINS, SENIORS, VOLUNTEER_TASKS
from app.ml import assess_speech_deviation, extract_demo_embedding_note
from app.models import (
    CallRecord,
    CheckInSession,
    ProviderResult,
    RiskAssessment,
    RiskSignal,
    SavedCallResponse,
    Senior,
    SpeechProfile,
    SpeechDeviationRequest,
    Symptoms,
    TranscriptSegment,
    TranscriptMessage,
    VolunteerTask,
)
from app.providers import GoogleTranslateProvider, clean_transcript_text, transcribe_with_fallback


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BACKEND_ROOT = Path(__file__).resolve().parents[1]
CALL_STORAGE_ROOT = BACKEND_ROOT / "storage" / "calls"
SPEECH_MODEL_ARTIFACT_ROOT = BACKEND_ROOT / "models" / "speech"
app = FastAPI(title="EarlyCare API", version="0.1.0")
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".webm"}
CONTENT_TYPE_AUDIO_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionRequest(BaseModel):
    language: str
    audioHint: str


class ElevenLabsSessionRequest(BaseModel):
    seniorId: str
    seniorName: str
    preferredLanguage: str
    caregiverContact: str
    checkInReason: str


class ElevenLabsSessionResponse(BaseModel):
    configured: bool
    signedUrl: str | None = None
    agentId: str | None = None
    message: str


def _parse_iso(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        normalized = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _relative_seconds(timestamp: str | None, started_at: str) -> float | None:
    current = _parse_iso(timestamp)
    start = _parse_iso(started_at)
    if not current or not start:
        return None
    return max(0, round((current - start).total_seconds(), 3))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_metadata_path(call_id: str) -> Path:
    return CALL_STORAGE_ROOT / call_id / "metadata.json"


def _audio_upload_extension(audio: UploadFile) -> str:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix in SUPPORTED_AUDIO_EXTENSIONS:
        return suffix
    content_type = (audio.content_type or "").split(";")[0].lower()
    return CONTENT_TYPE_AUDIO_EXTENSIONS.get(content_type, ".wav")


async def _save_uploaded_audio(upload: UploadFile | None, call_dir: Path, stem: str) -> tuple[str | None, Path | None]:
    if upload is None:
        return None, None
    audio_path = call_dir / f"{stem}{_audio_upload_extension(upload)}"
    audio_path.write_bytes(await upload.read())
    return str(audio_path), audio_path


def _audio_file_response(audio_path: Path, call_id: str) -> FileResponse:
    media_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    if audio_path.suffix.lower() == ".wav":
        media_type = "audio/wav"
    return FileResponse(audio_path, media_type=media_type, filename=f"{call_id}-{audio_path.name}")


def _read_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise RuntimeError("Patient speech extraction requires 16-bit PCM WAV audio")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _write_wav_mono(audio_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1, 1)
    samples = (clipped * 32767).astype("<i2")
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _voiced_clips(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.012,
    padding_seconds: float = 0.04,
    merge_gap_seconds: float = 0.18,
    min_clip_seconds: float = 0.12,
) -> list[np.ndarray]:
    if not len(audio):
        return []
    frame_size = max(1, int(sample_rate * 0.03))
    hop = max(1, int(sample_rate * 0.01))
    pad = int(sample_rate * padding_seconds)
    voiced_ranges: list[tuple[int, int]] = []
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start : start + frame_size]
        if len(frame) and float(np.sqrt(np.mean(np.square(frame)))) >= threshold:
            voiced_ranges.append((max(0, start - pad), min(len(audio), start + frame_size + pad)))
    if not voiced_ranges:
        return []

    merge_gap = int(sample_rate * merge_gap_seconds)
    merged: list[tuple[int, int]] = []
    for start, end in voiced_ranges:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))

    min_samples = int(sample_rate * min_clip_seconds)
    return [audio[start:end] for start, end in merged if end - start >= min_samples]


def _transcript_to_text(messages: list[TranscriptMessage]) -> str:
    return "\n".join(_role_line(_display_role(message.role), message.text) for message in messages)


def _display_role(role: str) -> str:
    return "Patient" if role == "Senior" else role


def _clean_transcript_text(text: str) -> str:
    return clean_transcript_text(text)


def _strip_speaker_labels(text: str) -> str:
    cleaned = _clean_transcript_text(text)
    while True:
        stripped = re.sub(r"^(?:Agent|Patient|Senior)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if stripped == cleaned:
            return stripped
        cleaned = stripped


def _role_line(role: str, text: str) -> str:
    return f"{_display_role(role)}: {_strip_speaker_labels(text)}"


def _normalize_role_labeled_transcript(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        cleaned = _clean_transcript_text(raw_line)
        match = re.match(r"^(Agent|Patient|Senior)\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE)
        if not match:
            if cleaned:
                lines.append(cleaned)
            continue
        role = _display_role(match.group(1).capitalize())
        lines.append(_role_line(role, match.group(2)))
    return "\n".join(line for line in lines if line.strip())


def _clean_messages(messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
    return [message.model_copy(update={"text": _strip_speaker_labels(message.text)}) for message in messages]


def _word_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[\u4e00-\u9fff]", text)))


def _estimated_utterance_seconds(text: str) -> float:
    words = _word_count(_strip_speaker_labels(text))
    return round(min(12, max(0.9, words / 2.4)), 3)


def _estimate_current_speech_profile(
    messages: list[TranscriptMessage],
    started_at: str,
    completed_at: str,
    segments: list[TranscriptSegment] | None = None,
) -> SpeechProfile | None:
    senior_messages = [message for message in messages if message.role == "Senior" and message.text.strip()]
    timed_segments = [
        segment
        for segment in segments or []
        if segment.startTimeSeconds is not None and segment.endTimeSeconds is not None and (segment.originalText or segment.text).strip()
        and (segment.role in (None, "Patient") or segment.speaker in (None, "Patient") or (segment.originalText or segment.text).startswith("Patient:"))
    ]
    if not senior_messages and not timed_segments:
        return None

    call_start = _parse_iso(started_at)
    call_end = _parse_iso(completed_at)
    duration_seconds = (call_end - call_start).total_seconds() if call_start and call_end else 0

    if timed_segments:
        segment_words = sum(_word_count(segment.originalText or segment.text) for segment in timed_segments)
        spoken_seconds = sum(max(0.1, (segment.endTimeSeconds or 0) - (segment.startTimeSeconds or 0)) for segment in timed_segments)
        speech_rate = round(segment_words / max(spoken_seconds / 60, 0.25), 1)
        segment_starts = sorted(segment.startTimeSeconds for segment in timed_segments if segment.startTimeSeconds is not None)
        pause_values = [
            max(0, (timed_segments[index].startTimeSeconds or 0) - (timed_segments[index - 1].endTimeSeconds or 0)) * 1000
            for index in range(1, len(timed_segments))
        ]
    else:
        words = sum(_word_count(message.text) for message in senior_messages)
        speech_rate = round(words / max(duration_seconds / 60, 0.25), 1) if duration_seconds > 0 else 0
        segment_starts = []
        pause_values = []

    latency_values: list[float] = []
    previous_senior_at: datetime | None = None
    previous_agent_at: datetime | None = None
    for message in messages:
        timestamp = _parse_iso(message.timestamp)
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

    return SpeechProfile(
        speechRate=speech_rate,
        avgPauseMs=avg_pause_ms,
        responseLatencyMs=response_latency_ms,
        pitchVariability=round(min(1, len(set(segment_starts)) / 20), 2) if segment_starts else 0,
        phraseAccuracy=1.0,
        updatedAt=completed_at or _now_iso(),
    )


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_transcript_text(text)
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?。？！])\s+", cleaned) if part.strip()]


def _translate_message_text(language: str, text: str) -> str:
    if language.lower() == "english":
        return _clean_transcript_text(text)
    try:
        return GoogleTranslateProvider().transcribe(language=language, audio_hint=text, audio_path=None).translation
    except Exception:
        return _clean_transcript_text(text)


def _role_labeled_english_transcript(messages: list[TranscriptMessage], language: str, fallback_translation: str) -> str:
    if not messages:
        return _clean_transcript_text(fallback_translation)
    if language.lower() == "english":
        return _transcript_to_text(messages)

    lines: list[str] = []
    translated_any = False
    for message in messages:
        if message.role == "System" or not message.text.strip():
            continue
        translated = _translate_message_text(language, message.text)
        if translated and translated != message.text:
            translated_any = True
        lines.append(_role_line(_display_role(message.role), translated or message.text))

    if translated_any or not fallback_translation:
        return "\n".join(lines)
    if _has_role_labeled_lines(fallback_translation):
        return _normalize_role_labeled_transcript(fallback_translation)

    fallback_sentences = _split_sentences(fallback_translation)
    if not fallback_sentences:
        return "\n".join(lines)
    spoken_messages = [message for message in messages if message.role != "System" and message.text.strip()]
    if len(fallback_sentences) < len(spoken_messages):
        return "\n".join(lines)
    mapped_lines: list[str] = []
    for index, message in enumerate(spoken_messages):
        sentence_index = min(len(fallback_sentences) - 1, round(index * (len(fallback_sentences) - 1) / max(1, len(spoken_messages) - 1)))
        mapped_lines.append(_role_line(_display_role(message.role), fallback_sentences[sentence_index]))
    return "\n".join(mapped_lines)


def _role_labeled_original_transcript(messages: list[TranscriptMessage], fallback_transcript: str) -> str:
    if not messages:
        return _clean_transcript_text(fallback_transcript)

    spoken_messages = [message for message in messages if message.role != "System" and message.text.strip()]
    if not spoken_messages:
        return _transcript_to_text(messages)

    speaker_parts = re.findall(r"<(Speaker\d+)>:\s*(.*?)(?=<Speaker\d+>:|$)", fallback_transcript, flags=re.IGNORECASE | re.DOTALL)
    if len(speaker_parts) >= 2:
        ordered_speakers: list[str] = []
        for speaker, _ in speaker_parts:
            if speaker not in ordered_speakers:
                ordered_speakers.append(speaker)
        role_order: list[str] = []
        for message in spoken_messages:
            role = _display_role(message.role)
            if not role_order or role_order[-1] != role:
                role_order.append(role)
        if len(ordered_speakers) <= len(role_order):
            speaker_roles = {speaker: role_order[index] for index, speaker in enumerate(ordered_speakers)}
            return "\n".join(
                _role_line(speaker_roles.get(speaker, "Patient"), text)
                for speaker, text in speaker_parts
                if _clean_transcript_text(text)
            )

    return _transcript_to_text(spoken_messages)


def _has_explicit_agent_patient_roles(segments: list[TranscriptSegment]) -> bool:
    roles = {
        role
        for segment in segments
        for role in [segment.role, segment.speaker]
        if role in {"Agent", "Patient"}
    }
    text_roles = {
        label
        for segment in segments
        for text in [segment.englishText or segment.text]
        for label in ["Agent", "Patient"]
        if text.startswith(f"{label}:")
    }
    return {"Agent", "Patient"}.issubset(roles | text_roles)


def _has_role_labeled_lines(text: str) -> bool:
    labels = {match.group(1) for match in re.finditer(r"(?m)^\s*(Agent|Patient):", text)}
    return {"Agent", "Patient"}.issubset(labels)


def _role_segments_from_messages(messages: list[TranscriptMessage], started_at: str, english_transcript: str) -> list[TranscriptSegment]:
    english_lines = [line.strip() for line in english_transcript.splitlines() if line.strip()]
    spoken_messages = [message for message in messages if message.role != "System" and message.text.strip()]
    if not spoken_messages or not english_lines:
        return []

    segments: list[TranscriptSegment] = []
    for index, message in enumerate(spoken_messages):
        line = english_lines[index] if index < len(english_lines) else f"{_display_role(message.role)}: {message.text}"
        role = _display_role(message.role)
        english_text = _strip_speaker_labels(line)
        event_time = _relative_seconds(message.timestamp, started_at)
        start = event_time
        end = None
        if role == "Patient" and event_time is not None:
            start = max(0, round(event_time - _estimated_utterance_seconds(message.text), 3))
            end = event_time
        elif role == "Agent" and event_time is not None:
            estimated_end = round(event_time + _estimated_utterance_seconds(english_text or message.text), 3)
            if index + 1 < len(spoken_messages) and spoken_messages[index + 1].role == "Senior":
                next_event = _relative_seconds(spoken_messages[index + 1].timestamp, started_at)
                if next_event is not None:
                    next_patient_start = max(0, round(next_event - _estimated_utterance_seconds(spoken_messages[index + 1].text), 3))
                    estimated_end = min(estimated_end, next_patient_start)
            end = max(event_time, estimated_end)
        segments.append(
            TranscriptSegment(
                text=_role_line(role, english_text),
                originalText=_role_line(role, message.text),
                englishText=_role_line(role, english_text),
                startTimeSeconds=start,
                endTimeSeconds=end,
                role=role,
                speaker=role,
            )
        )
    return segments


def _sync_english_segments(segments: list[TranscriptSegment], original_transcript: str, english_transcript: str) -> list[TranscriptSegment]:
    if not segments:
        english_sentences = _split_sentences(english_transcript)
        original_sentences = _split_sentences(original_transcript)
        return [
            TranscriptSegment(
                text=english,
                originalText=original_sentences[index] if index < len(original_sentences) else None,
                englishText=english,
            )
            for index, english in enumerate(english_sentences or [english_transcript])
        ]

    english_sentences = _split_sentences(english_transcript)
    if len(english_sentences) == len(segments):
        for index, segment in enumerate(segments):
            segment.englishText = english_sentences[index]
            segment.text = english_sentences[index]
        return segments

    if len(segments) == 1:
        segments[0].englishText = english_transcript
        return segments

    for segment in segments:
        if not segment.englishText or segment.englishText == segment.originalText or segment.englishText == segment.text:
            segment.englishText = None
    return segments


def _timed_segments_from_messages(messages: list[TranscriptMessage], started_at: str, english_transcript: str) -> list[TranscriptSegment]:
    english_sentences = _split_sentences(english_transcript)
    timed_messages = [
        (index, message, start)
        for index, message in enumerate(messages)
        for start in [_relative_seconds(message.timestamp, started_at)]
        if start is not None and message.text.strip()
    ]
    if not english_sentences or not timed_messages:
        return []

    starts = [item[2] for item in timed_messages]
    segments: list[TranscriptSegment] = []
    for sentence_index, sentence in enumerate(english_sentences):
        mapped_index = min(len(timed_messages) - 1, round(sentence_index * (len(timed_messages) - 1) / max(1, len(english_sentences) - 1)))
        _, message, start = timed_messages[mapped_index]
        next_start = starts[mapped_index + 1] if mapped_index + 1 < len(starts) else None
        role = _display_role(message.role)
        segments.append(
            TranscriptSegment(
                text=sentence,
                originalText=f"{role}: {message.text}",
                englishText=sentence,
                startTimeSeconds=start,
                endTimeSeconds=next_start,
                role=role,
                speaker=role,
            )
        )
    return segments


def _empty_assessment(risk_level: str, reasons: list[str]) -> RiskAssessment:
    return RiskAssessment(
        speechDeviationScore=0,
        parkinsonsWatchScore=0,
        postFallConcernScore=0,
        missedCheckInScore=0,
        riskLevel=risk_level,  # type: ignore[arg-type]
        reasons=reasons or ["No notable deviation from available baseline context."],
    )


def _risk_schema() -> dict[str, object]:
    signal_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "severity": {"type": "string", "enum": ["Green", "Watch", "Amber", "Red"]},
            "quotedText": {"type": "string"},
            "highlightText": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "sentenceIndex": {"type": ["integer", "null"]},
            "startTimeSeconds": {"type": ["number", "null"]},
            "endTimeSeconds": {"type": ["number", "null"]},
        },
        "required": ["id", "label", "severity", "quotedText", "highlightText", "reason", "sentenceIndex", "startTimeSeconds", "endTimeSeconds"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "riskLevel": {"type": "string", "enum": ["Green", "Watch", "Amber", "Red"]},
            "summary": {"type": "string"},
            "recommendedAction": {"type": "string"},
            "reasons": {"type": "array", "items": {"type": "string"}},
            "symptoms": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fall": {"type": "boolean"},
                    "headImpact": {"type": "boolean"},
                    "headache": {"type": "boolean"},
                    "dizziness": {"type": "boolean"},
                    "vomiting": {"type": "boolean"},
                    "confusion": {"type": "boolean"},
                    "slurredSpeech": {"type": "boolean"},
                    "weakness": {"type": "boolean"},
                    "poorIntake": {"type": "boolean"},
                    "asksForHelp": {"type": "boolean"},
                    "missedCheckIn": {"type": "boolean"},
                },
                "required": [
                    "fall",
                    "headImpact",
                    "headache",
                    "dizziness",
                    "vomiting",
                    "confusion",
                    "slurredSpeech",
                    "weakness",
                    "poorIntake",
                    "asksForHelp",
                    "missedCheckIn",
                ],
            },
            "signals": {"type": "array", "items": signal_schema},
        },
        "required": ["riskLevel", "summary", "recommendedAction", "reasons", "symptoms", "signals"],
    }


def _extract_openai_text(payload: dict[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                    chunks.append(content_item["text"])
        if chunks:
            return "".join(chunks)
    return ""


def _attach_segment_timestamps(signals: list[RiskSignal], segments: list[TranscriptSegment]) -> list[RiskSignal]:
    patient_segments = [
        segment
        for segment in segments
        if (segment.role == "Patient" or segment.speaker == "Patient" or (segment.englishText or segment.text).startswith("Patient:"))
    ]
    search_segments = patient_segments or segments
    normalized_segments = [
        (" ".join(filter(None, [segment.text, segment.englishText])).lower(), index, segment)
        for index, segment in enumerate(search_segments)
        if (segment.text or segment.englishText) and segment.startTimeSeconds is not None
    ]
    next_signals: list[RiskSignal] = []
    for signal in signals:
        match: TranscriptSegment | None = None
        if signal.sentenceIndex is not None and 0 <= signal.sentenceIndex < len(search_segments):
            match = search_segments[signal.sentenceIndex]
        else:
            quoted = (signal.highlightText or signal.quotedText).lower().strip()
            found = next((item for text, _, item in normalized_segments if quoted and (quoted in text or text in quoted)), None)
            match = found
        if match is None or not (
            match.role == "Patient"
            or match.speaker == "Patient"
            or (match.englishText or match.text).startswith("Patient:")
        ):
            continue
        quoted = (signal.highlightText or signal.quotedText).lower().strip()
        match_text = " ".join(filter(None, [match.text, match.englishText])).lower()
        if quoted and quoted not in match_text and match_text not in quoted:
            continue
        next_signals.append(
            signal.model_copy(
                update={
                    "startTimeSeconds": match.startTimeSeconds if match else None,
                    "endTimeSeconds": match.endTimeSeconds if match else None,
                    "highlightText": signal.highlightText or signal.quotedText,
                }
            )
        )
    return next_signals


def _is_patient_segment(segment: TranscriptSegment) -> bool:
    text = segment.englishText or segment.originalText or segment.text
    return segment.role == "Patient" or segment.speaker == "Patient" or text.startswith("Patient:")


def _is_agent_segment(segment: TranscriptSegment) -> bool:
    text = segment.englishText or segment.originalText or segment.text
    return segment.role == "Agent" or segment.speaker == "Agent" or text.startswith("Agent:")


def _segment_start(segment: TranscriptSegment) -> float | None:
    if segment.startTimeSeconds is None:
        return None
    return max(0.0, float(segment.startTimeSeconds))


def _segment_end(segment: TranscriptSegment, raw_duration: float) -> float | None:
    start = _segment_start(segment)
    if start is None:
        return None
    if segment.endTimeSeconds is not None and float(segment.endTimeSeconds) > start:
        return min(raw_duration, float(segment.endTimeSeconds))
    estimated_end = start + _estimated_utterance_seconds(segment.englishText or segment.originalText or segment.text)
    return min(raw_duration, estimated_end)


def _slice_voiced_clips(audio: np.ndarray, sample_rate: int, start_seconds: float, end_seconds: float) -> list[np.ndarray]:
    if end_seconds <= start_seconds:
        return []
    start_index = max(0, int(start_seconds * sample_rate))
    end_index = min(len(audio), int(end_seconds * sample_rate))
    return _voiced_clips(audio[start_index:end_index], sample_rate)


def _agent_answer_windows(segments: list[TranscriptSegment], raw_duration: float) -> list[tuple[float, float]]:
    timed_segments = sorted(
        [segment for segment in segments if _segment_start(segment) is not None],
        key=lambda item: float(item.startTimeSeconds or 0),
    )
    windows: list[tuple[float, float]] = []
    for index, segment in enumerate(timed_segments):
        if not _is_agent_segment(segment):
            continue
        start = _segment_end(segment, raw_duration)
        if start is None:
            continue
        next_turn_start = next(
            (_segment_start(candidate) for candidate in timed_segments[index + 1 :] if _segment_start(candidate) is not None),
            None,
        )
        if next_turn_start is not None:
            start = min(start, next_turn_start)
        next_agent_start = next(
            (_segment_start(candidate) for candidate in timed_segments[index + 1 :] if _is_agent_segment(candidate) and _segment_start(candidate) is not None),
            None,
        )
        end = next_agent_start if next_agent_start is not None else raw_duration
        if end > start:
            windows.append((start, min(raw_duration, end)))
    return windows


def _patient_segment_windows(segments: list[TranscriptSegment], raw_duration: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for segment in segments:
        if not _is_patient_segment(segment):
            continue
        start = _segment_start(segment)
        end = _segment_end(segment, raw_duration)
        if start is None or end is None:
            continue
        start = max(0.0, start - 0.15)
        end = min(raw_duration, end + 0.25)
        if end > start:
            windows.append((start, end))
    return windows


def _build_patient_speech_audio(
    patient_audio_path: Path | None,
    segments: list[TranscriptSegment],
    output_path: Path,
) -> tuple[Path | None, list[str], dict[str, float | int | str | None]]:
    if patient_audio_path is None or not patient_audio_path.exists():
        return None, [], {}
    warnings: list[str] = []
    try:
        audio, sample_rate = _read_wav_mono(patient_audio_path)
    except Exception as exc:
        return None, [f"Patient speech extraction unavailable: {exc}"], {}

    raw_duration = len(audio) / sample_rate if sample_rate else 0.0
    patient_segments = [
        segment
        for segment in segments
        if _is_patient_segment(segment) and segment.startTimeSeconds is not None
    ]
    speech_parts: list[np.ndarray] = []

    extraction_mode = "agent-window-vad"
    windows = _agent_answer_windows(segments, raw_duration)
    for start_seconds, end_seconds in windows:
        speech_parts.extend(_slice_voiced_clips(audio, sample_rate, start_seconds, end_seconds))

    if not speech_parts:
        extraction_mode = "patient-segment-vad"
        windows = _patient_segment_windows(segments, raw_duration)
        for start_seconds, end_seconds in windows:
            speech_parts.extend(_slice_voiced_clips(audio, sample_rate, start_seconds, end_seconds))
        if windows:
            warnings.append("Patient speech extraction used patient segment VAD because agent-bounded windows were unavailable or silent.")

    if not speech_parts:
        extraction_mode = "full-audio-vad"
        windows = [(0.0, raw_duration)] if raw_duration > 0 else []
        warnings.append("Patient speech extraction used full-audio VAD because turn timings were unavailable.")
        speech_parts.extend(_voiced_clips(audio, sample_rate))

    if not speech_parts:
        return None, warnings + ["Patient speech extraction found no usable speech audio."], {
            "rawPatientAudioDurationSeconds": round(raw_duration, 3),
            "patientSpeechDurationSeconds": 0,
            "patientSpeechTurnCount": len(patient_segments),
            "patientSpeechWindowCount": len(windows),
            "patientSpeechVoicedClipCount": 0,
            "patientSpeechExtractionMode": extraction_mode,
        }

    gap = np.zeros(int(sample_rate * 0.04), dtype=np.float32)
    combined_parts: list[np.ndarray] = []
    for index, part in enumerate(speech_parts):
        if index:
            combined_parts.append(gap)
        combined_parts.append(part)
    patient_speech = np.concatenate(combined_parts)
    _write_wav_mono(output_path, patient_speech, sample_rate)
    speech_duration = len(patient_speech) / sample_rate if sample_rate else 0.0
    summary = {
        "rawPatientAudioDurationSeconds": round(raw_duration, 3),
        "patientSpeechDurationSeconds": round(speech_duration, 3),
        "patientSpeechRemovedSeconds": round(max(0, raw_duration - speech_duration), 3),
        "patientSpeechRemovalRatio": round(max(0, raw_duration - speech_duration) / max(raw_duration, 0.001), 4),
        "patientSpeechTurnCount": len(patient_segments) or len(speech_parts),
        "patientSpeechWindowCount": len(windows),
        "patientSpeechVoicedClipCount": len(speech_parts),
        "patientSpeechExtractionMode": extraction_mode,
    }
    return output_path, warnings, summary


def _manual_risk_review() -> tuple[Symptoms, RiskAssessment, list[RiskSignal], str, bool]:
    reasons = ["Manual review required because AI risk extraction is unavailable."]
    return Symptoms(), _empty_assessment("Watch", reasons), [], "Manual review required.", True


def _speech_model_review(audio_path: Path | None) -> tuple[str | None, float | None, list[str], dict[str, float | int | str | None] | None]:
    if audio_path is None or os.getenv("EARLYCARE_SPEECH_MODEL_ENABLED", "").lower() not in {"1", "true", "yes"}:
        return None, None, [], None
    try:
        from app.speech_ml.inference import predict_speech_marker

        result = predict_speech_marker(audio_path, SPEECH_MODEL_ARTIFACT_ROOT)
        return result.model_version, result.probability, result.warnings, result.features_summary
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        return None, None, [f"Speech model review unavailable: {detail}"], None


def _openai_risk_review(english_transcript: str, segments: list[TranscriptSegment]) -> tuple[Symptoms, RiskAssessment, list[RiskSignal], str, bool]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _manual_risk_review()

    patient_segments = [
        segment
        for segment in segments
        if (segment.role == "Patient" or segment.speaker == "Patient" or (segment.englishText or segment.text).startswith("Patient:"))
    ]
    review_segments = patient_segments or segments
    patient_transcript = "\n".join(segment.englishText or segment.text for segment in review_segments)

    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "input": [
                    {
                        "role": "system",
                        "content": (
                            "You are an EarlyCare clinical safety review assistant. Extract decision-support risk signals from an elderly "
                            "wellbeing check-in transcript. Do not diagnose. Highlight only transcript evidence that suggests the patient "
                            "may be at risk, such as falls, sickness, confusion, weakness, dizziness, poor intake, missed check-ins, requests "
                            "for help, unsafe home situations, or other details needing earlier caregiver action. Review patient speech only. "
                            "Ignore agent questions, agent summaries, and any risk wording that the patient did not say. Use exact English patient evidence text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "englishTranscript": english_transcript,
                                "patientOnlyTranscript": patient_transcript,
                                "sentences": [
                                    {
                                        "sentenceIndex": index,
                                        "englishText": segment.englishText or segment.text,
                                        "startTimeSeconds": segment.startTimeSeconds,
                                        "endTimeSeconds": segment.endTimeSeconds,
                                    }
                                    for index, segment in enumerate(review_segments)
                                ],
                                "timestampInstruction": "Set sentenceIndex using only the patient sentence list. Copy that patient sentence timestamp if available; otherwise use null.",
                            }
                        ),
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "earlycare_risk_review",
                        "schema": _risk_schema(),
                        "strict": True,
                    }
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        result = json.loads(_extract_openai_text(payload))
        symptoms = Symptoms.model_validate(result["symptoms"])
        risk_level = result["riskLevel"]
        reasons = result.get("reasons") or [result.get("summary", "AI review completed.")]
        assessment = _empty_assessment(risk_level, reasons)
        signals = [RiskSignal.model_validate(signal) for signal in result.get("signals", [])]
        signals = _attach_segment_timestamps(signals, review_segments)
        recommended_action = result.get("recommendedAction") or "Review highlighted details and continue routine follow-up."
        return symptoms, assessment, signals, recommended_action, False
    except Exception:
        return _manual_risk_review()


def _load_call_record(path: Path) -> CallRecord:
    return CallRecord.model_validate_json(path.read_text())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "product": "EarlyCare"}


@app.get("/seniors", response_model=list[Senior])
def get_seniors() -> list[Senior]:
    return SENIORS


@app.get("/seniors/{senior_id}", response_model=Senior)
def get_senior(senior_id: str) -> Senior:
    for senior in SENIORS:
        if senior.id == senior_id:
            return senior
    raise HTTPException(status_code=404, detail="Senior not found")


@app.get("/checkins", response_model=list[CheckInSession])
def get_checkins() -> list[CheckInSession]:
    return CHECKINS


@app.get("/calls", response_model=list[CallRecord])
def get_calls() -> list[CallRecord]:
    if not CALL_STORAGE_ROOT.exists():
        return []
    records = [_load_call_record(path) for path in CALL_STORAGE_ROOT.glob("*/metadata.json")]
    return sorted(records, key=lambda record: record.completedAt, reverse=True)


@app.get("/calls/{call_id}", response_model=CallRecord)
def get_call(call_id: str) -> CallRecord:
    path = _call_metadata_path(call_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Call not found")
    return _load_call_record(path)


@app.get("/calls/{call_id}/audio")
def get_call_audio(call_id: str) -> FileResponse:
    call_dir = CALL_STORAGE_ROOT / call_id
    audio_path = next(
        (candidate for candidate in [*(call_dir.glob("full-call.*")), call_dir / "mic-audio.webm"] if candidate.exists()),
        None,
    )
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio recording not found")
    return _audio_file_response(audio_path, call_id)


@app.get("/calls/{call_id}/patient-audio")
def get_call_patient_audio(call_id: str) -> FileResponse:
    call_dir = CALL_STORAGE_ROOT / call_id
    audio_path = next((candidate for candidate in call_dir.glob("patient-audio.*") if candidate.exists()), None)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Patient audio recording not found")
    return _audio_file_response(audio_path, call_id)


@app.get("/calls/{call_id}/patient-speech-audio")
def get_call_patient_speech_audio(call_id: str) -> FileResponse:
    call_dir = CALL_STORAGE_ROOT / call_id
    audio_path = next((candidate for candidate in call_dir.glob("patient-speech.*") if candidate.exists()), None)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Patient speech audio recording not found")
    return _audio_file_response(audio_path, call_id)


@app.post("/checkins/start", response_model=CheckInSession)
def start_checkin(senior_id: str) -> CheckInSession:
    senior = get_senior(senior_id)
    return CheckInSession(
        id=f"c-demo-{senior.id}",
        seniorId=senior.id,
        scheduledAt="2026-07-04T12:00:00+08:00",
        status="Needs follow-up",
        language=senior.preferredLanguage,
        riskLevel="Watch",
        summary="Demo check-in started.",
        originalTranscript="",
        englishTranscript="",
        riskAssessment={
            "speechDeviationScore": 0,
            "parkinsonsWatchScore": 0,
            "postFallConcernScore": 0,
            "missedCheckInScore": 0,
            "riskLevel": "Watch",
            "reasons": ["Check-in in progress"],
        },
    )


@app.post("/checkins/{checkin_id}/audio", response_model=ProviderResult)
def transcribe_audio(checkin_id: str, request: TranscriptionRequest) -> ProviderResult:
    _ = checkin_id
    return transcribe_with_fallback(language=request.language, audio_hint=request.audioHint)


@app.post("/elevenlabs/signed-url", response_model=ElevenLabsSessionResponse)
def create_elevenlabs_signed_url(request: ElevenLabsSessionRequest) -> ElevenLabsSessionResponse:
    _ = request
    api_key = os.getenv("ELEVENLABS_API_KEY")
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")

    if not api_key or not agent_id:
        return ElevenLabsSessionResponse(
            configured=False,
            agentId=agent_id,
            message="ElevenLabs is not configured. Add ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID to backend/.env.",
        )

    try:
        response = httpx.get(
            "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url",
            params={"agent_id": agent_id},
            headers={"xi-api-key": api_key},
            timeout=12,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to create ElevenLabs signed URL: {exc}") from exc

    signed_url = response.json().get("signed_url")
    if not signed_url:
        raise HTTPException(status_code=502, detail="ElevenLabs response did not include signed_url")

    return ElevenLabsSessionResponse(
        configured=True,
        signedUrl=signed_url,
        agentId=agent_id,
        message="ElevenLabs signed URL created.",
    )


@app.post("/calls", response_model=SavedCallResponse)
async def save_call(
    seniorId: str = Form(...),
    status: str = Form("Complete"),
    startedAt: str = Form(...),
    completedAt: str = Form(...),
    transcriptMessages: str = Form(...),
    agentAudioCaptured: bool = Form(False),
    audio: UploadFile | None = File(None),
    patientAudio: UploadFile | None = File(None),
) -> SavedCallResponse:
    senior = get_senior(seniorId)
    try:
        raw_messages = json.loads(transcriptMessages)
        messages = [TranscriptMessage.model_validate(message) for message in raw_messages]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid transcriptMessages JSON") from exc

    messages = _clean_messages(messages)
    call_id = f"call-{uuid4().hex[:10]}"
    call_dir = CALL_STORAGE_ROOT / call_id
    call_dir.mkdir(parents=True, exist_ok=True)

    audio_file_path, audio_path = await _save_uploaded_audio(audio, call_dir, "full-call")
    patient_audio_file_path, _ = await _save_uploaded_audio(patientAudio, call_dir, "patient-audio")

    dialogue_transcript = _transcript_to_text(messages)
    translation = transcribe_with_fallback(senior.preferredLanguage, dialogue_transcript, audio_path)
    if translation.fallbackUsed:
        original_transcript = _clean_transcript_text(dialogue_transcript or translation.transcript)
        english_transcript = _clean_transcript_text(
            _role_labeled_english_transcript(messages, senior.preferredLanguage, translation.translation or original_transcript)
        )
    else:
        original_transcript = _clean_transcript_text(_role_labeled_original_transcript(messages, translation.transcript or dialogue_transcript))
        provider_english = _clean_transcript_text(translation.translation or translation.transcript or original_transcript)
        if _has_role_labeled_lines(provider_english):
            english_transcript = provider_english
        else:
            english_transcript = _clean_transcript_text(
                _role_labeled_english_transcript(messages, senior.preferredLanguage, provider_english)
            )
    for segment in translation.segments:
        segment.text = _clean_transcript_text(segment.text)
        segment.originalText = _clean_transcript_text(segment.originalText or segment.text)
        segment.englishText = _clean_transcript_text(segment.englishText or segment.text)
    translation.segments = _sync_english_segments(translation.segments, original_transcript, english_transcript)
    role_source_transcript = (
        english_transcript
        if translation.fallbackUsed
        else _role_labeled_english_transcript(messages, senior.preferredLanguage, english_transcript)
    )
    role_segments = _role_segments_from_messages(messages, startedAt, role_source_transcript)
    if role_segments and (translation.fallbackUsed or not _has_explicit_agent_patient_roles(translation.segments)):
        translation.segments = role_segments
    elif not any(segment.startTimeSeconds is not None for segment in translation.segments):
        timed_segments = _timed_segments_from_messages(messages, startedAt, english_transcript)
        if timed_segments:
            translation.segments = timed_segments
    _, assessment, risk_signals, recommended_action, ai_fallback_used = _openai_risk_review(english_transcript, translation.segments)
    audio_url = f"/calls/{call_id}/audio" if audio_file_path else None
    patient_audio_url = f"/calls/{call_id}/patient-audio" if patient_audio_file_path else None
    current_speech_profile = _estimate_current_speech_profile(messages, startedAt, completedAt, translation.segments)
    patient_speech_file_path: str | None = None
    patient_speech_url: str | None = None
    speech_extraction_warnings: list[str] = []
    speech_extraction_summary: dict[str, float | int | str | None] = {}
    patient_speech_path, speech_extraction_warnings, speech_extraction_summary = _build_patient_speech_audio(
        Path(patient_audio_file_path) if patient_audio_file_path else None,
        translation.segments,
        call_dir / "patient-speech.wav",
    )
    if patient_speech_path is not None:
        patient_speech_file_path = str(patient_speech_path)
        patient_speech_url = f"/calls/{call_id}/patient-speech-audio"
    speech_model_version, speech_model_probability, speech_model_warnings, speech_model_features = _speech_model_review(
        patient_speech_path
    )
    speech_model_warnings = [*speech_extraction_warnings, *speech_model_warnings]
    if speech_model_features is not None:
        speech_model_features = {**speech_model_features, **speech_extraction_summary}
    elif speech_extraction_summary:
        speech_model_features = speech_extraction_summary

    (call_dir / "transcript-original.json").write_text(json.dumps([message.model_dump() for message in messages], indent=2))
    (call_dir / "transcript-english.txt").write_text(english_transcript)

    call = CallRecord(
        id=call_id,
        seniorId=senior.id,
        seniorName=senior.name,
        startedAt=startedAt,
        completedAt=completedAt or _now_iso(),
        status="Complete" if status not in {"Failed", "Saved"} else status,  # type: ignore[arg-type]
        riskLevel=assessment.riskLevel,
        originalTranscript=original_transcript,
        englishTranscript=english_transcript,
        transcriptMessages=messages,
        translationProvider=translation.provider,
        translationFallbackUsed=translation.fallbackUsed,
        transcriptionAttempts=translation.attempts,
        audioFilePath=audio_file_path,
        audioUrl=audio_url,
        audioAvailable=audio_file_path is not None,
        patientAudioFilePath=patient_audio_file_path,
        patientAudioUrl=patient_audio_url,
        patientAudioAvailable=patient_audio_file_path is not None,
        patientSpeechAudioFilePath=patient_speech_file_path,
        patientSpeechAudioUrl=patient_speech_url,
        patientSpeechAudioAvailable=patient_speech_file_path is not None,
        agentAudioCaptured=agentAudioCaptured,
        currentSpeechProfile=current_speech_profile,
        transcriptSegments=translation.segments,
        riskSignals=risk_signals,
        aiRiskFallbackUsed=ai_fallback_used,
        speechModelVersion=speech_model_version,
        speechModelProbability=speech_model_probability,
        speechModelWarnings=speech_model_warnings,
        speechModelFeaturesSummary=speech_model_features,
        riskAssessment=assessment,
        recommendedAction=recommended_action,
    )
    _call_metadata_path(call_id).write_text(call.model_dump_json(indent=2))
    return SavedCallResponse(call=call)


@app.post("/checkins/{checkin_id}/complete", response_model=CheckInSession)
def complete_checkin(checkin_id: str) -> CheckInSession:
    for checkin in CHECKINS:
        if checkin.id == checkin_id:
            return checkin
    raise HTTPException(status_code=404, detail="Demo check-in not found")


@app.post("/ml/speech-deviation")
def speech_deviation(request: SpeechDeviationRequest) -> dict[str, object]:
    senior = get_senior(request.seniorId)
    assessment = assess_speech_deviation(senior.baselineSpeechProfile, request)
    return {
        "assessment": assessment,
        "modelNote": extract_demo_embedding_note(),
    }


@app.get("/volunteer-tasks", response_model=list[VolunteerTask])
def get_volunteer_tasks() -> list[VolunteerTask]:
    return VOLUNTEER_TASKS


@app.patch("/volunteer-tasks/{task_id}", response_model=VolunteerTask)
def update_volunteer_task(task_id: str, status: str) -> VolunteerTask:
    for task in VOLUNTEER_TASKS:
        if task.id == task_id:
            task.status = status  # type: ignore[assignment]
            return task
    raise HTTPException(status_code=404, detail="Volunteer task not found")
