import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import httpx
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
app = FastAPI(title="EarlyCare API", version="0.1.0")

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


def _transcript_to_text(messages: list[TranscriptMessage]) -> str:
    return "\n".join(f"{'Patient' if message.role == 'Senior' else message.role}: {message.text}" for message in messages)


def _display_role(role: str) -> str:
    return "Patient" if role == "Senior" else role


def _clean_transcript_text(text: str) -> str:
    return clean_transcript_text(text)


def _clean_messages(messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
    return [message.model_copy(update={"text": _clean_transcript_text(message.text)}) for message in messages]


def _word_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[\u4e00-\u9fff]", text)))


def _estimated_utterance_seconds(text: str) -> float:
    words = _word_count(re.sub(r"^(Agent|Patient):\s*", "", text))
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
    repeat_phrase = "today i am safe at home and i can ask for help"
    combined = " ".join(message.text.lower() for message in senior_messages)
    phrase_accuracy = 0.96 if repeat_phrase in combined else 0
    if not phrase_accuracy and timed_segments:
        segment_text = " ".join((segment.englishText or segment.originalText or segment.text).lower() for segment in timed_segments)
        phrase_accuracy = 0.96 if repeat_phrase in segment_text else 0

    return SpeechProfile(
        speechRate=speech_rate,
        avgPauseMs=avg_pause_ms,
        responseLatencyMs=response_latency_ms,
        pitchVariability=round(min(1, len(set(segment_starts)) / 20), 2) if segment_starts else 0,
        phraseAccuracy=phrase_accuracy,
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
        lines.append(f"{_display_role(message.role)}: {translated or message.text}")

    if translated_any or not fallback_translation:
        return "\n".join(lines)

    fallback_sentences = _split_sentences(fallback_translation)
    if not fallback_sentences:
        return "\n".join(lines)
    spoken_messages = [message for message in messages if message.role != "System" and message.text.strip()]
    mapped_lines: list[str] = []
    for index, message in enumerate(spoken_messages):
        sentence_index = min(len(fallback_sentences) - 1, round(index * (len(fallback_sentences) - 1) / max(1, len(spoken_messages) - 1)))
        mapped_lines.append(f"{_display_role(message.role)}: {fallback_sentences[sentence_index]}")
    return "\n".join(mapped_lines)


def _role_segments_from_messages(messages: list[TranscriptMessage], started_at: str, english_transcript: str) -> list[TranscriptSegment]:
    english_lines = [line.strip() for line in english_transcript.splitlines() if line.strip()]
    spoken_messages = [message for message in messages if message.role != "System" and message.text.strip()]
    if not spoken_messages or not english_lines:
        return []

    segments: list[TranscriptSegment] = []
    for index, message in enumerate(spoken_messages):
        line = english_lines[index] if index < len(english_lines) else f"{_display_role(message.role)}: {message.text}"
        role = _display_role(message.role)
        english_text = re.sub(r"^(Agent|Patient):\s*", "", line).strip()
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
                text=f"{role}: {english_text}",
                originalText=f"{role}: {message.text}",
                englishText=f"{role}: {english_text}",
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


def _manual_risk_review() -> tuple[Symptoms, RiskAssessment, list[RiskSignal], str, bool]:
    reasons = ["Manual review required because AI risk extraction is unavailable."]
    return Symptoms(), _empty_assessment("Watch", reasons), [], "Manual review required.", True


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
    audio_path = CALL_STORAGE_ROOT / call_id / "full-call.webm"
    if not audio_path.exists():
        audio_path = CALL_STORAGE_ROOT / call_id / "mic-audio.webm"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio recording not found")
    return FileResponse(audio_path, media_type="audio/webm", filename=f"{call_id}-{audio_path.name}")


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

    audio_file_path: str | None = None
    audio_path: Path | None = None
    if audio is not None:
        audio_path = call_dir / "full-call.webm"
        audio_path.write_bytes(await audio.read())
        audio_file_path = str(audio_path)

    dialogue_transcript = _transcript_to_text(messages)
    translation = transcribe_with_fallback(senior.preferredLanguage, dialogue_transcript, audio_path)
    original_transcript = _clean_transcript_text(dialogue_transcript or translation.transcript)
    english_transcript = _clean_transcript_text(
        _role_labeled_english_transcript(messages, senior.preferredLanguage, translation.translation or original_transcript)
    )
    for segment in translation.segments:
        segment.text = _clean_transcript_text(segment.text)
        segment.originalText = _clean_transcript_text(segment.originalText or segment.text)
        segment.englishText = _clean_transcript_text(segment.englishText or segment.text)
    translation.segments = _sync_english_segments(translation.segments, original_transcript, english_transcript)
    role_segments = _role_segments_from_messages(messages, startedAt, english_transcript)
    if role_segments:
        translation.segments = role_segments
    elif not any(segment.startTimeSeconds is not None for segment in translation.segments):
        timed_segments = _timed_segments_from_messages(messages, startedAt, english_transcript)
        if timed_segments:
            translation.segments = timed_segments
    _, assessment, risk_signals, recommended_action, ai_fallback_used = _openai_risk_review(english_transcript, translation.segments)
    audio_url = f"/calls/{call_id}/audio" if audio_file_path else None
    current_speech_profile = _estimate_current_speech_profile(messages, startedAt, completedAt, translation.segments)

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
        audioFilePath=audio_file_path,
        audioUrl=audio_url,
        audioAvailable=audio_file_path is not None,
        agentAudioCaptured=agentAudioCaptured,
        currentSpeechProfile=current_speech_profile,
        transcriptSegments=translation.segments,
        riskSignals=risk_signals,
        aiRiskFallbackUsed=ai_fallback_used,
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
