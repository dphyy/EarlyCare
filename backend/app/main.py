import base64
import hashlib
import hmac
import os
import json
import mimetypes
import re
import sqlite3
import time
import wave
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.data import CHECKINS, SENIORS, VOLUNTEER_TASKS
from app.concussion_speech import not_applicable_concussion_review, review_concussion_speech
from app.ml import assess_speech_deviation, extract_demo_embedding_note
from app.models import (
    CallRecord,
    ConsultationMemoryItem,
    ConcussionSpeechReview,
    CheckInSession,
    CrisisResource,
    EmotionConcernLevel,
    EmotionProviderResult,
    EmotionSegment,
    ModelExplanationItem,
    ParkinsonsSpeechReview,
    ProviderResult,
    RiskAssessment,
    RiskSignal,
    SafeguardLevel,
    SavedCallResponse,
    Senior,
    SpeechProfile,
    SpeechDeviationRequest,
    Symptoms,
    TranscriptSegment,
    TranscriptMessage,
    TranscriptionAttempt,
    VolunteerTaskUpdate,
    VolunteerTask,
)
from app.providers import GoogleTranslateProvider, _safe_error_reason, clean_transcript_text, transcribe_with_fallback
from app.readiness import readiness_report


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BACKEND_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path(os.getenv("EARLYCARE_STORAGE_ROOT", BACKEND_ROOT / "storage"))
CALL_STORAGE_ROOT = Path(os.getenv("EARLYCARE_CALL_STORAGE_ROOT", STORAGE_ROOT / "calls"))
PARKINSONS_SPEECH_MODEL_ARTIFACT_ROOT = BACKEND_ROOT / "models" / "parkinsons_speech"
FRONTEND_DIST_ROOT = Path(os.getenv("EARLYCARE_FRONTEND_DIST", BACKEND_ROOT.parent / "frontend" / "dist"))
app = FastAPI(title="EarlyCare API", version="0.1.0")
AUTH_COOKIE_NAME = "earlycare_session"
AUTH_PROTECTED_PREFIXES = ("/seniors", "/checkins", "/calls", "/elevenlabs", "/ml", "/volunteer-tasks")
AUTH_PUBLIC_PATHS = {"/health", "/readiness", "/auth/login", "/auth/logout", "/auth/me"}
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}
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
SINGAPORE_CRISIS_RESOURCES = [
    CrisisResource(
        name="Emergency medical services",
        phone="995",
        description="Call for immediate medical danger or urgent ambulance support in Singapore.",
    ),
    CrisisResource(
        name="Police emergency",
        phone="999",
        description="Call if there is immediate danger, violence, or urgent police assistance is needed in Singapore.",
    ),
    CrisisResource(
        name="Samaritans of Singapore hotline",
        phone="1767",
        description="24-hour emotional support and crisis hotline in Singapore.",
    ),
    CrisisResource(
        name="Samaritans of Singapore CareText",
        text="WhatsApp 9151 1767",
        url="https://www.sos.org.sg/",
        description="24-hour WhatsApp text support for emotional support or crisis-related concerns.",
    ),
]
SAFEGUARD_LEVEL_RISK: dict[str, str] = {
    "None": "Green",
    "Support": "Watch",
    "Urgent": "Amber",
    "Emergency": "Red",
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


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthStatus(BaseModel):
    authEnabled: bool
    authenticated: bool
    username: str | None = None
    message: str | None = None


def _operator_username() -> str:
    return os.getenv("EARLYCARE_OPERATOR_USERNAME", "operator")


def _operator_password() -> str | None:
    return os.getenv("EARLYCARE_OPERATOR_PASSWORD")


def _auth_secret() -> str | None:
    return os.getenv("EARLYCARE_AUTH_SECRET")


def _auth_enabled() -> bool:
    disabled = os.getenv("EARLYCARE_AUTH_DISABLED", "").strip().lower() in {"1", "true", "yes"}
    return not disabled and bool(_operator_password() and _auth_secret())


def _session_ttl_seconds() -> int:
    raw_value = os.getenv("EARLYCARE_SESSION_TTL_SECONDS", "43200")
    try:
        return max(300, int(raw_value))
    except ValueError:
        return 43200


def _base64url_encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _base64url_decode(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)


def _sign_session_payload(payload: str) -> str:
    secret = _auth_secret()
    if not secret:
        raise RuntimeError("Auth secret is not configured")
    digest = hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest()
    return _base64url_encode(digest)


def _create_session_token(username: str) -> str:
    now = int(time.time())
    payload = _base64url_encode(
        json.dumps({"sub": username, "iat": now, "exp": now + _session_ttl_seconds()}, separators=(",", ":")).encode("utf-8")
    )
    signature = _sign_session_payload(payload)
    return f"{payload}.{signature}"


def _verify_session_token(token: str | None) -> str | None:
    if not token or "." not in token or not _auth_enabled():
        return None
    payload, signature = token.rsplit(".", 1)
    expected_signature = _sign_session_payload(payload)
    if not hmac.compare_digest(signature, expected_signature):
        return None
    try:
        data = json.loads(_base64url_decode(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(data.get("exp", 0)) < int(time.time()):
        return None
    username = data.get("sub")
    if username != _operator_username():
        return None
    return username


def _request_operator(request: Request) -> str | None:
    return _verify_session_token(request.cookies.get(AUTH_COOKIE_NAME))


def _is_protected_api_path(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in AUTH_PROTECTED_PREFIXES)


def _is_public_path(path: str) -> bool:
    return path in AUTH_PUBLIC_PATHS or path.startswith("/assets/")


def _cookie_secure(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"


def _rate_limit_enabled() -> bool:
    return os.getenv("EARLYCARE_RATE_LIMIT_DISABLED", "").strip().lower() not in {"1", "true", "yes"}


def _env_int(name: str, fallback: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(fallback))))
    except ValueError:
        return fallback


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit_identity(request: Request) -> str:
    username = _request_operator(request)
    return f"user:{username}" if username else f"ip:{_client_ip(request)}"


def _rate_limit_rule(request: Request) -> tuple[str, int, int] | None:
    path = request.url.path
    method = request.method.upper()
    if method == "POST" and path == "/auth/login":
        return "auth-login", _env_int("EARLYCARE_RATE_LIMIT_LOGIN_PER_MINUTE", 8), 60
    if method == "POST" and path == "/elevenlabs/signed-url":
        return "elevenlabs-signed-url", _env_int("EARLYCARE_RATE_LIMIT_ELEVENLABS_PER_MINUTE", 6), 60
    if method == "POST" and path == "/calls":
        return "save-call", _env_int("EARLYCARE_RATE_LIMIT_CALL_SAVE_PER_MINUTE", 4), 60
    if method == "POST" and path.startswith("/checkins/"):
        return "checkin-audio", _env_int("EARLYCARE_RATE_LIMIT_CHECKIN_AUDIO_PER_MINUTE", 10), 60
    if method == "POST" and path == "/ml/speech-deviation":
        return "speech-ml", _env_int("EARLYCARE_RATE_LIMIT_ML_PER_MINUTE", 12), 60
    if _is_protected_api_path(path):
        return "api", _env_int("EARLYCARE_RATE_LIMIT_API_PER_MINUTE", 240), 60
    return None


def _rate_limit_retry_after(identity: str, bucket_name: str, limit: int, window_seconds: int) -> int | None:
    now = time.monotonic()
    bucket_key = f"{identity}:{bucket_name}"
    bucket = RATE_LIMIT_BUCKETS.setdefault(bucket_key, deque())
    while bucket and now - bucket[0] >= window_seconds:
        bucket.popleft()
    if len(bucket) >= limit:
        return max(1, int(window_seconds - (now - bucket[0])))
    bucket.append(now)
    return None


def _upload_too_large(request: Request) -> JSONResponse | None:
    if request.method.upper() != "POST" or request.url.path != "/calls":
        return None
    content_length = request.headers.get("content-length")
    if not content_length:
        return None
    max_bytes = _env_int("EARLYCARE_MAX_CALL_UPLOAD_MB", 30) * 1024 * 1024
    try:
        if int(content_length) > max_bytes:
            return JSONResponse({"detail": "Call upload is too large"}, status_code=413)
    except ValueError:
        return None
    return None


@app.middleware("http")
async def operator_auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    upload_response = _upload_too_large(request)
    if upload_response is not None:
        return upload_response

    if _rate_limit_enabled():
        rule = _rate_limit_rule(request)
        if rule is not None:
            bucket_name, limit, window_seconds = rule
            retry_after = _rate_limit_retry_after(_rate_limit_identity(request), bucket_name, limit, window_seconds)
            if retry_after is not None:
                return JSONResponse(
                    {"detail": "Rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )

    if not _auth_enabled() or _is_public_path(request.url.path):
        return await call_next(request)
    if _is_protected_api_path(request.url.path) and _request_operator(request) is None:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)


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


def _database_path() -> Path:
    default_root = CALL_STORAGE_ROOT.parent if CALL_STORAGE_ROOT.name == "calls" else CALL_STORAGE_ROOT
    return Path(os.getenv("EARLYCARE_DB_PATH", default_root / "earlycare.sqlite3"))


def _connect_database() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_database() -> None:
    with _connect_database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                senior_id TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS volunteer_task_status (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def _upsert_call_index(call: CallRecord) -> None:
    _ensure_database()
    with _connect_database() as connection:
        connection.execute(
            """
            INSERT INTO calls (id, senior_id, completed_at, risk_level, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                senior_id=excluded.senior_id,
                completed_at=excluded.completed_at,
                risk_level=excluded.risk_level,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (call.id, call.seniorId, call.completedAt, call.riskLevel, call.model_dump_json(indent=2), _now_iso()),
        )


def _persist_call_record(call: CallRecord) -> None:
    _call_metadata_path(call.id).write_text(call.model_dump_json(indent=2))
    _upsert_call_index(call)


def _load_indexed_call_records() -> list[CallRecord]:
    try:
        _ensure_database()
        with _connect_database() as connection:
            rows = connection.execute("SELECT metadata_json FROM calls ORDER BY completed_at DESC").fetchall()
    except sqlite3.Error:
        return []

    records: list[CallRecord] = []
    for row in rows:
        try:
            records.append(CallRecord.model_validate_json(row["metadata_json"]))
        except Exception:
            continue
    return records


def _load_indexed_call_record(call_id: str) -> CallRecord | None:
    try:
        _ensure_database()
        with _connect_database() as connection:
            row = connection.execute("SELECT metadata_json FROM calls WHERE id = ?", (call_id,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        return CallRecord.model_validate_json(row["metadata_json"])
    except Exception:
        return None


def _load_call_records_from_files() -> list[CallRecord]:
    if not CALL_STORAGE_ROOT.exists():
        return []
    records: list[CallRecord] = []
    for path in CALL_STORAGE_ROOT.glob("*/metadata.json"):
        try:
            records.append(_load_call_record(path))
        except Exception:
            continue
    return records


def _load_all_call_records() -> list[CallRecord]:
    records_by_id = {record.id: record for record in _load_indexed_call_records()}
    for record in _load_call_records_from_files():
        records_by_id[record.id] = record
        _upsert_call_index(record)
    return sorted(records_by_id.values(), key=lambda record: record.completedAt, reverse=True)


def _persist_volunteer_task_status(task: VolunteerTask) -> None:
    _ensure_database()
    with _connect_database() as connection:
        connection.execute(
            """
            INSERT INTO volunteer_task_status (task_id, status, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (task.id, task.status, _now_iso()),
        )


def _load_volunteer_task_statuses() -> dict[str, str]:
    try:
        _ensure_database()
        with _connect_database() as connection:
            rows = connection.execute("SELECT task_id, status FROM volunteer_task_status").fetchall()
    except sqlite3.Error:
        return {}
    return {row["task_id"]: row["status"] for row in rows}


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


def _contains_non_english_script(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\u0b80-\u0bff\u0d00-\u0d7f\u0600-\u06ff]", text))


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


def _live_spoken_messages(messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
    return [message for message in messages if message.role != "System" and message.text.strip()]


def _has_live_spoken_messages(messages: list[TranscriptMessage]) -> bool:
    return bool(_live_spoken_messages(messages))


def _role_labeled_original_from_live_messages(messages: list[TranscriptMessage]) -> str:
    return _transcript_to_text(_live_spoken_messages(messages))


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


def _canonical_transcript_for_comparison(text: str) -> str:
    normalized = _clean_transcript_text(text).replace("\r\n", "\n")
    normalized = re.sub(r"(?mi)^\s*Senior\s*:", "Patient:", normalized)
    normalized = re.sub(r"(?mi)^\s*(Agent|Patient)\s*:\s*", r"\1: ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


def _public_english_transcript(original_transcript: str, english_transcript: str) -> str:
    if not english_transcript.strip():
        return ""
    if _canonical_transcript_for_comparison(original_transcript) == _canonical_transcript_for_comparison(english_transcript):
        return ""
    return english_transcript


def _provider_text_units(text: str) -> list[str]:
    role_labeled = [
        _strip_speaker_labels(match.group(2))
        for match in re.finditer(r"(?m)^\s*(Agent|Patient|Senior)\s*:\s*(.+)$", text, flags=re.IGNORECASE)
    ]
    return [unit for unit in role_labeled if unit] or [_strip_speaker_labels(sentence) for sentence in _split_sentences(text)]


def _translate_message_text(language: str, text: str) -> str:
    cleaned = _clean_transcript_text(text)
    if language.lower() == "english" and not _contains_non_english_script(cleaned):
        return cleaned
    translate_language = "auto" if language.lower() == "english" else language
    if not _contains_non_english_script(cleaned) and translate_language == "auto":
        return cleaned
    try:
        return GoogleTranslateProvider().transcribe(language=translate_language, audio_hint=cleaned, audio_path=None).translation
    except Exception:
        translated = _openai_translate_text_to_english(cleaned)
        if translated != cleaned:
            return translated
        return _clean_transcript_text(text)


def _add_warning(warnings: list[str] | None, message: str) -> None:
    if warnings is not None and message not in warnings:
        warnings.append(message)


def _role_labeled_english_transcript(
    messages: list[TranscriptMessage],
    language: str,
    fallback_translation: str,
    warnings: list[str] | None = None,
) -> str:
    if not messages:
        _add_warning(warnings, "English transcript used provider text because no live transcript messages were available.")
        return _clean_transcript_text(fallback_translation)

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
        if translated_any:
            _add_warning(warnings, "English transcript was rebuilt from live role-labeled messages with per-turn translation.")
        return "\n".join(lines)

    fallback_units = _provider_text_units(fallback_translation)
    if not fallback_units:
        _add_warning(warnings, "Provider English transcript was empty or unsplittable; using live transcript text.")
        return "\n".join(lines)
    spoken_messages = _live_spoken_messages(messages)
    if len(fallback_units) != len(spoken_messages):
        _add_warning(
            warnings,
            "Provider English transcript sentence count did not match live turn count; using live transcript text to preserve speaker roles.",
        )
        return "\n".join(lines)
    mapped_lines: list[str] = []
    for index, message in enumerate(spoken_messages):
        mapped_lines.append(_role_line(_display_role(message.role), fallback_units[index]))
    if _has_role_labeled_lines(fallback_translation):
        _add_warning(warnings, "ElevenLabs live roles were used; MERaLiON speaker labels ignored.")
    else:
        _add_warning(warnings, "Provider English transcript was mapped to live transcript turns because provider roles were unavailable.")
    return "\n".join(mapped_lines)


def _role_labeled_original_transcript(
    messages: list[TranscriptMessage],
    fallback_transcript: str,
    warnings: list[str] | None = None,
) -> str:
    if _has_live_spoken_messages(messages):
        _add_warning(warnings, "ElevenLabs live roles were used for the original transcript; MERaLiON speaker labels ignored.")
        return _role_labeled_original_from_live_messages(messages)

    if not fallback_transcript:
        return _transcript_to_text(messages)
    if _has_role_labeled_lines(fallback_transcript):
        _add_warning(warnings, "Original transcript used provider role labels because no live spoken messages were available.")
        return _normalize_role_labeled_transcript(fallback_transcript)

    speaker_parts = re.findall(r"<(Speaker\d+)>:\s*(.*?)(?=<Speaker\d+>:|$)", fallback_transcript, flags=re.IGNORECASE | re.DOTALL)
    if speaker_parts:
        _add_warning(warnings, "Original transcript used provider speaker labels because no live spoken messages were available.")
        return "\n".join(
            f"{speaker}: {_strip_speaker_labels(text)}"
            for speaker, text in speaker_parts
            if _clean_transcript_text(text)
        )

    if not messages:
        return _clean_transcript_text(fallback_transcript)

    spoken_messages = _live_spoken_messages(messages)
    if not spoken_messages:
        return _transcript_to_text(messages)

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


def _sync_english_segments(
    segments: list[TranscriptSegment],
    original_transcript: str,
    english_transcript: str,
    warnings: list[str] | None = None,
) -> list[TranscriptSegment]:
    if not segments:
        _add_warning(warnings, "Transcript segments were estimated from transcript text because provider timing was unavailable.")
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
        _add_warning(warnings, "Provider returned one segment; English transcript timing is approximate.")
        segments[0].englishText = english_transcript
        return segments

    _add_warning(warnings, "Provider segment count did not match English sentence count; preserving provider segment timing without sentence remapping.")
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


def _safeguard_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "level": {"type": "string", "enum": ["None", "Support", "Urgent", "Emergency"]},
            "category": {
                "type": ["string", "null"],
                "enum": [
                    "emotional_distress",
                    "self_harm_or_suicidal_ideation",
                    "abuse_or_neglect",
                    "medical_emergency",
                    "unsafe_environment",
                    "other",
                    None,
                ],
            },
            "summary": {"type": "string"},
            "recommendedAction": {"type": ["string", "null"]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "resourceNames": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["level", "category", "summary", "recommendedAction", "evidence", "resourceNames"],
    }


def _consultation_memory_schema() -> dict[str, object]:
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "fall",
                    "medication",
                    "meal_intake",
                    "symptom",
                    "pain",
                    "sleep",
                    "mobility",
                    "mood",
                    "help_needed",
                    "appointment",
                    "other_medical",
                ],
            },
            "summary": {"type": "string"},
            "exactQuote": {"type": "string"},
            "sentenceIndex": {"type": ["integer", "null"]},
            "severity": {"type": "string", "enum": ["info", "watch", "urgent"]},
            "status": {"type": "string", "enum": ["new", "ongoing", "resolved", "unclear"]},
        },
        "required": ["category", "summary", "exactQuote", "sentenceIndex", "severity", "status"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"items": {"type": "array", "items": item_schema}},
        "required": ["items"],
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


def _openai_translate_text_to_english(text: str) -> str:
    cleaned = _clean_transcript_text(text)
    if not cleaned:
        return ""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return cleaned

    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("OPENAI_TRANSLATION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
                "input": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's text into natural English for a healthcare call transcript. "
                            "Preserve meaning only. Do not add explanation, labels, quotes, or extra context. "
                            "If the text is already English, return it unchanged."
                        ),
                    },
                    {"role": "user", "content": cleaned},
                ],
            },
            timeout=20,
        )
        response.raise_for_status()
        translated = _clean_transcript_text(_extract_openai_text(response.json()))
        return translated or cleaned
    except Exception:
        return cleaned


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


def _patient_review_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    return [segment for segment in segments if _is_patient_segment(segment)]


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


def _manual_risk_review(failure_reason: str | None = None) -> tuple[Symptoms, RiskAssessment, list[RiskSignal], str, bool, str | None]:
    reasons = ["Manual review required because AI risk extraction is unavailable."]
    return Symptoms(), _empty_assessment("Watch", reasons), [], "Manual review required.", True, failure_reason


def _parkinsons_speech_review(audio_path: Path | None) -> ParkinsonsSpeechReview:
    if audio_path is None:
        return ParkinsonsSpeechReview(
            qualityOk=False,
            failureReason="Patient speech audio was not available for Parkinson voice-feature review.",
            warnings=["Parkinson voice-feature model unavailable: patient speech audio was not available."],
        )
    try:
        from app.parkinsons_speech_model.inference import predict_speech_marker

        result = predict_speech_marker(audio_path, PARKINSONS_SPEECH_MODEL_ARTIFACT_ROOT)
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        return ParkinsonsSpeechReview(
            qualityOk=False,
            failureReason=f"Parkinson voice-feature model unavailable: {detail}",
            warnings=[f"Parkinson voice-feature model unavailable: {detail}"],
        )

    failure_reason = None
    if result.probability is None:
        failure_reason = next((warning for warning in result.warnings if "unavailable" in warning.lower()), None)
    risk_reason = None
    if result.probability is not None:
        risk_reason = f"Saved Parkinson voice-feature model returned {round(result.probability * 100)}% marker probability."
    return ParkinsonsSpeechReview(
        modelVersion=result.model_version,
        probability=result.probability,
        warnings=result.warnings,
        featuresSummary=result.features_summary,
        qualityOk=result.probability is not None,
        riskReason=risk_reason,
        failureReason=failure_reason,
    )


def _feature_number(features: dict[str, float | int | str | None] | None, key: str) -> float | None:
    if not features:
        return None
    value = features.get(key)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def _format_feature_value(key: str, value: float) -> str:
    if key.endswith("(Hz)") or key in {"HNR"}:
        suffix = " Hz" if key.endswith("(Hz)") else " dB"
        return f"{round(value, 1)}{suffix}"
    if "Jitter(Abs)" in key:
        return f"{value:.6f}"
    if "Jitter" in key or key in {"MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "NHR"}:
        return f"{value:.4f}"
    return f"{round(value, 3)}"


def _reference_range_score(value: float, limits: dict[str, float]) -> tuple[float, bool, str]:
    min_value = limits["min"]
    max_value = limits["max"]
    width = max(max_value - min_value, 1e-9)
    if value < min_value:
        return (min_value - value) / width + 1, True, "below"
    if value > max_value:
        return (value - max_value) / width + 1, True, "above"
    midpoint = (min_value + max_value) / 2
    return abs(value - midpoint) / max(width / 2, 1e-9), False, "within"


def _load_parkinsons_reference_ranges() -> dict[str, dict[str, float]]:
    ranges_path = PARKINSONS_SPEECH_MODEL_ARTIFACT_ROOT / "feature_reference_ranges.json"
    if not ranges_path.exists():
        return {}
    try:
        loaded = json.loads(ranges_path.read_text())
    except Exception:
        return {}
    ranges: dict[str, dict[str, float]] = {}
    if not isinstance(loaded, dict):
        return ranges
    for key, value in loaded.items():
        if not isinstance(value, dict):
            continue
        min_value = value.get("min")
        max_value = value.get("max")
        if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
            ranges[str(key)] = {"min": float(min_value), "max": float(max_value)}
    return ranges


def _parkinsons_explanations(review: ParkinsonsSpeechReview) -> list[ModelExplanationItem]:
    features = review.featuresSummary or {}
    ranges = _load_parkinsons_reference_ranges()
    if not features or not ranges:
        return [
            ModelExplanationItem(
                label="Voice features",
                value="Unavailable",
                status="unavailable",
                explanation="Patient-speech features were not available, so the Parkinson marker cannot be explained for this call.",
            )
        ]

    groups = [
        (
            "Pitch range",
            ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
            "Fundamental-frequency range in patient speech.",
        ),
        (
            "Jitter stability",
            ["MDVP:Jitter(%)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"],
            "Cycle-to-cycle pitch stability in voiced speech.",
        ),
        (
            "Harmonic-noise clarity",
            ["NHR", "HNR"],
            "Voice clarity versus noise in voiced speech.",
        ),
    ]
    scored: list[tuple[float, ModelExplanationItem]] = []
    for label, keys, plain_language in groups:
        best: tuple[float, str, float, bool, str, dict[str, float]] | None = None
        for key in keys:
            value = _feature_number(features, key)
            limits = ranges.get(key)
            if value is None or limits is None:
                continue
            score, outside, direction = _reference_range_score(value, limits)
            candidate = (score, key, value, outside, direction, limits)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            continue
        score, key, value, outside, direction, limits = best
        range_text = f"{_format_feature_value(key, limits['min'])}-{_format_feature_value(key, limits['max'])}"
        if outside:
            explanation = f"{plain_language} {key} was {direction} the reference range ({range_text})."
        else:
            explanation = f"{plain_language} Reference range: {range_text}."
        scored.append(
            (
                score,
                ModelExplanationItem(
                    label=label,
                    value=_format_feature_value(key, value),
                    status="watch" if outside or score >= 0.75 else "normal",
                    explanation=explanation,
                ),
            )
        )
    if not scored:
        return [
            ModelExplanationItem(
                label="Voice features",
                value="Unavailable",
                status="unavailable",
                explanation="Patient-speech features were not available, so the Parkinson marker cannot be explained for this call.",
            )
        ]
    return [item for _, item in sorted(scored, key=lambda row: row[0], reverse=True)[:3]]


def _volunteer_task_for_call(
    call_id: str,
    senior: Senior,
    assessment: RiskAssessment,
    recommended_action: str,
    safeguard_level: SafeguardLevel,
    safeguard_category: str | None,
    risk_signals: list[RiskSignal],
    created_at: str,
) -> VolunteerTask | None:
    if assessment.riskLevel == "Green" and safeguard_level == "None":
        return None

    if assessment.riskLevel in {"Red", "Amber"} or safeguard_level in {"Emergency", "Urgent"}:
        priority = "Urgent"
    elif assessment.riskLevel == "Watch" or safeguard_level == "Support":
        priority = "Today"
    else:
        priority = "Routine"

    if safeguard_level != "None":
        reason = f"Safeguard {safeguard_level.lower()} concern"
        if safeguard_category:
            reason = f"{reason}: {safeguard_category.replace('_', ' ')}"
    elif risk_signals:
        reason = risk_signals[0].label
    elif assessment.reasons:
        reason = assessment.reasons[0]
    else:
        reason = f"{assessment.riskLevel} follow-up"

    task_id = f"task-{call_id}"
    return VolunteerTask(
        id=task_id,
        seniorId=senior.id,
        priority=priority,
        reason=reason[:120],
        recommendedAction=recommended_action,
        assignedTo="Community volunteer follow-up team",
        status="Open",
        createdAt=created_at,
    )


def _upsert_volunteer_task_for_call(task: VolunteerTask | None) -> None:
    if task is None:
        return
    for index, existing in enumerate(VOLUNTEER_TASKS):
        if existing.id == task.id:
            VOLUNTEER_TASKS[index] = task
            _persist_volunteer_task_status(task)
            return
    VOLUNTEER_TASKS.insert(0, task)
    _persist_volunteer_task_status(task)


def _volunteer_task_from_record(record: CallRecord) -> VolunteerTask | None:
    if record.riskLevel == "Green" and record.safeguardLevel == "None":
        return None

    if record.riskLevel in {"Red", "Amber"} or record.safeguardLevel in {"Emergency", "Urgent"}:
        priority = "Urgent"
    elif record.riskLevel == "Watch" or record.safeguardLevel == "Support":
        priority = "Today"
    else:
        priority = "Routine"

    if record.safeguardLevel != "None":
        reason = f"Safeguard {record.safeguardLevel.lower()} concern"
        if record.safeguardCategory:
            reason = f"{reason}: {record.safeguardCategory.replace('_', ' ')}"
    elif record.riskSignals:
        reason = record.riskSignals[0].label
    elif record.riskAssessment.reasons:
        reason = record.riskAssessment.reasons[0]
    else:
        reason = f"{record.riskLevel} follow-up"

    return VolunteerTask(
        id=f"task-{record.id}",
        seniorId=record.seniorId,
        priority=priority,
        reason=reason[:120],
        recommendedAction=record.recommendedAction,
        assignedTo="Community volunteer follow-up team",
        status="Open",
        createdAt=record.completedAt,
    )


def _generated_volunteer_tasks() -> list[VolunteerTask]:
    tasks_by_id: dict[str, VolunteerTask] = {}
    for record in _load_all_call_records():
        task = _volunteer_task_from_record(record)
        if task is not None:
            tasks_by_id[task.id] = task
    persisted_statuses = _load_volunteer_task_statuses()
    for task_id, status in persisted_statuses.items():
        if task_id in tasks_by_id:
            tasks_by_id[task_id] = tasks_by_id[task_id].model_copy(update={"status": status})
    for task in VOLUNTEER_TASKS:
        existing = tasks_by_id.get(task.id)
        tasks_by_id[task.id] = task if existing is None else existing.model_copy(update={"status": task.status})
    return sorted(tasks_by_id.values(), key=lambda task: task.createdAt, reverse=True)


FALL_OR_NEAR_FALL_RE = re.compile(
    r"\b("
    r"fall|falls|fallen|falling|fell|"
    r"almost\s+fell|nearly\s+fell|nearly\s+fall|almost\s+fall|"
    r"slipped|tripped|lost\s+my\s+balance|"
    r"hit\s+my\s+head|knocked\s+my\s+head|bumped\s+my\s+head"
    r")\b",
    re.IGNORECASE,
)
FALL_NEGATION_RE = re.compile(
    r"\b("
    r"no|not|never|without|"
    r"did\s+not|didn't|"
    r"have\s+not|haven't|has\s+not|hasn't"
    r")\b[^.!?\n]{0,36}\b("
    r"fall|falls|fallen|falling|fell|"
    r"slipped|tripped|"
    r"hit\s+my\s+head|knocked\s+my\s+head|bumped\s+my\s+head"
    r")\b",
    re.IGNORECASE,
)


def _has_affirmed_fall_or_near_fall(text: str) -> bool:
    for sentence in _split_sentences(text) or [text]:
        if FALL_OR_NEAR_FALL_RE.search(sentence) and not FALL_NEGATION_RE.search(sentence):
            return True
    return False


def has_fall_or_near_fall_evidence(
    symptoms: Symptoms,
    segments: list[TranscriptSegment],
    signals: list[RiskSignal] | None = None,
) -> bool:
    if symptoms.fall or symptoms.headImpact:
        return True
    for segment in _patient_review_segments(segments):
        patient_text = _strip_speaker_labels(segment.englishText or segment.originalText or segment.text)
        if _has_affirmed_fall_or_near_fall(patient_text):
            return True
    for signal in signals or []:
        signal_text = " ".join(filter(None, [signal.label, signal.quotedText, signal.highlightText, signal.reason]))
        if _has_affirmed_fall_or_near_fall(signal_text):
            return True
    return False


NEGATIVE_EMOTION_LABELS = {
    "angry",
    "anger",
    "anxious",
    "anxiety",
    "distressed",
    "distress",
    "fear",
    "fearful",
    "frustrated",
    "sad",
    "sadness",
    "upset",
    "worried",
    "flat",
}


def _emotion_concern_level(segments: list[EmotionSegment]) -> EmotionConcernLevel:
    concerning = [segment for segment in segments if segment.label.lower() in NEGATIVE_EMOTION_LABELS and segment.confidence >= 0.6]
    if not concerning:
        return "None"
    high_confidence = [segment for segment in concerning if segment.confidence >= 0.8]
    if len(concerning) >= 2 or high_confidence:
        return "Review"
    return "Watch"


def _dominant_emotion(segments: list[EmotionSegment]) -> str | None:
    if not segments:
        return None
    best = max(segments, key=lambda segment: segment.confidence)
    return best.label


ELEVENLABS_EMOTION_DATA_KEY = "user_emotional_state"


def _jsonish(value: object) -> object:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return value
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return value
    return value


def _find_nested_key(payload: object, key: str) -> object | None:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_nested_key(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_nested_key(item, key)
            if found is not None:
                return found
    return None


def _extract_elevenlabs_data_value(payload: dict[str, object], key: str = ELEVENLABS_EMOTION_DATA_KEY) -> object | None:
    direct = _find_nested_key(payload, key)
    if direct is not None:
        if isinstance(direct, dict):
            for value_key in ["value", "result", "data", "answer"]:
                if value_key in direct:
                    return _jsonish(direct[value_key])
        return _jsonish(direct)

    for container_key in ["data_collection_results", "dataCollectionResults", "collected_data", "collectedData"]:
        container = _find_nested_key(payload, container_key)
        if isinstance(container, dict) and key in container:
            result = container[key]
            if isinstance(result, dict):
                for value_key in ["value", "result", "data", "answer"]:
                    if value_key in result:
                        return _jsonish(result[value_key])
            return _jsonish(result)
        if isinstance(container, list):
            for item in container:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("key") or item.get("id")
                if name == key:
                    for value_key in ["value", "result", "data", "answer"]:
                        if value_key in item:
                            return _jsonish(item[value_key])
                    return item
    return None


def _patient_segment_index_from_response_index(patient_segments: list[tuple[int, TranscriptSegment]], response_index: int | None) -> tuple[int | None, TranscriptSegment | None]:
    if response_index is None or response_index < 0 or response_index >= len(patient_segments):
        return None, None
    return patient_segments[response_index]


def _emotion_response_segment(
    item: dict[str, object],
    index: int,
    patient_segments: list[tuple[int, TranscriptSegment]],
    use_order_fallback: bool = False,
) -> EmotionSegment | None:
    label = item.get("emotion") or item.get("label") or item.get("emotionalState") or item.get("emotional_state")
    if not isinstance(label, str) or not label.strip():
        return None
    response_index_value = item.get("responseIndex") if "responseIndex" in item else item.get("response_index")
    response_index = int(response_index_value) if isinstance(response_index_value, (int, float)) else None
    if response_index is None and use_order_fallback:
        response_index = index
    transcript_segment_index, transcript_segment = _patient_segment_index_from_response_index(patient_segments, response_index)
    confidence_value = item.get("confidence") or item.get("score") or item.get("probability")
    confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else 0.65
    evidence = item.get("evidence") or item.get("text") or item.get("utterance")
    if not isinstance(evidence, str) or not evidence.strip():
        evidence = transcript_segment.englishText or transcript_segment.text if transcript_segment else f"Patient response {index + 1}"
    return EmotionSegment(
        id=f"emotion-{index + 1}",
        label=_clean_transcript_text(label).lower(),
        confidence=max(0.0, min(1.0, confidence)),
        startTimeSeconds=transcript_segment.startTimeSeconds if transcript_segment else None,
        endTimeSeconds=transcript_segment.endTimeSeconds if transcript_segment else None,
        transcriptSegmentIndex=transcript_segment_index,
        evidenceText=_clean_transcript_text(evidence),
    )


def _parse_elevenlabs_emotion_result(value: object, segments: list[TranscriptSegment]) -> EmotionProviderResult:
    value = _jsonish(value)
    patient_segments = [(index, segment) for index, segment in enumerate(segments) if _is_patient_segment(segment)]
    if isinstance(value, dict):
        raw_responses = value.get("responses")
        emotion_segments: list[EmotionSegment] = []
        if isinstance(raw_responses, list):
            has_indexes = any(isinstance(item, dict) and ("responseIndex" in item or "response_index" in item) for item in raw_responses)
            use_order_fallback = not has_indexes and len(raw_responses) == len(patient_segments)
            for index, item in enumerate(raw_responses):
                if isinstance(item, dict):
                    segment = _emotion_response_segment(item, index, patient_segments, use_order_fallback=use_order_fallback)
                    if segment is not None:
                        emotion_segments.append(segment)

        dominant = value.get("dominantEmotion") or value.get("dominant_emotion") or value.get("emotion") or value.get("label")
        dominant_emotion = _clean_transcript_text(dominant).lower() if isinstance(dominant, str) and dominant.strip() else _dominant_emotion(emotion_segments)
        failure_reason = None
        if dominant_emotion and not emotion_segments:
            failure_reason = "Tone summary available, but ElevenLabs did not return per-response emotion JSON for transcript tags."
        return EmotionProviderResult(
            provider="elevenlabs-data-collection",
            dominantEmotion=dominant_emotion,
            concernLevel=_emotion_concern_level(emotion_segments),
            segments=emotion_segments,
            attempts=[TranscriptionAttempt(provider="elevenlabs-data-collection", status="success")],
            failureReason=failure_reason,
        )

    if isinstance(value, str) and value.strip():
        return EmotionProviderResult(
            provider="elevenlabs-data-collection",
            dominantEmotion=_clean_transcript_text(value).lower(),
            concernLevel="None",
            segments=[],
            attempts=[TranscriptionAttempt(provider="elevenlabs-data-collection", status="success")],
            failureReason="Tone summary available, but ElevenLabs returned summary text instead of per-response emotion JSON for transcript tags.",
        )
    raise RuntimeError(f"ElevenLabs data collection key {ELEVENLABS_EMOTION_DATA_KEY} was empty or invalid")


def _elevenlabs_emotion_review(conversation_id: str | None, segments: list[TranscriptSegment]) -> EmotionProviderResult:
    if not conversation_id:
        return EmotionProviderResult(
            provider=None,
            attempts=[TranscriptionAttempt(provider="elevenlabs-data-collection", status="skipped", reason="ElevenLabs conversation ID was not captured")],
            failureReason="ElevenLabs conversation ID was not captured.",
        )
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return EmotionProviderResult(
            provider=None,
            attempts=[TranscriptionAttempt(provider="elevenlabs-data-collection", status="skipped", reason="ElevenLabs API key not configured")],
            failureReason="ElevenLabs API key not configured.",
        )

    last_error: Exception | None = None
    attempts: list[TranscriptionAttempt] = []
    for attempt_index in range(6):
        try:
            response = httpx.get(
                f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}",
                headers={"xi-api-key": api_key},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("ElevenLabs conversation response was not a JSON object")
            value = _extract_elevenlabs_data_value(payload)
            if value is not None:
                result = _parse_elevenlabs_emotion_result(value, segments)
                result.attempts = [*attempts, *result.attempts]
                return result
            attempts.append(TranscriptionAttempt(provider="elevenlabs-data-collection", status="skipped", reason="user_emotional_state not ready"))
        except Exception as exc:
            attempts.append(TranscriptionAttempt(provider="elevenlabs-data-collection", status="failed", reason=_safe_error_reason(exc)))
            last_error = exc
            break
        if attempt_index < 5:
            time.sleep(2)
    failure_reason = _safe_error_reason(last_error) if last_error else "ElevenLabs data collection result user_emotional_state was not available after polling."
    return EmotionProviderResult(provider=None, attempts=attempts, failureReason=failure_reason)


def _apply_emotion_modifier(assessment: RiskAssessment, emotion_result: EmotionProviderResult) -> RiskAssessment:
    if emotion_result.concernLevel != "Review" or not emotion_result.segments:
        return assessment
    reason = f"Patient vocal tone review suggested {emotion_result.dominantEmotion or 'distress'}; use as context for human follow-up, not diagnosis."
    if assessment.riskLevel == "Green":
        return assessment.model_copy(update={"riskLevel": "Watch", "reasons": [*assessment.reasons, reason]})
    if reason not in assessment.reasons:
        return assessment.model_copy(update={"reasons": [*assessment.reasons, reason]})
    return assessment


def _manual_safeguard_review(failure_reason: str | None = None) -> tuple[bool, SafeguardLevel, str | None, list[str], str | None, list[CrisisResource], str | None]:
    return False, "None", None, [], None, [], failure_reason


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", clean_transcript_text(text).lower()).strip()


def _validated_safeguard_evidence(evidence: list[str], segments: list[TranscriptSegment]) -> list[str]:
    patient_texts = [_normalize_match_text(segment.englishText or segment.text) for segment in segments]
    validated: list[str] = []
    for item in evidence:
        cleaned = clean_transcript_text(str(item))
        normalized = _normalize_match_text(cleaned)
        if not normalized:
            continue
        if any(normalized in patient_text or patient_text in normalized for patient_text in patient_texts):
            validated.append(cleaned)
    return validated


def _selected_safeguard_resources(level: SafeguardLevel, resource_names: list[str]) -> list[CrisisResource]:
    if level == "None":
        return []
    by_name = {resource.name.lower(): resource for resource in SINGAPORE_CRISIS_RESOURCES}
    selected: list[CrisisResource] = []
    for name in resource_names:
        resource = by_name.get(str(name).lower())
        if resource and resource.name not in {item.name for item in selected}:
            selected.append(resource)

    if selected:
        return selected
    if level == "Emergency":
        return SINGAPORE_CRISIS_RESOURCES[:3]
    return SINGAPORE_CRISIS_RESOURCES[2:]


CONSULTATION_MEMORY_CATEGORIES = {
    "fall",
    "medication",
    "meal_intake",
    "symptom",
    "pain",
    "sleep",
    "mobility",
    "mood",
    "help_needed",
    "appointment",
    "other_medical",
}


def _quote_matches_segment(quote: str, segment: TranscriptSegment) -> bool:
    normalized_quote = _normalize_match_text(_strip_speaker_labels(quote))
    normalized_segment = _normalize_match_text(_strip_speaker_labels(segment.englishText or segment.text))
    return bool(normalized_quote) and (normalized_quote in normalized_segment or normalized_segment in normalized_quote)


def _consultation_memory_item(
    senior_id: str,
    call_id: str,
    recorded_at: str,
    category: str,
    summary: str,
    exact_quote: str,
    segment: TranscriptSegment,
    index: int,
    severity: str = "info",
    status: str = "new",
) -> ConsultationMemoryItem | None:
    if not _quote_matches_segment(exact_quote, segment):
        return None
    safe_category = category if category in CONSULTATION_MEMORY_CATEGORIES else "other_medical"
    safe_severity = severity if severity in {"info", "watch", "urgent"} else "info"
    safe_status = status if status in {"new", "ongoing", "resolved", "unclear"} else "new"
    quote = _strip_speaker_labels(exact_quote)
    cleaned_summary = clean_transcript_text(summary) or quote
    return ConsultationMemoryItem(
        id=f"{call_id}-memory-{index}",
        seniorId=senior_id,
        callId=call_id,
        recordedAt=recorded_at,
        category=safe_category,  # type: ignore[arg-type]
        summary=cleaned_summary[:220],
        exactQuote=quote,
        startTimeSeconds=segment.startTimeSeconds,
        endTimeSeconds=segment.endTimeSeconds,
        severity=safe_severity,  # type: ignore[arg-type]
        status=safe_status,  # type: ignore[arg-type]
    )


def _fallback_consultation_memory(senior_id: str, call_id: str, recorded_at: str, segments: list[TranscriptSegment]) -> list[ConsultationMemoryItem]:
    rules: list[tuple[str, str, str]] = [
        ("fall", r"\b(fell|fall|fallen|slipped|tripped|hit my head|knocked my head)\b", "urgent"),
        ("medication", r"\b(medicine|medication|pills?|dose|tablet|insulin|missed.*med|forgot.*med|took my med)\b", "watch"),
        ("meal_intake", r"\b(ate|eat|meal|breakfast|lunch|dinner|appetite|hungry|water|drink|drank|dehydrat|skip.*meal)\b", "info"),
        ("pain", r"\b(pain|hurt|ache|sore|headache|chest pain|stomach pain)\b", "watch"),
        ("symptom", r"\b(dizzy|vomit|confus|weak|fever|breathless|nausea|blurred|numb|slurred)\b", "watch"),
        ("sleep", r"\b(sleep|slept|insomnia|awake|tired|fatigue)\b", "info"),
        ("mobility", r"\b(walk|walking|stand|standing|stairs|bath|toilet|cook|shower|move around)\b", "watch"),
        ("mood", r"\b(lonely|sad|afraid|scared|worried|anxious|distress|depressed)\b", "watch"),
        ("help_needed", r"\b(help|call my|caregiver|daughter|son|neighbour|neighbor|volunteer)\b", "info"),
        ("appointment", r"\b(appointment|clinic|doctor|hospital|visit|consultation)\b", "info"),
    ]
    summaries = {
        "fall": "Patient mentioned a fall or injury concern.",
        "medication": "Patient mentioned medication timing or adherence.",
        "meal_intake": "Patient mentioned eating, appetite, or fluid intake.",
        "pain": "Patient mentioned pain or discomfort.",
        "symptom": "Patient mentioned a symptom needing review.",
        "sleep": "Patient mentioned sleep or fatigue.",
        "mobility": "Patient mentioned mobility or daily-function changes.",
        "mood": "Patient mentioned mood, fear, loneliness, or distress.",
        "help_needed": "Patient mentioned help-seeking or care support.",
        "appointment": "Patient mentioned an appointment or care visit.",
    }
    items: list[ConsultationMemoryItem] = []
    seen: set[tuple[str, str]] = set()
    for segment in segments:
        quote = _strip_speaker_labels(segment.englishText or segment.text)
        normalized = quote.lower()
        if not normalized:
            continue
        for category, pattern, severity in rules:
            if not re.search(pattern, normalized):
                continue
            key = (category, _normalize_match_text(quote))
            if key in seen:
                continue
            seen.add(key)
            if re.search(r"\b(no|not|never|don't|do not|didn't|did not)\b.{0,25}\b(fall|dizzy|pain|headache|vomit|weak|missed|forgot)\b", normalized):
                item_status = "resolved"
                item_severity = "info"
            elif re.search(r"\b(not sure|can't remember|cannot remember|forgot|maybe|i think)\b", normalized):
                item_status = "unclear"
                item_severity = "watch"
            else:
                item_status = "new"
                item_severity = severity
            item = _consultation_memory_item(
                senior_id,
                call_id,
                recorded_at,
                category,
                summaries[category],
                quote,
                segment,
                len(items),
                item_severity,
                item_status,
            )
            if item:
                items.append(item)
                break
    return items


def _openai_consultation_memory(senior_id: str, call_id: str, recorded_at: str, segments: list[TranscriptSegment]) -> list[ConsultationMemoryItem]:
    patient_segments = _patient_review_segments(segments)
    if not patient_segments:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_consultation_memory(senior_id, call_id, recorded_at, patient_segments)

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
                            "Extract a longitudinal consultation memory for an AIC/community care coordinator. "
                            "Use only patient statements. Ignore agent questions and summaries. Return compact facts useful for a future clinic consultation. "
                            "Every item must include an exact quote copied from the patient sentence. Do not infer facts without exact evidence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "patientSentences": [
                                    {
                                        "sentenceIndex": index,
                                        "englishText": _strip_speaker_labels(segment.englishText or segment.text),
                                        "startTimeSeconds": segment.startTimeSeconds,
                                        "endTimeSeconds": segment.endTimeSeconds,
                                    }
                                    for index, segment in enumerate(patient_segments)
                                ],
                                "categories": sorted(CONSULTATION_MEMORY_CATEGORIES),
                            }
                        ),
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "earlycare_consultation_memory",
                        "schema": _consultation_memory_schema(),
                        "strict": True,
                    }
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        result = json.loads(_extract_openai_text(response.json()))
        items: list[ConsultationMemoryItem] = []
        for raw_item in result.get("items", []):
            sentence_index = raw_item.get("sentenceIndex")
            if not isinstance(sentence_index, int) or not 0 <= sentence_index < len(patient_segments):
                continue
            item = _consultation_memory_item(
                senior_id,
                call_id,
                recorded_at,
                str(raw_item.get("category", "other_medical")),
                str(raw_item.get("summary", "")),
                str(raw_item.get("exactQuote", "")),
                patient_segments[sentence_index],
                len(items),
                str(raw_item.get("severity", "info")),
                str(raw_item.get("status", "new")),
            )
            if item:
                items.append(item)
        return items
    except Exception:
        return _fallback_consultation_memory(senior_id, call_id, recorded_at, patient_segments)


def _risk_level_with_safeguard(risk_level: str, safeguard_level: SafeguardLevel) -> str:
    order = {"Green": 0, "Watch": 1, "Amber": 2, "Red": 3}
    safeguard_risk = SAFEGUARD_LEVEL_RISK.get(safeguard_level, "Green")
    return risk_level if order.get(risk_level, 0) >= order.get(safeguard_risk, 0) else safeguard_risk


def _risk_level_at_least(risk_level: str, minimum: str) -> str:
    order = {"Green": 0, "Watch": 1, "Amber": 2, "Red": 3}
    return risk_level if order.get(risk_level, 0) >= order.get(minimum, 0) else minimum


def _has_concussion_symptom_evidence(symptoms: Symptoms) -> bool:
    return any(
        [
            symptoms.fall,
            symptoms.headImpact,
            symptoms.headache,
            symptoms.dizziness,
            symptoms.vomiting,
            symptoms.confusion,
            symptoms.slurredSpeech,
            symptoms.weakness,
        ]
    )


def _apply_concussion_speech_modifier(
    assessment: RiskAssessment,
    symptoms: Symptoms,
    review: ConcussionSpeechReview | None,
) -> RiskAssessment:
    if review is None or review.riskContribution == "Green" or not review.qualityOk:
        return assessment

    reasons = list(assessment.reasons)
    if review.riskReason and review.riskReason not in reasons:
        reasons.append(review.riskReason)

    if _has_concussion_symptom_evidence(symptoms):
        reasons.append("Patient-reported concussion-relevant symptoms plus abnormal speech model output require human review.")
        minimum_level = "Red" if any([symptoms.vomiting, symptoms.confusion, symptoms.slurredSpeech, symptoms.weakness]) else "Amber"
        return assessment.model_copy(update={"riskLevel": _risk_level_at_least(assessment.riskLevel, minimum_level), "reasons": reasons})

    reasons.append("Speech-abnormality model output is present without patient-reported concussion symptoms; review audio before acting.")
    return assessment.model_copy(update={"riskLevel": _risk_level_at_least(assessment.riskLevel, "Watch"), "reasons": reasons})


def _openai_safeguard_review(
    segments: list[TranscriptSegment],
) -> tuple[bool, SafeguardLevel, str | None, list[str], str | None, list[CrisisResource], str | None]:
    patient_segments = _patient_review_segments(segments)
    if not patient_segments:
        return True, "None", None, [], None, [], None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _manual_safeguard_review("OPENAI_API_KEY is not configured.")

    patient_transcript = "\n".join(segment.englishText or segment.text for segment in patient_segments)
    resources_payload = [resource.model_dump() for resource in SINGAPORE_CRISIS_RESOURCES]

    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("OPENAI_SAFEGUARD_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
                "input": [
                    {
                        "role": "system",
                        "content": (
                            "You are an EarlyCare safeguard classifier for elderly wellbeing calls in Singapore. "
                            "Classify only what the patient says. Ignore agent questions, examples, safety instructions, and summaries. "
                            "Do not diagnose, counsel, or write a script for the agent. Use None when the patient denies distress, "
                            "the evidence is ambiguous, or the only concerning wording appears in an agent turn. "
                            "Use Support for non-emergency emotional distress, loneliness, grief, or anxiety where supportive outreach is appropriate. "
                            "Use Urgent for serious distress, self-harm ideation without immediate danger, abuse or neglect concern, or inability to stay safe without prompt human follow-up. "
                            "Use Emergency for immediate danger, stated intent or plan to self-harm, ongoing attempt, violence, or urgent medical danger. "
                            "Evidence must be exact English patient text from the supplied patient sentences. Select resourceNames only from the supplied resource list."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "patientOnlyTranscript": patient_transcript,
                                "sentences": [
                                    {
                                        "sentenceIndex": index,
                                        "englishText": segment.englishText or segment.text,
                                        "startTimeSeconds": segment.startTimeSeconds,
                                        "endTimeSeconds": segment.endTimeSeconds,
                                    }
                                    for index, segment in enumerate(patient_segments)
                                ],
                                "availableResources": resources_payload,
                            }
                        ),
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "earlycare_safeguard_review",
                        "schema": _safeguard_schema(),
                        "strict": True,
                    }
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        result = json.loads(_extract_openai_text(response.json()))
        level: SafeguardLevel = result.get("level", "None")
        category = result.get("category")
        evidence = _validated_safeguard_evidence(result.get("evidence", []), patient_segments)
        if level != "None" and not evidence:
            return True, "None", None, [], None, [], None
        resources = _selected_safeguard_resources(level, result.get("resourceNames", []))
        recommended_action = result.get("recommendedAction") if level != "None" else None
        return True, level, category if level != "None" else None, evidence, recommended_action, resources, None
    except Exception as exc:
        return _manual_safeguard_review(_safe_error_reason(exc))


def _openai_risk_review(english_transcript: str, segments: list[TranscriptSegment]) -> tuple[Symptoms, RiskAssessment, list[RiskSignal], str, bool, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _manual_risk_review("OPENAI_API_KEY is not configured.")

    patient_segments = _patient_review_segments(segments)
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
        return symptoms, assessment, signals, recommended_action, False, None
    except Exception as exc:
        return _manual_risk_review(_safe_error_reason(exc))


def _load_call_record(path: Path) -> CallRecord:
    return CallRecord.model_validate_json(path.read_text())


@app.get("/auth/me", response_model=AuthStatus)
def auth_me(request: Request) -> AuthStatus:
    if not _auth_enabled():
        return AuthStatus(authEnabled=False, authenticated=True, username="demo-operator")
    username = _request_operator(request)
    return AuthStatus(authEnabled=True, authenticated=username is not None, username=username)


@app.post("/auth/login", response_model=AuthStatus)
def auth_login(payload: LoginRequest, request: Request, response: Response) -> AuthStatus:
    if not _auth_enabled():
        return AuthStatus(authEnabled=False, authenticated=True, username="demo-operator")

    expected_username = _operator_username()
    expected_password = _operator_password() or ""
    if not hmac.compare_digest(payload.username, expected_username) or not hmac.compare_digest(payload.password, expected_password):
        raise HTTPException(status_code=401, detail="Invalid operator credentials")

    token = _create_session_token(expected_username)
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        max_age=_session_ttl_seconds(),
        path="/",
    )
    return AuthStatus(authEnabled=True, authenticated=True, username=expected_username)


@app.post("/auth/logout", response_model=AuthStatus)
def auth_logout(response: Response) -> AuthStatus:
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return AuthStatus(authEnabled=_auth_enabled(), authenticated=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "product": "EarlyCare",
        "storageRoot": str(CALL_STORAGE_ROOT.parent),
        "database": str(_database_path()),
    }


@app.get("/readiness")
def readiness() -> dict[str, object]:
    return readiness_report(BACKEND_ROOT, CALL_STORAGE_ROOT)


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
    return _load_all_call_records()


@app.get("/calls/{call_id}", response_model=CallRecord)
def get_call(call_id: str) -> CallRecord:
    path = _call_metadata_path(call_id)
    if path.exists():
        return _load_call_record(path)
    record = _load_indexed_call_record(call_id)
    if record is not None:
        return record
    raise HTTPException(status_code=404, detail="Call not found")


@app.get("/seniors/{senior_id}/consultation-memory", response_model=list[ConsultationMemoryItem])
def get_consultation_memory(senior_id: str) -> list[ConsultationMemoryItem]:
    get_senior(senior_id)
    items: list[ConsultationMemoryItem] = []
    for record in _load_all_call_records():
        if record.seniorId == senior_id:
            items.extend(record.consultationMemory)
    return sorted(items, key=lambda item: item.recordedAt, reverse=True)


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
    elevenLabsConversationId: str | None = Form(None),
    agentAudioCaptured: bool = Form(False),
    consentCaptured: bool = Form(False),
    consentVersion: str = Form("earlycare-demo-v1"),
    recordingNoticeShownAt: str | None = Form(None),
    retentionPolicy: str = Form("local-demo-delete-after-hackathon"),
    operatorId: str = Form("demo-operator"),
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
    completed_at_value = completedAt or _now_iso()

    audio_file_path, audio_path = await _save_uploaded_audio(audio, call_dir, "full-call")
    patient_audio_file_path, _ = await _save_uploaded_audio(patientAudio, call_dir, "patient-audio")

    transcript_alignment_warnings: list[str] = []
    dialogue_transcript = _transcript_to_text(messages)
    has_live_roles = _has_live_spoken_messages(messages)
    translation = transcribe_with_fallback(senior.preferredLanguage, dialogue_transcript, audio_path)
    if translation.fallbackUsed:
        original_transcript = _clean_transcript_text(
            _role_labeled_original_from_live_messages(messages) if has_live_roles else (dialogue_transcript or translation.transcript)
        )
        english_transcript = _clean_transcript_text(
            _role_labeled_english_transcript(
                messages,
                senior.preferredLanguage,
                translation.translation or original_transcript,
                transcript_alignment_warnings,
            )
        )
    else:
        original_transcript = _clean_transcript_text(
            _role_labeled_original_transcript(messages, translation.transcript or dialogue_transcript, transcript_alignment_warnings)
        )
        provider_english = _clean_transcript_text(translation.translation or translation.transcript or original_transcript)
        if _has_role_labeled_lines(provider_english) and not has_live_roles:
            english_transcript = provider_english
        else:
            english_transcript = _clean_transcript_text(
                _role_labeled_english_transcript(
                    messages,
                    senior.preferredLanguage,
                    provider_english,
                    transcript_alignment_warnings,
                )
            )
    for segment in translation.segments:
        segment.text = _clean_transcript_text(segment.text)
        segment.originalText = _clean_transcript_text(segment.originalText or segment.text)
        segment.englishText = _clean_transcript_text(segment.englishText or segment.text)
    translation.segments = _sync_english_segments(translation.segments, original_transcript, english_transcript, transcript_alignment_warnings)
    role_source_transcript = (
        english_transcript
        if translation.fallbackUsed
        else _role_labeled_english_transcript(messages, senior.preferredLanguage, english_transcript, transcript_alignment_warnings)
    )
    role_segments = _role_segments_from_messages(messages, startedAt, role_source_transcript)
    if role_segments and has_live_roles:
        if translation.fallbackUsed:
            _add_warning(transcript_alignment_warnings, "Transcript segment roles were rebuilt from live transcript messages because provider fallback was used.")
        else:
            _add_warning(transcript_alignment_warnings, "ElevenLabs live roles were used for transcript segments; MERaLiON speaker labels ignored.")
        translation.segments = role_segments
    elif role_segments and (translation.fallbackUsed or not _has_explicit_agent_patient_roles(translation.segments)):
        if translation.fallbackUsed:
            _add_warning(transcript_alignment_warnings, "Transcript segment roles were rebuilt from live transcript messages because provider fallback was used.")
        else:
            _add_warning(transcript_alignment_warnings, "Provider speaker labels were missing or generic; live transcript roles were used for segment roles.")
        translation.segments = role_segments
    elif not any(segment.startTimeSeconds is not None for segment in translation.segments):
        timed_segments = _timed_segments_from_messages(messages, startedAt, english_transcript)
        if timed_segments:
            _add_warning(transcript_alignment_warnings, "Provider segment timing was unavailable; segment times were estimated from live transcript event times.")
            translation.segments = timed_segments
    symptoms, assessment, risk_signals, recommended_action, ai_fallback_used, ai_failure_reason = _openai_risk_review(english_transcript, translation.segments)
    (
        safeguard_review_available,
        safeguard_level,
        safeguard_category,
        safeguard_evidence,
        safeguard_recommended_action,
        safeguard_resources,
        safeguard_failure_reason,
    ) = _openai_safeguard_review(translation.segments)
    combined_risk_level = _risk_level_with_safeguard(assessment.riskLevel, safeguard_level)
    if combined_risk_level != assessment.riskLevel:
        safeguard_reason = f"Safeguard review flagged {safeguard_level.lower()} concern"
        if safeguard_category:
            safeguard_reason = f"{safeguard_reason}: {safeguard_category.replace('_', ' ')}"
        assessment = assessment.model_copy(update={"riskLevel": combined_risk_level, "reasons": [*assessment.reasons, safeguard_reason]})
    if safeguard_level != "None" and safeguard_recommended_action:
        recommended_action = f"{recommended_action} Safeguard: {safeguard_recommended_action}"
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
    emotion_result = _elevenlabs_emotion_review(elevenLabsConversationId, translation.segments)
    assessment = _apply_emotion_modifier(assessment, emotion_result)
    parkinsons_speech_review = _parkinsons_speech_review(patient_speech_path)
    if has_fall_or_near_fall_evidence(symptoms, translation.segments, risk_signals):
        concussion_speech_review = review_concussion_speech(patient_speech_path)
    else:
        concussion_speech_review = not_applicable_concussion_review()
    previous_risk_level = assessment.riskLevel
    assessment = _apply_concussion_speech_modifier(assessment, symptoms, concussion_speech_review)
    if concussion_speech_review and assessment.riskLevel != previous_risk_level:
        recommended_action = (
            f"{recommended_action} Speech abnormality model also flagged a research-only review signal; "
            "compare the patient audio with the transcript and escalate to a clinician or emergency service if symptoms are acute."
        )
    consultation_memory = _openai_consultation_memory(senior.id, call_id, completed_at_value, translation.segments)
    speech_model_warnings = [*speech_extraction_warnings, *parkinsons_speech_review.warnings]
    speech_model_features = parkinsons_speech_review.featuresSummary
    if speech_model_features is not None:
        speech_model_features = {**speech_model_features, **speech_extraction_summary}
    elif speech_extraction_summary:
        speech_model_features = speech_extraction_summary
    parkinsons_speech_review = parkinsons_speech_review.model_copy(
        update={
            "warnings": speech_model_warnings,
            "featuresSummary": speech_model_features,
            "explanations": _parkinsons_explanations(
                parkinsons_speech_review.model_copy(update={"featuresSummary": speech_model_features})
            ),
        }
    )

    display_english_transcript = _public_english_transcript(original_transcript, english_transcript)
    (call_dir / "transcript-original.json").write_text(json.dumps([message.model_dump() for message in messages], indent=2))
    (call_dir / "transcript-english.txt").write_text(display_english_transcript)

    call = CallRecord(
        id=call_id,
        seniorId=senior.id,
        seniorName=senior.name,
        startedAt=startedAt,
        completedAt=completed_at_value,
        status="Complete" if status not in {"Failed", "Saved"} else status,  # type: ignore[arg-type]
        riskLevel=assessment.riskLevel,
        originalTranscript=original_transcript,
        englishTranscript=display_english_transcript,
        transcriptMessages=messages,
        elevenLabsConversationId=elevenLabsConversationId,
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
        transcriptAlignmentWarnings=transcript_alignment_warnings,
        riskSignals=risk_signals,
        aiRiskFallbackUsed=ai_fallback_used,
        aiRiskFailureReason=ai_failure_reason,
        emotionReviewAvailable=emotion_result.provider is not None,
        emotionProvider=emotion_result.provider,
        emotionFallbackUsed=emotion_result.fallbackUsed,
        emotionFailureReason=emotion_result.failureReason,
        dominantPatientEmotion=emotion_result.dominantEmotion,
        emotionConcernLevel=emotion_result.concernLevel,
        emotionSegments=emotion_result.segments,
        emotionAttempts=emotion_result.attempts,
        safeguardReviewAvailable=safeguard_review_available,
        safeguardLevel=safeguard_level,
        safeguardCategory=safeguard_category,
        safeguardEvidence=safeguard_evidence,
        safeguardRecommendedAction=safeguard_recommended_action,
        safeguardResources=safeguard_resources,
        safeguardFailureReason=safeguard_failure_reason,
        consultationMemory=consultation_memory,
        parkinsonsSpeechReview=parkinsons_speech_review,
        speechModelVersion=parkinsons_speech_review.modelVersion,
        speechModelProbability=parkinsons_speech_review.probability,
        speechModelWarnings=speech_model_warnings,
        speechModelFeaturesSummary=speech_model_features,
        concussionSpeechReview=concussion_speech_review,
        consentCaptured=consentCaptured,
        consentVersion=consentVersion or "earlycare-demo-v1",
        recordingNoticeShownAt=recordingNoticeShownAt,
        retentionPolicy=retentionPolicy or "local-demo-delete-after-hackathon",
        operatorId=operatorId or "demo-operator",
        riskAssessment=assessment,
        recommendedAction=recommended_action,
    )
    _upsert_volunteer_task_for_call(
        _volunteer_task_for_call(
            call_id,
            senior,
            assessment,
            recommended_action,
            safeguard_level,
            safeguard_category,
            risk_signals,
            completed_at_value,
        )
    )
    _persist_call_record(call)
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
    return _generated_volunteer_tasks()


@app.patch("/volunteer-tasks/{task_id}", response_model=VolunteerTask)
def update_volunteer_task(task_id: str, status: str | None = None, payload: VolunteerTaskUpdate | None = Body(None)) -> VolunteerTask:
    next_status = payload.status if payload is not None else status
    if next_status not in {"Open", "In progress", "Closed"}:
        raise HTTPException(status_code=422, detail="status must be one of: Open, In progress, Closed")
    for task in VOLUNTEER_TASKS:
        if task.id == task_id:
            task.status = next_status  # type: ignore[assignment]
            _persist_volunteer_task_status(task)
            return task
    for task in _generated_volunteer_tasks():
        if task.id == task_id:
            updated = task.model_copy(update={"status": next_status})
            _upsert_volunteer_task_for_call(updated)
            return updated
    raise HTTPException(status_code=404, detail="Volunteer task not found")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str) -> FileResponse:
    if not FRONTEND_DIST_ROOT.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found")

    requested_path = (FRONTEND_DIST_ROOT / full_path).resolve()
    try:
        requested_path.relative_to(FRONTEND_DIST_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Frontend asset not found") from exc

    if requested_path.is_file():
        return FileResponse(requested_path)

    index_path = FRONTEND_DIST_ROOT / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend entrypoint not found")
