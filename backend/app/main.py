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

from app.data import CHECKINS, SCENARIOS, SENIORS, VOLUNTEER_TASKS
from app.ml import assess_speech_deviation, extract_demo_embedding_note
from app.models import (
    CallRecord,
    CheckInSession,
    ProviderResult,
    RiskAssessment,
    RiskSignal,
    SavedCallResponse,
    Scenario,
    ScenarioRunResponse,
    Senior,
    SpeechDeviationRequest,
    Symptoms,
    TranscriptSegment,
    TranscriptMessage,
    VolunteerTask,
)
from app.providers import transcribe_with_fallback
from app.risk import (
    assessment_from_symptoms,
    build_conversation_categories,
    build_escalation_plan,
    detect_symptoms_from_text,
    recommended_action_for,
    risk_signals_from_categories,
)


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BACKEND_ROOT = Path(__file__).resolve().parents[1]
CALL_STORAGE_ROOT = BACKEND_ROOT / "storage" / "calls"
STATE_STORAGE_ROOT = BACKEND_ROOT / "storage" / "state"
CHECKINS_STATE_PATH = STATE_STORAGE_ROOT / "checkins.json"
TASKS_STATE_PATH = STATE_STORAGE_ROOT / "volunteer-tasks.json"
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_json_default(path: Path, fallback: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fallback)


def _load_checkins() -> list[CheckInSession]:
    _state_json_default(CHECKINS_STATE_PATH, json.dumps([checkin.model_dump() for checkin in CHECKINS], indent=2))
    return [CheckInSession.model_validate(item) for item in json.loads(CHECKINS_STATE_PATH.read_text())]


def _save_checkins(checkins: list[CheckInSession]) -> None:
    STATE_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKINS_STATE_PATH.write_text(json.dumps([checkin.model_dump() for checkin in checkins], indent=2))


def _load_tasks() -> list[VolunteerTask]:
    _state_json_default(TASKS_STATE_PATH, json.dumps([task.model_dump() for task in VOLUNTEER_TASKS], indent=2))
    return [VolunteerTask.model_validate(item) for item in json.loads(TASKS_STATE_PATH.read_text())]


def _save_tasks(tasks: list[VolunteerTask]) -> None:
    STATE_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    TASKS_STATE_PATH.write_text(json.dumps([task.model_dump() for task in tasks], indent=2))


def _call_metadata_path(call_id: str) -> Path:
    return CALL_STORAGE_ROOT / call_id / "metadata.json"


def _transcript_to_text(messages: list[TranscriptMessage]) -> str:
    return "\n".join(f"{message.role}: {message.text}" for message in messages)


def _clean_transcript_text(text: str) -> str:
    text = re.sub(r"\s*\[(happy|relieved|good|sad|angry|calm|cheerful|concerned|empathetic|laughs?|sighs?|pause|thinking)\]\s*", " ", text, flags=re.I)
    return re.sub(r"[ \t]+", " ", text).strip()


def _clean_messages(messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
    return [message.model_copy(update={"text": _clean_transcript_text(message.text)}) for message in messages]


def _empty_assessment(risk_level: str, reasons: list[str]) -> RiskAssessment:
    return RiskAssessment(
        speechDeviationScore=0,
        parkinsonsWatchScore=0,
        postFallConcernScore=0,
        missedCheckInScore=0,
        riskLevel=risk_level,  # type: ignore[arg-type]
        reasons=reasons or ["No notable deviation from available baseline context."],
    )


def _enrich_session(session: CheckInSession) -> CheckInSession:
    if session.categories and session.escalationPlan:
        return session
    senior = get_senior(session.seniorId)
    symptoms = detect_symptoms_from_text(session.englishTranscript or session.originalTranscript)
    categories = build_conversation_categories(session.englishTranscript or session.originalTranscript, symptoms, session.riskAssessment, senior)
    escalation_plan = build_escalation_plan(session.riskAssessment, categories, senior)
    recommended_action = session.recommendedAction or recommended_action_for(session.riskAssessment, categories, senior)
    return session.model_copy(update={"categories": categories, "escalationPlan": escalation_plan, "recommendedAction": recommended_action})


def _status_for_risk(assessment: RiskAssessment, symptoms: Symptoms) -> str:
    if symptoms.missedCheckIn:
        return "Missed"
    if assessment.riskLevel == "Red":
        return "Urgent"
    if assessment.riskLevel in {"Watch", "Amber"}:
        return "Needs follow-up"
    return "Checked in"


def _task_priority(risk_level: str) -> str:
    if risk_level == "Red":
        return "Urgent"
    if risk_level in {"Amber", "Watch"}:
        return "Today"
    return "Routine"


def _task_reason(session: CheckInSession) -> str:
    elevated = [category.label for category in session.categories if category.severity != "Green"]
    if elevated:
        return ", ".join(elevated[:2])
    return session.summary


def _upsert_task_for_session(session: CheckInSession, senior: Senior) -> list[VolunteerTask]:
    if session.riskLevel == "Green":
        return _load_tasks()

    tasks = _load_tasks()
    task = VolunteerTask(
        id=f"task-{session.id}",
        seniorId=senior.id,
        priority=_task_priority(session.riskLevel),  # type: ignore[arg-type]
        reason=_task_reason(session),
        recommendedAction=session.recommendedAction,
        assignedTo="Community response team",
        status="Open",
        createdAt=session.completedAt or _now_iso(),
        sourceSessionId=session.id,
        escalationStep="volunteer-social-task" if session.riskLevel != "Red" else "emergency-alert",
    )
    tasks = [existing for existing in tasks if existing.id != task.id and existing.sourceSessionId != session.id]
    tasks.append(task)
    _save_tasks(tasks)
    return tasks


def _upsert_task_for_call(call: CallRecord, senior: Senior) -> list[VolunteerTask]:
    if call.riskLevel == "Green":
        return _load_tasks()
    tasks = _load_tasks()
    task = VolunteerTask(
        id=f"task-{call.id}",
        seniorId=senior.id,
        priority=_task_priority(call.riskLevel),  # type: ignore[arg-type]
        reason=", ".join(category.label for category in call.categories if category.severity != "Green") or call.recommendedAction,
        recommendedAction=call.recommendedAction,
        assignedTo="Community response team",
        status="Open",
        createdAt=call.completedAt,
        sourceCallId=call.id,
        escalationStep="volunteer-social-task" if call.riskLevel != "Red" else "emergency-alert",
    )
    tasks = [existing for existing in tasks if existing.id != task.id and existing.sourceCallId != call.id]
    tasks.append(task)
    _save_tasks(tasks)
    return tasks


def _risk_schema() -> dict[str, object]:
    signal_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "severity": {"type": "string", "enum": ["Green", "Watch", "Amber", "Red"]},
            "quotedText": {"type": "string"},
            "reason": {"type": "string"},
            "startTimeSeconds": {"type": ["number", "null"]},
            "endTimeSeconds": {"type": ["number", "null"]},
        },
        "required": ["id", "label", "severity", "quotedText", "reason", "startTimeSeconds", "endTimeSeconds"],
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
                    "whiplashOrJolt": {"type": "boolean"},
                    "headache": {"type": "boolean"},
                    "worseningHeadache": {"type": "boolean"},
                    "dizziness": {"type": "boolean"},
                    "vomiting": {"type": "boolean"},
                    "confusion": {"type": "boolean"},
                    "slurredSpeech": {"type": "boolean"},
                    "weakness": {"type": "boolean"},
                    "numbness": {"type": "boolean"},
                    "unusualBehavior": {"type": "boolean"},
                    "drowsinessOrUnwakeable": {"type": "boolean"},
                    "poorIntake": {"type": "boolean"},
                    "asksForHelp": {"type": "boolean"},
                    "missedCheckIn": {"type": "boolean"},
                    "loneliness": {"type": "boolean"},
                    "lowMood": {"type": "boolean"},
                    "medicationMissed": {"type": "boolean"},
                    "chronicConcern": {"type": "boolean"},
                    "ckdConcern": {"type": "boolean"},
                    "diabetesConcern": {"type": "boolean"},
                    "highBloodPressureConcern": {"type": "boolean"},
                },
                "required": [
                    "fall",
                    "headImpact",
                    "whiplashOrJolt",
                    "headache",
                    "worseningHeadache",
                    "dizziness",
                    "vomiting",
                    "confusion",
                    "slurredSpeech",
                    "weakness",
                    "numbness",
                    "unusualBehavior",
                    "drowsinessOrUnwakeable",
                    "poorIntake",
                    "asksForHelp",
                    "missedCheckIn",
                    "loneliness",
                    "lowMood",
                    "medicationMissed",
                    "chronicConcern",
                    "ckdConcern",
                    "diabetesConcern",
                    "highBloodPressureConcern",
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
    normalized_segments = [(segment.text.lower(), segment) for segment in segments if segment.text and segment.startTimeSeconds is not None]
    next_signals: list[RiskSignal] = []
    for signal in signals:
        if signal.startTimeSeconds is not None:
            next_signals.append(signal)
            continue
        quoted = signal.quotedText.lower().strip()
        match = next((segment for text, segment in normalized_segments if quoted and (quoted in text or text in quoted)), None)
        next_signals.append(
            signal.model_copy(
                update={
                    "startTimeSeconds": match.startTimeSeconds if match else None,
                    "endTimeSeconds": match.endTimeSeconds if match else None,
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
                            "wellbeing check-in transcript. Do not diagnose. Only mark Amber or Red when the transcript itself supports "
                            "earlier volunteer or caregiver action. Track living-alone missed check-ins, fall/head impact, whiplash or body jolts, "
                            "possible concussion danger signs, possible Parkinson's speech-watch signals, chronic illness concerns, poor intake, "
                            "medication issues, loneliness, and help requests."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "englishTranscript": english_transcript,
                                "segments": [segment.model_dump() for segment in segments],
                                "timestampInstruction": "Use segment timestamps only when the quoted risky detail clearly matches a segment; otherwise use null.",
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
        assessment = assessment_from_symptoms(symptoms, risk_level, reasons)
        signals = [RiskSignal.model_validate(signal) for signal in result.get("signals", [])]
        signals = _attach_segment_timestamps(signals, segments)
        recommended_action = result.get("recommendedAction") or "Review highlighted details and continue routine follow-up."
        return symptoms, assessment, signals, recommended_action, False
    except Exception:
        return _manual_risk_review()


def _load_call_record(path: Path) -> CallRecord:
    return CallRecord.model_validate_json(path.read_text())


def _enrich_call_record(call: CallRecord) -> CallRecord:
    if call.categories and call.escalationPlan:
        return call
    senior = get_senior(call.seniorId)
    symptoms = detect_symptoms_from_text(call.englishTranscript or call.originalTranscript)
    categories = build_conversation_categories(call.englishTranscript or call.originalTranscript, symptoms, call.riskAssessment, senior)
    escalation_plan = build_escalation_plan(call.riskAssessment, categories, senior)
    return call.model_copy(update={"categories": categories, "escalationPlan": escalation_plan})


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
    records = [_enrich_session(checkin) for checkin in _load_checkins()]
    return sorted(records, key=lambda record: record.completedAt or record.scheduledAt, reverse=True)


@app.get("/scenarios", response_model=list[Scenario])
def get_scenarios() -> list[Scenario]:
    return SCENARIOS


@app.post("/scenarios/{scenario_id}/run", response_model=ScenarioRunResponse)
def run_scenario(scenario_id: str) -> ScenarioRunResponse:
    scenario = next((item for item in SCENARIOS if item.id == scenario_id), None)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")

    senior = get_senior(scenario.seniorId)
    assessment = assess_speech_deviation(
        senior.baselineSpeechProfile,
        SpeechDeviationRequest(seniorId=senior.id, currentSpeechProfile=scenario.speechMetrics, symptoms=scenario.symptoms),
    )
    if scenario.symptoms.missedCheckIn:
        assessment = assessment_from_symptoms(scenario.symptoms, "Amber", assessment.reasons)

    categories = build_conversation_categories(scenario.englishTranscript, scenario.symptoms, assessment, senior)
    recommended_action = recommended_action_for(assessment, categories, senior)
    escalation_plan = build_escalation_plan(assessment, categories, senior)
    now = _now_iso()
    session = CheckInSession(
        id=f"scenario-{scenario.id}-{uuid4().hex[:8]}",
        seniorId=senior.id,
        scenarioId=scenario.id,
        scenarioName=scenario.name,
        scheduledAt=now,
        completedAt=None if scenario.symptoms.missedCheckIn else now,
        status=_status_for_risk(assessment, scenario.symptoms),  # type: ignore[arg-type]
        language=senior.preferredLanguage,
        riskLevel=assessment.riskLevel,
        summary=scenario.description,
        recommendedAction=recommended_action,
        originalTranscript=scenario.originalTranscript,
        englishTranscript=scenario.englishTranscript,
        riskAssessment=assessment,
        categories=categories,
        escalationPlan=escalation_plan,
        modelNote=extract_demo_embedding_note(),
    )
    checkins = _load_checkins()
    checkins.append(session)
    _save_checkins(checkins)
    tasks = _upsert_task_for_session(session, senior)
    return ScenarioRunResponse(session=session, tasks=tasks)


@app.get("/calls", response_model=list[CallRecord])
def get_calls() -> list[CallRecord]:
    if not CALL_STORAGE_ROOT.exists():
        return []
    records = [_enrich_call_record(_load_call_record(path)) for path in CALL_STORAGE_ROOT.glob("*/metadata.json")]
    return sorted(records, key=lambda record: record.completedAt, reverse=True)


@app.get("/calls/{call_id}", response_model=CallRecord)
def get_call(call_id: str) -> CallRecord:
    path = _call_metadata_path(call_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Call not found")
    return _enrich_call_record(_load_call_record(path))


@app.get("/calls/{call_id}/audio")
def get_call_audio(call_id: str) -> FileResponse:
    audio_path = CALL_STORAGE_ROOT / call_id / "mic-audio.webm"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio recording not found")
    return FileResponse(audio_path, media_type="audio/webm", filename=f"{call_id}-mic-audio.webm")


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
        audio_path = call_dir / "mic-audio.webm"
        audio_path.write_bytes(await audio.read())
        audio_file_path = str(audio_path)

    original_transcript = _transcript_to_text(messages)
    translation = transcribe_with_fallback(senior.preferredLanguage, original_transcript, audio_path)
    symptoms, assessment, risk_signals, ai_recommended_action, ai_fallback_used = _openai_risk_review(translation.translation, translation.segments)
    if ai_fallback_used:
        symptoms = detect_symptoms_from_text(translation.translation)
        assessment = assessment_from_symptoms(symptoms, "Watch", assessment.reasons)
    categories = build_conversation_categories(translation.translation, symptoms, assessment, senior)
    escalation_plan = build_escalation_plan(assessment, categories, senior)
    recommended_action = ai_recommended_action if not ai_fallback_used and assessment.riskLevel != "Red" else recommended_action_for(assessment, categories, senior)
    if not risk_signals:
        risk_signals = risk_signals_from_categories(categories)
    audio_url = f"/calls/{call_id}/audio" if audio_file_path else None

    (call_dir / "transcript-original.json").write_text(json.dumps([message.model_dump() for message in messages], indent=2))
    (call_dir / "transcript-english.txt").write_text(translation.translation)

    call = CallRecord(
        id=call_id,
        seniorId=senior.id,
        seniorName=senior.name,
        startedAt=startedAt,
        completedAt=completedAt or _now_iso(),
        status="Complete" if status not in {"Failed", "Saved"} else status,  # type: ignore[arg-type]
        riskLevel=assessment.riskLevel,
        originalTranscript=original_transcript,
        englishTranscript=translation.translation,
        transcriptMessages=messages,
        translationProvider=translation.provider,
        translationFallbackUsed=translation.fallbackUsed,
        audioFilePath=audio_file_path,
        audioUrl=audio_url,
        audioAvailable=audio_file_path is not None,
        transcriptSegments=translation.segments,
        riskSignals=risk_signals,
        aiRiskFallbackUsed=ai_fallback_used,
        riskAssessment=assessment,
        recommendedAction=recommended_action,
        categories=categories,
        escalationPlan=escalation_plan,
    )
    _call_metadata_path(call_id).write_text(call.model_dump_json(indent=2))
    _upsert_task_for_call(call, senior)
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
    return sorted(_load_tasks(), key=lambda task: task.createdAt, reverse=True)


@app.patch("/volunteer-tasks/{task_id}", response_model=VolunteerTask)
def update_volunteer_task(task_id: str, status: str) -> VolunteerTask:
    tasks = _load_tasks()
    if status not in {"Open", "In progress", "Closed"}:
        raise HTTPException(status_code=400, detail="Invalid task status")
    for task in tasks:
        if task.id == task_id:
            task.status = status  # type: ignore[assignment]
            _save_tasks(tasks)
            return task
    raise HTTPException(status_code=404, detail="Volunteer task not found")
