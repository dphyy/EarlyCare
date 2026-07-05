import os
import json
import re
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta, timezone
from typing import TypeVar
from uuid import uuid4

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from app.data import CHECKINS, SCENARIOS, SENIORS, VOLUNTEER_TASKS
from app.ml import assess_speech_deviation, extract_demo_embedding_note
from app.models import (
    CallPlan,
    CallPlanQuestion,
    CallRecord,
    CheckInScheduleItem,
    CheckInSession,
    CompleteCheckInRequest,
    ConversationCategory,
    MissedCheckInRequest,
    OperationsQueueItem,
    ProviderResult,
    RiskAssessment,
    RiskSignal,
    SavedCallResponse,
    Scenario,
    ScenarioRunResponse,
    SeniorRecord,
    SeniorRecordCategory,
    SeniorRecordEvent,
    SpeechModelCardGate,
    SpeechEnrichmentRequest,
    SpeechModelProvenance,
    Senior,
    SpeechProfile,
    SpeechDeviationRequest,
    Symptoms,
    TranscriptSegment,
    TranscriptMessage,
    VolunteerTask,
)
from app.providers import GoogleTranslateProvider, clean_transcript_text, transcribe_with_fallback
from app.risk import (
    assessment_from_symptoms,
    build_conversation_categories,
    build_escalation_plan,
    detect_symptoms_from_text,
    recommended_action_for,
    risk_signals_from_categories,
)
from app.speech_features import (
    DemoSpeechFeatureExtractor,
    SpeechFeatureInput,
    estimated_utterance_seconds,
    parse_iso,
    word_count,
)


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BACKEND_ROOT = Path(__file__).resolve().parents[1]
CALL_STORAGE_ROOT = BACKEND_ROOT / "storage" / "calls"
STATE_STORAGE_ROOT = BACKEND_ROOT / "storage" / "state"
CHECKINS_STATE_PATH = STATE_STORAGE_ROOT / "checkins.json"
TASKS_STATE_PATH = STATE_STORAGE_ROOT / "volunteer-tasks.json"
app = FastAPI(title="EarlyCare API", version="0.1.0")
StateRecord = TypeVar("StateRecord", bound=BaseModel)


def _cors_origins() -> list[str]:
    configured = os.getenv("FRONTEND_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
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
    return parse_iso(timestamp)


def _relative_seconds(timestamp: str | None, started_at: str) -> float | None:
    current = _parse_iso(timestamp)
    start = _parse_iso(started_at)
    if not current or not start:
        return None
    return max(0, round((current - start).total_seconds(), 3))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_parent_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | directory_flag)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
        _fsync_parent_directory(path)
    finally:
        if temporary_path and temporary_path.exists():
            with suppress(OSError):
                temporary_path.unlink()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
        _fsync_parent_directory(path)
    finally:
        if temporary_path and temporary_path.exists():
            with suppress(OSError):
                temporary_path.unlink()


def _state_json_default(path: Path, fallback: str) -> None:
    if path.exists():
        return
    _write_text_atomic(path, fallback)


def _corrupt_backup_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for index in range(20):
        suffix = f".{index}" if index else ""
        candidate = path.with_name(f"{path.name}.corrupt-{stamp}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.name}.corrupt-{stamp}.{uuid4().hex[:8]}")


def _quarantine_corrupt_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = _corrupt_backup_path(path)
    with suppress(OSError):
        path.replace(backup_path)
        return backup_path
    return None


def _load_state_records(path: Path, defaults: list[StateRecord], model: type[StateRecord]) -> list[StateRecord]:
    fallback_payload = [record.model_dump() for record in defaults]
    fallback_json = json.dumps(fallback_payload, indent=2)
    _state_json_default(path, fallback_json)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("State file must contain a JSON list")
        return [model.model_validate(item) for item in payload]
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        _quarantine_corrupt_file(path)
        _write_text_atomic(path, fallback_json)
        return [model.model_validate(item) for item in fallback_payload]


def _load_checkins() -> list[CheckInSession]:
    return _load_state_records(CHECKINS_STATE_PATH, CHECKINS, CheckInSession)


def _save_checkins(checkins: list[CheckInSession]) -> None:
    _write_text_atomic(CHECKINS_STATE_PATH, json.dumps([checkin.model_dump() for checkin in checkins], indent=2))


def _load_tasks() -> list[VolunteerTask]:
    return _load_state_records(TASKS_STATE_PATH, VOLUNTEER_TASKS, VolunteerTask)


def _save_tasks(tasks: list[VolunteerTask]) -> None:
    _write_text_atomic(TASKS_STATE_PATH, json.dumps([task.model_dump() for task in tasks], indent=2))


def _call_metadata_path(call_id: str) -> Path:
    return CALL_STORAGE_ROOT / call_id / "metadata.json"


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _load_calls() -> list[CallRecord]:
    if not CALL_STORAGE_ROOT.exists():
        return []
    calls: list[CallRecord] = []
    for path in sorted(CALL_STORAGE_ROOT.glob("*/metadata.json")):
        try:
            calls.append(_load_call_record(path))
        except (OSError, ValueError, ValidationError):
            _quarantine_corrupt_file(path)
    return calls


def _latest_schedule_contact(
    senior_id: str,
    checkins: list[CheckInSession],
    calls: list[CallRecord],
) -> tuple[datetime | None, str]:
    candidates: list[tuple[datetime, str]] = []
    for call in calls:
        if call.seniorId != senior_id:
            continue
        completed_at = _parse_iso(call.completedAt)
        if completed_at:
            candidates.append((_aware_datetime(completed_at), "call"))
    for checkin in checkins:
        if checkin.seniorId != senior_id or checkin.status == "Missed":
            continue
        contact_time = _parse_iso(checkin.completedAt or checkin.scheduledAt)
        if contact_time and (checkin.completedAt or checkin.englishTranscript or checkin.originalTranscript):
            candidates.append((_aware_datetime(contact_time), "check-in"))
    if not candidates:
        return None, "none"
    return max(candidates, key=lambda item: item[0])


def _latest_schedule_attempt(senior_id: str, checkins: list[CheckInSession], calls: list[CallRecord]) -> tuple[datetime | None, str | None]:
    attempts: list[tuple[datetime, str]] = []
    for call in calls:
        if call.seniorId != senior_id:
            continue
        completed_at = _parse_iso(call.completedAt)
        if completed_at:
            attempts.append((_aware_datetime(completed_at), call.status))
    for checkin in checkins:
        if checkin.seniorId != senior_id:
            continue
        attempt_time = _parse_iso(checkin.completedAt or checkin.scheduledAt)
        if attempt_time:
            attempts.append((_aware_datetime(attempt_time), checkin.status))
    if not attempts:
        return None, None
    return max(attempts, key=lambda item: item[0])


def _schedule_status_for(next_due: datetime, now: datetime, has_contact: bool, latest_attempt_status: str | None) -> tuple[str, float, float]:
    delta_hours = round((next_due - now).total_seconds() / 3600, 1)
    overdue_hours = round(max(0, (now - next_due).total_seconds() / 3600), 1)
    if not has_contact:
        return ("Overdue" if latest_attempt_status == "Missed" else "Due now", delta_hours, overdue_hours)
    if delta_hours > 24:
        return "On track", delta_hours, overdue_hours
    if delta_hours > 0:
        return "Due soon", delta_hours, overdue_hours
    if overdue_hours <= 6:
        return "Due now", delta_hours, overdue_hours
    return "Overdue", delta_hours, overdue_hours


def _schedule_action_for(senior: Senior, status: str, hours_until_due: float) -> str:
    if status == "Overdue":
        return f"Retry the scheduled call now, notify {senior.caregiverContact}, then assign same-day volunteer follow-up if unanswered."
    if status == "Due now":
        return "Start the scheduled check-in now and retry once if there is no answer."
    if status == "Due soon":
        return f"Prepare the next scheduled call within {max(1, round(hours_until_due))} hours."
    return "Continue routine cadence and keep the next scheduled check-in queued."


def _build_schedule_items(now: datetime | None = None) -> list[CheckInScheduleItem]:
    current = _aware_datetime(now or datetime.now(timezone.utc))
    checkins = [_enrich_session(checkin) for checkin in _load_checkins()]
    calls = [_enrich_call_record(call) for call in _load_calls()]
    items: list[CheckInScheduleItem] = []
    for senior in SENIORS:
        last_contact_at, last_contact_kind = _latest_schedule_contact(senior.id, checkins, calls)
        last_attempt_at, last_attempt_status = _latest_schedule_attempt(senior.id, checkins, calls)
        next_due = last_contact_at + timedelta(days=senior.checkInFrequencyDays) if last_contact_at else current
        status, hours_until_due, overdue_hours = _schedule_status_for(next_due, current, last_contact_at is not None, last_attempt_status)
        items.append(
            CheckInScheduleItem(
                seniorId=senior.id,
                seniorName=senior.name,
                checkInFrequencyDays=senior.checkInFrequencyDays,
                lastContactAt=last_contact_at.isoformat() if last_contact_at else None,
                lastContactKind=last_contact_kind,  # type: ignore[arg-type]
                lastAttemptAt=last_attempt_at.isoformat() if last_attempt_at else None,
                lastAttemptStatus=last_attempt_status,
                nextDueAt=next_due.isoformat(),
                status=status,  # type: ignore[arg-type]
                hoursUntilDue=hours_until_due,
                overdueHours=overdue_hours,
                recommendedAction=_schedule_action_for(senior, status, hours_until_due),
            )
        )
    return sorted(items, key=lambda item: ({"Overdue": 0, "Due now": 1, "Due soon": 2, "On track": 3}[item.status], item.nextDueAt))


def _record_sort_time(value: str) -> datetime:
    parsed = _parse_iso(value)
    return _aware_datetime(parsed) if parsed else datetime.min.replace(tzinfo=timezone.utc)


def _record_category_is_relevant(category: ConversationCategory) -> bool:
    return category.severity != "Green" or bool(category.evidence)


def _record_event_categories(categories: list[ConversationCategory]) -> list[ConversationCategory]:
    return [category for category in categories if _record_category_is_relevant(category)]


def _highest_risk(levels: list[str]) -> str:
    order = {"Green": 0, "Watch": 1, "Amber": 2, "Red": 3}
    if not levels:
        return "Green"
    return max(levels, key=lambda level: order.get(level, 0))


def _build_record_events(senior_id: str, checkins: list[CheckInSession], calls: list[CallRecord]) -> list[SeniorRecordEvent]:
    events: list[SeniorRecordEvent] = []
    for checkin in checkins:
        if checkin.seniorId != senior_id:
            continue
        events.append(
            SeniorRecordEvent(
                id=checkin.id,
                source="check-in",
                occurredAt=checkin.completedAt or checkin.scheduledAt,
                riskLevel=checkin.riskLevel,
                status=checkin.status,
                summary=checkin.summary,
                recommendedAction=checkin.recommendedAction,
                categories=_record_event_categories(checkin.categories),
            )
        )
    for call in calls:
        if call.seniorId != senior_id:
            continue
        events.append(
            SeniorRecordEvent(
                id=call.id,
                source="call",
                occurredAt=call.completedAt,
                riskLevel=call.riskLevel,
                status=call.status,
                summary=call.englishTranscript.splitlines()[0][:180] if call.englishTranscript else "Saved Agents call.",
                recommendedAction=call.recommendedAction,
                categories=_record_event_categories(call.categories),
            )
        )
    return sorted(events, key=lambda event: _record_sort_time(event.occurredAt), reverse=True)


def _aggregate_record_categories(events: list[SeniorRecordEvent]) -> list[SeniorRecordCategory]:
    aggregates: dict[str, dict[str, object]] = {}
    order = {"Green": 0, "Watch": 1, "Amber": 2, "Red": 3}
    for event in events:
        for category in event.categories:
            current = aggregates.setdefault(
                category.id,
                {
                    "label": category.label,
                    "highestSeverity": category.severity,
                    "recordCount": 0,
                    "latestAt": event.occurredAt,
                    "latestEvidence": [],
                    "recommendedAction": category.recommendedAction,
                },
            )
            current["recordCount"] = int(current["recordCount"]) + 1
            if order[category.severity] > order[str(current["highestSeverity"])]:
                current["highestSeverity"] = category.severity
            if _record_sort_time(event.occurredAt) >= _record_sort_time(str(current["latestAt"])):
                current["latestAt"] = event.occurredAt
                current["recommendedAction"] = category.recommendedAction
            evidence = list(current["latestEvidence"])
            for item in category.evidence:
                if item not in evidence:
                    evidence.append(item)
            current["latestEvidence"] = evidence[:4]

    return sorted(
        [
            SeniorRecordCategory(
                id=category_id,  # type: ignore[arg-type]
                label=str(values["label"]),
                highestSeverity=values["highestSeverity"],  # type: ignore[arg-type]
                recordCount=int(values["recordCount"]),
                latestAt=values["latestAt"],  # type: ignore[arg-type]
                latestEvidence=values["latestEvidence"],  # type: ignore[arg-type]
                recommendedAction=str(values["recommendedAction"]),
            )
            for category_id, values in aggregates.items()
        ],
        key=lambda category: (-order[category.highestSeverity], -(category.recordCount), category.label),
    )


def _build_senior_records() -> list[SeniorRecord]:
    checkins = [_enrich_session(checkin) for checkin in _load_checkins()]
    calls = [_enrich_call_record(call) for call in _load_calls()]
    tasks = _load_tasks()
    records: list[SeniorRecord] = []
    for senior in SENIORS:
        events = _build_record_events(senior.id, checkins, calls)
        open_task_count = len([task for task in tasks if task.seniorId == senior.id and task.status != "Closed"])
        records.append(
            SeniorRecord(
                seniorId=senior.id,
                seniorName=senior.name,
                livingAlone=senior.livingAlone,
                checkInFrequencyDays=senior.checkInFrequencyDays,
                totalRecords=len(events),
                openTaskCount=open_task_count,
                highestRiskLevel=_highest_risk([event.riskLevel for event in events]),  # type: ignore[arg-type]
                latestRecordAt=events[0].occurredAt if events else None,
                categories=_aggregate_record_categories(events),
                timeline=events,
            )
        )
    return records


def _highest_priority_task(tasks: list[VolunteerTask]) -> VolunteerTask | None:
    priority_order = {"Urgent": 0, "Today": 1, "Routine": 2}
    open_tasks = [task for task in tasks if task.status != "Closed"]
    if not open_tasks:
        return None
    return sorted(open_tasks, key=lambda task: (priority_order[task.priority], _record_sort_time(task.createdAt)))[0]


def _queue_priority(schedule: CheckInScheduleItem, record: SeniorRecord, task: VolunteerTask | None) -> str:
    if record.highestRiskLevel == "Red" or (task is not None and task.priority == "Urgent"):
        return "Emergency"
    if schedule.status in {"Overdue", "Due now"} or record.highestRiskLevel in {"Amber", "Watch"} or (task is not None and task.priority == "Today"):
        return "Today"
    if schedule.status == "Due soon":
        return "Due"
    return "Routine"


def _queue_reason(schedule: CheckInScheduleItem, record: SeniorRecord, task: VolunteerTask | None) -> str:
    if task:
        return task.reason
    if schedule.status in {"Overdue", "Due now", "Due soon"}:
        return schedule.recommendedAction
    if record.timeline:
        return record.timeline[0].summary
    return "Routine scheduled check-in queued."


def _build_operations_queue() -> list[OperationsQueueItem]:
    schedules = {item.seniorId: item for item in _build_schedule_items()}
    records = {record.seniorId: record for record in _build_senior_records()}
    tasks_by_senior: dict[str, list[VolunteerTask]] = {}
    for task in _load_tasks():
        tasks_by_senior.setdefault(task.seniorId, []).append(task)

    priority_order = {"Emergency": 0, "Today": 1, "Due": 2, "Routine": 3}
    schedule_order = {"Overdue": 0, "Due now": 1, "Due soon": 2, "On track": 3}
    risk_order = {"Red": 0, "Amber": 1, "Watch": 2, "Green": 3}
    queue: list[OperationsQueueItem] = []

    for senior in SENIORS:
        schedule = schedules[senior.id]
        record = records[senior.id]
        task = _highest_priority_task(tasks_by_senior.get(senior.id, []))
        priority = _queue_priority(schedule, record, task)
        reason = _queue_reason(schedule, record, task)
        queue.append(
            OperationsQueueItem(
                seniorId=senior.id,
                seniorName=senior.name,
                priority=priority,  # type: ignore[arg-type]
                reason=reason,
                recommendedAction=task.recommendedAction if task else schedule.recommendedAction,
                scheduleStatus=schedule.status,
                riskLevel=record.highestRiskLevel,
                openTaskCount=record.openTaskCount,
                nextDueAt=schedule.nextDueAt,
                dueInHours=schedule.hoursUntilDue,
                lastContactAt=schedule.lastContactAt,
                assignedTo=task.assignedTo if task else None,
                taskId=task.id if task else None,
                queueRank=0,
            )
        )

    ranked = sorted(
        queue,
        key=lambda item: (
            priority_order[item.priority],
            schedule_order[item.scheduleStatus],
            risk_order[item.riskLevel],
            _record_sort_time(item.nextDueAt),
            -item.openTaskCount,
            item.seniorName,
        ),
    )
    return [item.model_copy(update={"queueRank": index + 1}) for index, item in enumerate(ranked)]


def _add_call_plan_question(
    questions: list[CallPlanQuestion],
    seen: set[str],
    question_id: str,
    priority: str,
    topic: str,
    prompt: str,
    rationale: str,
) -> None:
    if question_id in seen:
        return
    seen.add(question_id)
    questions.append(
        CallPlanQuestion(
            id=question_id,
            priority=priority,  # type: ignore[arg-type]
            topic=topic,
            prompt=prompt,
            rationale=rationale,
        )
    )


def _condition_text(senior: Senior) -> str:
    return ", ".join(senior.knownConditions) if senior.knownConditions else "no listed long-term condition"


def _build_call_plan_for_senior(senior: Senior, schedule: CheckInScheduleItem, record: SeniorRecord) -> CallPlan:
    questions: list[CallPlanQuestion] = []
    seen: set[str] = set()
    record_categories = {category.id: category for category in record.categories}
    focus_text = " ".join(senior.promptFocus + senior.knownConditions).lower()

    _add_call_plan_question(
        questions,
        seen,
        "basic-wellbeing",
        "Routine",
        "Basic wellbeing",
        "How are you feeling today compared with our last check-in?",
        "Start with a broad wellbeing question before symptom-specific prompts.",
    )
    _add_call_plan_question(
        questions,
        seen,
        "fall-head-impact",
        "Watch" if "fall_head_impact" in record_categories or "fall" in focus_text else "Routine",
        "Falls / head impact / jolts",
        "Since the last call, did you fall, bump your head, have a body jolt, or feel whiplash?",
        "Falls and jolts are core EarlyCare risk checks for seniors living alone.",
    )
    _add_call_plan_question(
        questions,
        seen,
        "concussion-danger",
        "Urgent" if "concussion_danger" in record_categories else "Routine",
        "Possible concussion danger signs",
        "Any worsening headache, vomiting, confusion, slurred speech, weakness, numbness, unusual behaviour, or trouble waking?",
        "These are danger signs after a fall, head impact, blow, or jolt and should trigger human escalation.",
    )
    _add_call_plan_question(
        questions,
        seen,
        "food-water-medication",
        "Watch" if "medication_food_water" in record_categories else "Routine",
        "Medication / food / water",
        "Have you eaten, drunk enough water, and taken your medication today?",
        "Food, water, and medication status are basic check-in anchors.",
    )

    if "parkinson" in focus_text or "speech" in focus_text or "parkinsons_watch" in record_categories:
        _add_call_plan_question(
            questions,
            seen,
            "speech-watch",
            "Watch",
            "Speech watch",
            "Please say pa-ta-ka three times, then repeat: Today I am safe at home and I can ask for help.",
            "Repeated speech prompts help compare pace, pauses, and phrase clarity against the senior's baseline.",
        )

    if "ckd" in focus_text or "kidney" in focus_text or "chronic_illness" in record_categories:
        _add_call_plan_question(
            questions,
            seen,
            "ckd-hydration",
            "Watch",
            "CKD / hydration",
            "Have you had enough water today, and is there any swelling, dizziness, or appointment concern?",
            "CKD and hydration concerns are listed in this senior profile or history.",
        )

    if "diabetes" in focus_text:
        _add_call_plan_question(
            questions,
            seen,
            "diabetes-food-medicine",
            "Watch",
            "Diabetes routine",
            "Did you eat today and take your diabetes medicine as planned?",
            "Diabetes check-ins should confirm food and medication together.",
        )

    if "blood pressure" in focus_text or "hypertension" in focus_text:
        _add_call_plan_question(
            questions,
            seen,
            "blood-pressure",
            "Watch",
            "Blood pressure",
            "Did you take your blood pressure medicine, and do you feel dizzy, weak, or unusually tired?",
            "Blood pressure or hypertension is listed in the senior profile.",
        )

    if "social_isolation" in record_categories or "mental_wellbeing" in record_categories or "loneliness" in focus_text:
        _add_call_plan_question(
            questions,
            seen,
            "mood-loneliness",
            "Watch",
            "Mood / loneliness",
            "How has your mood been, and would you like a befriender call, neighbour check-in, or volunteer visit this week?",
            "The senior record or prompt focus includes loneliness, low mood, or a help request.",
        )

    if schedule.status in {"Due now", "Overdue"} or "missed_checkin" in record_categories:
        _add_call_plan_question(
            questions,
            seen,
            "contact-reliability",
            "Watch" if schedule.status == "Due now" else "Urgent",
            "Contact reliability",
            "If we cannot reach you next time, should we call your caregiver or listed neighbour first?",
            "Due, overdue, or missed check-ins need a clear human contact path.",
        )

    priority_order = {"Urgent": 0, "Watch": 1, "Routine": 2}
    questions = sorted(questions, key=lambda question: (priority_order[question.priority], question.topic))[:8]
    opening_status = "overdue" if schedule.status == "Overdue" else "due now" if schedule.status == "Due now" else "scheduled"
    return CallPlan(
        seniorId=senior.id,
        seniorName=senior.name,
        preferredLanguage=senior.preferredLanguage,
        generatedAt=_now_iso(),
        scheduleStatus=schedule.status,
        openingScript=(
            f"Hello {senior.name}, this is EarlyCare. This is your {opening_status} "
            f"{senior.checkInFrequencyDays}-day wellbeing check-in. Are you safe to talk now?"
        ),
        questions=questions,
        escalationReminder=(
            f"Use {senior.caregiverContact}"
            f"{f' and {senior.neighborContact}' if senior.neighborContact else ''} for follow-up. "
            "For red danger signs after a fall, head impact, blow, or jolt, escalate for urgent human help."
        ),
    )


def _build_call_plans() -> list[CallPlan]:
    schedules = {item.seniorId: item for item in _build_schedule_items()}
    records = {record.seniorId: record for record in _build_senior_records()}
    plans: list[CallPlan] = []
    for senior in SENIORS:
        schedule = schedules[senior.id]
        record = records[senior.id]
        plans.append(_build_call_plan_for_senior(senior, schedule, record))
    return plans


def _transcript_to_text(messages: list[TranscriptMessage]) -> str:
    return "\n".join(f"{'Patient' if message.role == 'Senior' else message.role}: {message.text}" for message in messages)


def _display_role(role: str) -> str:
    return "Patient" if role == "Senior" else role


def _clean_transcript_text(text: str) -> str:
    return clean_transcript_text(text)


def _clean_messages(messages: list[TranscriptMessage]) -> list[TranscriptMessage]:
    return [message.model_copy(update={"text": _clean_transcript_text(message.text)}) for message in messages]


def _word_count(text: str) -> int:
    return word_count(text)


def _estimated_utterance_seconds(text: str) -> float:
    return estimated_utterance_seconds(text)


def _estimate_current_speech_profile(
    messages: list[TranscriptMessage],
    started_at: str,
    completed_at: str,
    segments: list[TranscriptSegment] | None = None,
    audio_path: Path | None = None,
) -> SpeechProfile | None:
    return DemoSpeechFeatureExtractor().extract(
        SpeechFeatureInput(
            audio_path=audio_path,
            messages=messages,
            started_at=started_at,
            completed_at=completed_at or _now_iso(),
            segments=segments,
        )
    )


def _demo_speech_provenance(generated_at: str | None = None) -> SpeechModelProvenance:
    return SpeechModelProvenance(
        runtimeMode="demo metrics",
        featureExtractor="transcript timing metrics",
        modelName="EarlyCare demo speech metrics",
        generatedAt=generated_at or _now_iso(),
        validated=False,
        notes=[
            "Fast call-save path calculated from transcript timestamps and patient segments.",
            "No diagnostic classifier or model weights were used.",
        ],
    )


def _str_from_provenance(provenance: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = provenance.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


BLOCKED_MODEL_COPY_PHRASES = (
    "parkinson's detected",
    "parkinson detected",
    "detected parkinson",
    "concussion detected",
    "detected concussion",
    "disease diagnosis",
    "medical certainty",
    "emergency confirmed",
)


def _contains_blocked_model_copy(text: str | None) -> bool:
    normalized = (text or "").lower()
    return any(phrase in normalized for phrase in BLOCKED_MODEL_COPY_PHRASES)


def _validate_model_card_gate(model_card: SpeechModelCardGate | None) -> None:
    if model_card is None:
        raise HTTPException(status_code=400, detail="Validated speech models require modelCard release-gate evidence")

    required_flags = [
        "datasetAccessReviewed",
        "speakerSplitVerified",
        "evaluationMetricsRecorded",
        "subgroupChecksReviewed",
        "failureModesDocumented",
        "uiCopyReviewed",
        "humanFollowUpActionDefined",
        "rollbackPathDocumented",
    ]
    missing = [field for field in required_flags if not getattr(model_card, field)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Validated speech model is missing release-gate checks: {', '.join(missing)}")
    if not model_card.humanFollowUpAction or not model_card.humanFollowUpAction.strip():
        raise HTTPException(status_code=400, detail="Validated speech model requires a humanFollowUpAction")
    if _contains_blocked_model_copy(model_card.humanFollowUpAction):
        raise HTTPException(status_code=400, detail="Validated speech model humanFollowUpAction uses blocked diagnosis language")


def _speech_profile_from_enrichment(call: CallRecord, request: SpeechEnrichmentRequest) -> SpeechProfile:
    profile = request.speechMetrics or call.currentSpeechProfile
    if profile is None:
        raise HTTPException(status_code=400, detail="Speech enrichment needs speech_metrics or an existing currentSpeechProfile")
    embedding = request.embedding or profile.embedding
    if embedding is None:
        raise HTTPException(status_code=400, detail="Speech enrichment needs an embedding")
    return profile.model_copy(update={"embedding": embedding, "updatedAt": profile.updatedAt or _now_iso()})


def _speech_provenance_from_enrichment(request: SpeechEnrichmentRequest) -> SpeechModelProvenance:
    provenance = request.provenance
    runtime_mode = request.runtimeMode
    validated = runtime_mode == "validated model"
    model_name = request.modelName or _str_from_provenance(provenance, "model_name", "model")
    feature_extractor = request.featureExtractor or _str_from_provenance(provenance, "feature_extractor")
    generated_at = _str_from_provenance(provenance, "extracted_at", "generated_at") or _now_iso()
    artifact_uri = request.artifactUri or _str_from_provenance(provenance, "artifact_uri", "artifact", "source_id")
    model_version = request.modelVersion or _str_from_provenance(provenance, "model_version")
    if validated:
        missing_fields = [
            field
            for field, value in {
                "modelName": model_name,
                "modelVersion": model_version,
                "featureExtractor": feature_extractor,
                "artifactUri": artifact_uri,
            }.items()
            if not value
        ]
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Validated speech model is missing required provenance: {', '.join(missing_fields)}")
        _validate_model_card_gate(request.modelCard)
    model_name = model_name or "offline speech embedding"
    feature_extractor = feature_extractor or _str_from_provenance(provenance, "model_name", "model") or model_name
    notes = [
        "Offline enrichment is stored as decision-support context.",
        "Unvalidated enrichment does not change emergency routing by itself.",
    ]
    if validated:
        notes = [
            "Validated speech model-card gate was supplied.",
            "Model output remains decision support and still requires human follow-up.",
        ]
    else:
        notes.append("Model-card release gate is required before this can be treated as a validated model.")
    return SpeechModelProvenance(
        runtimeMode=runtime_mode,
        featureExtractor=feature_extractor,
        modelName=model_name,
        modelVersion=model_version,
        artifactUri=artifact_uri,
        generatedAt=generated_at,
        validated=validated,
        modelCard=request.modelCard,
        notes=notes,
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


def _missed_checkin_transcript(request: MissedCheckInRequest) -> str:
    parts = [f"No answer after {request.attemptCount} scheduled call attempt{'s' if request.attemptCount != 1 else ''}."]
    if request.retryAt:
        parts.append(f"Latest retry attempt was at {request.retryAt}.")
    if request.note:
        parts.append(request.note.strip())
    return " ".join(part for part in parts if part)


def _create_missed_checkin_session(request: MissedCheckInRequest) -> ScenarioRunResponse:
    senior = get_senior(request.seniorId)
    scheduled_at = request.scheduledAt or _now_iso()
    symptoms = Symptoms(missedCheckIn=True)
    assessment = assessment_from_symptoms(
        symptoms,
        "Amber",
        [
            "Scheduled check-in was missed after retry.",
            "Volunteer follow-up needed because the senior lives alone.",
        ],
    )
    transcript = _missed_checkin_transcript(request)
    categories = build_conversation_categories(transcript, symptoms, assessment, senior)
    recommended_action = recommended_action_for(assessment, categories, senior)
    escalation_plan = build_escalation_plan(assessment, categories, senior)
    session = CheckInSession(
        id=f"missed-{senior.id}-{uuid4().hex[:8]}",
        seniorId=senior.id,
        scheduledAt=scheduled_at,
        completedAt=None,
        status="Missed",
        language=senior.preferredLanguage,
        riskLevel="Amber",
        summary=(
            f"Scheduled check-in missed after {request.attemptCount} attempt"
            f"{'s' if request.attemptCount != 1 else ''}."
        ),
        recommendedAction=recommended_action,
        originalTranscript=transcript,
        englishTranscript=transcript,
        riskAssessment=assessment,
        categories=categories,
        escalationPlan=escalation_plan,
        modelNote="No speech metrics were produced because the senior did not answer the scheduled check-in.",
    )
    checkins = _load_checkins()
    checkins.append(session)
    _save_checkins(checkins)
    tasks = _upsert_task_for_session(session, senior)
    return ScenarioRunResponse(session=session, tasks=tasks)


def _create_started_checkin_session(senior_id: str) -> CheckInSession:
    senior = get_senior(senior_id)
    now = _now_iso()
    symptoms = Symptoms()
    assessment = _empty_assessment("Green", ["Scheduled check-in started."])
    categories = build_conversation_categories("", symptoms, assessment, senior)
    escalation_plan = build_escalation_plan(assessment, categories, senior)
    session = CheckInSession(
        id=f"checkin-{uuid4().hex[:10]}",
        seniorId=senior.id,
        scheduledAt=now,
        completedAt=None,
        status="In progress",
        language=senior.preferredLanguage,
        riskLevel="Green",
        summary="Scheduled check-in started.",
        recommendedAction="Complete the call after speaking with the senior, or log it as missed if unanswered.",
        originalTranscript="",
        englishTranscript="",
        riskAssessment=assessment,
        categories=categories,
        escalationPlan=escalation_plan,
    )
    checkins = _load_checkins()
    checkins.append(session)
    _save_checkins(checkins)
    return session


def _completion_reasons(symptoms: Symptoms, transcript_present: bool) -> list[str]:
    reasons = ["Completed scheduled check-in reviewed from transcript." if transcript_present else "Completed scheduled check-in recorded manually."]
    if symptoms.fall or symptoms.headImpact or symptoms.whiplashOrJolt:
        reasons.append("Fall, head impact, blow, or jolt mentioned.")
    if symptoms.confusion or symptoms.vomiting or symptoms.slurredSpeech or symptoms.weakness or symptoms.numbness or symptoms.drowsinessOrUnwakeable or symptoms.worseningHeadache:
        reasons.append("Post-impact danger signs mentioned.")
    if symptoms.lowMood or symptoms.loneliness or symptoms.asksForHelp:
        reasons.append("Wellbeing or help-request signal mentioned.")
    if symptoms.chronicConcern or symptoms.medicationMissed or symptoms.poorIntake:
        reasons.append("Chronic illness, medication, food, or water follow-up signal mentioned.")
    if len(reasons) == 1:
        reasons.append("No danger signs were recorded.")
    return reasons


def _complete_started_checkin_session(checkin_id: str, request: CompleteCheckInRequest | None) -> CheckInSession:
    request = request or CompleteCheckInRequest()
    checkins = _load_checkins()
    for index, existing in enumerate(checkins):
        if existing.id != checkin_id:
            continue

        senior = get_senior(existing.seniorId)
        completed_at = request.completedAt or _now_iso()
        default_transcript = "Patient completed the check-in and reported no immediate concern."
        english_transcript = _clean_transcript_text(request.englishTranscript or existing.englishTranscript or default_transcript)
        original_transcript = _clean_transcript_text(request.originalTranscript or existing.originalTranscript or english_transcript)
        review_text = english_transcript or original_transcript
        symptoms = detect_symptoms_from_text(review_text)
        transcript_present = bool((request.englishTranscript or request.originalTranscript or existing.englishTranscript or existing.originalTranscript or "").strip())
        assessment = assessment_from_symptoms(symptoms, "Green", _completion_reasons(symptoms, transcript_present))
        categories = build_conversation_categories(review_text, symptoms, assessment, senior)
        recommended_action = recommended_action_for(assessment, categories, senior)
        escalation_plan = build_escalation_plan(assessment, categories, senior)
        elevated_categories = [category.label for category in categories if category.severity != "Green"]
        summary = request.summary or (
            f"Completed scheduled check-in with follow-up signals: {', '.join(elevated_categories[:2])}."
            if elevated_categories
            else "Completed scheduled check-in with no concerning symptoms recorded."
        )
        completed = existing.model_copy(
            update={
                "completedAt": completed_at,
                "status": _status_for_risk(assessment, symptoms),
                "riskLevel": assessment.riskLevel,
                "summary": summary,
                "recommendedAction": recommended_action,
                "originalTranscript": original_transcript,
                "englishTranscript": english_transcript,
                "riskAssessment": assessment,
                "categories": categories,
                "escalationPlan": escalation_plan,
            }
        )
        checkins[index] = completed
        _save_checkins(checkins)
        if completed.riskLevel != "Green":
            _upsert_task_for_session(completed, senior)
        return completed

    raise HTTPException(status_code=404, detail="Check-in not found")


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
                            "may be at risk, such as living-alone missed check-ins, falls, head impact, whiplash or body jolts, possible "
                            "concussion danger signs, possible Parkinson's speech-watch signals, chronic illness concerns, poor intake, "
                            "medication issues, loneliness, help requests, unsafe home situations, or other details needing earlier caregiver "
                            "or volunteer action. Review patient speech only. Ignore agent questions, agent summaries, and any risk wording "
                            "that the patient did not say. Use exact English patient evidence text."
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
        assessment = assessment_from_symptoms(symptoms, risk_level, reasons)
        signals = [RiskSignal.model_validate(signal) for signal in result.get("signals", [])]
        signals = _attach_segment_timestamps(signals, review_segments)
        recommended_action = result.get("recommendedAction") or "Review highlighted details and continue routine follow-up."
        return symptoms, assessment, signals, recommended_action, False
    except Exception:
        return _manual_risk_review()


def _load_call_record(path: Path) -> CallRecord:
    return CallRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _enrich_call_record(call: CallRecord) -> CallRecord:
    updates: dict[str, object] = {}
    if call.currentSpeechProfile and call.speechModelProvenance is None:
        updates["speechModelProvenance"] = _demo_speech_provenance(call.currentSpeechProfile.updatedAt or call.completedAt)
    if call.categories and call.escalationPlan:
        return call.model_copy(update=updates) if updates else call
    senior = get_senior(call.seniorId)
    symptoms = detect_symptoms_from_text(call.englishTranscript or call.originalTranscript)
    categories = build_conversation_categories(call.englishTranscript or call.originalTranscript, symptoms, call.riskAssessment, senior)
    escalation_plan = build_escalation_plan(call.riskAssessment, categories, senior)
    updates.update({"categories": categories, "escalationPlan": escalation_plan})
    return call.model_copy(update=updates)


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


@app.get("/senior-records", response_model=list[SeniorRecord])
def get_senior_records() -> list[SeniorRecord]:
    return _build_senior_records()


@app.get("/seniors/{senior_id}/record", response_model=SeniorRecord)
def get_senior_record(senior_id: str) -> SeniorRecord:
    for record in _build_senior_records():
        if record.seniorId == senior_id:
            return record
    raise HTTPException(status_code=404, detail="Senior record not found")


@app.get("/call-plans", response_model=list[CallPlan])
def get_call_plans() -> list[CallPlan]:
    return _build_call_plans()


@app.get("/seniors/{senior_id}/call-plan", response_model=CallPlan)
def get_call_plan(senior_id: str) -> CallPlan:
    for plan in _build_call_plans():
        if plan.seniorId == senior_id:
            return plan
    raise HTTPException(status_code=404, detail="Call plan not found")


@app.get("/schedule", response_model=list[CheckInScheduleItem])
def get_schedule() -> list[CheckInScheduleItem]:
    return _build_schedule_items()


@app.get("/operations-queue", response_model=list[OperationsQueueItem])
def get_operations_queue() -> list[OperationsQueueItem]:
    return _build_operations_queue()


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


@app.post("/checkins/missed", response_model=ScenarioRunResponse)
def record_missed_checkin(request: MissedCheckInRequest) -> ScenarioRunResponse:
    return _create_missed_checkin_session(request)


@app.get("/calls", response_model=list[CallRecord])
def get_calls() -> list[CallRecord]:
    records = [_enrich_call_record(call) for call in _load_calls()]
    return sorted(records, key=lambda record: record.completedAt, reverse=True)


@app.get("/calls/{call_id}", response_model=CallRecord)
def get_call(call_id: str) -> CallRecord:
    path = _call_metadata_path(call_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Call not found")
    return _enrich_call_record(_load_call_record(path))


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
    return _create_started_checkin_session(senior_id)


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
        _write_bytes_atomic(audio_path, await audio.read())
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
    symptoms, assessment, risk_signals, ai_recommended_action, ai_fallback_used = _openai_risk_review(english_transcript, translation.segments)
    if ai_fallback_used:
        symptoms = detect_symptoms_from_text(english_transcript)
        assessment = assessment_from_symptoms(symptoms, "Watch", assessment.reasons)
    categories = build_conversation_categories(english_transcript, symptoms, assessment, senior)
    escalation_plan = build_escalation_plan(assessment, categories, senior)
    recommended_action = ai_recommended_action if not ai_fallback_used and assessment.riskLevel != "Red" else recommended_action_for(assessment, categories, senior)
    if not risk_signals:
        risk_signals = risk_signals_from_categories(categories)
    audio_url = f"/calls/{call_id}/audio" if audio_file_path else None
    current_speech_profile = _estimate_current_speech_profile(messages, startedAt, completedAt, translation.segments, audio_path)

    _write_text_atomic(call_dir / "transcript-original.json", json.dumps([message.model_dump() for message in messages], indent=2))
    _write_text_atomic(call_dir / "transcript-english.txt", english_transcript)

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
        speechModelProvenance=_demo_speech_provenance(current_speech_profile.updatedAt or completedAt) if current_speech_profile else None,
        transcriptSegments=translation.segments,
        riskSignals=risk_signals,
        aiRiskFallbackUsed=ai_fallback_used,
        riskAssessment=assessment,
        recommendedAction=recommended_action,
        categories=categories,
        escalationPlan=escalation_plan,
    )
    _write_text_atomic(_call_metadata_path(call_id), call.model_dump_json(indent=2))
    _upsert_task_for_call(call, senior)
    return SavedCallResponse(call=call)


@app.patch("/calls/{call_id}/speech-enrichment", response_model=CallRecord)
def enrich_call_speech(call_id: str, request: SpeechEnrichmentRequest) -> CallRecord:
    path = _call_metadata_path(call_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Call not found")

    call = _load_call_record(path)
    current_speech_profile = _speech_profile_from_enrichment(call, request)
    speech_model_provenance = _speech_provenance_from_enrichment(request)
    updated = call.model_copy(
        update={
            "currentSpeechProfile": current_speech_profile,
            "speechModelProvenance": speech_model_provenance,
        }
    )
    _write_text_atomic(path, updated.model_dump_json(indent=2))
    return _enrich_call_record(updated)


@app.post("/checkins/{checkin_id}/complete", response_model=CheckInSession)
def complete_checkin(checkin_id: str, request: CompleteCheckInRequest | None = None) -> CheckInSession:
    return _complete_started_checkin_session(checkin_id, request)


@app.post("/ml/speech-deviation")
def speech_deviation(request: SpeechDeviationRequest) -> dict[str, object]:
    senior = get_senior(request.seniorId)
    assessment = assess_speech_deviation(senior.baselineSpeechProfile, request)
    return {
        "assessment": assessment,
        "modelNote": extract_demo_embedding_note(),
        "humanFollowUpAction": "Review speech deviation alongside symptoms and check-in history; use it only to guide human follow-up.",
        "safetyLabel": "decision support, not diagnosis",
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
