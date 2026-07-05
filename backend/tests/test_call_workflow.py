import unittest
import json
from contextlib import redirect_stdout
from datetime import timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import main, providers
from app.models import ProviderResult, RiskAssessment, SpeechProfile, TranscriptMessage, TranscriptSegment
from research.speech_ml import make_enrichment_payload


class CallWorkflowTests(unittest.TestCase):
    def test_clean_transcript_text_removes_bracket_cues(self) -> None:
        self.assertEqual(providers.clean_transcript_text("Agent: [happy] hello [concerned] there"), "Agent: hello there")

    def test_cors_origins_cover_vite_fallback_ports_and_env_override(self) -> None:
        with patch.dict(main.os.environ, {}, clear=True):
            origins = main._cors_origins()
            self.assertIn("http://localhost:5173", origins)
            self.assertIn("http://localhost:5174", origins)
            self.assertIn("http://127.0.0.1:5175", origins)

        with patch.dict(main.os.environ, {"FRONTEND_ORIGINS": "http://example.test, http://localhost:3000"}):
            self.assertEqual(main._cors_origins(), ["http://example.test", "http://localhost:3000"])

    def test_atomic_text_write_preserves_existing_file_when_replace_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "state" / "checkins.json"
            path.parent.mkdir(parents=True)
            path.write_text("existing", encoding="utf-8")

            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    main._write_text_atomic(path, "replacement")

            self.assertEqual(path.read_text(encoding="utf-8"), "existing")
            self.assertEqual(list(path.parent.glob(".checkins.json.*.tmp")), [])

    def test_checkin_state_remains_readable_when_atomic_replace_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            try:
                seeded = main._load_checkins()
                original_summary = seeded[0].summary
                changed = [seeded[0].model_copy(update={"summary": "should not replace existing state"})]

                with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                    with self.assertRaises(OSError):
                        main._save_checkins(changed)

                reloaded = main._load_checkins()
                self.assertEqual(reloaded[0].summary, original_summary)
                self.assertNotIn("should not replace", main.CHECKINS_STATE_PATH.read_text(encoding="utf-8"))
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path

    def test_corrupt_checkin_state_is_quarantined_and_reseeded(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            try:
                state_root.mkdir(parents=True)
                main.CHECKINS_STATE_PATH.write_text("{broken json", encoding="utf-8")

                checkins = main._load_checkins()

                self.assertEqual([checkin.id for checkin in checkins], [checkin.id for checkin in main.CHECKINS])
                self.assertTrue(list(state_root.glob("checkins.json.corrupt-*")))
                self.assertTrue(main.CHECKINS_STATE_PATH.read_text(encoding="utf-8").lstrip().startswith("["))
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path

    def test_invalid_task_state_is_quarantined_and_reseeded(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            try:
                state_root.mkdir(parents=True)
                main.TASKS_STATE_PATH.write_text(json.dumps([{"id": "missing-required-fields"}]), encoding="utf-8")

                tasks = main._load_tasks()

                self.assertEqual([task.id for task in tasks], [task.id for task in main.VOLUNTEER_TASKS])
                self.assertTrue(list(state_root.glob("volunteer-tasks.json.corrupt-*")))
                self.assertNotIn("missing-required-fields", main.TASKS_STATE_PATH.read_text(encoding="utf-8"))
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path

    def test_health_reports_storage_warnings_without_secret_values(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            call_root = Path(tmp) / "calls"
            call_dir = call_root / "call-bad"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = call_root
            try:
                state_root.mkdir(parents=True)
                call_dir.mkdir(parents=True)
                (state_root / "checkins.json.corrupt-20260705T000000Z").write_text("{broken", encoding="utf-8")
                (call_dir / "metadata.json.corrupt-20260705T000000Z").write_text("{broken", encoding="utf-8")

                response = TestClient(main.app).get("/health")

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["status"], "degraded")
                self.assertEqual(payload["storage"]["status"], "degraded")
                self.assertEqual(payload["storage"]["quarantinedStateFiles"], 1)
                self.assertEqual(payload["storage"]["quarantinedCallMetadataFiles"], 1)
                joined_warnings = " ".join(payload["storage"]["warnings"])
                self.assertIn("quarantined state file", joined_warnings)
                self.assertIn("quarantined call metadata file", joined_warnings)
                self.assertNotIn("sk-", joined_warnings)
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_schedule_items_use_frequency_and_last_contact(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = Path(tmp) / "calls"
            try:
                now = main._parse_iso("2026-07-05T10:00:00+08:00")
                assert now is not None
                schedule = {item.seniorId: item for item in main._build_schedule_items(now)}

                self.assertEqual(schedule["s-001"].status, "Due soon")
                self.assertEqual(schedule["s-001"].lastContactKind, "check-in")
                self.assertTrue(schedule["s-001"].nextDueAt.startswith("2026-07-06T09:00:00"))
                self.assertEqual(schedule["s-002"].status, "Due now")
                self.assertIsNone(schedule["s-002"].lastContactAt)
                self.assertEqual(schedule["s-003"].status, "Due soon")
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_senior_records_roll_up_categories_and_open_tasks(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = Path(tmp) / "calls"
            try:
                records = {record.seniorId: record for record in main._build_senior_records()}

                self.assertEqual(records["s-001"].highestRiskLevel, "Red")
                self.assertEqual(records["s-001"].totalRecords, 1)
                self.assertTrue(any(category.id == "concussion_danger" for category in records["s-001"].categories))
                self.assertEqual(records["s-001"].timeline[0].source, "check-in")
                self.assertEqual(records["s-002"].openTaskCount, 1)
                self.assertEqual(records["s-002"].totalRecords, 0)
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_call_plan_uses_schedule_profile_and_record_history(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = Path(tmp) / "calls"
            try:
                plans = {plan.seniorId: plan for plan in main._build_call_plans()}

                tan_questions = {question.id: question for question in plans["s-001"].questions}
                self.assertEqual(tan_questions["concussion-danger"].priority, "Urgent")
                self.assertIn("Daughter: Mei Ling", plans["s-001"].escalationReminder)

                raman_questions = {question.id: question for question in plans["s-002"].questions}
                self.assertEqual(plans["s-002"].scheduleStatus, "Due now")
                self.assertIn("speech-watch", raman_questions)
                self.assertIn("contact-reliability", raman_questions)

                ahmad_questions = {question.id: question for question in plans["s-003"].questions}
                self.assertIn("ckd-hydration", ahmad_questions)
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_operations_queue_ranks_emergency_tasks_before_due_calls(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = Path(tmp) / "calls"
            try:
                queue = main._build_operations_queue()

                self.assertEqual([item.queueRank for item in queue], [1, 2, 3])
                self.assertEqual(queue[0].seniorId, "s-001")
                self.assertEqual(queue[0].priority, "Emergency")
                self.assertEqual(queue[0].taskId, "t-001")
                self.assertEqual(queue[1].seniorId, "s-002")
                self.assertEqual(queue[1].priority, "Today")
                self.assertEqual(queue[2].seniorId, "s-003")
                self.assertEqual(queue[2].priority, "Due")
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_missing_call_follow_up_task_is_repaired_for_task_list_records_and_queue(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            call_root = Path(tmp) / "calls"
            call_dir = call_root / "call-red"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = call_root
            try:
                state_root.mkdir(parents=True)
                call_dir.mkdir(parents=True)
                main.CHECKINS_STATE_PATH.write_text("[]", encoding="utf-8")
                main.TASKS_STATE_PATH.write_text("[]", encoding="utf-8")
                call = main.CallRecord(
                    id="call-red",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Red",
                    originalTranscript="Patient: I fell and feel confused.",
                    englishTranscript="Patient: I fell and feel confused.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I fell and feel confused.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=0,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=100,
                        missedCheckInScore=0,
                        riskLevel="Red",
                        reasons=["Confusion after fall"],
                    ),
                    recommendedAction="Call caregiver and coordinate urgent medical assessment.",
                    categories=[
                        main.ConversationCategory(
                            id="concussion_danger",
                            label="Possible concussion danger signs",
                            severity="Red",
                            evidence=["Confusion reported."],
                            recommendedAction="Escalate for urgent assessment.",
                        )
                    ],
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2), encoding="utf-8")

                tasks = TestClient(main.app).get("/volunteer-tasks").json()
                repaired_task = next(task for task in tasks if task.get("sourceCallId") == "call-red")
                self.assertEqual(repaired_task["priority"], "Urgent")
                self.assertEqual(repaired_task["escalationStep"], "emergency-alert")

                records = {record.seniorId: record for record in main._build_senior_records()}
                self.assertEqual(records["s-001"].openTaskCount, 1)
                queue = main._build_operations_queue()
                self.assertEqual(queue[0].taskId, "task-call-red")
                self.assertEqual(queue[0].priority, "Emergency")
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_repaired_call_follow_up_task_can_be_updated(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            call_root = Path(tmp) / "calls"
            call_dir = call_root / "call-ack"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = call_root
            try:
                state_root.mkdir(parents=True)
                call_dir.mkdir(parents=True)
                main.CHECKINS_STATE_PATH.write_text("[]", encoding="utf-8")
                main.TASKS_STATE_PATH.write_text("[]", encoding="utf-8")
                call = main.CallRecord(
                    id="call-ack",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Amber",
                    originalTranscript="Patient: I slipped and feel dizzy.",
                    englishTranscript="Patient: I slipped and feel dizzy.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I slipped and feel dizzy.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=0,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=80,
                        missedCheckInScore=0,
                        riskLevel="Amber",
                        reasons=["Fall with dizziness"],
                    ),
                    recommendedAction="Notify caregiver and arrange same-day follow-up.",
                    categories=[
                        main.ConversationCategory(
                            id="fall_head_impact",
                            label="Fall / head impact / whiplash",
                            severity="Amber",
                            evidence=["Fall reported."],
                            recommendedAction="Ask what happened and whether the senior can move safely.",
                        )
                    ],
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2), encoding="utf-8")

                client = TestClient(main.app)
                acknowledged = client.patch("/volunteer-tasks/task-call-ack", params={"status": "In progress"})

                self.assertEqual(acknowledged.status_code, 200)
                self.assertEqual(acknowledged.json()["status"], "In progress")
                self.assertEqual(acknowledged.json()["sourceCallId"], "call-ack")
                saved_tasks = json.loads(main.TASKS_STATE_PATH.read_text(encoding="utf-8"))
                self.assertEqual(saved_tasks[0]["id"], "task-call-ack")
                self.assertEqual(saved_tasks[0]["status"], "In progress")

                closed = client.patch("/volunteer-tasks/task-call-ack", params={"status": "Closed"})

                self.assertEqual(closed.status_code, 200)
                self.assertEqual(closed.json()["status"], "Closed")
                records = {record.seniorId: record for record in main._build_senior_records()}
                self.assertEqual(records["s-001"].openTaskCount, 0)
                self.assertEqual(json.loads(main.TASKS_STATE_PATH.read_text(encoding="utf-8"))[0]["status"], "Closed")
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_closed_call_follow_up_task_is_not_reopened_by_repair(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            call_root = Path(tmp) / "calls"
            call_dir = call_root / "call-closed"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = call_root
            try:
                state_root.mkdir(parents=True)
                call_dir.mkdir(parents=True)
                main.CHECKINS_STATE_PATH.write_text("[]", encoding="utf-8")
                main.TASKS_STATE_PATH.write_text(
                    json.dumps(
                        [
                            {
                                "id": "task-call-closed",
                                "seniorId": "s-001",
                                "priority": "Urgent",
                                "reason": "Previously handled.",
                                "recommendedAction": "No further action.",
                                "assignedTo": "Community response team",
                                "status": "Closed",
                                "createdAt": "2026-07-04T10:05:00+08:00",
                                "sourceCallId": "call-closed",
                                "escalationStep": "emergency-alert",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                call = main.CallRecord(
                    id="call-closed",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Red",
                    originalTranscript="Patient: I fell and feel confused.",
                    englishTranscript="Patient: I fell and feel confused.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I fell and feel confused.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=0,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=100,
                        missedCheckInScore=0,
                        riskLevel="Red",
                        reasons=["Confusion after fall"],
                    ),
                    recommendedAction="Call caregiver and coordinate urgent medical assessment.",
                    categories=[],
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2), encoding="utf-8")

                tasks = main._repair_missing_follow_up_tasks()

                self.assertEqual(len([task for task in tasks if task.sourceCallId == "call-closed"]), 1)
                self.assertEqual(tasks[0].status, "Closed")
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_task_repair_returns_in_memory_tasks_when_persist_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            call_root = Path(tmp) / "calls"
            call_dir = call_root / "call-unwritten"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = call_root
            try:
                state_root.mkdir(parents=True)
                call_dir.mkdir(parents=True)
                main.CHECKINS_STATE_PATH.write_text("[]", encoding="utf-8")
                main.TASKS_STATE_PATH.write_text("[]", encoding="utf-8")
                call = main.CallRecord(
                    id="call-unwritten",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Amber",
                    originalTranscript="Patient: I fell and feel dizzy.",
                    englishTranscript="Patient: I fell and feel dizzy.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I fell and feel dizzy.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=0,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=80,
                        missedCheckInScore=0,
                        riskLevel="Amber",
                        reasons=["Fall with dizziness"],
                    ),
                    recommendedAction="Notify caregiver and arrange same-day follow-up.",
                    categories=[],
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2), encoding="utf-8")

                with patch.object(main, "_save_tasks", side_effect=OSError("disk unavailable")):
                    response = TestClient(main.app).get("/volunteer-tasks")

                self.assertEqual(response.status_code, 200)
                task = next(item for item in response.json() if item.get("sourceCallId") == "call-unwritten")
                self.assertEqual(task["priority"], "Today")
                self.assertEqual(json.loads(main.TASKS_STATE_PATH.read_text(encoding="utf-8")), [])
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_record_missed_checkin_persists_session_task_and_schedule_attempt(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = Path(tmp) / "calls"
            try:
                client = TestClient(main.app)
                response = client.post(
                    "/checkins/missed",
                    json={
                        "seniorId": "s-002",
                        "scheduledAt": "2026-07-05T08:00:00+08:00",
                        "retryAt": "2026-07-05T08:20:00+08:00",
                        "attemptCount": 2,
                        "note": "Phone rang but nobody answered.",
                    },
                )

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                session = payload["session"]
                self.assertEqual(session["seniorId"], "s-002")
                self.assertEqual(session["status"], "Missed")
                self.assertEqual(session["riskLevel"], "Amber")
                self.assertEqual(session["riskAssessment"]["missedCheckInScore"], 100)
                self.assertTrue(any(category["id"] == "missed_checkin" for category in session["categories"]))
                self.assertTrue(any(step["id"] == "retry-call" and step["status"] == "Triggered" for step in session["escalationPlan"]))

                task = next(task for task in payload["tasks"] if task.get("sourceSessionId") == session["id"])
                self.assertEqual(task["seniorId"], "s-002")
                self.assertEqual(task["priority"], "Today")
                self.assertEqual(task["status"], "Open")

                stored = {checkin.id: checkin for checkin in main._load_checkins()}
                self.assertIn(session["id"], stored)

                now = main._parse_iso("2026-07-05T10:00:00+08:00")
                assert now is not None
                schedule = {item.seniorId: item for item in main._build_schedule_items(now)}
                self.assertEqual(schedule["s-002"].status, "Overdue")
                self.assertEqual(schedule["s-002"].lastAttemptStatus, "Missed")
                self.assertTrue(schedule["s-002"].lastAttemptAt.startswith("2026-07-05T08:00:00"))
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_start_and_complete_checkin_persist_state_schedule_and_task(self) -> None:
        with TemporaryDirectory() as tmp:
            original_state_root = main.STATE_STORAGE_ROOT
            original_checkins_path = main.CHECKINS_STATE_PATH
            original_tasks_path = main.TASKS_STATE_PATH
            original_call_root = main.CALL_STORAGE_ROOT
            state_root = Path(tmp) / "state"
            main.STATE_STORAGE_ROOT = state_root
            main.CHECKINS_STATE_PATH = state_root / "checkins.json"
            main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"
            main.CALL_STORAGE_ROOT = Path(tmp) / "calls"
            try:
                client = TestClient(main.app)
                started_response = client.post("/checkins/start", params={"senior_id": "s-002"})

                self.assertEqual(started_response.status_code, 200)
                started = started_response.json()
                self.assertTrue(started["id"].startswith("checkin-"))
                self.assertEqual(started["status"], "In progress")
                self.assertEqual(started["riskLevel"], "Green")

                started_at = main._parse_iso(started["scheduledAt"])
                assert started_at is not None
                schedule = {item.seniorId: item for item in main._build_schedule_items(started_at)}
                self.assertEqual(schedule["s-002"].lastAttemptStatus, "In progress")
                stale_schedule = {item.seniorId: item for item in main._build_schedule_items(started_at + main.ACTIVE_CHECKIN_ATTEMPT_WINDOW + timedelta(minutes=1))}
                self.assertEqual(stale_schedule["s-002"].status, "Due now")
                self.assertIsNone(stale_schedule["s-002"].lastAttemptAt)
                self.assertIsNone(stale_schedule["s-002"].lastAttemptStatus)
                started_record = {record.seniorId: record for record in main._build_senior_records()}["s-002"]
                self.assertEqual(started_record.totalRecords, 0)
                self.assertFalse(any(checkin["id"] == started["id"] for checkin in client.get("/checkins").json()))

                completed_response = client.post(
                    f"/checkins/{started['id']}/complete",
                    json={
                        "completedAt": "2026-07-05T10:30:00+08:00",
                        "originalTranscript": "I fell and hit my head. I am confused and my left hand is weak.",
                        "englishTranscript": "I fell and hit my head. I am confused and my left hand is weak.",
                    },
                )

                self.assertEqual(completed_response.status_code, 200)
                completed = completed_response.json()
                self.assertEqual(completed["status"], "Urgent")
                self.assertEqual(completed["riskLevel"], "Red")
                self.assertTrue(any(category["id"] == "concussion_danger" and category["severity"] == "Red" for category in completed["categories"]))
                self.assertTrue(any(step["id"] == "emergency-alert" and step["status"] == "Triggered" for step in completed["escalationPlan"]))

                tasks = main._load_tasks()
                task = next(task for task in tasks if task.sourceSessionId == completed["id"])
                self.assertEqual(task.priority, "Urgent")
                self.assertEqual(task.escalationStep, "emergency-alert")

                stored = {checkin.id: checkin for checkin in main._load_checkins()}
                self.assertEqual(stored[completed["id"]].status, "Urgent")
                self.assertEqual(stored[completed["id"]].completedAt, "2026-07-05T10:30:00+08:00")
                completed_record = {record.seniorId: record for record in main._build_senior_records()}["s-002"]
                self.assertEqual(completed_record.totalRecords, 1)
                self.assertEqual(completed_record.timeline[0].id, completed["id"])
                self.assertTrue(any(checkin["id"] == completed["id"] for checkin in client.get("/checkins").json()))

                schedule = {item.seniorId: item for item in main._build_schedule_items(main._parse_iso("2026-07-05T10:31:00+08:00"))}
                self.assertEqual(schedule["s-002"].lastAttemptStatus, "Urgent")
                self.assertEqual(schedule["s-002"].lastContactKind, "check-in")
            finally:
                main.STATE_STORAGE_ROOT = original_state_root
                main.CHECKINS_STATE_PATH = original_checkins_path
                main.TASKS_STATE_PATH = original_tasks_path
                main.CALL_STORAGE_ROOT = original_call_root

    def test_transcript_to_text_uses_patient_label(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="Are you okay?", timestamp=None),
            main.TranscriptMessage(role="Senior", text="I feel dizzy.", timestamp=None),
        ]

        self.assertEqual(main._transcript_to_text(messages), "Agent: Are you okay?\nPatient: I feel dizzy.")

    def test_meralion_segments_parse_choice_message_segments(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "I fell yesterday.",
                        "segments": [{"start": 1.2, "end": 3.4, "text": "I fell yesterday.", "speaker": "SPEAKER_00"}],
                    }
                }
            ]
        }
        segments = providers._parse_segments(payload)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].startTimeSeconds, 1.2)
        self.assertEqual(segments[0].speaker, "SPEAKER_00")

    def test_fallback_order_uses_elevenlabs_then_google_for_non_english(self) -> None:
        elevenlabs_result = ProviderResult(
            provider="elevenlabs-stt",
            language="Mandarin",
            transcript="我跌倒了",
            translation="我跌倒了",
            confidence=0.74,
            fallbackUsed=True,
            segments=[TranscriptSegment(text="我跌倒了", startTimeSeconds=2.0)],
        )
        google_result = ProviderResult(
            provider="google-translate",
            language="Mandarin",
            transcript="我跌倒了",
            translation="I fell down.",
            confidence=0.78,
            fallbackUsed=True,
        )

        with (
            patch.object(providers.MeralionProvider, "transcribe", side_effect=RuntimeError("meralion unavailable")),
            patch.object(providers.ElevenLabsSpeechToTextProvider, "transcribe", return_value=elevenlabs_result),
            patch.object(providers.GoogleTranslateProvider, "transcribe", return_value=google_result),
        ):
            result = providers.transcribe_with_fallback("Mandarin", "dialogue hint", Path("full-call.webm"))

        self.assertEqual(result.provider, "elevenlabs-stt+google-translate")
        self.assertEqual(result.translation, "I fell down.")
        self.assertEqual(result.segments[0].startTimeSeconds, 2.0)
        self.assertEqual(result.segments[0].englishText, "I fell down.")

    def test_meralion_is_attempted_before_fallbacks(self) -> None:
        calls: list[str] = []

        def meralion_fail(*args, **kwargs):
            calls.append("meralion")
            raise RuntimeError("meralion unavailable")

        def elevenlabs_success(*args, **kwargs):
            calls.append("elevenlabs")
            return ProviderResult(
                provider="elevenlabs-stt",
                language="English",
                transcript="I am okay.",
                translation="I am okay.",
                confidence=0.74,
                fallbackUsed=True,
            )

        with (
            patch.object(providers.MeralionProvider, "transcribe", side_effect=meralion_fail),
            patch.object(providers.ElevenLabsSpeechToTextProvider, "transcribe", side_effect=elevenlabs_success),
        ):
            result = providers.transcribe_with_fallback("English", "dialogue hint", Path("full-call.webm"))

        self.assertEqual(calls, ["meralion", "elevenlabs"])
        self.assertEqual(result.provider, "elevenlabs-stt")

    def test_speech_timing_estimate_uses_message_timestamps(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="How are you?", timestamp="2026-07-04T10:00:00+08:00"),
            main.TranscriptMessage(role="Senior", text="I am okay", timestamp="2026-07-04T10:00:02+08:00"),
            main.TranscriptMessage(role="Agent", text="Any falls?", timestamp="2026-07-04T10:00:05+08:00"),
            main.TranscriptMessage(role="Senior", text="I almost fell", timestamp="2026-07-04T10:00:09+08:00"),
        ]

        profile = main._estimate_current_speech_profile(messages, "2026-07-04T10:00:00+08:00", "2026-07-04T10:00:20+08:00")

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertGreater(profile.speechRate, 0)
        self.assertEqual(profile.responseLatencyMs, 3000)
        self.assertEqual(profile.avgPauseMs, 7000)

    def test_speech_timing_prefers_audio_segments(self) -> None:
        messages = [
            main.TranscriptMessage(role="Senior", text="I am okay", timestamp="2026-07-04T10:00:02+08:00"),
        ]
        segments = [
            TranscriptSegment(text="I am okay", originalText="I am okay", startTimeSeconds=1.0, endTimeSeconds=2.0),
            TranscriptSegment(text="I feel dizzy", originalText="I feel dizzy", startTimeSeconds=5.0, endTimeSeconds=6.0),
        ]

        profile = main._estimate_current_speech_profile(messages, "2026-07-04T10:00:00+08:00", "2026-07-04T10:00:20+08:00", segments)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertGreater(profile.speechRate, 0)
        self.assertEqual(profile.avgPauseMs, 3000)

    def test_sync_english_segments_updates_single_fallback_segment(self) -> None:
        segments = [TranscriptSegment(text="我跌倒了", originalText="我跌倒了", englishText="我跌倒了", startTimeSeconds=2.0)]

        synced = main._sync_english_segments(segments, "我跌倒了", "I fell down.")

        self.assertEqual(synced[0].englishText, "I fell down.")
        self.assertEqual(synced[0].startTimeSeconds, 2.0)

    def test_timed_segments_from_messages_adds_sentence_start_times(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="How are you?", timestamp="2026-07-04T10:00:00+08:00"),
            main.TranscriptMessage(role="Senior", text="I almost fell.", timestamp="2026-07-04T10:00:05+08:00"),
            main.TranscriptMessage(role="Agent", text="Any pain?", timestamp="2026-07-04T10:00:10+08:00"),
            main.TranscriptMessage(role="Senior", text="My head hurts.", timestamp="2026-07-04T10:00:15+08:00"),
        ]

        segments = main._timed_segments_from_messages(messages, "2026-07-04T10:00:00+08:00", "I almost fell. My head hurts.")

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].startTimeSeconds, 0)
        self.assertEqual(segments[1].startTimeSeconds, 15)

    def test_role_labeled_english_transcript_preserves_agent_patient_cues(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="How are you?", timestamp=None),
            main.TranscriptMessage(role="Senior", text="我差点跌倒。", timestamp=None),
        ]

        def translate(language: str, text: str) -> str:
            return {"How are you?": "How are you?", "我差点跌倒。": "I almost fell."}[text]

        with patch.object(main, "_translate_message_text", side_effect=translate):
            transcript = main._role_labeled_english_transcript(messages, "Mandarin", "How are you? I almost fell.")

        self.assertEqual(transcript, "Agent: How are you?\nPatient: I almost fell.")

    def test_role_segments_estimate_patient_start_before_transcript_event_time(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="How are you?", timestamp="2026-07-04T10:00:00+08:00"),
            main.TranscriptMessage(role="Senior", text="I almost fell while showering.", timestamp="2026-07-04T10:00:10+08:00"),
            main.TranscriptMessage(role="Agent", text="Are you hurt?", timestamp="2026-07-04T10:00:12+08:00"),
        ]

        segments = main._role_segments_from_messages(
            messages,
            "2026-07-04T10:00:00+08:00",
            "Agent: How are you?\nPatient: I almost fell while showering.\nAgent: Are you hurt?",
        )

        patient = segments[1]
        self.assertEqual(patient.role, "Patient")
        self.assertLess(patient.startTimeSeconds, 10)
        self.assertEqual(patient.endTimeSeconds, 10)

    def test_role_segments_agent_end_is_before_following_patient_start(self) -> None:
        messages = [
            main.TranscriptMessage(
                role="Agent",
                text="Do you have any headaches, dizziness, vomiting, blurred vision, drowsiness, confusion, weakness, numbness, or difficulty speaking?",
                timestamp="2026-07-04T10:00:30+08:00",
            ),
            main.TranscriptMessage(role="Senior", text="I feel a little dizzy right now.", timestamp="2026-07-04T10:00:45+08:00"),
        ]

        segments = main._role_segments_from_messages(
            messages,
            "2026-07-04T10:00:00+08:00",
            "Agent: Do you have any headaches, dizziness, vomiting, blurred vision, drowsiness, confusion, weakness, numbness, or difficulty speaking?\nPatient: I feel a little dizzy right now.",
        )

        self.assertEqual(segments[0].role, "Agent")
        self.assertEqual(segments[1].role, "Patient")
        self.assertIsNotNone(segments[0].endTimeSeconds)
        self.assertIsNotNone(segments[1].startTimeSeconds)
        assert segments[0].endTimeSeconds is not None
        assert segments[1].startTimeSeconds is not None
        self.assertLessEqual(segments[0].endTimeSeconds, segments[1].startTimeSeconds)

    def test_openai_risk_review_sends_patient_only_sentences(self) -> None:
        segments = [
            TranscriptSegment(text="Agent: Did you fall?", englishText="Agent: Did you fall?", role="Agent", speaker="Agent", startTimeSeconds=0),
            TranscriptSegment(text="Patient: I almost fell.", englishText="Patient: I almost fell.", role="Patient", speaker="Patient", startTimeSeconds=4),
        ]
        payload = {
            "output_text": json.dumps(
                {
                    "riskLevel": "Amber",
                    "summary": "Near fall.",
                    "recommendedAction": "Check in.",
                    "reasons": ["Patient reported near fall."],
                    "symptoms": {
                        "fall": True,
                        "headImpact": False,
                        "headache": False,
                        "dizziness": False,
                        "vomiting": False,
                        "confusion": False,
                        "slurredSpeech": False,
                        "weakness": False,
                        "poorIntake": False,
                        "asksForHelp": False,
                        "missedCheckIn": False,
                    },
                    "signals": [
                        {
                            "id": "1",
                            "label": "Near fall",
                            "severity": "Amber",
                            "quotedText": "I almost fell.",
                            "highlightText": "I almost fell.",
                            "reason": "Patient reported near fall.",
                            "sentenceIndex": 0,
                            "startTimeSeconds": None,
                            "endTimeSeconds": None,
                        }
                    ],
                }
            )
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload

        with patch.dict(main.os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(main.httpx, "post", return_value=response) as post:
            _, _, signals, _, fallback = main._openai_risk_review("Agent: Did you fall?\nPatient: I almost fell.", segments)

        request_json = post.call_args.kwargs["json"]
        user_payload = json.loads(request_json["input"][1]["content"])
        self.assertFalse(fallback)
        self.assertEqual(user_payload["patientOnlyTranscript"], "Patient: I almost fell.")
        self.assertEqual(len(user_payload["sentences"]), 1)
        self.assertEqual(signals[0].startTimeSeconds, 4)

    def test_attach_segment_timestamps_drops_agent_only_signal(self) -> None:
        signals = [
            main.RiskSignal(
                id="agent",
                label="Agent question",
                severity="Amber",
                quotedText="Did you eat or drink anything today?",
                highlightText="Did you eat or drink anything today?",
                reason="Agent asked about intake.",
                sentenceIndex=0,
            )
        ]
        segments = [
            TranscriptSegment(text="Agent: Did you eat or drink anything today?", englishText="Agent: Did you eat or drink anything today?", role="Agent", speaker="Agent", startTimeSeconds=10),
            TranscriptSegment(text="Patient: I ate chicken rice.", englishText="Patient: I ate chicken rice.", role="Patient", speaker="Patient", startTimeSeconds=20),
        ]

        attached = main._attach_segment_timestamps(signals, segments)

        self.assertEqual(attached, [])

    def test_enrich_call_record_adds_demo_speech_provenance_for_older_metadata(self) -> None:
        call = main.CallRecord(
            id="call-legacy",
            seniorId="s-001",
            seniorName="Mdm Tan Bee Hoon",
            startedAt="2026-07-04T10:00:00+08:00",
            completedAt="2026-07-04T10:05:00+08:00",
            status="Complete",
            riskLevel="Green",
            originalTranscript="Patient: I am okay.",
            englishTranscript="Patient: I am okay.",
            transcriptMessages=[TranscriptMessage(role="Senior", text="I am okay.", timestamp="2026-07-04T10:01:00+08:00")],
            translationProvider="test",
            translationFallbackUsed=False,
            audioAvailable=False,
            currentSpeechProfile=SpeechProfile(
                speechRate=120,
                avgPauseMs=500,
                responseLatencyMs=1000,
                pitchVariability=0.5,
                phraseAccuracy=0.95,
                updatedAt="2026-07-04T10:05:00+08:00",
            ),
            riskAssessment=RiskAssessment(
                speechDeviationScore=0,
                parkinsonsWatchScore=0,
                postFallConcernScore=0,
                missedCheckInScore=0,
                riskLevel="Green",
                reasons=["No concerning symptoms and speech remains close to baseline"],
            ),
            recommendedAction="Continue routine scheduled check-ins.",
        )

        enriched = main._enrich_call_record(call)

        self.assertIsNotNone(enriched.speechModelProvenance)
        assert enriched.speechModelProvenance is not None
        self.assertEqual(enriched.speechModelProvenance.runtimeMode, "demo metrics")
        self.assertFalse(enriched.speechModelProvenance.validated)

    def test_load_calls_quarantines_corrupt_metadata_and_returns_valid_records(self) -> None:
        with TemporaryDirectory() as tmp:
            original_storage_root = main.CALL_STORAGE_ROOT
            main.CALL_STORAGE_ROOT = Path(tmp)
            good_dir = main.CALL_STORAGE_ROOT / "call-good"
            bad_dir = main.CALL_STORAGE_ROOT / "call-bad"
            good_dir.mkdir(parents=True)
            bad_dir.mkdir(parents=True)
            try:
                call = main.CallRecord(
                    id="call-good",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Green",
                    originalTranscript="Patient: I am okay.",
                    englishTranscript="Patient: I am okay.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I am okay.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=0,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=0,
                        missedCheckInScore=0,
                        riskLevel="Green",
                        reasons=["No concerning symptoms and speech remains close to baseline"],
                    ),
                    recommendedAction="Continue routine scheduled check-ins.",
                )
                (good_dir / "metadata.json").write_text(call.model_dump_json(indent=2), encoding="utf-8")
                (bad_dir / "metadata.json").write_text("{broken json", encoding="utf-8")

                calls = main._load_calls()

                self.assertEqual([item.id for item in calls], ["call-good"])
                self.assertFalse((bad_dir / "metadata.json").exists())
                self.assertTrue(list(bad_dir.glob("metadata.json.corrupt-*")))
            finally:
                main.CALL_STORAGE_ROOT = original_storage_root

    def test_failed_call_save_removes_partial_audio_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            original_storage_root = main.CALL_STORAGE_ROOT
            main.CALL_STORAGE_ROOT = Path(tmp)
            try:
                client = TestClient(main.app, raise_server_exceptions=False)
                with patch.object(main, "transcribe_with_fallback", side_effect=RuntimeError("provider failed")):
                    response = client.post(
                        "/calls",
                        data={
                            "seniorId": "s-001",
                            "status": "Complete",
                            "startedAt": "2026-07-04T10:00:00+08:00",
                            "completedAt": "2026-07-04T10:05:00+08:00",
                            "transcriptMessages": json.dumps(
                                [
                                    {
                                        "role": "Senior",
                                        "text": "I am okay.",
                                        "timestamp": "2026-07-04T10:01:00+08:00",
                                    }
                                ]
                            ),
                            "agentAudioCaptured": "false",
                        },
                        files={"audio": ("full-call.webm", b"fake audio", "audio/webm")},
                    )

                self.assertEqual(response.status_code, 500)
                self.assertEqual(list(main.CALL_STORAGE_ROOT.glob("call-*")), [])
            finally:
                main.CALL_STORAGE_ROOT = original_storage_root

    def test_speech_enrichment_keeps_existing_metadata_when_atomic_replace_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            original_storage_root = main.CALL_STORAGE_ROOT
            main.CALL_STORAGE_ROOT = Path(tmp)
            call_dir = main.CALL_STORAGE_ROOT / "call-atomic"
            call_dir.mkdir(parents=True)
            metadata_path = call_dir / "metadata.json"
            try:
                call = main.CallRecord(
                    id="call-atomic",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Green",
                    originalTranscript="Patient: I am okay.",
                    englishTranscript="Patient: I am okay.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I am okay.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=0,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=0,
                        missedCheckInScore=0,
                        riskLevel="Green",
                        reasons=["No concerning symptoms and speech remains close to baseline"],
                    ),
                    recommendedAction="Continue routine scheduled check-ins.",
                )
                metadata_path.write_text(call.model_dump_json(indent=2), encoding="utf-8")
                request = main.SpeechEnrichmentRequest.model_validate(
                    {
                        "embedding": [0.1, 0.2, 0.3],
                        "speech_metrics": {
                            "speechRate": 118,
                            "avgPauseMs": 520,
                            "responseLatencyMs": 990,
                            "pitchVariability": 0.48,
                            "phraseAccuracy": 0.94,
                            "embedding": [0.1, 0.2, 0.3],
                            "updatedAt": "2026-07-04T10:06:00+08:00",
                        },
                        "provenance": {
                            "model": "demo",
                            "model_name": "demo-standard-library",
                            "extracted_at": "2026-07-04T10:06:00+08:00",
                        },
                    }
                )

                with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                    with self.assertRaises(OSError):
                        main.enrich_call_speech("call-atomic", request)

                stored = main._load_call_record(metadata_path)
                self.assertIsNone(stored.currentSpeechProfile)
                self.assertIsNone(stored.speechModelProvenance)
                self.assertEqual(list(call_dir.glob(".metadata.json.*.tmp")), [])
            finally:
                main.CALL_STORAGE_ROOT = original_storage_root

    def test_speech_enrichment_endpoint_stores_offline_embedding_without_changing_risk(self) -> None:
        with TemporaryDirectory() as tmp:
            original_storage_root = main.CALL_STORAGE_ROOT
            main.CALL_STORAGE_ROOT = Path(tmp)
            call_dir = main.CALL_STORAGE_ROOT / "call-enrich"
            call_dir.mkdir(parents=True)
            try:
                call = main.CallRecord(
                    id="call-enrich",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Green",
                    originalTranscript="Patient: I am okay.",
                    englishTranscript="Patient: I am okay.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I am okay.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    currentSpeechProfile=SpeechProfile(
                        speechRate=120,
                        avgPauseMs=500,
                        responseLatencyMs=1000,
                        pitchVariability=0.5,
                        phraseAccuracy=0.95,
                        updatedAt="2026-07-04T10:05:00+08:00",
                    ),
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=12,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=0,
                        missedCheckInScore=0,
                        riskLevel="Green",
                        reasons=["No concerning symptoms and speech remains close to baseline"],
                    ),
                    recommendedAction="Continue routine scheduled check-ins.",
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2))
                client = TestClient(main.app)

                response = client.patch(
                    "/calls/call-enrich/speech-enrichment",
                    json={
                        "embedding": [0.1, 0.2, 0.3],
                        "speech_metrics": {
                            "speechRate": 118,
                            "avgPauseMs": 520,
                            "responseLatencyMs": 990,
                            "pitchVariability": 0.48,
                            "phraseAccuracy": 0.94,
                            "embedding": [0.1, 0.2, 0.3],
                            "updatedAt": "2026-07-04T10:06:00+08:00",
                        },
                        "provenance": {
                            "model": "demo",
                            "model_name": "demo-standard-library",
                            "source_id": "sample-row-1",
                            "extracted_at": "2026-07-04T10:06:00+08:00",
                        },
                    },
                )

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["currentSpeechProfile"]["embedding"], [0.1, 0.2, 0.3])
                self.assertEqual(payload["speechModelProvenance"]["runtimeMode"], "offline embedding")
                self.assertEqual(payload["speechModelProvenance"]["modelName"], "demo-standard-library")
                self.assertFalse(payload["speechModelProvenance"]["validated"])
                self.assertEqual(payload["riskAssessment"]["speechDeviationScore"], 12)

                stored = main._load_call_record(call_dir / "metadata.json")
                self.assertEqual(stored.currentSpeechProfile.embedding, [0.1, 0.2, 0.3])
                self.assertEqual(stored.speechModelProvenance.runtimeMode, "offline embedding")
            finally:
                main.CALL_STORAGE_ROOT = original_storage_root

    def test_generated_speech_enrichment_payload_patches_call(self) -> None:
        with TemporaryDirectory() as tmp:
            original_storage_root = main.CALL_STORAGE_ROOT
            main.CALL_STORAGE_ROOT = Path(tmp) / "calls"
            call_dir = main.CALL_STORAGE_ROOT / "call-generated-payload"
            call_dir.mkdir(parents=True)
            try:
                call = main.CallRecord(
                    id="call-generated-payload",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Green",
                    originalTranscript="Patient: I am okay.",
                    englishTranscript="Patient: I am okay.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I am okay.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    currentSpeechProfile=SpeechProfile(
                        speechRate=120,
                        avgPauseMs=500,
                        responseLatencyMs=1000,
                        pitchVariability=0.5,
                        phraseAccuracy=0.95,
                        updatedAt="2026-07-04T10:05:00+08:00",
                    ),
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=12,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=0,
                        missedCheckInScore=0,
                        riskLevel="Green",
                        reasons=["No concerning symptoms and speech remains close to baseline"],
                    ),
                    recommendedAction="Continue routine scheduled check-ins.",
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2))

                rows_path = Path(tmp) / "embeddings.jsonl"
                payload_path = Path(tmp) / "payload.json"
                rows_path.write_text(
                    json.dumps(
                        {
                            "dataset": "sample",
                            "speaker_id": "s-001",
                            "label": "control",
                            "task": "repeat_phrase",
                            "embedding": [0.1, 0.2, 0.3],
                            "speech_metrics": {
                                "speechRate": 118,
                                "avgPauseMs": 520,
                                "responseLatencyMs": 990,
                                "pitchVariability": 0.48,
                                "phraseAccuracy": 0.94,
                            },
                            "provenance": {
                                "source_id": "sample-row-1",
                                "model": "demo",
                                "model_name": "demo-standard-library",
                                "extracted_at": "2026-07-04T10:06:00+08:00",
                            },
                        }
                    )
                    + "\n"
                )
                with redirect_stdout(StringIO()):
                    make_enrichment_payload.main(
                        ["--input", str(rows_path), "--output", str(payload_path), "--speaker-id", "s-001"]
                    )
                generated_payload = json.loads(payload_path.read_text())

                client = TestClient(main.app)
                response = client.patch("/calls/call-generated-payload/speech-enrichment", json=generated_payload)

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["currentSpeechProfile"]["embedding"], [0.1, 0.2, 0.3])
                self.assertEqual(payload["currentSpeechProfile"]["speechRate"], 118)
                self.assertEqual(payload["speechModelProvenance"]["runtimeMode"], "offline embedding")
                self.assertEqual(payload["speechModelProvenance"]["modelName"], "demo-standard-library")
                self.assertEqual(payload["speechModelProvenance"]["artifactUri"], f"{rows_path}#row=1")
                self.assertEqual(payload["speechModelProvenance"]["generatedAt"], "2026-07-04T10:06:00+08:00")
                self.assertFalse(payload["speechModelProvenance"]["validated"])
            finally:
                main.CALL_STORAGE_ROOT = original_storage_root

    def test_validated_speech_enrichment_requires_model_card_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            original_storage_root = main.CALL_STORAGE_ROOT
            main.CALL_STORAGE_ROOT = Path(tmp)
            call_dir = main.CALL_STORAGE_ROOT / "call-validated-gate"
            call_dir.mkdir(parents=True)
            try:
                call = main.CallRecord(
                    id="call-validated-gate",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Green",
                    originalTranscript="Patient: I am okay.",
                    englishTranscript="Patient: I am okay.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I am okay.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    currentSpeechProfile=SpeechProfile(
                        speechRate=120,
                        avgPauseMs=500,
                        responseLatencyMs=1000,
                        pitchVariability=0.5,
                        phraseAccuracy=0.95,
                        updatedAt="2026-07-04T10:05:00+08:00",
                    ),
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=12,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=0,
                        missedCheckInScore=0,
                        riskLevel="Green",
                        reasons=["No concerning symptoms and speech remains close to baseline"],
                    ),
                    recommendedAction="Continue routine scheduled check-ins.",
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2))
                client = TestClient(main.app)

                response = client.patch(
                    "/calls/call-validated-gate/speech-enrichment",
                    json={
                        "runtimeMode": "validated model",
                        "modelName": "speech-watch-v1",
                        "modelVersion": "1.0.0",
                        "featureExtractor": "MERaLiON SpeechEncoder",
                        "artifactUri": "research/artifacts/model-card-speech-watch-v1.md",
                        "embedding": [0.1, 0.2, 0.3],
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertIn("modelCard", response.json()["detail"])
            finally:
                main.CALL_STORAGE_ROOT = original_storage_root

    def test_validated_speech_enrichment_stores_model_card_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            original_storage_root = main.CALL_STORAGE_ROOT
            main.CALL_STORAGE_ROOT = Path(tmp)
            call_dir = main.CALL_STORAGE_ROOT / "call-validated"
            call_dir.mkdir(parents=True)
            try:
                call = main.CallRecord(
                    id="call-validated",
                    seniorId="s-001",
                    seniorName="Mdm Tan Bee Hoon",
                    startedAt="2026-07-04T10:00:00+08:00",
                    completedAt="2026-07-04T10:05:00+08:00",
                    status="Complete",
                    riskLevel="Green",
                    originalTranscript="Patient: I am okay.",
                    englishTranscript="Patient: I am okay.",
                    transcriptMessages=[TranscriptMessage(role="Senior", text="I am okay.", timestamp="2026-07-04T10:01:00+08:00")],
                    translationProvider="test",
                    translationFallbackUsed=False,
                    audioAvailable=False,
                    currentSpeechProfile=SpeechProfile(
                        speechRate=120,
                        avgPauseMs=500,
                        responseLatencyMs=1000,
                        pitchVariability=0.5,
                        phraseAccuracy=0.95,
                        updatedAt="2026-07-04T10:05:00+08:00",
                    ),
                    riskAssessment=RiskAssessment(
                        speechDeviationScore=12,
                        parkinsonsWatchScore=0,
                        postFallConcernScore=0,
                        missedCheckInScore=0,
                        riskLevel="Green",
                        reasons=["No concerning symptoms and speech remains close to baseline"],
                    ),
                    recommendedAction="Continue routine scheduled check-ins.",
                )
                (call_dir / "metadata.json").write_text(call.model_dump_json(indent=2))
                client = TestClient(main.app)

                response = client.patch(
                    "/calls/call-validated/speech-enrichment",
                    json={
                        "runtimeMode": "validated model",
                        "modelName": "speech-watch-v1",
                        "modelVersion": "1.0.0",
                        "featureExtractor": "MERaLiON SpeechEncoder",
                        "artifactUri": "research/artifacts/model-card-speech-watch-v1.md",
                        "embedding": [0.1, 0.2, 0.3],
                        "modelCard": {
                            "datasetAccessReviewed": True,
                            "speakerSplitVerified": True,
                            "evaluationMetricsRecorded": True,
                            "subgroupChecksReviewed": True,
                            "failureModesDocumented": True,
                            "uiCopyReviewed": True,
                            "humanFollowUpActionDefined": True,
                            "rollbackPathDocumented": True,
                            "humanFollowUpAction": "Schedule caregiver review if the speech watch pattern persists.",
                        },
                    },
                )

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["speechModelProvenance"]["runtimeMode"], "validated model")
                self.assertTrue(payload["speechModelProvenance"]["validated"])
                self.assertEqual(
                    payload["speechModelProvenance"]["modelCard"]["humanFollowUpAction"],
                    "Schedule caregiver review if the speech watch pattern persists.",
                )
            finally:
                main.CALL_STORAGE_ROOT = original_storage_root

    def test_model_card_gate_rejects_blocked_diagnosis_copy(self) -> None:
        model_card = main.SpeechModelCardGate(
            datasetAccessReviewed=True,
            speakerSplitVerified=True,
            evaluationMetricsRecorded=True,
            subgroupChecksReviewed=True,
            failureModesDocumented=True,
            uiCopyReviewed=True,
            humanFollowUpActionDefined=True,
            rollbackPathDocumented=True,
            humanFollowUpAction="Tell the caregiver Parkinson's detected.",
        )

        with self.assertRaises(main.HTTPException) as context:
            main._validate_model_card_gate(model_card)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("blocked diagnosis language", context.exception.detail)

    def test_speech_deviation_response_includes_human_follow_up_action(self) -> None:
        client = TestClient(main.app)
        response = client.post(
            "/ml/speech-deviation",
            json={
                "seniorId": "s-001",
                "currentSpeechProfile": {
                    "speechRate": 120,
                    "avgPauseMs": 600,
                    "responseLatencyMs": 1000,
                    "pitchVariability": 0.6,
                    "phraseAccuracy": 0.95,
                },
                "symptoms": {},
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("humanFollowUpAction", payload)
        self.assertEqual(payload["safetyLabel"], "decision support, not diagnosis")


if __name__ == "__main__":
    unittest.main()
