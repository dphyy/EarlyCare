import unittest
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import main, providers
from app.models import ProviderResult, RiskAssessment, SpeechProfile, TranscriptMessage, TranscriptSegment


class CallWorkflowTests(unittest.TestCase):
    def test_clean_transcript_text_removes_bracket_cues(self) -> None:
        self.assertEqual(providers.clean_transcript_text("Agent: [happy] hello [concerned] there"), "Agent: hello there")

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
