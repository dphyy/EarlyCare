import unittest
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import main, providers
from app.models import ProviderResult, TranscriptSegment


class CallWorkflowTests(unittest.TestCase):
    def test_clean_transcript_text_removes_bracket_cues(self) -> None:
        self.assertEqual(providers.clean_transcript_text("Agent: [happy] hello [concerned] there"), "Agent: hello there")

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

    def test_meralion_asr_sends_data_url_without_boundary_mode(self) -> None:
        asr_response = Mock()
        asr_response.raise_for_status.return_value = None
        asr_response.json.return_value = {"choices": [{"message": {"content": "I am okay."}}]}

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "full-call.wav"
            audio_path.write_bytes(b"audio bytes")
            with patch.dict(providers.os.environ, {"MERALION_API_KEY": "test-key"}), patch.object(providers.httpx, "post", return_value=asr_response) as post:
                result = providers.MeralionProvider().transcribe("English", "fallback hint", audio_path)

        request_json = post.call_args.kwargs["json"]
        self.assertEqual(result.provider, "meralion")
        self.assertTrue(request_json["audio_url"].startswith("data:audio/wav;base64,"))
        self.assertEqual(request_json["return_diarization"], True)
        self.assertNotIn("boundary_mode", request_json)
        self.assertNotIn("return_timestamps", request_json)

    def test_meralion_translation_sends_target_language_only(self) -> None:
        asr_response = Mock()
        asr_response.raise_for_status.return_value = None
        asr_response.json.return_value = {"choices": [{"message": {"content": "我跌倒了"}}]}
        translation_response = Mock()
        translation_response.raise_for_status.return_value = None
        translation_response.json.return_value = {"choices": [{"message": {"content": "I fell down."}}]}

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "full-call.wav"
            audio_path.write_bytes(b"audio bytes")
            with patch.dict(providers.os.environ, {"MERALION_API_KEY": "test-key"}), patch.object(
                providers.httpx, "post", side_effect=[asr_response, translation_response]
            ) as post:
                result = providers.MeralionProvider().transcribe("Mandarin", "fallback hint", audio_path)

        translation_json = post.call_args_list[1].kwargs["json"]
        self.assertEqual(result.provider, "meralion-audio-translation")
        self.assertTrue(translation_json["audio_url"].startswith("data:audio/wav;base64,"))
        self.assertEqual(translation_json["translation_params"], {"target_language": "English"})
        self.assertNotIn("source_language", translation_json["translation_params"])

    def test_save_call_prefers_meralion_audio_transcripts_for_patient_overview(self) -> None:
        messages = [
            {"role": "Agent", "text": "Live agent text", "timestamp": "2026-07-04T10:00:00+08:00"},
            {"role": "Senior", "text": "Live senior text", "timestamp": "2026-07-04T10:00:05+08:00"},
        ]
        meralion_result = ProviderResult(
            provider="meralion-audio-translation",
            language="Mandarin",
            transcript="MERaLiON original transcript.",
            translation="MERaLiON English transcript.",
            confidence=0.86,
            fallbackUsed=False,
            segments=[TranscriptSegment(text="MERaLiON original transcript.", originalText="MERaLiON original transcript.", englishText="MERaLiON English transcript.")],
        )

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)),
                patch.object(main, "transcribe_with_fallback", return_value=meralion_result) as transcribe,
                patch.object(main, "_openai_risk_review", return_value=main._manual_risk_review()),
            ):
                client = TestClient(main.app)
                response = client.post(
                    "/calls",
                    data={
                        "seniorId": "s-001",
                        "status": "Complete",
                        "startedAt": "2026-07-04T10:00:00+08:00",
                        "completedAt": "2026-07-04T10:01:00+08:00",
                        "transcriptMessages": json.dumps(messages),
                        "agentAudioCaptured": "true",
                    },
                    files={"audio": ("full-call.wav", b"audio bytes", "audio/wav")},
                )

        self.assertEqual(response.status_code, 200)
        call = response.json()["call"]
        transcribe.assert_called_once()
        self.assertEqual(call["originalTranscript"], "Agent: MERaLiON original transcript.\nPatient: MERaLiON original transcript.")
        self.assertEqual(call["englishTranscript"], "MERaLiON English transcript.")
        self.assertEqual(call["transcriptMessages"][0]["text"], "Live agent text")
        self.assertEqual(call["translationProvider"], "meralion-audio-translation")
        self.assertFalse(call["translationFallbackUsed"])
        self.assertTrue(call["audioFilePath"].endswith("full-call.wav"))

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

    def test_role_labeled_original_transcript_replaces_meralion_speaker_tags(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="How are you?", timestamp=None),
            main.TranscriptMessage(role="Senior", text="不错。", timestamp=None),
        ]

        transcript = main._role_labeled_original_transcript(messages, "<Speaker1>: 您今天感觉怎么样？ <Speaker2>: 今天不错。")

        self.assertEqual(transcript, "Agent: 您今天感觉怎么样？\nPatient: 今天不错。")
        self.assertNotIn("<Speaker", transcript)

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


if __name__ == "__main__":
    unittest.main()
