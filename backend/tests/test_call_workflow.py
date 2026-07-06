import unittest
from pathlib import Path
import json
import math
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import wave

import numpy as np

from fastapi.testclient import TestClient

from app import main, providers
from app.models import ProviderResult, TranscriptSegment, TranscriptionAttempt


def _test_wav_bytes(seconds: float = 4.0, speech_ranges: list[tuple[float, float]] | None = None, sample_rate: int = 16_000) -> bytes:
    sample_count = int(seconds * sample_rate)
    samples = np.zeros(sample_count, dtype=np.float32)
    for start_seconds, end_seconds in speech_ranges or [(0, seconds)]:
        start = int(start_seconds * sample_rate)
        end = min(sample_count, int(end_seconds * sample_rate))
        for index in range(start, end):
            samples[index] = 0.25 * math.sin(2 * math.pi * 180 * index / sample_rate)
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "test.wav"
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((samples * 32767).astype("<i2").tobytes())
        return path.read_bytes()


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def _read_wav_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0, sample_rate


def _longest_silent_run_seconds(samples: np.ndarray, sample_rate: int, threshold: float = 0.01) -> float:
    silent = np.abs(samples) < threshold
    longest = 0
    current = 0
    for is_silent in silent:
        if bool(is_silent):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / sample_rate


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
        self.assertEqual([attempt.provider for attempt in result.attempts], ["meralion", "elevenlabs-stt", "google-translate"])
        self.assertEqual([attempt.status for attempt in result.attempts], ["failed", "success", "success"])

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
        self.assertEqual([attempt.provider for attempt in result.attempts], ["meralion", "elevenlabs-stt"])
        self.assertEqual([attempt.status for attempt in result.attempts], ["failed", "success"])

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

    def test_save_call_preserves_live_roles_when_provider_lacks_speaker_roles(self) -> None:
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
                        "elevenLabsConversationId": "conv-123",
                        "agentAudioCaptured": "true",
                    },
                    files={"audio": ("full-call.wav", b"audio bytes", "audio/wav")},
                )

        self.assertEqual(response.status_code, 200)
        call = response.json()["call"]
        transcribe.assert_called_once()
        self.assertEqual(call["originalTranscript"], "Agent: Live agent text\nPatient: Live senior text")
        self.assertEqual(call["englishTranscript"], "Agent: Live agent text\nPatient: Live senior text")
        self.assertEqual(call["elevenLabsConversationId"], "conv-123")
        self.assertEqual(call["transcriptMessages"][0]["text"], "Live agent text")
        self.assertEqual(call["translationProvider"], "meralion-audio-translation")
        self.assertFalse(call["translationFallbackUsed"])
        self.assertEqual(call["transcriptionAttempts"], [])
        self.assertTrue(call["transcriptAlignmentWarnings"])
        self.assertEqual(call["aiRiskFailureReason"], None)
        self.assertTrue(call["audioFilePath"].endswith("full-call.wav"))
        self.assertFalse(call["patientAudioAvailable"])
        self.assertIsNone(call["speechModelProbability"])

    def test_save_call_uses_live_role_segments_when_provider_speakers_are_generic(self) -> None:
        messages = [
            {"role": "Agent", "text": "This is your routine well-being check-in.", "timestamp": "2026-07-04T10:00:00+08:00"},
            {"role": "Senior", "text": "Yes.", "timestamp": "2026-07-04T10:00:04+08:00"},
            {"role": "Agent", "text": "How are you feeling?", "timestamp": "2026-07-04T10:00:08+08:00"},
            {"role": "Senior", "text": "I feel okay.", "timestamp": "2026-07-04T10:00:11+08:00"},
        ]
        provider_result = ProviderResult(
            provider="meralion-audio-translation",
            language="English",
            transcript="This is your routine well-being check-in. Yes. How are you feeling? I feel okay.",
            translation="This is your routine well-being check-in. Yes. How are you feeling? I feel okay.",
            confidence=0.86,
            fallbackUsed=False,
            segments=[
                TranscriptSegment(text="This is your routine well-being check-in.", speaker="SPEAKER_00", startTimeSeconds=0, endTimeSeconds=3),
                TranscriptSegment(text="Yes.", speaker="SPEAKER_01", startTimeSeconds=4, endTimeSeconds=5),
            ],
        )

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)),
                patch.object(main, "transcribe_with_fallback", return_value=provider_result),
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
                    },
                    files={"audio": ("full-call.wav", _test_wav_bytes(2), "audio/wav")},
                )

        call = response.json()["call"]
        self.assertEqual(response.status_code, 200)
        self.assertIn("Agent: This is your routine well-being check-in.", call["englishTranscript"])
        self.assertIn("Patient: Yes.", call["englishTranscript"])
        self.assertEqual(call["transcriptSegments"][0]["role"], "Agent")
        self.assertEqual(call["transcriptSegments"][1]["role"], "Patient")
        self.assertNotEqual(call["transcriptSegments"][0]["speaker"], "SPEAKER_00")
        self.assertTrue(any("MERaLiON speaker labels ignored" in warning for warning in call["transcriptAlignmentWarnings"]))

    def test_save_call_ignores_swapped_provider_role_labels_when_live_roles_exist(self) -> None:
        messages = [
            {"role": "Agent", "text": "Did you fall?", "timestamp": "2026-07-04T10:00:00+08:00"},
            {"role": "Senior", "text": "I fell in the bathroom.", "timestamp": "2026-07-04T10:00:04+08:00"},
        ]
        provider_result = ProviderResult(
            provider="meralion-audio-translation",
            language="English",
            transcript="<Speaker1>: Did you fall? <Speaker2>: I fell in the bathroom.",
            translation="Patient: Did you fall?\nAgent: I fell in the bathroom.",
            confidence=0.86,
            fallbackUsed=False,
            segments=[
                TranscriptSegment(text="Did you fall?", role="Patient", speaker="SPEAKER_00", startTimeSeconds=0, endTimeSeconds=2),
                TranscriptSegment(text="I fell in the bathroom.", role="Agent", speaker="SPEAKER_01", startTimeSeconds=3, endTimeSeconds=5),
            ],
        )

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)),
                patch.object(main, "transcribe_with_fallback", return_value=provider_result),
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
                    },
                    files={"audio": ("full-call.wav", _test_wav_bytes(2), "audio/wav")},
                )

        self.assertEqual(response.status_code, 200)
        call = response.json()["call"]
        self.assertEqual(call["originalTranscript"], "Agent: Did you fall?\nPatient: I fell in the bathroom.")
        self.assertEqual(call["englishTranscript"], "Agent: Did you fall?\nPatient: I fell in the bathroom.")
        self.assertEqual([segment["role"] for segment in call["transcriptSegments"]], ["Agent", "Patient"])
        self.assertEqual([segment["speaker"] for segment in call["transcriptSegments"]], ["Agent", "Patient"])
        self.assertTrue(any("MERaLiON speaker labels ignored" in warning for warning in call["transcriptAlignmentWarnings"]))

    def test_dialogue_fallback_preserves_live_roles_without_duplicate_prefixes(self) -> None:
        messages = [
            {"role": "Agent", "text": "Agent: Hello, this is EarlyCare. This is your routine well-being check-in.", "timestamp": "2026-07-04T10:00:00+08:00"},
            {"role": "Senior", "text": "Patient: Yes.", "timestamp": "2026-07-04T10:00:05+08:00"},
            {"role": "Agent", "text": "How are you feeling today?", "timestamp": "2026-07-04T10:00:08+08:00"},
            {"role": "Senior", "text": "I am feeling okay.", "timestamp": "2026-07-04T10:00:12+08:00"},
        ]
        provider_result = ProviderResult(
            provider="dialogue-transcript",
            language="English",
            transcript="Agent: Hello, this is EarlyCare. This is your routine well-being check-in.\nPatient: Yes.\nAgent: How are you feeling today?\nPatient: I am feeling okay.",
            translation="Agent: Hello, this is EarlyCare. This is your routine well-being check-in.\nPatient: Yes.\nAgent: How are you feeling today?\nPatient: I am feeling okay.",
            confidence=0.55,
            fallbackUsed=True,
            attempts=[
                TranscriptionAttempt(provider="meralion", status="failed", reason="MERaLiON unavailable"),
                TranscriptionAttempt(provider="elevenlabs-stt", status="failed", reason="ElevenLabs unavailable"),
                TranscriptionAttempt(provider="google-translate", status="skipped", reason="Google Translate fallback skipped for English transcript"),
                TranscriptionAttempt(provider="dialogue-transcript", status="success"),
            ],
        )

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)),
                patch.object(main, "transcribe_with_fallback", return_value=provider_result),
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
                    },
                    files={"audio": ("full-call.wav", _test_wav_bytes(2), "audio/wav")},
                )

        call = response.json()["call"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            call["englishTranscript"],
            "Agent: Hello, this is EarlyCare. This is your routine well-being check-in.\n"
            "Patient: Yes.\n"
            "Agent: How are you feeling today?\n"
            "Patient: I am feeling okay.",
        )
        combined_segments = "\n".join(segment["englishText"] for segment in call["transcriptSegments"])
        self.assertNotIn("Agent: Agent:", call["englishTranscript"])
        self.assertNotIn("Agent: Patient:", combined_segments)
        self.assertEqual(call["transcriptSegments"][0]["role"], "Agent")
        self.assertEqual(call["transcriptSegments"][1]["role"], "Patient")
        self.assertEqual(call["transcriptionAttempts"][0]["provider"], "meralion")
        self.assertEqual(call["transcriptionAttempts"][-1]["status"], "success")

    def test_save_call_stores_patient_only_audio_when_uploaded(self) -> None:
        messages = [
            {"role": "Agent", "text": "How are you?", "timestamp": "2026-07-04T10:00:00+08:00"},
            {"role": "Senior", "text": "I am okay.", "timestamp": "2026-07-04T10:00:02+08:00"},
        ]
        provider_result = ProviderResult(
            provider="fallback",
            language="English",
            transcript="Agent: How are you?\nPatient: I am okay.",
            translation="Agent: How are you?\nPatient: I am okay.",
            confidence=0.5,
            fallbackUsed=True,
        )

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)),
                patch.object(main, "transcribe_with_fallback", return_value=provider_result),
                patch.object(main, "_openai_risk_review", return_value=main._manual_risk_review()),
                patch.dict(main.os.environ, {"EARLYCARE_SPEECH_MODEL_ENABLED": "false"}),
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
                    },
                    files={
                        "audio": ("full-call.wav", _test_wav_bytes(2), "audio/wav"),
                        "patientAudio": ("patient-audio.wav", _test_wav_bytes(4, [(1.2, 2.3)]), "audio/wav"),
                    },
                )
                call = response.json()["call"]
                audio_response = client.get(call["audioUrl"])
                patient_audio_response = client.get(call["patientAudioUrl"])
                patient_speech_response = client.get(call["patientSpeechAudioUrl"])
                patient_speech_duration = _wav_duration(Path(call["patientSpeechAudioFilePath"]))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(call["audioAvailable"])
        self.assertTrue(call["patientAudioAvailable"])
        self.assertTrue(call["patientSpeechAudioAvailable"])
        self.assertTrue(call["patientAudioFilePath"].endswith("patient-audio.wav"))
        self.assertTrue(call["patientSpeechAudioFilePath"].endswith("patient-speech.wav"))
        self.assertLess(patient_speech_duration, 2.0)
        self.assertGreater(patient_speech_duration, 0.5)
        self.assertEqual(call["speechModelWarnings"], [])
        self.assertTrue(audio_response.content)
        self.assertTrue(patient_audio_response.content)
        self.assertTrue(patient_speech_response.content)

    def test_patient_audio_endpoint_returns_404_when_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)):
                client = TestClient(main.app)
                response = client.get("/calls/missing/patient-audio")

        self.assertEqual(response.status_code, 404)

    def test_speech_model_receives_derived_patient_speech_audio(self) -> None:
        messages = [
            {"role": "Agent", "text": "How are you?", "timestamp": "2026-07-04T10:00:00+08:00"},
            {"role": "Senior", "text": "I am okay.", "timestamp": "2026-07-04T10:00:02+08:00"},
        ]
        provider_result = ProviderResult(
            provider="fallback",
            language="English",
            transcript="Agent: How are you?\nPatient: I am okay.",
            translation="Agent: How are you?\nPatient: I am okay.",
            confidence=0.5,
            fallbackUsed=True,
        )
        reviewed_paths: list[Path | None] = []

        def capture_speech_model_path(audio_path: Path | None):
            reviewed_paths.append(audio_path)
            return "test-model", 0.42, [], {"speechModelUsable": "true"}

        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)),
                patch.object(main, "transcribe_with_fallback", return_value=provider_result),
                patch.object(main, "_openai_risk_review", return_value=main._manual_risk_review()),
                patch.object(main, "_speech_model_review", side_effect=capture_speech_model_path),
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
                    },
                    files={
                        "audio": ("full-call.wav", _test_wav_bytes(2), "audio/wav"),
                        "patientAudio": ("patient-audio.wav", _test_wav_bytes(4, [(1.2, 2.3)]), "audio/wav"),
                    },
                )

        call = response.json()["call"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(call["speechModelProbability"], 0.42)
        self.assertEqual(len(reviewed_paths), 1)
        self.assertIsNotNone(reviewed_paths[0])
        assert reviewed_paths[0] is not None
        self.assertTrue(reviewed_paths[0].name.endswith("patient-speech.wav"))

    def test_patient_speech_audio_endpoint_returns_404_when_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch.object(main, "CALL_STORAGE_ROOT", Path(temp_dir)):
                client = TestClient(main.app)
                response = client.get("/calls/missing/patient-speech-audio")

        self.assertEqual(response.status_code, 404)

    def test_patient_speech_extraction_uses_agent_bounded_voiced_windows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patient_audio = root / "patient-audio.wav"
            patient_audio.write_bytes(_test_wav_bytes(8, [(3.0, 4.0), (6.0, 6.7)]))
            output_path = root / "patient-speech.wav"
            segments = [
                TranscriptSegment(text="Agent: question", role="Agent", startTimeSeconds=0.0, endTimeSeconds=2.0),
                TranscriptSegment(text="Patient: answer", role="Patient", startTimeSeconds=2.8, endTimeSeconds=4.2),
                TranscriptSegment(text="Agent: question", role="Agent", startTimeSeconds=4.3, endTimeSeconds=5.5),
                TranscriptSegment(text="Patient: yes", role="Patient", startTimeSeconds=5.8, endTimeSeconds=6.9),
            ]

            speech_path, warnings, summary = main._build_patient_speech_audio(patient_audio, segments, output_path)
            speech_duration = _wav_duration(output_path)
            speech_samples, sample_rate = _read_wav_float(output_path)

        self.assertEqual(speech_path, output_path)
        self.assertFalse(warnings)
        self.assertLess(summary["patientSpeechDurationSeconds"], summary["rawPatientAudioDurationSeconds"])
        self.assertLess(speech_duration, 2.2)
        self.assertGreater(speech_duration, 1.4)
        edge_samples = int(sample_rate * 0.12)
        self.assertGreater(float(np.max(np.abs(speech_samples[:edge_samples]))), 0.05)
        self.assertGreater(float(np.max(np.abs(speech_samples[-edge_samples:]))), 0.05)
        self.assertLess(_longest_silent_run_seconds(speech_samples, sample_rate), 0.18)
        self.assertEqual(summary["patientSpeechExtractionMode"], "agent-window-vad")
        self.assertEqual(summary["patientSpeechTurnCount"], 2)
        self.assertEqual(summary["patientSpeechWindowCount"], 2)
        self.assertEqual(summary["patientSpeechVoicedClipCount"], 2)

    def test_patient_speech_extraction_falls_back_to_patient_segment_vad(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patient_audio = root / "patient-audio.wav"
            patient_audio.write_bytes(_test_wav_bytes(5, [(2.0, 2.9)]))
            output_path = root / "patient-speech.wav"
            segments = [
                TranscriptSegment(text="Patient: answer", role="Patient", startTimeSeconds=0.5, endTimeSeconds=4.5),
            ]

            speech_path, warnings, summary = main._build_patient_speech_audio(patient_audio, segments, output_path)
            speech_duration = _wav_duration(output_path)

        self.assertEqual(speech_path, output_path)
        self.assertIn("patient segment VAD", " ".join(warnings))
        self.assertLess(speech_duration, 1.3)
        self.assertGreater(speech_duration, 0.7)
        self.assertEqual(summary["patientSpeechExtractionMode"], "patient-segment-vad")
        self.assertEqual(summary["patientSpeechWindowCount"], 1)
        self.assertEqual(summary["patientSpeechVoicedClipCount"], 1)

    def test_patient_speech_extraction_caps_estimated_agent_end_before_next_turn(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patient_audio = root / "patient-audio.wav"
            patient_audio.write_bytes(_test_wav_bytes(4, [(1.2, 1.8)]))
            output_path = root / "patient-speech.wav"
            segments = [
                TranscriptSegment(
                    text="Agent: this is a very long question with many words that would otherwise overrun the answer",
                    role="Agent",
                    startTimeSeconds=0.0,
                ),
                TranscriptSegment(text="Patient: yes", role="Patient", startTimeSeconds=1.1, endTimeSeconds=2.0),
                TranscriptSegment(text="Agent: next question", role="Agent", startTimeSeconds=3.0, endTimeSeconds=3.5),
            ]

            speech_path, warnings, summary = main._build_patient_speech_audio(patient_audio, segments, output_path)
            speech_duration = _wav_duration(output_path)

        self.assertEqual(speech_path, output_path)
        self.assertFalse(warnings)
        self.assertGreater(speech_duration, 0.4)
        self.assertLess(speech_duration, 0.9)
        self.assertEqual(summary["patientSpeechExtractionMode"], "agent-window-vad")

    def test_patient_speech_extraction_falls_back_to_full_audio_vad_without_timings(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patient_audio = root / "patient-audio.wav"
            patient_audio.write_bytes(_test_wav_bytes(4, [(1.0, 1.8)]))
            output_path = root / "patient-speech.wav"

            speech_path, warnings, summary = main._build_patient_speech_audio(patient_audio, [], output_path)
            speech_duration = _wav_duration(output_path)

        self.assertEqual(speech_path, output_path)
        self.assertIn("full-audio VAD", " ".join(warnings))
        self.assertLess(speech_duration, 1.2)
        self.assertGreater(speech_duration, 0.6)
        self.assertEqual(summary["patientSpeechExtractionMode"], "full-audio-vad")
        self.assertEqual(summary["patientSpeechWindowCount"], 1)
        self.assertEqual(summary["patientSpeechVoicedClipCount"], 1)

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

    def test_role_labeled_english_transcript_does_not_map_mismatched_sentence_counts(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="你好吗？", timestamp=None),
            main.TranscriptMessage(role="Senior", text="我头痛。", timestamp=None),
            main.TranscriptMessage(role="Agent", text="你跌倒了吗？", timestamp=None),
            main.TranscriptMessage(role="Senior", text="没有。", timestamp=None),
        ]
        warnings: list[str] = []

        with patch.object(main, "_translate_message_text", side_effect=lambda language, text: text):
            transcript = main._role_labeled_english_transcript(
                messages,
                "Mandarin",
                "How are you? I have a headache. Did you fall?",
                warnings,
            )

        self.assertEqual(transcript, "Agent: 你好吗？\nPatient: 我头痛。\nAgent: 你跌倒了吗？\nPatient: 没有。")
        self.assertTrue(any("sentence count did not match" in warning for warning in warnings))

    def test_role_labeled_original_transcript_replaces_meralion_speaker_tags(self) -> None:
        messages = [
            main.TranscriptMessage(role="Agent", text="How are you?", timestamp=None),
            main.TranscriptMessage(role="Senior", text="不错。", timestamp=None),
        ]

        transcript = main._role_labeled_original_transcript(messages, "<Speaker1>: 您今天感觉怎么样？ <Speaker2>: 今天不错。")

        self.assertEqual(transcript, "Agent: How are you?\nPatient: 不错。")
        self.assertNotIn("<Speaker", transcript)

    def test_role_labeled_original_transcript_uses_provider_speakers_without_live_messages(self) -> None:
        transcript = main._role_labeled_original_transcript(
            [],
            "<Speaker1>: 您今天感觉怎么样？ <Speaker2>: 今天不错。",
        )

        self.assertEqual(transcript, "Speaker1: 您今天感觉怎么样？\nSpeaker2: 今天不错。")

    def test_role_labeled_original_transcript_uses_live_roles_without_speaker_tags(self) -> None:
        messages = [
            main.TranscriptMessage(
                role="Agent",
                text="Hello, this is Early Care. This is your routine well-being check-in. Is now a good time to continue?",
                timestamp=None,
            ),
            main.TranscriptMessage(role="Senior", text="Yes.", timestamp=None),
        ]

        transcript = main._role_labeled_original_transcript(
            messages,
            "Hello, this is Early Care. This is your routine well-being check-in. Is now a good time to continue? Yes.",
        )

        self.assertIn("Agent: Hello, this is Early Care. This is your routine well-being check-in. Is now a good time to continue?", transcript)
        self.assertIn("Patient: Yes.", transcript)
        self.assertNotIn("Patient: This is your routine well-being check-in.", transcript)

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
            _, _, signals, _, fallback, failure_reason = main._openai_risk_review("Agent: Did you fall?\nPatient: I almost fell.", segments)

        request_json = post.call_args.kwargs["json"]
        user_payload = json.loads(request_json["input"][1]["content"])
        self.assertFalse(fallback)
        self.assertIsNone(failure_reason)
        self.assertEqual(user_payload["patientOnlyTranscript"], "Patient: I almost fell.")
        self.assertEqual(len(user_payload["sentences"]), 1)
        self.assertEqual(signals[0].startTimeSeconds, 4)

    def test_openai_risk_review_records_sanitized_failure_reason(self) -> None:
        segments = [
            TranscriptSegment(text="Patient: I feel dizzy.", englishText="Patient: I feel dizzy.", role="Patient", speaker="Patient", startTimeSeconds=4),
        ]
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("Authorization: Bearer secret-token failed")
        response.json.return_value = {}

        with patch.dict(main.os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(main.httpx, "post", return_value=response):
            _, _, signals, _, fallback, failure_reason = main._openai_risk_review("Patient: I feel dizzy.", segments)

        self.assertTrue(fallback)
        self.assertEqual(signals, [])
        self.assertIsNotNone(failure_reason)
        assert failure_reason is not None
        self.assertIn("redacted", failure_reason)
        self.assertNotIn("secret-token", failure_reason)

    def test_openai_safeguard_review_flags_patient_distress_with_resources(self) -> None:
        segments = [
            TranscriptSegment(text="Agent: Are you safe?", englishText="Agent: Are you safe?", role="Agent", speaker="Agent", startTimeSeconds=0),
            TranscriptSegment(
                text="Patient: I want to hurt myself tonight.",
                englishText="Patient: I want to hurt myself tonight.",
                role="Patient",
                speaker="Patient",
                startTimeSeconds=4,
            ),
        ]
        payload = {
            "output_text": json.dumps(
                {
                    "level": "Emergency",
                    "category": "self_harm_or_suicidal_ideation",
                    "summary": "Patient stated imminent self-harm intent.",
                    "recommendedAction": "Encourage immediate emergency help and alert a human responder.",
                    "evidence": ["I want to hurt myself tonight."],
                    "resourceNames": ["Emergency medical services", "Samaritans of Singapore hotline"],
                }
            )
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload

        with patch.dict(main.os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(main.httpx, "post", return_value=response) as post:
            available, level, category, evidence, action, resources, failure_reason = main._openai_safeguard_review(segments)

        request_json = post.call_args.kwargs["json"]
        user_payload = json.loads(request_json["input"][1]["content"])
        self.assertTrue(available)
        self.assertEqual(level, "Emergency")
        self.assertEqual(category, "self_harm_or_suicidal_ideation")
        self.assertEqual(evidence, ["I want to hurt myself tonight."])
        self.assertIn("emergency", action.lower())
        self.assertEqual([resource.name for resource in resources], ["Emergency medical services", "Samaritans of Singapore hotline"])
        self.assertIsNone(failure_reason)
        self.assertEqual(user_payload["patientOnlyTranscript"], "Patient: I want to hurt myself tonight.")
        self.assertEqual(len(user_payload["sentences"]), 1)

    def test_openai_safeguard_review_drops_flags_without_patient_evidence(self) -> None:
        segments = [
            TranscriptSegment(text="Agent: Are you thinking of hurting yourself?", englishText="Agent: Are you thinking of hurting yourself?", role="Agent", speaker="Agent"),
            TranscriptSegment(text="Patient: No, I am okay.", englishText="Patient: No, I am okay.", role="Patient", speaker="Patient"),
        ]
        payload = {
            "output_text": json.dumps(
                {
                    "level": "Emergency",
                    "category": "self_harm_or_suicidal_ideation",
                    "summary": "Incorrectly used agent wording.",
                    "recommendedAction": "Call emergency services.",
                    "evidence": ["Are you thinking of hurting yourself?"],
                    "resourceNames": ["Emergency medical services"],
                }
            )
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload

        with patch.dict(main.os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(main.httpx, "post", return_value=response):
            available, level, category, evidence, action, resources, failure_reason = main._openai_safeguard_review(segments)

        self.assertTrue(available)
        self.assertEqual(level, "None")
        self.assertIsNone(category)
        self.assertEqual(evidence, [])
        self.assertIsNone(action)
        self.assertEqual(resources, [])
        self.assertIsNone(failure_reason)

    def test_openai_safeguard_review_records_sanitized_failure_reason(self) -> None:
        segments = [
            TranscriptSegment(text="Patient: I feel hopeless.", englishText="Patient: I feel hopeless.", role="Patient", speaker="Patient"),
        ]
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("Authorization: Bearer secret-token failed")
        response.json.return_value = {}

        with patch.dict(main.os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(main.httpx, "post", return_value=response):
            available, level, _, evidence, _, resources, failure_reason = main._openai_safeguard_review(segments)

        self.assertFalse(available)
        self.assertEqual(level, "None")
        self.assertEqual(evidence, [])
        self.assertEqual(resources, [])
        self.assertIsNotNone(failure_reason)
        assert failure_reason is not None
        self.assertIn("redacted", failure_reason)
        self.assertNotIn("secret-token", failure_reason)

    def test_safeguard_level_can_lift_visible_risk_level(self) -> None:
        self.assertEqual(main._risk_level_with_safeguard("Green", "Emergency"), "Red")
        self.assertEqual(main._risk_level_with_safeguard("Red", "Support"), "Red")

    def test_elevenlabs_emotion_review_fetches_data_collection(self) -> None:
        segments = [
            TranscriptSegment(text="Agent: Are you okay?", englishText="Agent: Are you okay?", role="Agent", speaker="Agent", startTimeSeconds=0.2, endTimeSeconds=1.0),
            TranscriptSegment(text="Patient: I am scared.", englishText="Patient: I am scared.", role="Patient", speaker="Patient", startTimeSeconds=1.3, endTimeSeconds=2.0),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "analysis": {
                "data_collection_results": {
                    "user_emotional_state": {
                        "value": json.dumps(
                            {
                                "responses": [
                                    {"responseIndex": 0, "emotion": "anxious", "confidence": 0.82, "evidence": "I am scared."}
                                ],
                                "dominantEmotion": "anxious",
                            }
                        )
                    }
                }
            }
        }

        with patch.dict(main.os.environ, {"ELEVENLABS_API_KEY": "test-key"}), patch.object(main.httpx, "get", return_value=response) as get:
            result = main._elevenlabs_emotion_review("conv-123", segments)

        self.assertEqual(get.call_args.args[0], "https://api.elevenlabs.io/v1/convai/conversations/conv-123")
        self.assertEqual(get.call_args.kwargs["headers"], {"xi-api-key": "test-key"})
        self.assertEqual(result.provider, "elevenlabs-data-collection")
        self.assertEqual(result.dominantEmotion, "anxious")
        self.assertEqual(result.concernLevel, "Review")
        self.assertEqual(result.segments[0].label, "anxious")
        self.assertEqual(result.segments[0].startTimeSeconds, 1.3)
        self.assertEqual(result.segments[0].transcriptSegmentIndex, 1)

    def test_elevenlabs_emotion_review_polls_until_data_collection_ready(self) -> None:
        missing = Mock()
        missing.raise_for_status.return_value = None
        missing.json.return_value = {"analysis": {"data_collection_results": {}}}
        ready = Mock()
        ready.raise_for_status.return_value = None
        ready.json.return_value = {"analysis": {"data_collection_results": {"user_emotional_state": {"value": {"dominantEmotion": "calm", "responses": []}}}}}

        with (
            patch.dict(main.os.environ, {"ELEVENLABS_API_KEY": "test-key"}),
            patch.object(main.httpx, "get", side_effect=[missing, ready]) as get,
            patch.object(main.time, "sleep") as sleep,
        ):
            result = main._elevenlabs_emotion_review("conv-123", [])

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(2)
        self.assertEqual(result.provider, "elevenlabs-data-collection")
        self.assertEqual(result.dominantEmotion, "calm")
        self.assertEqual([attempt.status for attempt in result.attempts], ["skipped", "success"])

    def test_elevenlabs_emotion_review_summary_only_has_no_segments(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data_collection_results": {"user_emotional_state": {"value": "calm"}}}

        with patch.dict(main.os.environ, {"ELEVENLABS_API_KEY": "test-key"}), patch.object(main.httpx, "get", return_value=response):
            result = main._elevenlabs_emotion_review("conv-123", [])

        self.assertEqual(result.dominantEmotion, "calm")
        self.assertEqual(result.segments, [])
        self.assertIsNotNone(result.failureReason)
        assert result.failureReason is not None
        self.assertIn("summary", result.failureReason.lower())

    def test_elevenlabs_emotion_review_maps_missing_indexes_by_order_when_counts_match(self) -> None:
        segments = [
            TranscriptSegment(text="Patient: First answer.", englishText="Patient: First answer.", role="Patient", speaker="Patient", startTimeSeconds=1, endTimeSeconds=2),
            TranscriptSegment(text="Agent: Next question.", englishText="Agent: Next question.", role="Agent", speaker="Agent", startTimeSeconds=3, endTimeSeconds=4),
            TranscriptSegment(text="Patient: Second answer.", englishText="Patient: Second answer.", role="Patient", speaker="Patient", startTimeSeconds=5, endTimeSeconds=6),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data_collection_results": {
                "user_emotional_state": {
                    "value": {
                        "dominantEmotion": "calm",
                        "responses": [
                            {"emotion": "calm", "confidence": 0.7, "evidence": "First answer."},
                            {"emotion": "frustrated", "confidence": 0.8, "evidence": "Second answer."},
                        ],
                    }
                }
            }
        }

        with patch.dict(main.os.environ, {"ELEVENLABS_API_KEY": "test-key"}), patch.object(main.httpx, "get", return_value=response):
            result = main._elevenlabs_emotion_review("conv-123", segments)

        self.assertEqual([segment.transcriptSegmentIndex for segment in result.segments], [0, 2])
        self.assertEqual([segment.startTimeSeconds for segment in result.segments], [1, 5])

    def test_elevenlabs_emotion_review_summary_dict_records_helpful_reason(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data_collection_results": {
                "user_emotional_state": {
                    "value": {
                        "dominantEmotion": "calm",
                        "summary": "The user sounded calm overall.",
                    }
                }
            }
        }

        with patch.dict(main.os.environ, {"ELEVENLABS_API_KEY": "test-key"}), patch.object(main.httpx, "get", return_value=response):
            result = main._elevenlabs_emotion_review("conv-123", [])

        self.assertEqual(result.dominantEmotion, "calm")
        self.assertEqual(result.segments, [])
        self.assertIsNotNone(result.failureReason)
        assert result.failureReason is not None
        self.assertIn("per-response", result.failureReason)

    def test_elevenlabs_emotion_review_records_sanitized_failure_reason(self) -> None:
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("Authorization: Bearer secret-token failed")
        response.json.return_value = {}

        with patch.dict(main.os.environ, {"ELEVENLABS_API_KEY": "test-key"}), patch.object(main.httpx, "get", return_value=response):
            result = main._elevenlabs_emotion_review("conv-123", [])

        self.assertIsNone(result.provider)
        self.assertIsNotNone(result.failureReason)
        assert result.failureReason is not None
        self.assertIn("redacted", result.failureReason)
        self.assertNotIn("secret-token", result.failureReason)

    def test_emotion_modifier_lifts_green_only_to_watch(self) -> None:
        assessment = main._empty_assessment("Green", ["No notable risk."])
        result = main.EmotionProviderResult(
            provider="meralion-emotion",
            dominantEmotion="sad",
            concernLevel="Review",
            segments=[main.EmotionSegment(id="emotion-1", label="sad", confidence=0.86, evidenceText="Patient sounded sad.")],
        )
        lifted = main._apply_emotion_modifier(assessment, result)
        self.assertEqual(lifted.riskLevel, "Watch")

        red_assessment = main._empty_assessment("Red", ["Emergency clinical risk."])
        unchanged = main._apply_emotion_modifier(red_assessment, result)
        self.assertEqual(unchanged.riskLevel, "Red")

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

    def test_update_volunteer_task_rejects_invalid_status(self) -> None:
        client = TestClient(main.app)

        response = client.patch("/volunteer-tasks/t-001", params={"status": "Done"})

        self.assertEqual(response.status_code, 422)

    def test_update_volunteer_task_accepts_body_status(self) -> None:
        client = TestClient(main.app)
        original_status = main.VOLUNTEER_TASKS[0].status

        response = client.patch("/volunteer-tasks/t-001", json={"status": "Closed"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "Closed")
        main.VOLUNTEER_TASKS[0].status = original_status


if __name__ == "__main__":
    unittest.main()
