import unittest
from pathlib import Path

from app.models import TranscriptMessage, TranscriptSegment
from app.speech_features import DemoSpeechFeatureExtractor, SpeechFeatureInput, estimated_utterance_seconds, word_count


class FixedEmbeddingExtractor:
    provider = "fixed"

    def __init__(self) -> None:
        self.seen_audio_path: Path | None = None

    def extract_embedding(self, payload: SpeechFeatureInput) -> list[float]:
        self.seen_audio_path = payload.audio_path
        return [0.1, 0.2, 0.3]


class SpeechFeatureTests(unittest.TestCase):
    def test_word_count_supports_english_and_chinese_characters(self) -> None:
        self.assertEqual(word_count("I am okay"), 3)
        self.assertEqual(word_count("我跌倒了"), 4)

    def test_estimated_utterance_seconds_removes_speaker_label(self) -> None:
        self.assertEqual(estimated_utterance_seconds("Patient: I am safe at home"), 2.083)

    def test_message_timestamps_produce_deterministic_profile_without_embedding(self) -> None:
        messages = [
            TranscriptMessage(role="Agent", text="How are you?", timestamp="2026-07-04T10:00:00+08:00"),
            TranscriptMessage(role="Senior", text="I am okay", timestamp="2026-07-04T10:00:02+08:00"),
            TranscriptMessage(role="Agent", text="Any falls?", timestamp="2026-07-04T10:00:05+08:00"),
            TranscriptMessage(role="Senior", text="I almost fell", timestamp="2026-07-04T10:00:09+08:00"),
        ]

        profile = DemoSpeechFeatureExtractor().extract(
            SpeechFeatureInput(
                audio_path=None,
                messages=messages,
                started_at="2026-07-04T10:00:00+08:00",
                completed_at="2026-07-04T10:00:20+08:00",
            )
        )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.speechRate, 18.0)
        self.assertEqual(profile.responseLatencyMs, 3000)
        self.assertEqual(profile.avgPauseMs, 7000)
        self.assertIsNone(profile.embedding)

    def test_timed_patient_segments_override_message_pause_estimates(self) -> None:
        messages = [
            TranscriptMessage(role="Agent", text="Any dizziness?", timestamp="2026-07-04T10:00:00+08:00"),
            TranscriptMessage(role="Senior", text="I feel dizzy", timestamp="2026-07-04T10:00:05+08:00"),
            TranscriptMessage(role="Senior", text="Ignore duplicate timestamp gap", timestamp="2026-07-04T10:00:20+08:00"),
        ]
        segments = [
            TranscriptSegment(text="Agent: Any dizziness?", role="Agent", speaker="Agent", startTimeSeconds=0, endTimeSeconds=1),
            TranscriptSegment(text="Patient: I feel dizzy", originalText="Patient: I feel dizzy", role="Patient", speaker="Patient", startTimeSeconds=2, endTimeSeconds=3),
            TranscriptSegment(text="Patient: My head hurts", originalText="Patient: My head hurts", role="Patient", speaker="Patient", startTimeSeconds=7, endTimeSeconds=8),
        ]

        profile = DemoSpeechFeatureExtractor().extract(
            SpeechFeatureInput(
                audio_path=None,
                messages=messages,
                started_at="2026-07-04T10:00:00+08:00",
                completed_at="2026-07-04T10:00:30+08:00",
                segments=segments,
            )
        )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.avgPauseMs, 4000)
        self.assertEqual(profile.pitchVariability, 0.1)

    def test_repeat_phrase_can_be_detected_from_messages_or_segments(self) -> None:
        profile_from_message = DemoSpeechFeatureExtractor().extract(
            SpeechFeatureInput(
                audio_path=None,
                messages=[
                    TranscriptMessage(
                        role="Senior",
                        text="Today I am safe at home and I can ask for help.",
                        timestamp="2026-07-04T10:00:02+08:00",
                    )
                ],
                started_at="2026-07-04T10:00:00+08:00",
                completed_at="2026-07-04T10:00:05+08:00",
            )
        )
        profile_from_segment = DemoSpeechFeatureExtractor().extract(
            SpeechFeatureInput(
                audio_path=None,
                messages=[],
                started_at="2026-07-04T10:00:00+08:00",
                completed_at="2026-07-04T10:00:05+08:00",
                segments=[
                    TranscriptSegment(
                        text="Patient: 今天我在家安全",
                        englishText="Patient: Today I am safe at home and I can ask for help.",
                        role="Patient",
                        speaker="Patient",
                        startTimeSeconds=1,
                        endTimeSeconds=3,
                    )
                ],
            )
        )

        self.assertIsNotNone(profile_from_message)
        self.assertIsNotNone(profile_from_segment)
        assert profile_from_message is not None and profile_from_segment is not None
        self.assertEqual(profile_from_message.phraseAccuracy, 0.96)
        self.assertEqual(profile_from_segment.phraseAccuracy, 0.96)

    def test_custom_embedding_extractor_receives_audio_path(self) -> None:
        embedding_extractor = FixedEmbeddingExtractor()
        audio_path = Path("backend/storage/calls/call-test/full-call.webm")

        profile = DemoSpeechFeatureExtractor(embedding_extractor).extract(
            SpeechFeatureInput(
                audio_path=audio_path,
                messages=[TranscriptMessage(role="Senior", text="I am okay", timestamp=None)],
                started_at="2026-07-04T10:00:00+08:00",
                completed_at="2026-07-04T10:00:10+08:00",
            )
        )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(embedding_extractor.seen_audio_path, audio_path)

    def test_empty_payload_returns_none(self) -> None:
        profile = DemoSpeechFeatureExtractor().extract(
            SpeechFeatureInput(
                audio_path=None,
                messages=[],
                started_at="2026-07-04T10:00:00+08:00",
                completed_at="2026-07-04T10:00:10+08:00",
            )
        )

        self.assertIsNone(profile)


if __name__ == "__main__":
    unittest.main()
