import base64
import html
import os
import re
from pathlib import Path
from typing import Any

import httpx

from app.models import ProviderResult, TranscriptSegment


GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
MERALION_ASR_URL = "http://meralion.org:8010/audio/transcription"
MERALION_TRANSLATION_URL = "http://meralion.org:8010/audio/translation"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
BRACKET_CUE_RE = re.compile(r"\s*\[[^\]\r\n]{1,40}\]\s*")


def clean_transcript_text(text: str) -> str:
    text = BRACKET_CUE_RE.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _first_text(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return clean_transcript_text(value)
    return ""


def _choice_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return {}


def _message_content(payload: dict[str, Any]) -> str:
    message = _choice_message(payload)
    content = message.get("content")
    if isinstance(content, str):
        return clean_transcript_text(content)
    return _first_text(payload, ["transcript", "text", "source_text", "originalTranscript", "translation", "translated_text"])


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _segment_text(item: dict[str, Any]) -> str:
    return _first_text(item, ["text", "transcript", "translation", "englishText", "originalText"])


def _parse_segments(payload: dict[str, Any]) -> list[TranscriptSegment]:
    message = _choice_message(payload)
    raw_segments = (
        message.get("segments")
        or payload.get("segments")
        or payload.get("chunks")
        or payload.get("utterances")
        or []
    )
    if not isinstance(raw_segments, list):
        return []

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = _segment_text(item)
        if not text:
            continue
        speaker = item.get("speaker") or item.get("role")
        segments.append(
            TranscriptSegment(
                text=text,
                originalText=clean_transcript_text(str(item.get("originalText") or text)),
                englishText=clean_transcript_text(str(item.get("englishText") or item.get("translation") or text)),
                startTimeSeconds=_number(item.get("start") or item.get("start_time") or item.get("startTimeSeconds")),
                endTimeSeconds=_number(item.get("end") or item.get("end_time") or item.get("endTimeSeconds")),
                role=str(item.get("role")) if item.get("role") else None,
                speaker=str(speaker) if speaker else None,
            )
        )
    return segments


def _segments_from_words(payload: dict[str, Any], transcript: str) -> list[TranscriptSegment]:
    message = _choice_message(payload)
    words = message.get("words") or payload.get("words") or []
    if not isinstance(words, list) or not words:
        return []

    sentence_parts: list[str] = []
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    segments: list[TranscriptSegment] = []

    def flush() -> None:
        nonlocal sentence_parts, start, end, speaker
        text = clean_transcript_text(" ".join(sentence_parts))
        if text:
            segments.append(
                TranscriptSegment(
                    text=text,
                    originalText=text,
                    englishText=text,
                    startTimeSeconds=start,
                    endTimeSeconds=end,
                    speaker=speaker,
                )
            )
        sentence_parts = []
        start = None
        end = None
        speaker = None

    for item in words:
        if not isinstance(item, dict):
            continue
        word = item.get("display") or item.get("word")
        if not isinstance(word, str) or not word.strip():
            continue
        word = word.strip()
        word_start = _number(item.get("start"))
        word_end = _number(item.get("end"))
        if start is None:
            start = word_start
        end = word_end if word_end is not None else end
        speaker = str(item.get("speaker")) if item.get("speaker") else speaker
        sentence_parts.append(word)
        if re.search(r"[.!?。？！]$", word):
            flush()

    flush()
    if segments:
        return segments

    text = clean_transcript_text(transcript)
    return [TranscriptSegment(text=text, originalText=text, englishText=text)] if text else []


def _fallback_segments(transcript: str, translation: str) -> list[TranscriptSegment]:
    text = clean_transcript_text(translation or transcript)
    original = clean_transcript_text(transcript)
    if not text and not original:
        return []
    return [TranscriptSegment(text=text or original, originalText=original or text, englishText=text or original)]


def _audio_base64(audio_path: Path) -> str:
    return base64.b64encode(audio_path.read_bytes()).decode("ascii")


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key, "Content-Type": "application/json"}


class MeralionProvider:
    name = "meralion"

    def transcribe(self, language: str, audio_hint: str, audio_path: Path | None) -> ProviderResult:
        api_key = os.getenv("MERALION_API_KEY")
        endpoint = os.getenv("MERALION_ASR_URL", MERALION_ASR_URL)
        translation_endpoint = os.getenv("MERALION_TRANSLATION_URL", MERALION_TRANSLATION_URL)
        if not api_key or audio_path is None:
            raise RuntimeError("MERaLiON credentials or audio file not configured")

        audio_url = _audio_base64(audio_path)
        response = httpx.post(
            endpoint,
            headers=_auth_headers(api_key),
            json={
                "audio_url": audio_url,
                "return_timestamps": True,
                "return_diarization": True,
                "boundary_mode": "sequential",
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("MERaLiON transcription response was not a JSON object")

        transcript = _message_content(payload) or clean_transcript_text(audio_hint)
        segments = _parse_segments(payload) or _segments_from_words(payload, transcript)
        translation = transcript
        provider = self.name

        if language.lower() != "english":
            translation_response = httpx.post(
                translation_endpoint,
                headers=_auth_headers(api_key),
                json={
                    "audio_url": audio_url,
                    "translation_params": {
                        "source_language": language,
                        "target_language": "English",
                    },
                },
                timeout=90,
            )
            translation_response.raise_for_status()
            translation_payload = translation_response.json()
            if not isinstance(translation_payload, dict):
                raise RuntimeError("MERaLiON translation response was not a JSON object")
            translation = _message_content(translation_payload)
            if not translation:
                raise RuntimeError("MERaLiON response did not include an English translation")
            provider = f"{self.name}-audio-translation"
            for segment in segments:
                if segment.englishText is None:
                    segment.englishText = segment.text

        confidence = payload.get("confidence")
        return ProviderResult(
            provider=provider,
            language=language,
            transcript=clean_transcript_text(transcript),
            translation=clean_transcript_text(translation),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.86,
            fallbackUsed=False,
            segments=segments or _fallback_segments(transcript, translation),
        )


class ElevenLabsSpeechToTextProvider:
    name = "elevenlabs-stt"

    def transcribe(self, language: str, audio_hint: str, audio_path: Path | None) -> ProviderResult:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key or audio_path is None:
            raise RuntimeError("ElevenLabs API key or audio file not configured")

        with audio_path.open("rb") as audio_file:
            response = httpx.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": api_key},
                data={"model_id": os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")},
                files={"file": (audio_path.name, audio_file, "audio/webm")},
                timeout=90,
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("ElevenLabs STT response was not a JSON object")

        transcript = _first_text(payload, ["text", "transcript"]) or clean_transcript_text(audio_hint)
        return ProviderResult(
            provider=self.name,
            language=language,
            transcript=transcript,
            translation=transcript,
            confidence=0.74,
            fallbackUsed=True,
            segments=_parse_segments(payload) or _fallback_segments(transcript, transcript),
        )


class GoogleTranslateProvider:
    name = "google-translate"

    def transcribe(self, language: str, audio_hint: str, audio_path: Path | None) -> ProviderResult:
        _ = audio_path
        api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        endpoint = os.getenv("GOOGLE_TRANSLATE_URL", GOOGLE_TRANSLATE_URL)
        transcript = clean_transcript_text(audio_hint)
        if not api_key:
            raise RuntimeError("Google Translate API key not configured")
        if not transcript:
            raise RuntimeError("No transcript text available for Google Translate fallback")
        if language.lower() == "english":
            raise RuntimeError("Google Translate fallback skipped for English transcript")

        response = httpx.post(
            endpoint,
            params={"key": api_key},
            json={"q": transcript, "target": "en", "format": "text"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        translations = payload.get("data", {}).get("translations", []) if isinstance(payload, dict) else []
        if not translations or not isinstance(translations[0], dict):
            raise RuntimeError("Google Translate response did not include translations")

        translated_text = translations[0].get("translatedText")
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise RuntimeError("Google Translate response was empty")
        translated_text = clean_transcript_text(html.unescape(translated_text))

        return ProviderResult(
            provider=self.name,
            language=language,
            transcript=transcript,
            translation=translated_text,
            confidence=0.78,
            fallbackUsed=True,
            segments=_fallback_segments(transcript, translated_text),
        )


class TranscriptFallbackProvider:
    name = "dialogue-transcript"

    def transcribe(self, language: str, audio_hint: str, audio_path: Path | None) -> ProviderResult:
        _ = audio_path
        transcript = clean_transcript_text(audio_hint)
        return ProviderResult(
            provider=self.name,
            language=language,
            transcript=transcript,
            translation=transcript,
            confidence=0.55,
            fallbackUsed=True,
            segments=_fallback_segments(transcript, transcript),
        )


def transcribe_with_fallback(language: str, audio_hint: str, audio_path: Path | None = None) -> ProviderResult:
    providers = [MeralionProvider(), ElevenLabsSpeechToTextProvider(), GoogleTranslateProvider(), TranscriptFallbackProvider()]
    last_error: Exception | None = None
    current_hint = clean_transcript_text(audio_hint)

    for provider in providers:
        try:
            result = provider.transcribe(language=language, audio_hint=current_hint, audio_path=audio_path)
            if provider.name == "elevenlabs-stt" and language.lower() != "english":
                try:
                    translated = GoogleTranslateProvider().transcribe(language=language, audio_hint=result.transcript, audio_path=None)
                    segments = result.segments or translated.segments
                    if len(segments) == 1:
                        segments[0].englishText = translated.translation
                    else:
                        for segment in segments:
                            if segment.englishText == segment.text or segment.englishText == segment.originalText:
                                segment.englishText = None
                    return translated.model_copy(
                        update={
                            "provider": f"{result.provider}+{translated.provider}",
                            "confidence": min(result.confidence, translated.confidence),
                            "segments": segments,
                        }
                    )
                except Exception as exc:
                    last_error = exc
            return result
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"No ASR or translation provider available: {last_error}")
