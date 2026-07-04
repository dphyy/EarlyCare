import os
from pathlib import Path
from typing import Any

import httpx

from app.models import ProviderResult, TranscriptSegment


GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"


def _first_text(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_segments(payload: dict[str, Any]) -> list[TranscriptSegment]:
    raw_segments = payload.get("segments") or payload.get("chunks") or payload.get("utterances") or []
    if not isinstance(raw_segments, list):
        return []

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = _first_text(item, ["text", "transcript", "translation"])
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                text=text,
                startTimeSeconds=item.get("start") or item.get("start_time") or item.get("startTimeSeconds"),
                endTimeSeconds=item.get("end") or item.get("end_time") or item.get("endTimeSeconds"),
                role=item.get("role") or item.get("speaker"),
            )
        )
    return segments


class MeralionProvider:
    name = "meralion"

    def transcribe(self, language: str, audio_hint: str, audio_path: Path | None) -> ProviderResult:
        api_key = os.getenv("MERALION_API_KEY")
        endpoint = os.getenv("MERALION_ASR_URL")
        if not api_key or not endpoint or audio_path is None:
            raise RuntimeError("MERaLiON credentials, endpoint, or audio file not configured")

        with audio_path.open("rb") as audio_file:
            response = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "x-api-key": api_key},
                data={"language": language, "target_language": "en", "task": "translate"},
                files={"audio": (audio_path.name, audio_file, "audio/webm")},
                timeout=45,
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("MERaLiON response was not a JSON object")

        transcript = _first_text(payload, ["transcript", "text", "source_text", "originalTranscript"]) or audio_hint
        translation = _first_text(payload, ["translation", "english", "english_text", "translated_text", "englishTranscript"])
        if not translation:
            raise RuntimeError("MERaLiON response did not include an English translation")

        confidence = payload.get("confidence")
        return ProviderResult(
            provider=self.name,
            language=language,
            transcript=transcript,
            translation=translation,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.86,
            fallbackUsed=False,
            segments=_parse_segments(payload),
        )


class GoogleTranslateProvider:
    name = "google-translate"

    def transcribe(self, language: str, audio_hint: str, audio_path: Path | None) -> ProviderResult:
        _ = audio_path
        api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        endpoint = os.getenv("GOOGLE_TRANSLATE_URL", GOOGLE_TRANSLATE_URL)
        if not api_key:
            raise RuntimeError("Google Translate API key not configured")
        if not audio_hint.strip():
            raise RuntimeError("No transcript text available for Google Translate fallback")

        if language.lower() == "english":
            raise RuntimeError("Google Translate fallback skipped for English transcript")

        response = httpx.post(
            endpoint,
            params={"key": api_key},
            json={"q": audio_hint, "target": "en", "format": "text"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        translations = payload.get("data", {}).get("translations", []) if isinstance(payload, dict) else []
        if not translations or not isinstance(translations[0], dict):
            raise RuntimeError("Google Translate response did not include translations")

        translated_text = translations[0].get("translatedText")
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise RuntimeError("Google Translate response was empty")

        return ProviderResult(
            provider=self.name,
            language=language,
            transcript=audio_hint,
            translation=translated_text.strip(),
            confidence=0.78,
            fallbackUsed=True,
        )


class ElevenLabsFallbackProvider:
    name = "elevenlabs-original"

    def transcribe(self, language: str, audio_hint: str, audio_path: Path | None) -> ProviderResult:
        _ = audio_path
        return ProviderResult(
            provider=self.name,
            language=language,
            transcript=audio_hint,
            translation=audio_hint,
            confidence=0.65,
            fallbackUsed=True,
        )


def transcribe_with_fallback(language: str, audio_hint: str, audio_path: Path | None = None) -> ProviderResult:
    providers = [MeralionProvider(), GoogleTranslateProvider(), ElevenLabsFallbackProvider()]
    last_error: Exception | None = None

    for provider in providers:
        try:
            return provider.transcribe(language=language, audio_hint=audio_hint, audio_path=audio_path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"No ASR or translation provider available: {last_error}")
