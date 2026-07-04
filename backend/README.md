# EarlyCare Backend

FastAPI service for EarlyCare call sessions, call artifact storage, transcription, translation, speech timing, and OpenAI-assisted patient-risk review.

## What It Does

- Creates signed ElevenLabs Agents session URLs without exposing secrets to the frontend.
- Saves full-call audio uploaded by the browser as `full-call.webm`.
- Stores cleaned live dialogue messages with `Agent:` and `Patient:` labels.
- Removes bracketed delivery cues such as `[happy]`, `[concerned]`, and `[sighs]`.
- Uses MERaLiON first for timestamped transcription and audio translation.
- Falls back to ElevenLabs speech-to-text, Google Translate, and finally the saved dialogue transcript.
- Uses OpenAI structured output to detect patient-only risk signals.
- Attaches risk evidence to patient transcript segments and audio seek times.
- Serves saved call metadata and audio to the Patient overview.

## Call Save Flow

When the frontend posts to `POST /calls`, the backend:

1. Validates the selected senior.
2. Parses and cleans transcript messages.
3. Saves uploaded mixed audio as `backend/storage/calls/{call_id}/full-call.webm`.
4. Builds an original transcript with `Agent:` and `Patient:` speaker labels.
5. Builds an English transcript with the same speaker labels.
6. Creates timestamped transcript segments from provider output or live message timing.
7. Estimates speech timing metrics for the latest call.
8. Sends patient speech only to OpenAI for structured risk review.
9. Drops any risk signal that cannot be validated against a patient segment.
10. Saves `metadata.json`, `transcript-original.json`, `transcript-english.txt`, and audio.
11. Returns the saved call record to the frontend.

Generated call artifacts are intentionally local-only and ignored by git.

## Transcription And Translation Chain

| Priority | Provider | Use |
| --- | --- | --- |
| 1 | MERaLiON ASR | `POST http://meralion.org:8010/audio/transcription` with base64 audio, timestamps, and diarization. |
| 2 | MERaLiON audio translation | `POST http://meralion.org:8010/audio/translation` for non-English English transcript generation. |
| 3 | ElevenLabs speech-to-text | Original transcript fallback when MERaLiON fails. |
| 4 | Google Translate | English translation fallback for non-English transcript text. |
| 5 | Saved dialogue transcript | Final demo fallback so calls still save. |

If a fallback is used, the saved call marks `translationFallbackUsed=true`.

## AI Risk Review

OpenAI is used for structured decision-support extraction when `OPENAI_API_KEY` is configured. The default model is controlled by `OPENAI_MODEL` and is currently `gpt-4o-mini`.

The risk review prompt is constrained to patient speech only. Agent questions and agent summaries are ignored. Returned signals are validated against patient transcript segments before saving.

OpenAI returns:

- risk level: `Green`, `Watch`, `Amber`, or `Red`
- concise reasons
- recommended follow-up action
- exact English patient evidence text
- patient sentence index and timestamp when available

If OpenAI is unavailable, the call still saves with `aiRiskFallbackUsed=true` and a manual-review recommendation.

EarlyCare does not diagnose medical conditions.

## Audio Seeking

For Patient overview verification, risk-signal timestamps are tied to patient segments. The frontend starts playback from immediately after the previous agent question/statement so the caregiver hears the full patient answer in context.

The backend stores estimated segment timing for new calls. When precise provider word timestamps are unavailable, it estimates:

- patient utterance start from live transcript event time minus estimated speaking duration
- agent utterance end from estimated speaking duration, bounded by the following patient turn

## Local Setup

Create the local env file:

```bash
cp .env.example .env
```

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

API health check:

```bash
curl http://127.0.0.1:8000/health
```

## Environment Variables

| Variable | Required For | Notes |
| --- | --- | --- |
| `ELEVENLABS_API_KEY` | Agents website call, STT fallback | Used only by FastAPI. |
| `ELEVENLABS_AGENT_ID` | Agents website call | Agent configured in the ElevenLabs dashboard. |
| `ELEVENLABS_STT_MODEL` | STT fallback | Defaults to `scribe_v2` in `.env.example`. |
| `MERALION_API_KEY` | MERaLiON ASR/translation | Sent to MERaLiON endpoints. |
| `MERALION_ASR_URL` | MERaLiON ASR | Defaults to `http://meralion.org:8010/audio/transcription`. |
| `MERALION_TRANSLATION_URL` | MERaLiON translation | Defaults to `http://meralion.org:8010/audio/translation`. |
| `GOOGLE_TRANSLATE_API_KEY` | Google fallback | Used if MERaLiON translation fails and transcript is non-English. |
| `GOOGLE_TRANSLATE_URL` | Google fallback | Defaults to Cloud Translation v2. |
| `OPENAI_API_KEY` | AI risk review | Used for structured patient-risk extraction. |
| `OPENAI_MODEL` | AI risk review | Defaults to `gpt-4o-mini`. |

Never commit real `.env` files.

## Main Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /health` | API health check. |
| `GET /seniors` | Demo senior roster. |
| `GET /calls` | Saved call records, newest first. |
| `GET /calls/{call_id}` | One saved call record. |
| `GET /calls/{call_id}/audio` | Replayable saved full-call recording. |
| `POST /calls` | Save transcript messages and uploaded audio. |
| `POST /elevenlabs/signed-url` | Create a signed Agents session URL. |
| `GET /volunteer-tasks` | Demo volunteer task list. |

## Storage

Saved calls are written to:

```text
backend/storage/calls/{call_id}/
```

Each call can include:

- `metadata.json`
- `full-call.webm`
- `transcript-original.json`
- `transcript-english.txt`

This is intentionally simple for hackathon speed. Use a database and object storage before any real pilot.

## Smoke Checks

From the repository root:

```bash
backend/.venv/bin/python -m py_compile backend/app/*.py
PYTHONPATH=backend backend/.venv/bin/python -m unittest discover backend/tests
```

With the API running:

```bash
curl http://127.0.0.1:8000/health
```
