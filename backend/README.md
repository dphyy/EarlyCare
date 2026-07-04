# EarlyCare Backend

FastAPI service for EarlyCare call sessions, transcript processing, audio storage, translation, and AI-assisted risk review.

## What It Does

- Creates signed ElevenLabs Agents session URLs without exposing secrets to the frontend.
- Serves the current seven-scenario demo runner.
- Persists scripted check-in runs and volunteer task status under local hackathon storage.
- Saves call artifacts from the website call demo.
- Stores browser microphone recordings as `mic-audio.webm`.
- Cleans transcript text by removing agent delivery tags such as `[happy]` or `[relieved]`.
- Translates or normalizes transcripts to English.
- Uses OpenAI structured output for decision-support symptom and risk extraction.
- Builds categorized conversation evidence and escalation trails for Patient overview.
- Serves saved call metadata and replayable audio to the Patient overview.

## Scenario Runner Flow

When the frontend posts to `POST /scenarios/{scenario_id}/run`, the backend:

1. Loads the selected senior profile and scripted scenario.
2. Compares the scenario speech metrics with the senior's baseline demo profile.
3. Categorizes the record across eight care categories.
4. Builds the escalation trail from routine note through emergency alert.
5. Persists the check-in under `backend/storage/state/checkins.json`.
6. Creates or updates a volunteer task when the result is Watch, Amber, Red, or missed check-in.

## Call Save Flow

When the frontend posts to `POST /calls`, the backend:

1. Validates the selected senior.
2. Parses and cleans transcript messages.
3. Saves the uploaded microphone recording under `backend/storage/calls/{call_id}/`.
4. Creates the original transcript.
5. Runs the translation fallback chain.
6. Runs OpenAI structured risk review on the English transcript.
7. Saves `metadata.json`, `transcript-original.json`, `transcript-english.txt`, and audio.
8. Returns the saved call record to the frontend.

Generated call artifacts are intentionally local-only and ignored by git.

## Translation Fallback Chain

| Priority | Provider | Use |
| --- | --- | --- |
| 1 | MERaLiON | Audio translation or ASR using `MERALION_ASR_URL`. |
| 2 | Google Translate | Text translation fallback for non-English transcripts. |
| 3 | ElevenLabs/original transcript | Last-resort fallback so the demo still saves calls. |

If translation falls back, the saved call marks `translationFallbackUsed=true`.

## AI Risk Review

OpenAI is used for structured decision-support extraction. It returns:

- risk level: `Green`, `Watch`, `Amber`, or `Red`
- concise reasons
- recommended follow-up action
- risk-signal evidence
- optional timestamps for audio seeking

If OpenAI is unavailable, the call still saves with `aiRiskFallbackUsed=true` and a manual-review recommendation.

EarlyCare does not diagnose medical conditions.

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
| `ELEVENLABS_API_KEY` | Agents website call | Used only by FastAPI to create signed session URLs. |
| `ELEVENLABS_AGENT_ID` | Agents website call | Agent configured in the ElevenLabs dashboard. |
| `MERALION_API_KEY` | MERaLiON translation | Sent to `MERALION_ASR_URL`. |
| `MERALION_ASR_URL` | MERaLiON translation | Generic multipart audio endpoint. |
| `GOOGLE_TRANSLATE_API_KEY` | Google fallback | Used if MERaLiON fails and transcript is non-English. |
| `GOOGLE_TRANSLATE_URL` | Google fallback | Defaults to Cloud Translation v2. |
| `OPENAI_API_KEY` | AI risk review | Used for structured risk-signal extraction. |

Never commit real `.env` files.

## Main Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /health` | API health check. |
| `GET /seniors` | Demo senior roster. |
| `GET /checkins` | Historical check-in records, including persisted scenario runs. |
| `GET /scenarios` | Scripted demo scenarios. |
| `POST /scenarios/{scenario_id}/run` | Run and persist a scripted scenario. |
| `GET /calls` | Saved call records, newest first. |
| `GET /calls/{call_id}` | One saved call record. |
| `GET /calls/{call_id}/audio` | Replayable saved microphone recording. |
| `POST /calls` | Save transcript messages and uploaded audio. |
| `POST /elevenlabs/signed-url` | Create a signed Agents session URL. |
| `GET /volunteer-tasks` | Demo volunteer task list. |
| `PATCH /volunteer-tasks/{task_id}` | Update task status to `Open`, `In progress`, or `Closed`. |

## Storage

Saved calls are written to:

```text
backend/storage/calls/{call_id}/
```

Each call can include:

- `metadata.json`
- `mic-audio.webm`
- `transcript-original.json`
- `transcript-english.txt`

This is intentionally simple for hackathon speed. Use a database and object storage before any real pilot.

Scripted check-ins and task status updates are written to:

```text
backend/storage/state/
```

This keeps the demo persistent across refreshes without committing generated records.

## Smoke Checks

From the repository root:

```bash
python3 -m py_compile backend/app/*.py
pnpm backend:smoke
```
