# EarlyCare Backend

FastAPI service for EarlyCare call sessions, call artifact storage, transcription, translation, OpenAI-assisted patient-risk review, safeguard classification, ElevenLabs tone ingestion, and conversational speech-marker scoring.

## What It Does

- Creates signed ElevenLabs Agents session URLs without exposing secrets to the frontend.
- Saves full-call audio uploaded by the browser as `full-call.wav`.
- Saves patient-only microphone audio as `patient-audio.wav` when the frontend provides it; the frontend requests browser echo cancellation, noise suppression, and auto gain control before recording.
- Derives `patient-speech.wav` from voiced patient answer regions between agent turns and uses it for speech-marker scoring.
- Stores cleaned live dialogue messages with `Agent:` and `Patient:` labels.
- Removes bracketed delivery cues such as `[happy]`, `[concerned]`, and `[sighs]`.
- Uses MERaLiON first for timestamped transcription and audio translation.
- Falls back to ElevenLabs speech-to-text, Google Translate, and finally the saved dialogue transcript.
- Saves sanitized provider attempt diagnostics so fallback reasons are visible in `metadata.json` and the dashboard.
- Uses OpenAI structured output to detect patient-only risk signals.
- Uses a separate OpenAI safeguard classifier for patient-stated distress, self-harm, abuse/neglect, unsafe environment, and emergency cues.
- Reads ElevenLabs `user_emotional_state` data collection results and attaches per-response tone evidence when available.
- Attaches risk evidence to patient transcript segments and audio seek times.
- Applies safeguard/tone modifiers to the visible call risk level while preserving the underlying reasons in `riskAssessment`.
- Scores optional conversational speech-marker features from `patient-speech.wav` when enabled.
- Serves saved call metadata and audio to the Patient overview.

## Call Save Flow

When the frontend posts to `POST /calls`, the backend:

1. Validates the selected senior.
2. Parses and cleans transcript messages.
3. Saves uploaded mixed audio as `backend/storage/calls/{call_id}/full-call.wav`.
4. Saves uploaded patient-only audio as `backend/storage/calls/{call_id}/patient-audio.wav` when available.
5. Builds an original transcript with `Agent:` and `Patient:` speaker labels.
6. Builds an English transcript with the same speaker labels.
7. Creates timestamped transcript segments from provider output or live message timing.
8. Sends patient speech only to OpenAI for structured risk review.
9. Sends patient speech only to a separate OpenAI safeguard classifier and lifts the visible risk level when the safeguard level is higher.
10. Estimates speech profile metrics and speech signal quality for the latest call.
11. Derives `patient-speech.wav` by finding agent-bounded answer windows, falling back to patient-segment or full-audio VAD when needed, and stitching voiced patient clips from `patient-audio.wav`.
12. Queries ElevenLabs conversation data collection for `user_emotional_state` and maps tone evidence to patient segments when possible.
13. Scores the optional conversational speech-marker model when `EARLYCARE_SPEECH_MODEL_ENABLED=true`.
14. Drops any risk or safeguard signal that cannot be validated against patient evidence.
15. Saves `metadata.json`, `transcript-original.json`, `transcript-english.txt`, provider attempt history, and audio.
16. Returns the saved call record to the frontend.

Generated call artifacts are intentionally local-only and ignored by git.

## Transcription And Translation Chain

| Priority | Provider | Use |
| --- | --- | --- |
| 1 | MERaLiON ASR | `POST http://meralion.org:8010/audio/transcription` with base64 audio, timestamps, and diarization. |
| 2 | MERaLiON audio translation | `POST http://meralion.org:8010/audio/translation` for non-English English transcript generation. |
| 3 | ElevenLabs speech-to-text | Original transcript fallback when MERaLiON fails. |
| 4 | Google Translate | English translation fallback for non-English transcript text. |
| 5 | Saved dialogue transcript | Final demo fallback so calls still save. |

If a fallback is used, the saved call marks `translationFallbackUsed=true`. Every saved call also includes `transcriptionAttempts`, with each provider marked `success`, `failed`, or `skipped` plus a sanitized reason when available.

## AI Risk Review

OpenAI is used for structured decision-support extraction when `OPENAI_API_KEY` is configured. The default model is controlled by `OPENAI_MODEL` and is currently `gpt-4o-mini`.

The risk review prompt is constrained to patient speech only. Agent questions and agent summaries are ignored. Returned signals are validated against patient transcript segments before saving. Live ElevenLabs transcript message roles are the source of truth when provider speaker labels are missing, generic, or ambiguous.

OpenAI returns:

- risk level: `Green`, `Watch`, `Amber`, or `Red`
- concise reasons
- recommended follow-up action
- exact English patient evidence text
- patient sentence index and timestamp when available

If OpenAI is unavailable, the call still saves with `aiRiskFallbackUsed=true` and a manual-review recommendation.

EarlyCare does not diagnose medical conditions.

## Safeguard And Tone Review

The safeguard review is separate from the clinical risk review. It uses OpenAI structured output and only patient speech to classify:

- `None`
- `Support`
- `Urgent`
- `Emergency`

It returns a category, exact patient evidence, recommended action, and optional Singapore resources from the backend's configured resource list. If the safeguard level maps to a higher visible risk level than the clinical risk review, the call risk is lifted.

Tone context comes from ElevenLabs conversation data collection, not OpenAI. The backend looks for the `user_emotional_state` result, accepts summary text or JSON, and maps per-response emotion entries to patient transcript segments by `responseIndex` or by order when counts match. Negative, high-confidence tone can lift a green assessment to `Watch`; it does not lower higher clinical or safeguard risk.

## Optional Speech ML Research

The backend includes a trained conversational-compatible Parkinsonian speech-marker model path. Runtime scoring runs when `EARLYCARE_SPEECH_MODEL_ENABLED=true`; optional ML imports are lazy so missing speech dependencies produce warnings instead of blocking normal call saving.

The runtime model uses exactly these 10 features:

```text
MDVP:Fo(Hz)
MDVP:Fhi(Hz)
MDVP:Flo(Hz)
MDVP:Jitter(%)
MDVP:Jitter(Abs)
MDVP:RAP
MDVP:PPQ
Jitter:DDP
NHR
HNR
```

The runtime extractor no longer computes shimmer or nonlinear UCI fields for model input. The excluded source fields are recorded in `model_card.json`.

Install optional dependencies:

```bash
pip install -r requirements-ml.txt
```

Retrain and evaluate tabular models:

```bash
PYTHONPATH=. python scripts/train_parkinsons_tabular_model.py data/parkinsons.data --output-dir models/speech
```

The bundled `data/parkinsons.data` uses the [Kaggle Parkinson's Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set), mirrored from the source [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/174/parkinsons). The trainer compares logistic regression, random forest, gradient boosting, SVM, XGBoost, and LightGBM when installed, then saves `parkinsons_tabular_model.joblib`, `feature_schema.json`, `feature_reference_ranges.json`, `metrics.json`, and `model_card.json`. It uses grouped splits by subject ID from the `name` column when possible. The checked-in model is `earlycare-conversational-parkinsons-marker-random_forest-v0`.

Runtime inference scores derived `patient-speech.wav`, which contains stitched voiced patient clips rather than the full mic-only timeline. Longer derived clips are scored in shorter patient-speech chunks and aggregated with a median probability. `feature_reference_ranges.json` gates the selected feature space; one out-of-range feature yields low confidence, while three or more out-of-range features make the score unavailable. Returned model values are screening research signals, not a Parkinson's diagnosis, and do not currently determine the main call risk level.

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
| `OPENAI_SAFEGUARD_MODEL` | Safeguard review | Optional override; falls back to `OPENAI_MODEL`. |
| `EARLYCARE_SPEECH_MODEL_ENABLED` | Speech marker scoring | Example config is `true`; install optional ML dependencies for full scoring, otherwise calls still save with model warnings. |

Never commit real `.env` files.

## Main Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /health` | API health check. |
| `GET /seniors` | Demo senior roster. |
| `GET /calls` | Saved call records, newest first. |
| `GET /calls/{call_id}` | One saved call record. |
| `GET /calls/{call_id}/audio` | Replayable saved full-call recording. |
| `GET /calls/{call_id}/patient-audio` | Saved patient-only microphone recording for ML research. |
| `GET /calls/{call_id}/patient-speech-audio` | Derived patient-turn-only audio used for speech-marker scoring. |
| `POST /calls` | Save transcript messages and uploaded audio. |
| `POST /elevenlabs/signed-url` | Create a signed Agents session URL. |
| `GET /checkins` | Demo historical check-ins. |
| `POST /checkins/start` | Start a demo check-in record. |
| `POST /checkins/{checkin_id}/audio` | Legacy/demo transcription endpoint. |
| `POST /checkins/{checkin_id}/complete` | Complete a demo check-in record. |
| `GET /volunteer-tasks` | Demo volunteer task list. |

## Storage

Saved calls are written to:

```text
backend/storage/calls/{call_id}/
```

Each call can include:

- `metadata.json`
- `full-call.wav`
- `patient-audio.wav`
- `patient-speech.wav`
- `transcript-original.json`
- `transcript-english.txt`

`metadata.json` includes the final transcript provider, fallback flag, provider attempt trail, live transcript messages, role-labeled transcript segments, AI risk fallback status, safeguard fields, tone/emotion fields, speech profile metrics, and speech-marker quality fields when available.

Generated storage is ignored by git and can be cleared between demo runs. This is intentionally simple for hackathon speed; use a database and object storage before any real pilot.

## Smoke Checks

From the repository root:

```bash
backend/.venv/bin/python -m py_compile backend/app/*.py
PYTHONPATH=backend backend/.venv/bin/python -m unittest discover backend/tests
PYTHONPATH=backend backend/.venv/bin/python -m unittest backend.tests.test_speech_ml
```

With the API running:

```bash
curl http://127.0.0.1:8000/health
```
