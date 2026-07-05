# EarlyCare

**Preventive care and patient engagement for elderly people living alone.**

EarlyCare is a hackathon prototype for routine wellbeing calls. A patient starts a simulated browser call, an ElevenLabs agent conducts the check-in, the app records the full conversation, and the Patient overview shows the recording, original transcript, English transcript, patient-only AI risk highlights, and speech timing context for caregiver review.

EarlyCare is decision support, not diagnosis. It helps volunteers and officers notice missed check-ins, reported danger signs, possible post-fall/concussion concerns, Parkinson's watch signals, poor intake, confusion, weakness, requests for help, and meaningful speech changes sooner.

`PRODUCT_CONTEXT.md` is the current source of truth for product framing, source links, safety language, and demo scope. Older planning notes outside this repository may contain stale stroke/clinic framing.

`docs/ml/implementation-plan.md` captures the current ML direction, dataset findings, and staged implementation plan.

## Why It Matters

Older adults living alone may go days without anyone noticing a fall, head impact, confusion, poor intake, or a sudden change in speech. EarlyCare creates a lightweight check-in workflow that can be run every 2-3 days and escalated when the conversation suggests that follow-up is needed.

## Key Features

| Area | What EarlyCare Does |
| --- | --- |
| Scenario runner | Runs seven scripted demo paths: stable, missed check-in, Parkinson's watch, Post-Fall Amber, Post-Fall Red, chronic illness, and loneliness/wellbeing. |
| Check-in schedule | Computes next due time, due/overdue status, last contact, and next action from each senior's 2-3 day cadence. |
| Agents website call | Starts an ElevenLabs Agents-powered browser call from the EarlyCare website and lets the patient speak in a comfortable language. |
| Full-call recording | Records patient microphone audio and ElevenLabs agent audio into one replayable `full-call.webm`. |
| Patient overview | Shows historical check-ins, saved calls, translated transcripts, original recordings, categorized evidence, escalation trails, volunteer tasks, speech timing, and risk highlights. |
| Transcription and translation | Uses MERaLiON first, ElevenLabs speech-to-text and Google Translate as fallback, and saved dialogue transcript only as the final demo fallback. |
| Inline risk highlights | Uses OpenAI structured output to detect patient-only risk signals and highlights exact English evidence inline. |
| Audio verification | Clicking a highlighted patient phrase seeks playback to immediately after the previous agent question, so caregivers can hear the patient answer in context. |
| Neurological watch | Flags speech and symptom patterns that may justify earlier follow-up for Parkinson's watch or post-fall/concussion review without presenting diagnosis. |
| Speech provenance | Labels speech timing as `demo metrics`, `offline embedding`, or `validated model`, with offline enrichment kept separate from live call saving. |
| Volunteer workflow | Creates hackathon-scope follow-up tasks for missed check-ins and elevated risk, with acknowledge/close actions backed by the API. |
| Safety stance | Avoids diagnosis language and frames alerts as prompts for volunteer or caregiver follow-up. |

## Workflow

1. Patient starts the simulated call from the **Agents call** page.
2. The ElevenLabs agent conducts the wellbeing check-in.
3. The frontend captures:
   - live dialogue messages in the original spoken language
   - mixed full-call audio containing patient and agent voice
4. The backend saves `full-call.webm` and call metadata.
5. The backend attempts transcript generation in this order:
   - MERaLiON `http://meralion.org:8010/audio/transcription`
   - MERaLiON `http://meralion.org:8010/audio/translation`
   - ElevenLabs speech-to-text for original transcript fallback
   - Google Translate for English translation fallback
   - saved dialogue transcript as final demo fallback
6. The backend stores:
   - original transcript with `Agent:` and `Patient:` speaker labels
   - English transcript with `Agent:` and `Patient:` speaker labels
   - timestamped transcript segments
   - provider/fallback metadata
   - speech timing metrics
7. OpenAI reviews patient speech only and returns structured risk signals.
8. The Patient overview renders the English transcript above the original transcript and highlights risk evidence inline.
9. Clicking a highlight plays the saved audio from immediately after the previous agent prompt.

## Architecture

| Layer | Stack | Role |
| --- | --- | --- |
| Frontend | React, Vite, TypeScript | Scenario runner, Agents call experience, mixed audio capture, Patient overview, audio playback, risk-signal UI. |
| Backend | FastAPI, Python | Signed Agents sessions, scenario persistence, volunteer task state, call artifact storage, transcription, translation, risk review, API routes. |
| Voice agent | ElevenLabs Agents React SDK | Live browser-based voice check-in and transcript events. |
| Transcription | MERaLiON, ElevenLabs STT | Primary and fallback speech-to-text. |
| Translation | MERaLiON, Google Translate | English transcript normalization for officer review. |
| AI review | OpenAI API | Structured patient-only risk extraction and highlighted evidence. |
| Persistence | Local filesystem | Hackathon-friendly storage under `backend/storage/calls/` and `backend/storage/state/`. |

## Setup

### 1. Create Env Files

```bash
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env
```

`frontend/.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

`backend/.env`:

```bash
ELEVENLABS_API_KEY=
ELEVENLABS_AGENT_ID=
ELEVENLABS_STT_MODEL=scribe_v2
MERALION_API_KEY=
MERALION_ASR_URL=http://meralion.org:8010/audio/transcription
MERALION_TRANSLATION_URL=http://meralion.org:8010/audio/translation
GOOGLE_TRANSLATE_API_KEY=
GOOGLE_TRANSLATE_URL=https://translation.googleapis.com/language/translate/v2
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

Never commit real `.env` files.

### 2. Install Frontend

```bash
pnpm --dir frontend install
```

### 3. Install Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run Locally

Start the backend:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Start the frontend:

```bash
pnpm dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Demo Flow

### Scripted Scenario Demo

1. Open **Demo runner**.
2. Run one of the seven scripted scenarios.
3. Open **Patient overview**.
4. Review historical check-ins, categorized evidence, risk scores, escalation steps, transcripts, and volunteer tasks.
5. Acknowledge or close a task to confirm PATCH-backed status persistence.

### Live Agents Call Demo

1. Open **Agents call**.
2. Choose a senior and click **Start call**.
3. Allow microphone permission.
4. Speak with the agent in any comfortable language.
5. Click **End & save**.
6. Open **Patient overview**.
7. Review the full-call recording, English transcript, original transcript, speech timing, and inline risk highlights.
8. Click a highlighted risk phrase to replay the patient answer from immediately after the previous agent question.

## Commands

| Command | Description |
| --- | --- |
| `pnpm dev` | Start the frontend dev server. |
| `pnpm lint` | Type-check the frontend. |
| `pnpm build` | Build the frontend. |
| `pnpm frontend:smoke` | Validate frontend demo data and key UI hooks. |
| `pnpm backend:smoke` | Run backend API smoke coverage with FastAPI TestClient. |
| `pnpm safety:copy` | Block diagnosis-style wording in user-facing app and README copy. |
| `uvicorn app.main:app --reload --port 8000` | Start the backend from the `backend/` folder. |
## Repository Guide

- `frontend/` contains the React + Vite interface.
- `backend/` contains the FastAPI service and provider integrations.
- `backend/tests/` contains backend workflow tests.
- `backend/storage/` contains generated local call artifacts and is ignored.
- `.env.example` files document configuration without secrets.

## Safety Positioning

EarlyCare does not diagnose Parkinson's disease, concussion, stroke, or any other medical condition. It surfaces concerning patient statements, missed check-ins, and changes from available speech baselines so a human volunteer, caregiver, or officer can follow up sooner.

## Roadmap

- Replace heuristic speech timing estimates with validated audio-derived features.
- Follow `docs/ml/implementation-plan.md` for speech-deviation model work and dataset validation.
- Keep the schedule endpoint and Patient overview schedule panel aligned with the 2-3 day living-alone check-in workflow.
- Improve audio/transcript alignment with provider word-level timestamps when available.
- Validate risk categories with clinicians and labelled datasets before real-world deployment.
- Add persistent database/object storage for multi-user demos.
- Add consent, retention, audit, and access-control controls before any real pilot.
