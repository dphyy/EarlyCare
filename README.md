# EarlyCare

**Preventive care and patient engagement for elderly people living alone.**

EarlyCare is a hackathon prototype that turns regular wellbeing check-ins into earlier volunteer action. It simulates an in-browser voice call with an elderly resident, saves the transcript and recording, translates the conversation to English, and highlights risk signals for care teams to review.

EarlyCare is designed for triage support, not diagnosis. It helps volunteers and officers notice missed check-ins, reported danger signs, possible post-fall/concussion concerns, Parkinson's watch signals, and meaningful speech changes sooner.

## Why It Matters

Older adults living alone may go days without anyone noticing a fall, head impact, confusion, poor intake, or a sudden change in speech. EarlyCare creates a lightweight check-in workflow that can be run every 2-3 days and escalated when the conversation suggests that follow-up is needed.

## Key Features

| Area | What EarlyCare Does |
| --- | --- |
| Agents website call | Starts an ElevenLabs Agents-powered browser call from the EarlyCare website. |
| Patient overview | Shows saved calls, translated transcripts, original recordings, risk signals, and volunteer tasks. |
| Audio capture | Records the browser microphone in parallel and stores replayable call audio locally. |
| Translation pipeline | Uses MERaLiON first, Google Translate second, and transcript fallback last. |
| AI risk review | Uses OpenAI structured output to identify decision-support risk signals from the English transcript. |
| Neurological watch | Flags speech and symptom patterns that may justify earlier follow-up for Parkinson's watch or post-fall/concussion review. |
| Safety stance | Avoids diagnosis language and frames alerts as prompts for volunteer or caregiver follow-up. |

## Early-Warning Logic

EarlyCare combines regular engagement with speech and transcript review. The current prototype focuses on whether a check-in suggests that a volunteer, caregiver, or officer should follow up sooner.

| Signal | What EarlyCare Looks For | Output |
| --- | --- | --- |
| Speech deviation from baseline | The senior's current speech is compared against their own usual pattern, with future support for speech embeddings from models such as wav2vec 2.0, WavLM, or MERaLiON SpeechEncoder. | A deviation-focused risk summary, not a diagnosis. |
| Parkinson's watch | Repeated or meaningful changes such as slower speech, longer pauses, reduced clarity, or reduced vocal variation can be surfaced as watch signals for human review. | Earlier monitoring or caregiver follow-up if the pattern persists. |
| Post-fall/concussion concern | Mentions of falls, head impact, headache, dizziness, vomiting, confusion, weakness, or unusual speech after a fall are highlighted for urgent review. | Amber or Red follow-up recommendation depending on severity. |
| Missed check-ins | Silence or failure to complete scheduled check-ins can indicate that a volunteer should check in. | Volunteer task or escalation prompt. |

These signals are intentionally explainable. The app highlights the evidence, lets officers replay the original audio, and keeps the final decision with a human.

## Architecture

| Layer | Stack | Role |
| --- | --- | --- |
| Frontend | React, Vite, TypeScript | Agents call experience, Patient overview, audio playback, risk-signal UI. |
| Backend | FastAPI, Python | Signed Agents sessions, call artifact storage, translation, risk review, API routes. |
| Voice agent | ElevenLabs Agents React SDK | Live browser-based voice check-in. |
| Translation | MERaLiON, Google Translate | English transcript normalization for officer review. |
| AI review | OpenAI API | Structured symptom/risk extraction and highlighted evidence. |
| Persistence | Local filesystem | Hackathon-friendly storage under `backend/storage/calls/`. |

## Provider Flow

1. A volunteer starts an Agents website call.
2. The browser captures live transcript messages and microphone audio.
3. FastAPI saves the original transcript and `mic-audio.webm`.
4. The backend translates to English using:
   - MERaLiON audio translation
   - Google Translate text translation if MERaLiON fails
   - original transcript fallback if both providers are unavailable
5. OpenAI reviews the English transcript and returns structured risk signals for missed check-ins, reported symptoms, Parkinson's watch, and post-fall/concussion concern.
6. Patient overview shows the recording, transcripts, risk summary, and follow-up recommendation.

## Setup

### 1. Clone and Prepare Env Files

```bash
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env
```

`frontend/.env` should point to the backend:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

`backend/.env` may contain:

```bash
ELEVENLABS_API_KEY=
ELEVENLABS_AGENT_ID=
MERALION_API_KEY=
MERALION_ASR_URL=
GOOGLE_TRANSLATE_API_KEY=
GOOGLE_TRANSLATE_URL=https://translation.googleapis.com/language/translate/v2
OPENAI_API_KEY=
```

Never commit real `.env` files.

### 2. Install Frontend

```bash
cd frontend
npm install
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
cd frontend
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Demo Flow

1. Open **Agents call**.
2. Choose a senior and click **Start call**.
3. Allow microphone permission.
4. Speak with the agent and click **End & save**.
5. Open **Patient overview**.
6. Review the recording, original transcript, English transcript, AI risk review, and volunteer follow-up action.

## Commands

| Command | Description |
| --- | --- |
| `npm run dev --prefix frontend` | Start the frontend dev server. |
| `npm run lint --prefix frontend` | Type-check the frontend. |
| `npm run build --prefix frontend` | Build the frontend. |
| `uvicorn app.main:app --reload --port 8000` | Start the backend from the `backend/` folder. |

## Safety Positioning

EarlyCare does not diagnose Parkinson's disease, concussion, stroke, or any other medical condition. Parkinson's watch and post-fall/concussion concern are decision-support categories only. The product surfaces concerning statements, missed check-ins, and possible deviation from a personal baseline so a human volunteer, caregiver, or officer can follow up sooner.

## Roadmap

- Replace heuristic baseline placeholders with validated speech embeddings from models such as wav2vec 2.0, WavLM, or MERaLiON SpeechEncoder.
- Validate Parkinson's watch and post-fall/concussion markers with clinicians and labelled datasets before any real-world deployment.
- Add stronger timestamp alignment for sentence-level audio playback.
- Add persistent storage beyond local filesystem for multi-user demos.
- Add volunteer task assignment and acknowledgement flows.
- Add consent, retention, and audit controls before any real-world pilot.

## Repository Guide

- `frontend/` contains the React + Vite user interface.
- `backend/` contains the FastAPI service and provider integrations.
- `backend/storage/` is local generated data and is intentionally ignored.
- `.env.example` files document required configuration without secrets.
