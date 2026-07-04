# EarlyCare

**Preventive care and patient engagement for elderly people living alone.**

EarlyCare is a hackathon prototype for routine wellbeing calls. A patient starts a simulated browser call, an ElevenLabs agent conducts the check-in, the app records the full conversation, and the Patient overview shows the recording, original transcript, English transcript, patient-only AI risk highlights, and speech timing context for caregiver review.

EarlyCare is decision support, not diagnosis. It helps care teams notice risk signals such as falls, dizziness, sickness, confusion, weakness, poor intake, missed check-ins, or requests for help earlier.

## Key Features

| Area | What EarlyCare Does |
| --- | --- |
| Agents call | Starts an ElevenLabs Agents-powered browser call from the EarlyCare website. The patient can speak in the language they are comfortable with. |
| Full-call recording | Records patient microphone audio and ElevenLabs agent audio into one replayable `full-call.webm`. |
| Patient overview | Shows saved recordings, translated English transcript, original transcript, speech timing, risk review, and follow-up recommendation. |
| Transcription and translation | Uses MERaLiON first, ElevenLabs speech-to-text and Google Translate as fallback, and saved dialogue transcript only as the final demo fallback. |
| Inline risk highlights | Uses OpenAI structured output to detect patient-only risk signals and highlights the exact English evidence inline. |
| Audio verification | Clicking a highlighted patient phrase seeks playback to immediately after the previous agent question, so caregivers can hear the patient answer in context. |
| Speech timing | Tracks latest-call speech rate, average pause, response latency, pitch variability, and phrase accuracy against the latest available baseline. |

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
| Frontend | React, Vite, TypeScript | Agents call UI, full-call recording, Patient overview, inline highlights, audio seeking. |
| Backend | FastAPI, Python | Signed ElevenLabs sessions, call artifact storage, transcription, translation, OpenAI risk review. |
| Voice agent | ElevenLabs Agents React SDK | Live browser voice check-in and live transcript events. |
| Transcription | MERaLiON, ElevenLabs STT | Primary and fallback speech-to-text. |
| Translation | MERaLiON, Google Translate | English transcript generation for caregiver review. |
| AI review | OpenAI API | Structured patient-only risk extraction. |
| Persistence | Local filesystem | Hackathon-friendly storage under `backend/storage/calls/`. |

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
4. Speak with the agent in any comfortable language.
5. Click **End & save**.
6. Open **Patient overview**.
7. Review the full-call recording, English transcript, original transcript, speech timing, and inline risk highlights.
8. Click a highlighted risk phrase to replay the patient answer from immediately after the previous agent question.

## Commands

| Command | Description |
| --- | --- |
| `npm run dev --prefix frontend` | Start the frontend dev server. |
| `npm run build --prefix frontend` | Type-check and build the frontend. |
| `PYTHONPATH=backend backend/.venv/bin/python -m unittest discover backend/tests` | Run backend tests. |
| `backend/.venv/bin/python -m py_compile backend/app/*.py` | Compile-check backend modules. |
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
- Improve audio/transcript alignment with provider word-level timestamps when available.
- Validate risk categories with clinicians and labelled datasets before real-world deployment.
- Add persistent database/object storage for multi-user demos.
- Add consent, retention, audit, and access-control controls before any real pilot.
