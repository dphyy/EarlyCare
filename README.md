# EarlyCare

**Preventive care and patient engagement for elderly people living alone.**

EarlyCare is a hackathon prototype for routine wellbeing calls. A patient starts a simulated browser call, an ElevenLabs agent conducts the check-in, the app records the full conversation plus patient-only microphone audio, and the Patient overview shows the recording, original transcript, English transcript, patient-only AI risk highlights, and speech signal quality context for caregiver review.

EarlyCare is decision support, not diagnosis. It helps care teams notice risk signals such as falls, dizziness, sickness, confusion, weakness, poor intake, missed check-ins, or requests for help earlier.

## Key Features

| Area | What EarlyCare Does |
| --- | --- |
| Agents call | Starts an ElevenLabs Agents-powered browser call from a transcript-free animated call screen. The patient can speak in the language they are comfortable with. |
| Full-call recording | Requests browser echo cancellation, noise suppression, and auto gain control, then records patient microphone audio and ElevenLabs agent audio into one replayable `full-call.wav`. |
| Patient-only audio | Saves raw `patient-audio.wav` and derives `patient-speech.wav` by isolating voiced patient answers between agent turns for speech-model scoring. |
| Patient overview | Shows saved recordings, translated English transcript, original transcript, speech signal quality, risk review, and follow-up recommendation. |
| Transcription and translation | Uses MERaLiON first, ElevenLabs speech-to-text and Google Translate as fallback, and saved dialogue transcript only as the final demo fallback. |
| Inline risk highlights | Uses OpenAI structured output to detect patient-only risk signals and highlights exact English evidence inline when AI review succeeds. |
| Audio verification | Clicking a highlighted patient phrase seeks playback to immediately after the previous agent question, so caregivers can hear the patient answer in context. |
| Speech signal quality | Shows derived patient-speech duration, speech coverage, response latency, speaking rate, and model readiness against recent-call baselines. |

## Workflow

1. Patient starts the simulated call from the **Agents call** page.
2. The ElevenLabs agent conducts the wellbeing check-in with concise turn-by-turn questions and no required repeat phrase.
3. The frontend captures:
   - live dialogue messages internally for saved-call review, without rendering a live transcript during the call
   - mixed full-call audio containing patient and agent voice
   - patient-only microphone audio for downstream speech ML, using browser microphone cleanup when available
4. The backend saves `full-call.wav`, raw `patient-audio.wav`, derived `patient-speech.wav`, and call metadata.
5. The backend attempts transcript generation in this order:
   - MERaLiON `http://meralion.org:8010/audio/transcription`
   - MERaLiON `http://meralion.org:8010/audio/translation`
   - ElevenLabs speech-to-text for original transcript fallback
   - Google Translate for English translation fallback
   - saved dialogue transcript as final demo fallback
   - each provider attempt is saved with success/failure/skipped status for debugging
6. The backend stores:
   - original transcript with `Agent:` and `Patient:` speaker labels
   - English transcript with `Agent:` and `Patient:` speaker labels
   - timestamped transcript segments
   - provider/fallback metadata and sanitized provider attempt reasons
   - speech profile metrics
   - patient-only audio, derived patient-speech audio, and speech-model quality fields when configured
7. OpenAI reviews patient speech only and returns structured risk signals when configured; otherwise the dashboard shows manual review status without inline AI highlights.
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
| Speech model | UCI/Kaggle tabular voice-feature model | Optional high-risk speech-marker probability from patient-only audio. |
| Persistence | Local filesystem | Hackathon-friendly storage under `backend/storage/calls/`. |

## Speech ML Research Path

EarlyCare includes a Parkinsonian speech-marker research path trained on UCI/Kaggle tabular voice features. The repo includes `backend/data/parkinsons.data` and trained artifacts under `backend/models/speech/`; runtime ML imports are lazy so the core app can still save calls if optional speech dependencies are unavailable.

Recommended training path:

1. Use the bundled `backend/data/parkinsons.data`, downloaded from the [Kaggle Parkinson's Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set), or replace it with the source [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/174/parkinsons). Cite the dataset when using it.
2. Install optional ML dependencies:

```bash
cd backend
source .venv/bin/activate
pip install -r requirements-ml.txt
```

3. Train and evaluate tabular models:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/train_parkinsons_tabular_model.py backend/data/parkinsons.data --output-dir backend/models/speech
```

The current saved winner is a calibrated random forest selected by grouped cross-validation ROC-AUC. Runtime inference builds `patient-speech.wav` from voiced patient answer regions between agent turns, then scores manageable patient-speech chunks and aggregates the median probability. These features were designed for controlled voice recordings, so conversational EarlyCare audio is still quality-gated. When extracted patient speech is too short, silent, clipped, or severely unstable, the app shows "Speech marker unavailable" or "Speech marker low confidence" instead of a misleading percentage.

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
EARLYCARE_SPEECH_MODEL_ENABLED=true
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

Install optional speech-model dependencies when retraining or using runtime speech-marker scoring:

```bash
pip install -r requirements-ml.txt
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
7. Review the full-call recording, English transcript, original transcript, speech signal quality, and inline risk highlights.
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

- Replace heuristic speech-profile estimates with validated audio-derived features.
- Train and validate the optional UCI/Kaggle tabular speech-marker model with grouped evaluation when subject IDs are available.
- Improve audio/transcript alignment with provider word-level timestamps when available.
- Validate risk categories with clinicians and labelled datasets before real-world deployment.
- Add persistent database/object storage for multi-user demos.
- Add consent, retention, audit, and access-control controls before any real pilot.
