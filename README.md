# EarlyCare

**Preventive care and patient engagement for elderly people living alone.**

EarlyCare is a hackathon prototype for routine wellbeing calls operated as an AIC/community care-coordinator monitoring layer. A patient starts a simulated browser call, an ElevenLabs agent conducts the check-in, the app records the full conversation plus patient-only microphone audio, and the Care Desk shows the recording, original transcript, English transcript, patient-only AI risk highlights, distress safeguard status, tone context, patient speech quality, and a printable Doctor Brief for point-of-care handoff.

EarlyCare is decision support, not diagnosis. It helps care teams notice risk signals such as falls, dizziness, sickness, confusion, weakness, poor intake, missed check-ins, or requests for help earlier.

## Key Features

| Area | What EarlyCare Does |
| --- | --- |
| Agents call | Starts an ElevenLabs Agents-powered browser call from a transcript-free animated call screen. The patient can speak in the language they are comfortable with. |
| Full-call recording | Requests browser echo cancellation, noise suppression, and auto gain control, then records patient microphone audio and ElevenLabs agent audio into one replayable `full-call.wav`. |
| Patient-only audio | Saves raw `patient-audio.wav` and derives `patient-speech.wav` by isolating voiced patient answers for saved-model speech review. |
| Care Desk | Shows saved recordings, translated English transcript, original transcript, patient speech quality, explainable model review cards, risk/safeguard/tone review, and follow-up recommendation. |
| Consultation memory | Extracts dated, evidence-backed patient facts from check-ins, such as falls, medication/meal concerns, symptoms, mood, mobility, sleep, help-seeking, and appointment mentions. |
| Doctor Brief | Generates a printable one-page **EarlyCare Consultation Brief** for AIC/care coordinators to share before a clinic visit or when risk rises, without asking doctors to manage another dashboard. |
| Transcription and translation | Uses MERaLiON first, ElevenLabs speech-to-text and Google Translate as fallback, and saved dialogue transcript only as the final demo fallback. |
| Inline risk highlights | Uses OpenAI structured output to detect patient-only risk signals and highlights exact English evidence inline when AI review succeeds. |
| Distress safeguard | Uses a separate OpenAI structured review for patient-stated distress, self-harm, abuse/neglect, unsafe environment, or emergency cues; safeguard levels can lift visible call risk. |
| Tone context | Reads ElevenLabs `user_emotional_state` data collection results when available and highlights per-response emotion evidence in the transcript. |
| Audio verification | Clicking a highlighted patient phrase seeks playback to immediately after the previous agent question, so caregivers can hear the patient answer in context. |
| Patient speech quality | Shows derived patient-speech duration, speech coverage, response latency, speaking rate, Parkinson model readiness, and concussion review readiness. |
| Model explainability | Shows concise explanation bullets above Parkinson and concussion speech outputs, using top voice-feature signals for Parkinson and probability/audio-quality context for concussion without claiming diagnosis or WavLM feature attribution. |
| Demo | Opens a separate scripted demo view from the top navigation so judges can review curated demo cases and tailored demo volunteer tasks without overwriting real saved calls, audio, or Care Desk tasks. |

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
   - patient-only audio, derived patient-speech audio, and saved-model quality fields
7. OpenAI reviews patient speech only and returns structured risk signals when configured; otherwise the Care Desk shows manual review status without inline AI highlights.
8. A separate OpenAI safeguard review classifies patient-stated distress as `None`, `Support`, `Urgent`, or `Emergency`, attaches exact patient evidence, and can raise the visible risk level.
9. ElevenLabs data collection is queried for `user_emotional_state`; per-response emotion tags are attached to patient transcript segments when the returned JSON includes response indexes or can be mapped by order.
10. The backend extracts consultation-memory items from patient speech only. Each item must be backed by exact patient evidence and a dated check-in.
11. The backend scores derived `patient-speech.wav` with the saved Parkinson voice-feature model and, only after patient-stated fall or near-fall evidence, the saved concussion speech-abnormality model.
12. Parkinson explanations are generated from the top pitch, jitter, and harmonic/noise feature groups against `feature_reference_ranges.json`.
13. Concussion explanations summarize applicability, predicted label/probability gap, abnormal-class probability, and audio quality metrics without claiming WavLM feature attribution.
14. The Care Desk shows a **Patient speech quality** panel for shared audio/model readiness and separate Parkinson/concussion cards for each model's interpretation.
15. The Care Desk renders the English transcript above the original transcript and highlights risk, safeguard, and tone evidence inline.
16. The Care Desk includes a printable **EarlyCare Consultation Brief** with patient details, reporting window, risk trend, grouped memory items, exact quotes, and a decision-support disclaimer.
17. Clicking a highlight plays the saved audio from immediately after the previous agent prompt.

## Architecture

| Layer | Stack | Role |
| --- | --- | --- |
| Frontend | React, Vite, TypeScript | Agents call UI, full-call recording, AIC Care Desk, printable Doctor Brief, inline highlights, audio seeking. |
| Backend | FastAPI, Python | Signed ElevenLabs sessions, call artifact storage, transcription, translation, OpenAI risk/safeguard review, consultation-memory extraction, tone ingestion, Parkinson and concussion speech review. |
| Voice agent | ElevenLabs Agents React SDK | Live browser voice check-in and live transcript events. |
| Transcription | MERaLiON, ElevenLabs STT | Primary and fallback speech-to-text. |
| Translation | MERaLiON, Google Translate | English transcript generation for caregiver review. |
| AI review | OpenAI API | Structured patient-only risk extraction and separate distress safeguard classification. |
| Consultation memory | OpenAI structured output plus deterministic fallback | Dated, exact-quote-backed facts for AIC monitoring and doctor handoff. |
| Emotion/tone | ElevenLabs data collection | Optional `user_emotional_state` summary and per-response tags. |
| Parkinson speech marker | Saved conversational-compatible tabular voice-feature model | Post-call Parkinson marker probability from patient-only pitch, jitter, and noise features. |
| Concussion speech review | Vendored WavLM speech-abnormality inference path | Post-call `patient-speech.wav` review for `normal`, `dysarthria_like`, `dysphonia_like`, or `low_audio_quality` research labels. |
| Persistence | Local filesystem | Hackathon-friendly storage under `backend/storage/calls/`. |

## Parkinson Speech Research Path

EarlyCare includes a saved Parkinsonian speech-marker inference path trained on the conversational-compatible subset of UCI/Kaggle tabular voice features. The repo includes `backend/data/parkinsons.data` and trained artifacts under `backend/models/parkinsons_speech/`; new patient speech is passed through the saved model without retraining.

The checked-in runtime schema is exactly:

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

The model intentionally excludes shimmer fields (`MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`) and nonlinear UCI fields (`RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`) because the previous runtime approximations were not comparable enough for conversational audio.

Recommended training path:

1. Use the bundled `backend/data/parkinsons.data`, downloaded from the [Kaggle Parkinson's Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set), or replace it with the source [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/174/parkinsons). Cite the dataset when using it.
2. Install training extras:

```bash
cd backend
source .venv/bin/activate
pip install -r requirements-ml.txt
```

3. Train and evaluate tabular models:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/train_parkinsons_tabular_model.py backend/data/parkinsons.data --output-dir backend/models/parkinsons_speech
```

The current saved winner is `earlycare-conversational-parkinsons-marker-random_forest-v0`, selected by grouped cross-validation ROC-AUC using only the 10 transferable pitch, jitter, and harmonic/noise fields. Runtime inference builds `patient-speech.wav` from voiced patient answer regions between agent turns, then scores manageable patient-speech chunks and aggregates the median probability. `feature_reference_ranges.json` stores the selected training ranges, and inference reports low confidence or unavailable when extracted patient speech is too short, silent, clipped, severely unstable, or outside those ranges.

The Parkinson marker score is saved as `parkinsonsSpeechReview.probability` and mirrored to legacy `speechModelProbability` for older dashboard compatibility. Each saved review also stores `parkinsonsSpeechReview.explanations`, a short list of top feature-group explanations such as jitter stability, pitch range, and harmonic/noise clarity. It does not diagnose Parkinson's disease and does not currently determine the call's main `riskLevel`; the visible risk level comes from AI risk review, safeguard review, tone modifiers, and concussion speech review when relevant.

## Concussion Speech Review

The backend runs the bundled speech-abnormality model after a call is saved when
derived patient speech is available and the patient stated a fall or near-fall.
It scores the derived patient-only speech file, stores the result as
`concussionSpeechReview`, and shows it in the Care Desk.

The repo includes the runtime adapter and vendored inference code under
`backend/app/concussion_speech_model/`, plus trained pilot artifacts under
`backend/models/concussion_speech/`. Training datasets, embedding caches, and raw
TORGO/VOICED files are intentionally not required for local website inference and
should not be pushed. Cache the configured WavLM backbone locally during setup so
demos do not depend on a first-run network download.

If the patient does not state a fall or near-fall, EarlyCare intentionally skips
the concussion speech review and records
`concussionSpeechReview.applicability = "not_applicable"`. The Care Desk shows
**Not applicable**, not an unavailable model.

This is not concussion detection or diagnosis. The model returns research labels
only. If the patient reports concussion-relevant symptoms and the speech model
also flags abnormal speech, EarlyCare raises the call for human review. If the
model flags speech without reported symptoms, it is treated as a watch-level
audio review signal. Dataset, backbone, and classifier citations are listed in
the References section.

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
OPENAI_SAFEGUARD_MODEL=
EARLYCARE_CONCUSSION_SPEECH_DEVICE=cpu
```

Never commit real `.env` files.

### 2. Install Frontend

```bash
cd frontend
pnpm install
```

### 3. Install Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cache the WavLM backbone for deterministic demo readiness:

```bash
cd ..
backend/.venv/bin/python backend/scripts/cache_wavlm.py
```

This downloads `microsoft/wavlm-base` into the ignored local cache at
`backend/models/hf_cache/`. Do not commit that cache directory.

Install training extras only when retraining or experimenting:

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
pnpm dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Demo Flow

### Scripted Judge Demo

1. Click **Demo** in the top navigation.
2. Review the scripted frontend-only demo cases and tailored demo volunteer tasks. This view is separate from **Care Desk**, so real saved calls, audio, and live tasks remain untouched.
3. Check the fall escalation case, Parkinson/frailty watch case, safeguard support case, multilingual routine case, and concussion **Not applicable** cases.
4. Review the model explanation bullets above the Parkinson and concussion outputs.
5. Click transcript highlights to scroll to supporting patient evidence when demo audio is absent.

### Live Call Demo

1. Open **Agents call**.
2. Choose a senior and click **Start call**.
3. Confirm the recording notice and demo consent checkbox.
4. Allow microphone permission.
5. Speak with the agent in any comfortable language.
6. Click **End & save**.
7. Open **Care Desk**.
8. Review the full-call recording, English transcript, original transcript, patient speech quality, model review cards, and inline risk/safeguard/tone highlights.
9. Review or print the **EarlyCare Consultation Brief** as the doctor-facing handoff summary.
10. Click a highlighted patient phrase to replay the patient answer from immediately after the previous agent question.

## Commands

| Command | Description |
| --- | --- |
| `pnpm --dir frontend dev` | Start the frontend dev server. |
| `pnpm --dir frontend build` | Type-check and build the frontend. |
| `PYTHONPATH=backend backend/.venv/bin/python -m unittest discover backend/tests` | Run backend tests. |
| `PYTHONPATH=backend backend/.venv/bin/python -m unittest backend.tests.test_speech_ml` | Run focused Parkinson speech-marker tests. |
| `backend/.venv/bin/python backend/scripts/cache_wavlm.py` | Download `microsoft/wavlm-base` into the ignored local Hugging Face cache used by readiness and concussion inference. |
| `pnpm --dir frontend lint` | Run frontend TypeScript checks. |
| `backend/.venv/bin/python -m py_compile backend/app/*.py` | Compile-check backend modules. |
| `uvicorn app.main:app --reload --port 8000` | Start the backend from the `backend/` folder. |

## Repository Guide

- `frontend/` contains the React + Vite interface.
- `backend/` contains the FastAPI service and provider integrations.
- `backend/models/parkinsons_speech/` contains the checked-in Parkinson speech-marker artifacts: model, schema, metrics, model card, and reference ranges.
- `backend/models/concussion_speech/` contains the checked-in concussion speech-abnormality pilot artifacts needed for inference.
- `backend/models/hf_cache/` is an ignored local Hugging Face cache created by `backend/scripts/cache_wavlm.py`.
- `backend/app/concussion_speech_model/` contains the vendored speech-abnormality inference package.
- `backend/tests/` contains backend workflow tests.
- `backend/storage/` contains generated local call artifacts and is ignored.
- `.env.example` files document configuration without secrets.

## Safety Positioning

EarlyCare does not diagnose Parkinson's disease, concussion, stroke, or any other medical condition. It surfaces concerning patient statements, missed check-ins, and changes from available speech baselines so a human volunteer, caregiver, or officer can follow up sooner.

## Future Improvements

- Train and validate the Parkinson and concussion speech models on larger, more representative datasets before treating them as more than research signals.
- Include Singapore-context speech data where licensing and governance allow, such as IMDA's [National Speech Corpus](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus), so speech models and transcription checks are better calibrated to Singaporean accents, code-switching, local languages, and older-adult speech patterns.
- Personalize each agent call to the individual's medical history, care plan, medication list, known risks, preferred language/dialect, caregiver arrangement, and previous consultation-memory items.
- Validate risk categories, safeguard thresholds, and Doctor Brief content with clinicians, AIC/community care teams, patients, and caregivers before real-world deployment.
- Add persistent database/object storage for multi-user demos, including longitudinal reporting windows such as "since last clinic visit."
- Add consent, retention, audit, role-based access control, and export governance before any real pilot.

## References

### Parkinson Speech-Marker Dataset

- **Kaggle Parkinson's Disease Data Set**: bundled as `backend/data/parkinsons.data` for the saved Parkinson voice-feature model. Dataset page: [Kaggle Parkinson's Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set).
- **UCI Parkinsons Dataset**: source dataset mirrored by the Kaggle copy. Dataset page: [UCI Parkinsons](https://archive.ics.uci.edu/dataset/174/parkinsons).

### Concussion Speech-Abnormality Model

The bundled `backend/models/concussion_speech` artifacts are the runtime output of the internal `pilot_full` speech-abnormality model. The model is a research-only speech classifier, not a concussion, dysarthria, or dysphonia diagnostic system.

#### Training Datasets

- **TORGO Database**: acoustic and articulatory speech from speakers with dysarthria and matched controls. Used for `dysarthria_like` and `normal` training examples.
  - Site: [The TORGO Database: Acoustic and articulatory speech from speakers with dysarthria](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
  - Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. *Language Resources and Evaluation*, 46, 523-541.

- **VOICED Database v1.0.0**: healthy and pathological sustained-vowel voice samples from PhysioNet. Used for `dysphonia_like` and `normal` training examples.
  - Site: [VOICED Database v1.0.0](https://physionet.org/content/voiced/1.0.0/)
  - Cesari, U., De Pietro, G., Marciano, E., Niri, C., Sannino, G., & Verde, L. (2018). A new database of healthy and pathological voices. *Computers & Electrical Engineering*, 68, 310-321.
  - Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.

#### Feature Backbone And Classifier

- **WavLM Base**: frozen speech embedding backbone configured as `microsoft/wavlm-base` in `backend/models/concussion_speech/config.json`. Runtime audio is resampled to 16 kHz before embedding.
  - Model card: [microsoft/wavlm-base on Hugging Face](https://huggingface.co/microsoft/wavlm-base)
  - Paper: [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
  - Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE Journal of Selected Topics in Signal Processing*.

- **Classifier**: scikit-learn logistic regression with class balancing and calibration, saved in `backend/models/concussion_speech/model.joblib`.
  - API reference: [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

#### Runtime Labels

The checked-in model returns these research labels only: `normal`, `dysarthria_like`, `dysphonia_like`, and `low_audio_quality`. The labels are dataset-derived research categories. They are not medical diagnoses and must not be presented as concussion detection.
