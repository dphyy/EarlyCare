# EarlyCare Frontend

React + Vite interface for the EarlyCare hackathon prototype.

The frontend provides two main experiences:

- **Agents call**: an in-browser call simulation powered by ElevenLabs Agents.
- **Patient overview**: a care-team view for recordings, transcripts, speech timing, conversational speech-marker readiness, inline AI risk/safeguard/tone highlights, and follow-up recommendations.

## Agents Call

The Agents call page:

- lets a volunteer select a senior and start a browser-based voice check-in
- requests microphone permission
- starts an ElevenLabs Agents session through a backend signed URL
- keeps live transcript messages internally without rendering a live transcript during the call
- reminds the agent to respond in the language/dialect used most in the patient's previous response
- records patient-only microphone audio with browser echo cancellation, noise suppression, and auto gain control when available
- decodes ElevenLabs agent audio packets from the SDK
- mixes patient and agent audio into one full-call upload
- uploads transcript messages, full-call audio, patient-only audio, ElevenLabs conversation ID, and `agentAudioCaptured` metadata to FastAPI when the call ends

The call page does not translate the live transcript. Translation happens after saving, in the backend workflow.

## Patient Overview

The Patient overview:

- lists living-alone seniors and their saved calls
- shows the full-call recording in a browser audio player
- shows translated English transcript first
- shows original transcript below it
- preserves `Agent:` and `Patient:` speaker labels
- shows current speech timing and baseline context
- shows derived patient-speech duration, speech coverage, and conversational speech-marker readiness
- shows OpenAI risk review status and recommended action
- shows OpenAI distress safeguard status and Singapore resource text when returned
- shows ElevenLabs tone/emotion summary when `user_emotional_state` data collection is available
- renders risk evidence inline inside the English transcript
- renders safeguard and tone evidence inline when segment evidence can be mapped
- highlights patient speech only for AI-generated evidence
- starts audio playback from immediately after the previous agent question when a highlight is clicked

Risk, safeguard, and tone signals are not rendered as separate evidence cards. The transcript itself is the review surface.

## Speech Signal Display

The frontend does not run the Parkinsonian speech-marker model. It displays backend fields from the saved call:

- `speechModelProbability`
- `speechModelVersion`
- `speechModelWarnings`
- `speechModelFeaturesSummary`
- `patientSpeechAudioAvailable`

The displayed speech-marker description says the backend scored patient-only pitch, jitter, and noise features. It is shown as a research screening signal, not a diagnosis. If the backend reports low confidence or unavailable, the UI shows that state instead of inventing a percentage.

## Local Setup

Create the local env file:

```bash
cp .env.example .env
```

Expected value:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Install dependencies:

```bash
npm install
```

Start the dev server:

```bash
npm run dev
```

The app usually runs at `http://localhost:5173`.

## Commands

| Command | Description |
| --- | --- |
| `npm run dev` | Start the Vite dev server. |
| `npm run lint` | Run TypeScript checks without emitting files. |
| `npm run build` | Type-check and build the production frontend. |

## Main Files

| File | Purpose |
| --- | --- |
| `src/main.tsx` | Main React app, Agents call flow, Patient overview, audio mixing, transcript highlights. |
| `src/api.ts` | Backend API helpers and audio URL construction. |
| `src/types.ts` | Shared frontend types for seniors, calls, transcripts, speech profiles, risk signals, safeguards, and emotion/tone segments. |
| `src/styles.css` | Application styling. |

## ElevenLabs Notes

The frontend avoids restricted ElevenLabs config overrides such as `first_message`. Multilingual behavior is sent through runtime context updates and dynamic variables.

If the ElevenLabs agent has a dashboard prompt that says it must always respond in English, update that agent prompt to allow multilingual replies and language switching. The app will remind the agent to follow the patient's most recent dominant language, but the agent's base prompt still matters.

Tone context depends on an ElevenLabs data collection result named `user_emotional_state`. The backend can use a plain summary, but transcript tone tags require JSON responses with per-response emotion entries or a response count that can be mapped to patient turns.

## Provider Secrets

The frontend does not store provider secrets.

Keep these in `backend/.env`:

- ElevenLabs API key and agent ID
- MERaLiON API key
- Google Translate API key
- OpenAI API key and model
- optional OpenAI safeguard model override

If the backend is unavailable, the app can still load local demo data, but live call saving, saved audio playback, transcription, translation, risk review, safeguard review, tone ingestion, and speech-marker display require FastAPI.
