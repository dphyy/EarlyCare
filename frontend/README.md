# EarlyCare Frontend

React + Vite interface for the EarlyCare hackathon prototype.

The frontend provides two main experiences:

- **Agents call**: an in-browser call simulation powered by ElevenLabs Agents.
- **Patient overview**: a care-team view for recordings, transcripts, speech timing, inline AI risk highlights, and follow-up recommendations.

## Agents Call

The Agents call page:

- lets a volunteer select a senior and start a browser-based voice check-in
- requests microphone permission
- starts an ElevenLabs Agents session through a backend signed URL
- keeps the live transcript in the original spoken language
- reminds the agent to respond in the language/dialect used most in the patient's previous response
- records patient microphone audio
- decodes ElevenLabs agent audio packets from the SDK
- mixes patient and agent audio into one `full-call.webm`
- uploads transcript messages, audio, and `agentAudioCaptured` metadata to FastAPI when the call ends

The call page does not translate the live transcript. Translation happens after saving, in the backend workflow.

## Patient Overview

The Patient overview:

- lists living-alone seniors and their saved calls
- shows the full-call recording in a browser audio player
- shows translated English transcript first
- shows original transcript below it
- preserves `Agent:` and `Patient:` speaker labels
- shows current speech timing and baseline context
- shows OpenAI risk review status and recommended action
- renders risk evidence inline inside the English transcript
- highlights patient speech only
- starts audio playback from immediately after the previous agent question when a highlight is clicked

Risk signals are not rendered as separate cards. The transcript itself is the review surface.

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
| `src/types.ts` | Shared frontend types for seniors, calls, transcripts, speech profiles, and risk signals. |
| `src/styles.css` | Application styling. |

## ElevenLabs Notes

The frontend avoids restricted ElevenLabs config overrides such as `first_message`. Multilingual behavior is sent through runtime context updates and dynamic variables.

If the ElevenLabs agent has a dashboard prompt that says it must always respond in English, update that agent prompt to allow multilingual replies and language switching. The app will remind the agent to follow the patient's most recent dominant language, but the agent's base prompt still matters.

## Provider Secrets

The frontend does not store provider secrets.

Keep these in `backend/.env`:

- ElevenLabs API key and agent ID
- MERaLiON API key
- Google Translate API key
- OpenAI API key and model

If the backend is unavailable, the app can still load local demo data, but live call saving, saved audio playback, transcription, translation, and risk review require FastAPI.
