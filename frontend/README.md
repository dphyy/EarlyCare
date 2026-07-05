# EarlyCare Frontend

React + Vite interface for the EarlyCare hackathon prototype.

The frontend provides three main experiences:

- **Demo runner**: seven scripted check-in scenarios that persist records and tasks through the backend.
- **Agents call**: an in-browser call simulation powered by ElevenLabs Agents.
- **Patient overview**: a care-team view for roster triage, operations queue, check-in history, recordings, transcripts, categorized evidence, escalation trails, volunteer tasks, speech timing, inline AI risk highlights, answered/missed schedule logging, and follow-up recommendations.

The top bar shows whether the browser is connected to the live FastAPI backend or using local demo data. The command header also surfaces the top care-desk priority, last sync time, and a manual refresh action.

## Demo Runner

- Runs Stable check-in, Missed check-in, Parkinson's watch, Post-Fall Amber, Post-Fall Red, Chronic Illness Check-In, and Mental Wellbeing / Loneliness.
- Sends the selected scenario to FastAPI so the result is saved to check-in history.
- Shows demo baseline speech scoring, categorized evidence, recommended action, and escalation steps.
- Creates or updates volunteer tasks for missed check-ins and elevated-risk scenarios.

## Agents Call

- Lets a volunteer select a senior and start a browser-based voice check-in.
- Personalizes the prompt with language, living-alone status, known conditions, check-in frequency, caregiver, neighbour, and focus areas.
- Requests microphone permission and starts the ElevenLabs Agents session.
- Keeps the live transcript in the original spoken language.
- Reminds the agent to respond in the language or dialect used most in the patient's previous response.
- Records patient microphone audio and decodes ElevenLabs agent audio packets from the SDK.
- Mixes patient and agent audio into one `full-call.webm`.
- Uploads transcript messages, audio, and `agentAudioCaptured` metadata to FastAPI when the call ends.

The call page does not translate the live transcript. Translation happens after saving, in the backend workflow.

## Patient Overview

- Lists living-alone seniors and their saved calls.
- Shows an operations queue ranked by schedule status, senior risk, and open volunteer work.
- Filters the roster by due status, open work, elevated risk, or search text.
- Shows historical scripted check-ins as well as saved Agents calls.
- Shows full-call recordings through a browser audio player.
- Shows translated English transcripts and cleaned original transcripts with `Agent:` and `Patient:` speaker labels.
- Displays risk review, categorized evidence, escalation steps, and recommended follow-up action.
- Shows current speech timing and baseline context.
- Shows clickable risk-signal cards and inline risk highlights inside the English transcript.
- If a timestamp is available, clicking a signal seeks the audio recording to that part of the call.
- Lets users log answered or missed scheduled check-ins when the live backend is connected.
- Lets users acknowledge or close volunteer tasks through `PATCH /volunteer-tasks/{id}`.

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
pnpm install
```

Start the dev server:

```bash
pnpm dev
```

The app usually runs at `http://localhost:5173`.

## Commands

| Command | Description |
| --- | --- |
| `pnpm dev` | Start the Vite dev server. |
| `pnpm lint` | Run TypeScript checks without emitting files. |
| `pnpm build` | Type-check and build the production frontend. |
| `pnpm test:data` | Validate scenario data, category types, and key UI hooks. |

## Main Files

| File | Purpose |
| --- | --- |
| `src/main.tsx` | Main React app, Agents call flow, Patient overview, audio mixing, transcript highlights. |
| `src/api.ts` | Backend API helpers and audio URL construction. |
| `src/types.ts` | Shared frontend types for seniors, calls, transcripts, speech profiles, and risk signals. |
| `src/styles.css` | Application styling. |
| `scripts/validate-ui-data.mjs` | Lightweight frontend smoke check. |

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
