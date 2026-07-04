# EarlyCare Frontend

React + Vite interface for the EarlyCare hackathon prototype.

The frontend provides three main experiences:

- **Demo runner**: seven scripted check-in scenarios that persist records and tasks through the backend.
- **Agents call**: an in-browser call simulation powered by ElevenLabs Agents.
- **Patient overview**: a care-team view for check-in history, recordings, transcripts, categorized evidence, escalation trails, and volunteer follow-up tasks.

## What It Does

### Demo Runner

- Runs Stable check-in, Missed check-in, Parkinson's watch, Post-Fall Amber, Post-Fall Red, Chronic Illness Check-In, and Mental Wellbeing / Loneliness.
- Sends the selected scenario to FastAPI so the result is saved to check-in history.
- Shows demo baseline speech scoring, categorized evidence, recommended action, and escalation steps.
- Creates or updates volunteer tasks for missed check-ins and elevated-risk scenarios.

### Agents Call

- Lets a volunteer select a senior and start a browser-based voice check-in.
- Personalizes the prompt with language, living-alone status, known conditions, check-in frequency, caregiver, neighbour, and focus areas.
- Requests microphone permission and starts the ElevenLabs Agents session.
- Records browser microphone audio in parallel through `MediaRecorder`.
- Captures live transcript messages from the Agents SDK.
- Uploads transcript messages and the recording to the FastAPI backend when the call ends.

### Patient Overview

- Lists living-alone seniors and their saved calls.
- Shows historical scripted check-ins as well as saved Agents calls.
- Shows original call recordings through a browser audio player.
- Shows cleaned original transcripts and English transcripts.
- Displays risk review, categorized evidence, escalation steps, and recommended follow-up action.
- Shows clickable risk-signal cards. If a timestamp is available, clicking a signal seeks the audio recording to that part of the call.
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
| `src/main.tsx` | Main React app, Agents call flow, Patient overview. |
| `src/api.ts` | Backend API helpers and audio URL construction. |
| `src/types.ts` | Shared frontend types for seniors, calls, transcripts, and risk signals. |
| `src/styles.css` | Application styling. |
| `scripts/validate-ui-data.mjs` | Lightweight frontend smoke check. |

## Notes

- The frontend does not store provider secrets.
- ElevenLabs, MERaLiON, Google Translate, and OpenAI keys belong in `backend/.env`.
- If the backend is unavailable, the app can still load local demo data, but live call saving and audio playback require FastAPI.
