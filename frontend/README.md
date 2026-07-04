# EarlyCare Frontend

React + Vite interface for the EarlyCare hackathon prototype.

The frontend provides two main experiences:

- **Agents call**: an in-browser call simulation powered by ElevenLabs Agents.
- **Patient overview**: a care-team view for recordings, transcripts, AI risk signals, and volunteer follow-up tasks.

## What It Does

### Agents Call

- Lets a volunteer select a senior and start a browser-based voice check-in.
- Requests microphone permission and starts the ElevenLabs Agents session.
- Records browser microphone audio in parallel through `MediaRecorder`.
- Captures live transcript messages from the Agents SDK.
- Uploads transcript messages and the recording to the FastAPI backend when the call ends.

### Patient Overview

- Lists living-alone seniors and their saved calls.
- Shows original call recordings through a browser audio player.
- Shows cleaned original transcripts and English transcripts.
- Displays AI-generated risk review and recommended follow-up action.
- Shows clickable risk-signal cards. If a timestamp is available, clicking a signal seeks the audio recording to that part of the call.

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
| `src/main.tsx` | Main React app, Agents call flow, Patient overview. |
| `src/api.ts` | Backend API helpers and audio URL construction. |
| `src/types.ts` | Shared frontend types for seniors, calls, transcripts, and risk signals. |
| `src/styles.css` | Application styling. |

## Notes

- The frontend does not store provider secrets.
- ElevenLabs, MERaLiON, Google Translate, and OpenAI keys belong in `backend/.env`.
- If the backend is unavailable, the app can still load local demo data, but live call saving and audio playback require FastAPI.
