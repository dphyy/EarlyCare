# EarlyCare Deployment

EarlyCare is deployed as one Render web service. FastAPI serves the API and the built React frontend from the same domain.

## Runtime Shape

- Branch: `main`
- Platform: Render web service
- Runtime: Docker
- App command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --app-dir /app/backend`
- Frontend: Vite build copied into `/app/frontend/dist`
- Persistent data: Render disk mounted at `/var/data/earlycare`
- SQLite: `/var/data/earlycare/earlycare.sqlite3`
- Audio/transcripts: `/var/data/earlycare/calls`

Persistent disks require a paid Render web service. If billing is not available, remove the `disk` block from `render.yaml` and use the same image with ephemeral storage for the demo.

## Required Render Environment Variables

Set these as secrets in Render, not in Git:

```bash
MERALION_API_KEY=
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_AGENT_ID=
EARLYCARE_OPERATOR_PASSWORD=
EARLYCARE_AUTH_SECRET=
```

Optional:

```bash
GOOGLE_TRANSLATE_API_KEY=
OPENAI_SAFEGUARD_MODEL=
```

The non-secret defaults are already in `render.yaml`.

## Local Production Check

```bash
pnpm install
pnpm --dir frontend build
PYTHONPATH=backend python -m unittest backend.tests.test_call_workflow
EARLYCARE_FRONTEND_DIST=frontend/dist uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

Open `http://127.0.0.1:8000`.

## Render CLI Flow

The Render CLI is useful for login, Blueprint validation, service selection, deploys, logs, SSH, and restarts:

```bash
render login
render blueprints validate render.yaml --confirm -o text
render services -o text
render deploys create <service-id> --commit <commit-sha> --wait -o text
render logs <service-id> -o text
```

Blueprint validation and service operations require a logged-in Render CLI session.

## Post-Deploy Smoke Test

```bash
curl https://<render-host>/health
curl https://<render-host>/auth/me
```

Then sign in with the configured operator credentials and verify:

- roster loads
- Care Desk loads
- ElevenLabs signed session request returns configured status
- a saved call appears in the archive
- audio playback URL loads
- `/readiness` reports provider/model status
