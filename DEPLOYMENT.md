# EarlyCare Deployment

EarlyCare is deployed as one Render web service. FastAPI serves the API and the built React frontend from the same domain.

## Runtime Shape

- Branch: `main`
- Platform: Render web service
- Runtime: Docker
- App command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --app-dir /app/backend`
- Frontend: Vite build copied into `/app/frontend/dist`
- Python deps: `backend/requirements-deploy.txt` for a lean demo image
- Demo data: `/tmp/earlycare` on Render free instances
- SQLite: `/tmp/earlycare/earlycare.sqlite3`
- Audio/transcripts: `/tmp/earlycare/calls`

Persistent disks require a paid Render web service. The checked-in `render.yaml` uses free ephemeral storage so the hackathon demo can go live without payment info. For durable storage, switch the plan to `starter`, set `EARLYCARE_STORAGE_ROOT=/var/data/earlycare`, and add a disk mounted at `/var/data/earlycare`.

The deploy image intentionally excludes WavLM/Torch/Transformers to avoid multi-GB CUDA wheels on the free Render path. The app still runs, and concussion speech review reports unavailable/degraded when those optional model dependencies are absent.

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
render logs --resources <service-id> -o text
```

Blueprint validation and service operations require a logged-in Render CLI session.

If Render cannot access the GitHub repo through its GitHub integration, deploy a prebuilt image instead:

```bash
docker buildx build --platform linux/amd64 -t <registry>/earlycare:<tag> --push .
render services create \
  --name earlycare \
  --type web_service \
  --image <registry>/earlycare:<tag> \
  --plan free \
  --region oregon \
  --health-check-path /health \
  --env-var EARLYCARE_STORAGE_ROOT=/tmp/earlycare \
  --env-var EARLYCARE_FRONTEND_DIST=/app/frontend/dist \
  --confirm \
  -o json
```

Use a durable registry such as GHCR for repeatable deploys. A short-lived public registry is acceptable only for same-day demos because later redeploys need the image to remain pullable.

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
