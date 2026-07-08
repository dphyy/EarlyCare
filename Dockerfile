FROM node:22-bookworm-slim AS frontend

WORKDIR /app
RUN corepack enable && corepack prepare pnpm@11.7.0 --activate

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY frontend/package.json ./frontend/package.json
RUN pnpm install --frozen-lockfile

COPY frontend ./frontend
RUN pnpm --dir frontend build

FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EARLYCARE_FRONTEND_DIST=/app/frontend/dist \
    EARLYCARE_STORAGE_ROOT=/var/data/earlycare

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN mkdir -p /var/data/earlycare

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir /app/backend"]
