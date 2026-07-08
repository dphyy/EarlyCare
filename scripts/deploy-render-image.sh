#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SERVICE_ID="${SERVICE_ID:-srv-d96uaquq1p3s7382dfqg}"
RENDER_URL="${RENDER_URL:-https://earlycare.onrender.com}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/saaiaravindhraja/earlycare}"
REGISTRY_CREDENTIAL_ID="${REGISTRY_CREDENTIAL_ID:-rgc-d96ufpsvikkc73d95eg0}"
PLATFORM="${PLATFORM:-linux/amd64}"
BRANCH="${BRANCH:-main}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
IMAGE="${IMAGE_REPO}:${TAG}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command git
require_command docker
require_command gh
require_command render
require_command curl

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  echo "Refusing to deploy from '$current_branch'. Switch to '$BRANCH' first." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "Refusing to deploy with uncommitted changes. Commit first, or set ALLOW_DIRTY=1." >&2
  exit 1
fi

git fetch origin "$BRANCH" >/dev/null
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "origin/${BRANCH}")"
if [[ "$local_head" != "$remote_head" && "${SKIP_PUSH_CHECK:-0}" != "1" ]]; then
  echo "Refusing to deploy because local HEAD is not pushed to origin/${BRANCH}." >&2
  echo "Push first, or set SKIP_PUSH_CHECK=1." >&2
  exit 1
fi

echo "Deploying ${IMAGE} to Render service ${SERVICE_ID}"

gh auth status >/dev/null
gh auth token | docker login ghcr.io -u "${GHCR_USER:-SaaiAravindhRaja}" --password-stdin >/dev/null
render whoami -o text >/dev/null

docker buildx build \
  --platform "$PLATFORM" \
  -t "$IMAGE" \
  -t "${IMAGE_REPO}:latest" \
  --push \
  .

render_update_args=(
  services update "$SERVICE_ID"
  --image "$IMAGE"
  --confirm
  -o json
)
if [[ -n "$REGISTRY_CREDENTIAL_ID" ]]; then
  render_update_args+=(--registry-credential "$REGISTRY_CREDENTIAL_ID")
fi
render "${render_update_args[@]}" >/tmp/earlycare-render-service-update.json

render deploys create "$SERVICE_ID" \
  --image "$IMAGE" \
  --wait \
  --confirm \
  -o text

curl -fsS "${RENDER_URL}/health"
printf "\n"
curl -fsS "${RENDER_URL}/auth/me"
printf "\n"

echo "Deployed ${IMAGE} to ${RENDER_URL}"
