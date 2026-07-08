from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Literal


WAVLM_MODEL_NAME = "microsoft/wavlm-base"
WAVLM_REQUIRED_FILES = ["config.json", "preprocessor_config.json"]
WAVLM_WEIGHT_GLOBS = ["pytorch_model*.bin", "model*.safetensors"]
ReadinessStatus = Literal["ready", "degraded", "blocked"]
ComponentStatus = Literal["ready", "degraded", "missing", "blocked"]


def _component(name: str, status: ComponentStatus, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _env_component(name: str, env_names: list[str], required_for_demo: bool = False) -> dict[str, str]:
    missing = [env_name for env_name in env_names if not os.getenv(env_name)]
    if not missing:
        return _component(name, "ready", "Configured.")
    status: ComponentStatus = "blocked" if required_for_demo else "degraded"
    return _component(name, status, f"Missing {', '.join(missing)}.")


def _storage_component(storage_root: Path) -> dict[str, str]:
    try:
        storage_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=storage_root, prefix=".readiness-", delete=True) as temp_file:
            temp_file.write(b"ok")
        return _component("Local call storage", "ready", "Writable.")
    except Exception as exc:
        return _component("Local call storage", "blocked", f"Not writable: {exc.__class__.__name__}.")


def _is_under_path(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _storage_persistence_component(call_storage_root: Path) -> dict[str, str]:
    storage_root = call_storage_root.parent if call_storage_root.name == "calls" else call_storage_root
    configured_storage_root = os.getenv("EARLYCARE_STORAGE_ROOT")
    temp_roots = [Path(tempfile.gettempdir()), Path("/tmp"), Path("/var/tmp")]
    if any(_is_under_path(storage_root, temp_root) for temp_root in temp_roots):
        return _component(
            "Storage persistence",
            "degraded",
            f"Using temporary storage at {storage_root}; saved SQLite/audio data can disappear on redeploy or restart.",
        )
    if not configured_storage_root:
        return _component(
            "Storage persistence",
            "degraded",
            "EARLYCARE_STORAGE_ROOT is not configured; set it to the mounted Render disk path for production.",
        )
    if storage_root.is_mount():
        return _component("Storage persistence", "ready", f"{storage_root} is mounted as a filesystem.")
    return _component(
        "Storage persistence",
        "degraded",
        f"{storage_root} is configured, but this process cannot verify that it is a mounted persistent disk.",
    )


def _sqlite_component(call_storage_root: Path) -> dict[str, str]:
    default_root = call_storage_root.parent if call_storage_root.name == "calls" else call_storage_root
    database_path = Path(os.getenv("EARLYCARE_DB_PATH", default_root / "earlycare.sqlite3"))
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS readiness_probe (id INTEGER PRIMARY KEY, checked_at TEXT NOT NULL)")
            connection.execute("INSERT INTO readiness_probe (checked_at) VALUES (datetime('now'))")
            connection.execute("DELETE FROM readiness_probe WHERE id NOT IN (SELECT MAX(id) FROM readiness_probe)")
        return _component("SQLite metadata store", "ready", "Writable.")
    except Exception as exc:
        return _component("SQLite metadata store", "blocked", f"Not writable: {exc.__class__.__name__}.")


def _auth_component() -> dict[str, str]:
    if os.getenv("EARLYCARE_AUTH_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        return _component("Operator auth", "degraded", "Disabled by EARLYCARE_AUTH_DISABLED.")
    missing = [env_name for env_name in ["EARLYCARE_OPERATOR_PASSWORD", "EARLYCARE_AUTH_SECRET"] if not os.getenv(env_name)]
    if missing:
        return _component("Operator auth", "degraded", f"Missing {', '.join(missing)}.")
    return _component("Operator auth", "ready", "Configured.")


def _artifact_component(name: str, root: Path, required_files: list[str]) -> dict[str, str]:
    missing = [file_name for file_name in required_files if not (root / file_name).exists()]
    if not root.exists():
        return _component(name, "blocked", f"Missing artifact directory: {root.name}.")
    if missing:
        return _component(name, "blocked", f"Missing artifact files: {', '.join(missing)}.")
    return _component(name, "ready", "Artifacts present.")


def _wavlm_snapshot_root(cache_root: Path) -> Path:
    return cache_root / "hub" / f"models--{WAVLM_MODEL_NAME.replace('/', '--')}" / "snapshots"


def _wavlm_cache_ready(cache_root: Path) -> bool:
    snapshot_root = _wavlm_snapshot_root(cache_root)
    if not snapshot_root.exists():
        return False
    for snapshot in snapshot_root.iterdir():
        if not snapshot.is_dir():
            continue
        if not all((snapshot / file_name).exists() for file_name in WAVLM_REQUIRED_FILES):
            continue
        if any(list(snapshot.glob(pattern)) for pattern in WAVLM_WEIGHT_GLOBS):
            return True
    return False


def _wavlm_component(backend_root: Path) -> dict[str, str]:
    configured_cache = os.getenv("HF_HOME")
    local_cache = backend_root / "models" / "hf_cache"
    if _wavlm_cache_ready(local_cache):
        return _component("WavLM cache", "ready", f"{WAVLM_MODEL_NAME} is cached under backend/models/hf_cache.")
    if configured_cache and _wavlm_cache_ready(Path(configured_cache)):
        return _component("WavLM cache", "ready", f"{WAVLM_MODEL_NAME} is cached under HF_HOME.")
    if local_cache.exists() or configured_cache:
        return _component(
            "WavLM cache",
            "degraded",
            f"Cache directory exists, but {WAVLM_MODEL_NAME} files are incomplete. Run backend/.venv/bin/python backend/scripts/cache_wavlm.py.",
        )
    return _component(
        "WavLM cache",
        "degraded",
        f"{WAVLM_MODEL_NAME} is not cached locally. Run backend/.venv/bin/python backend/scripts/cache_wavlm.py before the demo.",
    )


def readiness_report(backend_root: Path, call_storage_root: Path) -> dict[str, object]:
    components = [
        _env_component("ElevenLabs Agents", ["ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID"], required_for_demo=True),
        _env_component("OpenAI structured review", ["OPENAI_API_KEY"]),
        _env_component("MERaLiON transcription", ["MERALION_API_KEY"]),
        _env_component("Google Translate fallback", ["GOOGLE_TRANSLATE_API_KEY"]),
        _auth_component(),
        _storage_persistence_component(call_storage_root),
        _storage_component(call_storage_root),
        _sqlite_component(call_storage_root),
        _artifact_component(
            "Parkinson speech marker",
            backend_root / "models" / "parkinsons_speech",
            ["parkinsons_tabular_model.joblib", "feature_schema.json", "feature_reference_ranges.json", "model_card.json"],
        ),
        _artifact_component(
            "Concussion speech review",
            backend_root / "models" / "concussion_speech",
            ["model.joblib", "config.json", "metrics.json"],
        ),
        _wavlm_component(backend_root),
    ]
    statuses = [str(component["status"]) for component in components]
    if "blocked" in statuses:
        status: ReadinessStatus = "blocked"
    elif "degraded" in statuses or "missing" in statuses:
        status = "degraded"
    else:
        status = "ready"
    return {
        "status": status,
        "components": components,
        "message": "Demo can run." if status == "ready" else "Demo has degraded or blocked components; review details before presenting.",
    }
