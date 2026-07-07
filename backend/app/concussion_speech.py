from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from app.models import ConcussionSpeechReview


RESEARCH_WARNING = (
    "Research-only speech abnormality signal. This is not a concussion diagnosis, "
    "dysarthria diagnosis, dysphonia diagnosis, or medical device output."
)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _backend_root().parent


def _resolve_configured_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base / path


def _model_dir() -> Path:
    configured = os.getenv("EARLYCARE_CONCUSSION_SPEECH_MODEL_DIR")
    if configured:
        return _resolve_configured_path(configured, _repo_root())
    return _backend_root() / "models" / "concussion_speech"


def _source_dir() -> Path:
    configured = os.getenv("EARLYCARE_CONCUSSION_SPEECH_SOURCE_DIR")
    if configured:
        return _resolve_configured_path(configured, _repo_root())
    return Path(__file__).resolve().parent / "concussion_speech_model"


def _configure_huggingface_cache() -> None:
    local_cache = _backend_root() / "models" / "hf_cache"
    if local_cache.exists():
        os.environ.setdefault("HF_HOME", str(local_cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _quality_value(quality: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in quality:
            return quality[name]
    return None


def review_concussion_speech(audio_path: Path | None) -> ConcussionSpeechReview | None:
    if audio_path is None:
        return ConcussionSpeechReview(
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason="Patient speech audio was not available for concussion speech review.",
        )
    if not audio_path.exists():
        return ConcussionSpeechReview(
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason="Patient speech audio was not available for concussion speech review.",
        )

    model_dir = _model_dir()
    source_dir = _source_dir()
    if not model_dir.exists():
        return ConcussionSpeechReview(
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason=f"Concussion speech model directory was not found: {model_dir}",
        )

    try:
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        _configure_huggingface_cache()
        from speech_abnormality.infer import predict_audio  # type: ignore

        prediction = predict_audio(audio_path, model_dir=model_dir, device=os.getenv("EARLYCARE_CONCUSSION_SPEECH_DEVICE", "cpu"))
    except Exception as exc:
        return ConcussionSpeechReview(
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason=f"Concussion speech model unavailable: {exc}",
        )

    quality = prediction.get("quality") or {}
    probabilities = prediction.get("probabilities") or {}
    if not isinstance(quality, dict):
        quality = {}
    if not isinstance(probabilities, dict):
        probabilities = {}

    predicted_label = str(prediction.get("label") or "")
    quality_ok = bool(_quality_value(quality, "ok"))
    risk_contribution = "Green"
    risk_reason = None
    abnormal_probability = max(
        float(probabilities.get("dysarthria_like") or 0.0),
        float(probabilities.get("dysphonia_like") or 0.0),
    )
    if quality_ok and predicted_label in {"dysarthria_like", "dysphonia_like"}:
        risk_contribution = "Watch"
        risk_reason = (
            f"Speech-abnormality model predicted {predicted_label} "
            f"with {round(abnormal_probability * 100)}% abnormal-class probability."
        )
    elif predicted_label == "low_audio_quality" or not quality_ok:
        risk_reason = "Speech-abnormality model could not score the audio reliably."

    return ConcussionSpeechReview(
        modelVersion=str(model_dir.name),
        predictedLabel=predicted_label or None,
        probabilities={str(label): float(value) for label, value in probabilities.items()},
        qualityOk=quality_ok,
        qualityReason=_quality_value(quality, "reason"),
        durationSec=_quality_value(quality, "duration_sec", "durationSeconds"),
        sampleRate=_quality_value(quality, "sample_rate", "sampleRate"),
        rms=_quality_value(quality, "rms"),
        clippingFraction=_quality_value(quality, "clipping_fraction", "clippingFraction"),
        riskContribution=risk_contribution,  # type: ignore[arg-type]
        riskReason=risk_reason,
        warning=RESEARCH_WARNING,
    )
