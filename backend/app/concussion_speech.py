from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from app.models import ConcussionSpeechReview, ModelExplanationItem


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
        os.environ["HF_HOME"] = str(local_cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _quality_value(quality: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in quality:
            return quality[name]
    return None


def not_applicable_concussion_review() -> ConcussionSpeechReview:
    return ConcussionSpeechReview(
        applicability="not_applicable",
        qualityOk=False,
        riskContribution="Green",
        warning=RESEARCH_WARNING,
        explanations=[
            ModelExplanationItem(
                label="Applicability",
                value="Not applicable",
                status="normal",
                explanation="No patient-stated fall or near-fall was found, so concussion speech review was not applied.",
            )
        ],
    )


def _format_probability(value: float) -> str:
    return f"{round(max(0.0, min(1.0, value)) * 100)}%"


def _concussion_explanations(
    predicted_label: str | None,
    probabilities: dict[str, float],
    quality_ok: bool,
    quality_reason: str | None,
    duration_sec: float | None,
    rms: float | None,
    clipping_fraction: float | None,
) -> list[ModelExplanationItem]:
    if not quality_ok:
        return [
            ModelExplanationItem(
                label="Audio quality",
                value=quality_reason or "Low quality",
                status="unavailable",
                explanation="The patient-speech clip did not pass audio quality checks, so the speech-abnormality model should not be interpreted.",
            )
        ]

    abnormal_probability = max(
        float(probabilities.get("dysarthria_like") or 0.0),
        float(probabilities.get("dysphonia_like") or 0.0),
    )
    normal_probability = float(probabilities.get("normal") or 0.0)
    probability_gap = abs(abnormal_probability - normal_probability)
    label_text = (predicted_label or "unknown").replace("_", " ")
    status = "watch" if predicted_label in {"dysarthria_like", "dysphonia_like"} else "normal"
    explanations = [
        ModelExplanationItem(
            label="Predicted speech pattern",
            value=label_text,
            status=status,
            explanation=(
                "The model compared the patient-speech clip with research speech-abnormality labels. "
                "This label is a review cue, not a concussion diagnosis."
            ),
        ),
        ModelExplanationItem(
            label="Abnormal-class probability",
            value=_format_probability(abnormal_probability),
            status="watch" if abnormal_probability >= normal_probability else "normal",
            explanation=(
                f"The strongest abnormal class was {_format_probability(abnormal_probability)} versus "
                f"{_format_probability(normal_probability)} for normal speech."
            ),
        ),
        ModelExplanationItem(
            label="Audio quality",
            value=f"{round(duration_sec or 0, 1)}s, RMS {round(rms or 0, 4)}, clipping {_format_probability(float(clipping_fraction or 0))}",
            status="normal" if probability_gap >= 0.15 else "watch",
            explanation=(
                "The clip passed duration, loudness, and clipping checks. "
                "A smaller probability gap means the model evidence is less decisive."
            ),
        ),
    ]
    return explanations


def review_concussion_speech(audio_path: Path | None) -> ConcussionSpeechReview | None:
    if audio_path is None:
        return ConcussionSpeechReview(
            applicability="applicable",
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason="Patient speech audio was not available for concussion speech review.",
            explanations=_concussion_explanations(None, {}, False, "Patient speech audio was not available", None, None, None),
        )
    if not audio_path.exists():
        return ConcussionSpeechReview(
            applicability="applicable",
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason="Patient speech audio was not available for concussion speech review.",
            explanations=_concussion_explanations(None, {}, False, "Patient speech audio was not available", None, None, None),
        )

    model_dir = _model_dir()
    source_dir = _source_dir()
    if not model_dir.exists():
        return ConcussionSpeechReview(
            applicability="applicable",
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason=f"Concussion speech model directory was not found: {model_dir}",
            explanations=_concussion_explanations(None, {}, False, "Concussion speech model directory was not found", None, None, None),
        )

    try:
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        _configure_huggingface_cache()
        from speech_abnormality.infer import predict_audio  # type: ignore

        prediction = predict_audio(audio_path, model_dir=model_dir, device=os.getenv("EARLYCARE_CONCUSSION_SPEECH_DEVICE", "cpu"))
    except Exception as exc:
        return ConcussionSpeechReview(
            applicability="applicable",
            qualityOk=False,
            warning=RESEARCH_WARNING,
            failureReason=f"Concussion speech model unavailable: {exc}",
            explanations=_concussion_explanations(None, {}, False, "Concussion speech model unavailable", None, None, None),
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
        applicability="applicable",
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
        explanations=_concussion_explanations(
            predicted_label or None,
            {str(label): float(value) for label, value in probabilities.items()},
            quality_ok,
            _quality_value(quality, "reason"),
            _quality_value(quality, "duration_sec", "durationSeconds"),
            _quality_value(quality, "rms"),
            _quality_value(quality, "clipping_fraction", "clippingFraction"),
        ),
        warning=RESEARCH_WARNING,
    )
