import math

from app.models import RiskAssessment, SpeechDeviationRequest, SpeechProfile, Symptoms


def _cosine_distance(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.3
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if not mag_a or not mag_b:
        return 0.3
    return max(0.0, 1.0 - dot / (mag_a * mag_b))


def _score(value: float) -> int:
    return max(0, min(100, round(value)))


def assess_speech_deviation(baseline: SpeechProfile, request: SpeechDeviationRequest) -> RiskAssessment:
    symptoms: Symptoms = request.symptoms
    current = request.currentSpeechProfile

    if symptoms.missedCheckIn:
        return RiskAssessment(
            speechDeviationScore=0,
            parkinsonsWatchScore=0,
            postFallConcernScore=0,
            missedCheckInScore=100,
            riskLevel="Amber",
            reasons=[
                "Scheduled check-in missed after retry",
                "Volunteer follow-up needed because the senior lives alone",
            ],
        )

    embedding_delta = _cosine_distance(baseline.embedding, current.embedding) * 100
    rate_delta = abs(current.speechRate - baseline.speechRate) / max(baseline.speechRate, 1)
    pause_delta = abs(current.avgPauseMs - baseline.avgPauseMs) / max(baseline.avgPauseMs, 1)
    latency_delta = abs(current.responseLatencyMs - baseline.responseLatencyMs) / max(baseline.responseLatencyMs, 1)
    pitch_delta = abs(current.pitchVariability - baseline.pitchVariability) / max(baseline.pitchVariability, 0.1)

    speech_deviation = _score(
        embedding_delta * 0.45
        + rate_delta * 35
        + pause_delta * 24
        + latency_delta * 18
        + pitch_delta * 20
    )

    parkinsons_watch = _score(
        (22 if current.speechRate < baseline.speechRate * 0.8 else 0)
        + (24 if current.avgPauseMs > baseline.avgPauseMs * 1.55 else 0)
        + (22 if current.pitchVariability < baseline.pitchVariability * 0.7 else 0)
        + min(18, speech_deviation * 0.22)
    )

    danger_signs = sum([symptoms.confusion, symptoms.vomiting, symptoms.slurredSpeech, symptoms.weakness])
    post_fall = _score(
        (24 if symptoms.fall else 0)
        + (24 if symptoms.headImpact else 0)
        + (12 if symptoms.headache else 0)
        + (10 if symptoms.dizziness else 0)
        + danger_signs * 18
        + (speech_deviation * 0.22 if symptoms.fall or symptoms.headImpact else 0)
    )

    reasons: list[str] = []
    if speech_deviation > 45:
        reasons.append("Speech differs meaningfully from personal baseline")
    if parkinsons_watch > 50:
        reasons.append("Gradual pattern resembles Parkinson's watch markers: slower rate, longer pauses, lower pitch variation")
    if symptoms.fall:
        reasons.append("Fall reported during check-in")
    if symptoms.headImpact:
        reasons.append("Head impact reported")
    if danger_signs:
        reasons.append("Danger signs reported: confusion, slurred speech, weakness, or vomiting")
    if not reasons:
        reasons.append("No concerning symptoms and speech remains close to baseline")

    risk_level = "Green"
    if post_fall >= 75:
        risk_level = "Red"
    elif post_fall >= 40 or speech_deviation >= 60:
        risk_level = "Amber"
    elif parkinsons_watch >= 50 or speech_deviation >= 35:
        risk_level = "Watch"

    return RiskAssessment(
        speechDeviationScore=speech_deviation,
        parkinsonsWatchScore=parkinsons_watch,
        postFallConcernScore=post_fall,
        missedCheckInScore=0,
        riskLevel=risk_level,
        reasons=reasons,
    )


def extract_demo_embedding_note() -> str:
    return (
        "Production path: save patient-only audio, extract UCI/Kaggle-style voice features, "
        "score the trained tabular speech-marker model, and compare speech timing against "
        "the senior's stable baseline. Outputs are screening signals, not diagnoses."
    )
