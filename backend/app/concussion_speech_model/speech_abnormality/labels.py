"""Canonical labels for the first speech-abnormality model."""

NORMAL = "normal"
DYSARTHRIA = "dysarthria_like"
DYSPHONIA = "dysphonia_like"
MIXED_OR_UNCERTAIN = "mixed_or_uncertain"
LOW_AUDIO_QUALITY = "low_audio_quality"

ALL_LABELS = [
    NORMAL,
    DYSARTHRIA,
    DYSPHONIA,
    MIXED_OR_UNCERTAIN,
    LOW_AUDIO_QUALITY,
]

ABNORMAL_LABELS = {DYSARTHRIA, DYSPHONIA, MIXED_OR_UNCERTAIN}
