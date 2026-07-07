#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parkinsons_speech_model.inference import predict_speech_marker


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict EarlyCare speech-marker probability for one patient-only audio file.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=Path("backend/models/parkinsons_speech"))
    args = parser.parse_args()

    result = predict_speech_marker(args.audio_path, args.artifact_dir)
    print(
        json.dumps(
            {
                "parkinsonsSpeechReview": {
                    "modelVersion": result.model_version,
                    "probability": result.probability,
                    "warnings": result.warnings,
                    "featuresSummary": result.features_summary,
                    "label": "Parkinson voice-feature marker, not diagnosis",
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
