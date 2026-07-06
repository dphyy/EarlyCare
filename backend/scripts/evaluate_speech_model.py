#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.speech_ml.evaluation import binary_metrics, speaker_level_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clip predictions as speaker-level speech-marker metrics.")
    parser.add_argument("predictions", type=Path, help="CSV with speaker_id,label,probability columns.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import csv

    speaker_ids: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    with args.predictions.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            speaker_ids.append(row["speaker_id"])
            labels.append(int(row["label"]))
            probabilities.append(float(row["probability"]))

    speakers, speaker_labels, speaker_probs = speaker_level_probabilities(speaker_ids, labels, probabilities)
    metrics = binary_metrics(speaker_labels, speaker_probs, args.threshold)
    metrics["speaker_count"] = len(speakers)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
