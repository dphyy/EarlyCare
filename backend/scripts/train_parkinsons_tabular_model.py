#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.speech_ml.parkinsons_features import UCI_PARKINSONS_FEATURE_NAMES
from app.speech_ml.tabular_training import candidate_estimators, evaluate_candidates, load_uci_dataframe, select_best_candidate, subject_id_from_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Parkinson speech-marker model on the Kaggle/UCI Parkinson dataset.")
    parser.add_argument("csv_path", type=Path, help="Local Kaggle/UCI CSV, for example parkinsons.data.")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/models/speech"))
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    import joblib  # type: ignore
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore

    frame = load_uci_dataframe(args.csv_path)
    features = frame[UCI_PARKINSONS_FEATURE_NAMES].astype(float).to_numpy()
    labels = frame["status"].astype(int).to_numpy()
    groups = [subject_id_from_name(value) for value in frame["name"].tolist()] if "name" in frame.columns else None

    results, warnings = evaluate_candidates(features, labels, groups, args.splits)
    best = select_best_candidate(results)
    candidates, candidate_warnings = candidate_estimators()
    warnings.extend(candidate_warnings)
    winner = candidates[best.name]
    calibrated_winner = CalibratedClassifierCV(winner, cv=3, method="sigmoid")
    calibrated_winner.fit(features, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_winner, args.output_dir / "parkinsons_tabular_model.joblib")
    (args.output_dir / "feature_schema.json").write_text(json.dumps(UCI_PARKINSONS_FEATURE_NAMES, indent=2))

    candidate_metrics = {
        result.name: {
            "skipped": result.skipped,
            "reason": result.reason,
            "metrics": result.metrics,
        }
        for result in results
    }
    metrics_payload = {
        "selected_model": best.name,
        "selection_metric": "roc_auc_then_balanced_accuracy",
        "candidate_metrics": candidate_metrics,
        "warnings": warnings,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (args.output_dir / "model_card.json").write_text(
        json.dumps(
            {
                "model_version": f"earlycare-uci-parkinsons-tabular-{best.name}-v0",
                "model_type": best.name,
                "calibration": {
                    "enabled": True,
                    "method": "CalibratedClassifierCV sigmoid",
                    "cv": 3,
                },
                "dataset_sources": [
                    "https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set",
                    "https://archive.ics.uci.edu/dataset/174/parkinsons",
                ],
                "label_semantics": "status 0 healthy, status 1 Parkinson's; EarlyCare surfaces only a high-risk speech marker.",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_schema": UCI_PARKINSONS_FEATURE_NAMES,
                "record_count": int(len(frame)),
                "positive_count": int(np.sum(labels)),
                "evaluation": metrics_payload,
                "disclaimer": "Research screening artifact only. Not a diagnosis and not clinically validated for conversational EarlyCare audio.",
            },
            indent=2,
        )
    )
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
