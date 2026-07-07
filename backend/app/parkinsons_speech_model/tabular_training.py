from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.parkinsons_speech_model.evaluation import binary_metrics, group_folds, speaker_level_probabilities
from app.parkinsons_speech_model.parkinsons_features import CONVERSATIONAL_PARKINSONS_FEATURE_NAMES


@dataclass
class CandidateResult:
    name: str
    metrics: dict[str, float | list[list[int]] | None]
    skipped: bool = False
    reason: str | None = None


def subject_id_from_name(name: str) -> str:
    normalized = str(name).strip()
    match = re.match(r"(.+?)(?:[_-]\d+)?$", normalized)
    return match.group(1) if match else normalized


def load_uci_dataframe(csv_path):
    import pandas as pd  # type: ignore

    frame = pd.read_csv(csv_path)
    missing = [column for column in [*CONVERSATIONAL_PARKINSONS_FEATURE_NAMES, "status"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required UCI/Kaggle Parkinson columns: {', '.join(missing)}")
    return frame


def candidate_estimators(random_state: int = 42) -> tuple[dict[str, Any], list[str]]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.svm import SVC  # type: ignore

    warnings: list[str] = []
    candidates: dict[str, Any] = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "svc_rbf": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=random_state)),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        candidates["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    except ImportError:
        warnings.append("xgboost is not installed; skipping XGBClassifier.")
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        candidates["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        warnings.append("lightgbm is not installed; skipping LGBMClassifier.")
    return candidates, warnings


def _probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    decision = model.decision_function(features)
    return 1 / (1 + np.exp(-decision))


def evaluate_candidates(features: np.ndarray, labels: np.ndarray, groups: list[str] | None, n_splits: int = 5) -> tuple[list[CandidateResult], list[str]]:
    from sklearn.model_selection import StratifiedKFold  # type: ignore

    candidates, warnings = candidate_estimators()
    split_count = min(n_splits, len(set(groups or [])) if groups else int(np.bincount(labels).min()))
    if groups and len(set(groups)) >= 2:
        folds = group_folds(labels.tolist(), groups, max(2, split_count))
        evaluation_mode = "grouped_by_subject"
    else:
        warnings.append("Subject groups were unavailable; using stratified CV and recording-level evaluation.")
        splitter = StratifiedKFold(n_splits=max(2, min(n_splits, int(np.bincount(labels).min()))), shuffle=True, random_state=42)
        folds = [(train.tolist(), test.tolist()) for train, test in splitter.split(features, labels)]
        evaluation_mode = "stratified_recording"

    results: list[CandidateResult] = []
    for name, model in candidates.items():
        probabilities = np.zeros(len(labels), dtype=np.float32)
        try:
            for train_indices, test_indices in folds:
                model.fit(features[train_indices], labels[train_indices])
                probabilities[test_indices] = _probabilities(model, features[test_indices])
        except Exception as exc:
            results.append(CandidateResult(name=name, metrics={}, skipped=True, reason=str(exc)))
            continue

        if evaluation_mode == "grouped_by_subject" and groups:
            _, eval_labels, eval_probs = speaker_level_probabilities(groups, labels.tolist(), probabilities.tolist())
        else:
            eval_labels, eval_probs = labels.tolist(), probabilities.tolist()
        metrics = binary_metrics(eval_labels, eval_probs)
        results.append(CandidateResult(name=name, metrics=metrics))
    warnings.append(f"Evaluation mode: {evaluation_mode}.")
    return results, warnings


def select_best_candidate(results: list[CandidateResult]) -> CandidateResult:
    available = [result for result in results if not result.skipped]
    if not available:
        raise RuntimeError("No model candidates trained successfully.")

    def score(result: CandidateResult) -> tuple[float, float]:
        roc_auc = result.metrics.get("roc_auc")
        balanced = result.metrics.get("balanced_accuracy")
        return (
            float(roc_auc) if isinstance(roc_auc, (int, float)) else -1.0,
            float(balanced) if isinstance(balanced, (int, float)) else -1.0,
        )

    return max(available, key=score)
