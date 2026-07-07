from __future__ import annotations

from collections import defaultdict

import numpy as np


def speaker_level_probabilities(speaker_ids: list[str], labels: list[int], probabilities: list[float]) -> tuple[list[str], list[int], list[float]]:
    grouped_probs: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for speaker_id, label, probability in zip(speaker_ids, labels, probabilities):
        grouped_probs[speaker_id].append(float(probability))
        grouped_labels[speaker_id] = int(label)
    ordered_speakers = sorted(grouped_probs)
    return (
        ordered_speakers,
        [grouped_labels[speaker_id] for speaker_id in ordered_speakers],
        [float(np.mean(grouped_probs[speaker_id])) for speaker_id in ordered_speakers],
    )


def group_folds(labels: list[int], groups: list[str], n_splits: int = 5) -> list[tuple[list[int], list[int]]]:
    try:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # type: ignore

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            return [(train.tolist(), test.tolist()) for train, test in splitter.split(np.zeros(len(labels)), labels, groups)]
        except ValueError:
            group_splitter = GroupKFold(n_splits=n_splits)
            return [(train.tolist(), test.tolist()) for train, test in group_splitter.split(np.zeros(len(labels)), labels, groups)]
    except ImportError:
        unique_groups = sorted(set(groups))
        folds: list[tuple[list[int], list[int]]] = []
        for fold_index in range(min(n_splits, len(unique_groups))):
            test_groups = set(unique_groups[fold_index::n_splits])
            train_indices = [index for index, group in enumerate(groups) if group not in test_groups]
            test_indices = [index for index, group in enumerate(groups) if group in test_groups]
            folds.append((train_indices, test_indices))
        return folds


def binary_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, float | list[list[int]] | None]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 1)
    tn = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 0)
    fp = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 1)
    fn = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 0)
    accuracy = (tp + tn) / max(1, len(labels))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = sensitivity
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    roc_auc: float | None = None
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        if len(set(labels)) > 1:
            roc_auc = float(roc_auc_score(labels, probabilities))
    except ImportError:
        roc_auc = None
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
