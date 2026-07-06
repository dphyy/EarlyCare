"""Single-audio inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from speech_abnormality.audio import preprocess_audio
from speech_abnormality.embeddings import FrozenSpeechEmbedder
from speech_abnormality.labels import LOW_AUDIO_QUALITY


RESEARCH_USE_WARNING = (
    "Research-use speech abnormality output only. This model is not a medical diagnostic "
    "device and must not be used to diagnose concussion, dysarthria, dysphonia, or any "
    "other condition without external clinical validation."
)


class SpeechAbnormalityPredictor:
    """Reusable inference helper that loads the classifier and embedder once."""

    def __init__(self, model_dir: str | Path, device: str = "auto") -> None:
        model_path = Path(model_dir) / "model.joblib"
        self.artifacts = joblib.load(model_path)
        self.config = self.artifacts["config"]
        self.classes = list(self.artifacts["classes"])
        self.embedder = FrozenSpeechEmbedder(self.config["model_name"], device=device)

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        processed = preprocess_audio(
            audio_path,
            target_rate=int(self.config["sample_rate"]),
            max_seconds=float(self.config["max_seconds"]),
            min_seconds=float(self.config["min_seconds"]),
            silence_rms_threshold=float(self.config["silence_rms_threshold"]),
            clipping_threshold=float(self.config["clipping_threshold"]),
            clipping_fraction_threshold=float(self.config["clipping_fraction_threshold"]),
        )
        if not processed.quality.ok:
            return {
                "label": LOW_AUDIO_QUALITY,
                "probabilities": {LOW_AUDIO_QUALITY: 1.0},
                "quality": processed.quality.__dict__,
                "warning": RESEARCH_USE_WARNING,
            }

        embedding = self.embedder.embed_waveform(processed.waveform, processed.sample_rate).reshape(1, -1)
        probabilities = self.artifacts["model"].predict_proba(embedding)
        encoder = self.artifacts["label_encoder"]
        model_classes = [encoder.classes_[idx] for idx in self.artifacts["model"].classes_]
        aligned = np.zeros(len(self.classes), dtype=np.float32)
        for src_idx, label in enumerate(model_classes):
            aligned[self.classes.index(label)] = probabilities[0, src_idx]
        if aligned.sum() > 0:
            aligned = aligned / aligned.sum()
        best_idx = int(np.argmax(aligned))
        return {
            "label": self.classes[best_idx],
            "probabilities": {label: float(aligned[idx]) for idx, label in enumerate(self.classes)},
            "quality": processed.quality.__dict__,
            "warning": RESEARCH_USE_WARNING,
        }


def predict_audio(audio_path: str | Path, model_dir: str | Path, device: str = "auto") -> dict[str, Any]:
    return SpeechAbnormalityPredictor(model_dir, device=device).predict(audio_path)


def batch_predict_audio(
    rows: list[dict[str, Any]],
    model_dir: str | Path,
    output_csv: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    predictor = SpeechAbnormalityPredictor(model_dir, device=device)
    results: list[dict[str, Any]] = []
    for row in rows:
        audio_path = str(row["path"])
        prediction = predictor.predict(audio_path)
        quality = prediction["quality"]
        flat = {
            "path": audio_path,
            "dataset": row.get("dataset", ""),
            "speaker_id": row.get("speaker_id", ""),
            "expected_label": row.get("label", row.get("expected_label", "")),
            "predicted_label": prediction["label"],
            "quality_ok": quality.get("ok", False),
            "quality_reason": quality.get("reason", ""),
            "duration_sec": quality.get("duration_sec", 0.0),
            "sample_rate": quality.get("sample_rate", 0),
            "rms": quality.get("rms", 0.0),
            "clipping_fraction": quality.get("clipping_fraction", 0.0),
            "warning": prediction["warning"],
        }
        for label, probability in prediction["probabilities"].items():
            flat[f"probability_{label}"] = probability
        results.append(flat)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    return summarize_batch_results(results, output_path)


def summarize_batch_results(results: list[dict[str, Any]], output_csv: Path) -> dict[str, Any]:
    total = len(results)
    quality_rejected = sum(1 for row in results if not row["quality_ok"])
    rows_with_expected = [row for row in results if row["expected_label"]]
    correct = sum(
        1
        for row in rows_with_expected
        if row["quality_ok"] and row["expected_label"] == row["predicted_label"]
    )
    normal_rows = [row for row in rows_with_expected if row["expected_label"] == "normal" and row["quality_ok"]]
    normal_false_positives = sum(1 for row in normal_rows if row["predicted_label"] != "normal")
    return {
        "output_csv": str(output_csv),
        "n_files": total,
        "n_low_audio_quality": quality_rejected,
        "n_with_expected_label": len(rows_with_expected),
        "accuracy_on_expected_quality_ok": correct / len(rows_with_expected) if rows_with_expected else None,
        "normal_false_positive_rate": (
            normal_false_positives / len(normal_rows) if normal_rows else None
        ),
        "warning": RESEARCH_USE_WARNING,
    }


def summarize_results_csv(results_csv: str | Path, group_by: list[str] | None = None) -> dict[str, Any]:
    frame = pd.read_csv(results_csv)
    quality_ok = frame["quality_ok"].astype(bool) if "quality_ok" in frame else pd.Series([], dtype=bool)
    expected_mask = frame["expected_label"].fillna("").astype(str) != "" if "expected_label" in frame else pd.Series(False, index=frame.index)
    quality_expected = frame[expected_mask & quality_ok]
    summary: dict[str, Any] = {
        "results_csv": str(results_csv),
        "n_files": int(len(frame)),
        "n_low_audio_quality": int((~quality_ok).sum()) if "quality_ok" in frame else 0,
        "predicted_label_counts": frame["predicted_label"].value_counts(dropna=False).to_dict(),
        "warning": RESEARCH_USE_WARNING,
    }
    if not quality_expected.empty:
        correct = quality_expected["expected_label"] == quality_expected["predicted_label"]
        summary["n_with_expected_label"] = int(len(quality_expected))
        summary["accuracy_on_expected_quality_ok"] = float(correct.mean())
        summary["confusion_matrix"] = pd.crosstab(
            quality_expected["expected_label"],
            quality_expected["predicted_label"],
        ).to_dict(orient="index")
        normal = quality_expected[quality_expected["expected_label"] == "normal"]
        if not normal.empty:
            summary["normal_false_positive_rate"] = float((normal["predicted_label"] != "normal").mean())
    if group_by:
        groups: dict[str, Any] = {}
        for column in group_by:
            if column not in frame.columns:
                continue
            rows: dict[str, Any] = {}
            for value, group in frame.groupby(column, dropna=False):
                key = str(value)
                group_expected = group[(group["expected_label"].fillna("").astype(str) != "") & group["quality_ok"].astype(bool)]
                rows[key] = {
                    "n_files": int(len(group)),
                    "n_low_audio_quality": int((~group["quality_ok"].astype(bool)).sum()),
                    "predicted_label_counts": group["predicted_label"].value_counts(dropna=False).to_dict(),
                }
                if not group_expected.empty:
                    rows[key]["accuracy_on_expected_quality_ok"] = float(
                        (group_expected["expected_label"] == group_expected["predicted_label"]).mean()
                    )
            groups[column] = rows
        summary["groups"] = groups
    return summary
