"""Frozen WavLM embedding extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from speech_abnormality.audio import preprocess_audio
from speech_abnormality.labels import LOW_AUDIO_QUALITY


def resolve_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FrozenSpeechEmbedder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = model_name
        self.device = resolve_device(device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            mask = attention_mask.squeeze(0)
            feature_len = hidden.shape[0]
            if mask.shape[0] != feature_len:
                mask = torch.nn.functional.interpolate(
                    mask.float().view(1, 1, -1),
                    size=feature_len,
                    mode="nearest",
                ).view(-1)
            mask = mask.to(hidden.device).bool()
            if mask.any():
                hidden = hidden[mask]

        mean = hidden.mean(dim=0)
        std = hidden.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0).detach().cpu().numpy().astype(np.float32)


def extract_embeddings(
    manifest,
    config: dict[str, Any],
    output_path: str | Path,
    device: str = "auto",
    limit: int | None = None,
) -> dict[str, Any]:
    embedder = FrozenSpeechEmbedder(config["model_name"], device=device)
    rows = manifest.head(limit).to_dict("records") if limit else manifest.to_dict("records")
    embeddings: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="Extracting embeddings"):
        processed = preprocess_audio(
            row["path"],
            target_rate=int(config["sample_rate"]),
            max_seconds=float(config["max_seconds"]),
            min_seconds=float(config["min_seconds"]),
            silence_rms_threshold=float(config["silence_rms_threshold"]),
            clipping_threshold=float(config["clipping_threshold"]),
            clipping_fraction_threshold=float(config["clipping_fraction_threshold"]),
        )
        if not processed.quality.ok:
            rejected = dict(row)
            rejected["quality_label"] = LOW_AUDIO_QUALITY
            rejected["quality_reason"] = processed.quality.reason
            rejected_rows.append(rejected)
            continue
        embedding = embedder.embed_waveform(processed.waveform, processed.sample_rate)
        embeddings.append(embedding)
        kept_rows.append(dict(row))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if embeddings:
        x = np.stack(embeddings)
    else:
        x = np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(output, embeddings=x, rows=np.array(kept_rows, dtype=object), rejected=np.array(rejected_rows, dtype=object))
    return {"embeddings": x, "rows": kept_rows, "rejected": rejected_rows}


def load_embedding_cache(path: str | Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    return {
        "embeddings": loaded["embeddings"],
        "rows": loaded["rows"].tolist(),
        "rejected": loaded["rejected"].tolist(),
    }
