from __future__ import annotations

import os
from pathlib import Path


MODEL_NAME = "microsoft/wavlm-base"


def backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cache_root() -> Path:
    return backend_root() / "models" / "hf_cache"


def cache_wavlm() -> Path:
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from transformers import AutoFeatureExtractor, AutoModel

    AutoFeatureExtractor.from_pretrained(MODEL_NAME, cache_dir=str(root / "hub"))
    AutoModel.from_pretrained(MODEL_NAME, cache_dir=str(root / "hub"))
    return root


def main() -> None:
    root = cache_wavlm()
    print(f"Cached {MODEL_NAME} under {root}")


if __name__ == "__main__":
    main()
