# EarlyCare Offline Speech ML

This folder contains local-only research scripts. Keep downloaded datasets under `research/datasets/` and generated outputs under `research/artifacts/`; both folders are ignored by git.

## Current Scope

The product does not need ML for the core MVP. These scripts only support optional speech-deviation research:

- Parkinson's watch validation across repeated calls
- personal-baseline speech deviation
- post-fall/head-impact support evidence

Do not use these scripts to claim Parkinson's disease, concussion, TBI, stroke, depression, or any other diagnosis.

## Manifest Format

Create a CSV or JSONL manifest under `research/datasets/`:

```csv
dataset,speaker_id,label,task,audio_path,language,transcript
sample,s-001,control,repeat_phrase,sample/control/s-001.wav,English,today i am safe at home and i can ask for help
sample,s-002,pd,repeat_phrase,sample/pd/s-002.wav,English,today i am safe at home and i can ask for help
```

`audio_path` can be absolute or relative to `--audio-root`.

## Extract Embeddings

Smoke-test the pipeline with the standard-library demo extractor:

```bash
python3 research/speech_ml/extract_embeddings.py \
  --manifest research/datasets/sample_manifest.csv \
  --audio-root research/datasets \
  --model demo \
  --output research/artifacts/sample_embeddings.jsonl
```

The JSONL output contains:

- `dataset`
- `speaker_id`
- `label`
- `task`
- `embedding`
- `speech_metrics`
- `provenance`

Heavy encoders are explicit research choices:

```bash
python3 research/speech_ml/extract_embeddings.py --model meralion
python3 research/speech_ml/extract_embeddings.py --model wavlm
python3 research/speech_ml/extract_embeddings.py --model wav2vec2
```

Those modes require optional `torch`, `transformers`, and `soundfile` packages in a separate research environment. Do not add those packages to the FastAPI runtime unless latency and deployment cost have been measured.

## Train Baseline

Train a small research-only baseline from extracted embeddings:

```bash
python3 research/speech_ml/train_baseline.py \
  --input research/artifacts/sample_embeddings.jsonl \
  --output research/artifacts/sample_baseline_model.json \
  --positive-labels pd,parkinson,parkinsonian
```

This writes a centroid baseline artifact for offline review. It is not loaded by the live app and is not a validated model.

## Evaluate

Run speaker-level evaluation:

```bash
python3 research/speech_ml/evaluate_baseline.py \
  --input research/artifacts/sample_embeddings.jsonl \
  --output research/artifacts/sample_eval.json \
  --positive-labels pd,parkinson,parkinsonian
```

For cross-dataset checks:

```bash
python3 research/speech_ml/evaluate_baseline.py \
  --input research/artifacts/combined_embeddings.jsonl \
  --train-dataset NeuroVoz \
  --test-dataset PC-GITA \
  --output research/artifacts/neurovoz_to_pcgita_eval.json
```

If there are too few positive or negative speakers, the report returns `insufficient-data` instead of pretending the model is valid.
