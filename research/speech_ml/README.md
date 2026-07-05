# EarlyCare Offline Speech ML

This folder contains local-only research scripts. Keep downloaded datasets under `research/datasets/` and generated outputs under `research/artifacts/`; both folders are ignored by git.

## Current Scope

The product does not need ML for the core MVP. These scripts only support optional speech-deviation research:

- Parkinson's watch validation across repeated calls
- personal-baseline speech deviation
- post-fall/head-impact support evidence

Do not use these scripts to claim Parkinson's disease, concussion, TBI, stroke, depression, or any other diagnosis.

## Inspect Dataset Readiness

Run the registry readiness check before fetching data or training:

```bash
python3 research/speech_ml/dataset_registry.py \
  --output research/artifacts/dataset_readiness.md \
  --json-output research/artifacts/dataset_readiness.json
```

The registry lives at `research/speech_ml/dataset_registry.json`. The report also inspects local `dataset_fetch_manifest.json` files under `research/datasets/` and marks datasets as feature-baseline trainable, progression-analysis ready, access-needed, or literature-only.

After local datasets have been fetched, run every locally ready feature-baseline experiment:

```bash
python3 research/speech_ml/run_ready_experiments.py \
  --output-dir research/artifacts
```

Use `--dry-run` to preview commands first. Add `--include-progression` when you also want progression-only analysis reports; those are not classifier training.

Audit generated artifacts before any app handoff:

```bash
python3 research/speech_ml/audit_model_artifacts.py \
  --artifacts-dir research/artifacts
```

This writes `model_artifact_audit.md` and `model_artifact_audit.json`. A trained artifact remains research-only unless every model-card gate is complete and safe human follow-up wording is present.

## Fetch Public Feature Datasets

Download supported public feature-only datasets into ignored local folders:

```bash
python3 research/speech_ml/fetch_public_datasets.py \
  --dataset uci-parkinson-speech
```

The fetcher writes `dataset_fetch_manifest.json` under `research/datasets/<dataset>/` with the source URL, table candidates, nested archives, extraction notes, and table readiness summaries. Check `table_summaries[].classification_ready` before running classifier training.

UCI's Parkinson speech package currently downloads as a zip containing a `.rar`. Python cannot extract that nested archive without an external tool, so the fetcher writes `EXTRACTION_REQUIRED.md` when `unar`, `unrar`, or `7z` is not available. After installing one of those tools, re-run with:

```bash
python3 research/speech_ml/fetch_public_datasets.py \
  --dataset uci-parkinson-speech \
  --allow-external-extractors
```

To fetch the UCI Parkinsons Telemonitoring feature table for longitudinal/progression analysis:

```bash
python3 research/speech_ml/fetch_public_datasets.py \
  --dataset uci-parkinsons-telemonitoring
```

Telemonitoring tables should show `progression_ready=true` and `classification_ready=false`; do not pass them into the PD/control classifier flow.

Analyze the progression-only table separately:

```bash
python3 research/speech_ml/analyze_progression_table.py \
  --dataset-fetch-manifest research/datasets/uci-parkinsons-telemonitoring/dataset_fetch_manifest.json \
  --output research/artifacts/uci-telemonitoring_progression.json
```

This writes subject-level UPDRS trend summaries and exploratory voice-feature associations. It is not a classifier and must not be used for diagnosis or app routing.
The analyzer also writes a markdown report next to the JSON output unless `--report-output` is provided.

After extraction, run an experiment directly from a fetch manifest. The runner selects the first `classification_ready=true` table and refuses progression-only manifests:

```bash
python3 research/speech_ml/run_experiment.py \
  --dataset-fetch-manifest research/datasets/uci-parkinson-speech/dataset_fetch_manifest.json \
  --output-dir research/artifacts \
  --experiment-name uci-parkinson-feature
```

## Manifest Format

Create a CSV or JSONL manifest under `research/datasets/`:

```csv
dataset,speaker_id,label,task,audio_path,language,transcript
sample,s-001,control,repeat_phrase,sample/control/s-001.wav,English,today i am safe at home and i can ask for help
sample,s-002,pd,repeat_phrase,sample/pd/s-002.wav,English,today i am safe at home and i can ask for help
```

`audio_path` can be absolute or relative to `--audio-root`.

## Prepare a Manifest

After downloading an approved raw-audio dataset locally, generate a first-pass manifest from its folder layout:

```bash
python3 research/speech_ml/prepare_manifest.py \
  --audio-root research/datasets/NeuroVoz \
  --dataset NeuroVoz \
  --language Spanish \
  --output research/datasets/neurovoz_manifest.csv
```

The preparer infers `label`, `speaker_id`, and `task` from path names, then marks uncertain rows as `needs-review`. Review and fix those rows before extraction or training.

The manifest `audio_path` values are relative to the `--audio-root` used above. Use the same audio root when extracting embeddings or running experiments.

## Run an Experiment

Run extraction, speaker-level evaluation, baseline training, and a markdown experiment report:

```bash
python3 research/speech_ml/run_experiment.py \
  --manifest research/datasets/neurovoz_manifest.csv \
  --audio-root research/datasets/NeuroVoz \
  --output-dir research/artifacts \
  --experiment-name neurovoz-demo \
  --model demo
```

The runner refuses manifests with `review_status=needs-review` unless `--allow-review-rows` is explicitly passed.
It also writes a draft model card and conservative `model_card_gate.json`; human-review fields stay false until they are manually reviewed.
Every run also writes `*_personal_baselines.json`, which estimates per-speaker normal embedding variation when enough repeated samples exist.

## Build an App Enrichment Payload

Convert one experiment row into the FastAPI payload accepted by `/calls/{call_id}/speech-enrichment`:

```bash
python3 research/speech_ml/make_enrichment_payload.py \
  --input research/artifacts/neurovoz-demo_embeddings.jsonl \
  --output research/artifacts/neurovoz-demo_payload.json \
  --speaker-id s-001
```

This defaults to `offline embedding`, which stores the embedding, speech metrics, and provenance as decision-support context. It does not mark the model as validated.

After the backend is running and a call has been saved:

```bash
CALL_ID=call-id-from-save-response
curl -X PATCH "http://localhost:8000/calls/${CALL_ID}/speech-enrichment" \
  -H "Content-Type: application/json" \
  --data @research/artifacts/neurovoz-demo_payload.json
```

Only use `--runtime-mode "validated model"` after the model-card review is complete. That mode requires `--model-card-gate`, `--model-version`, a stable artifact URI, and safe human follow-up wording:

```bash
python3 research/speech_ml/make_enrichment_payload.py \
  --input research/artifacts/neurovoz-demo_embeddings.jsonl \
  --output research/artifacts/neurovoz-demo_validated_payload.json \
  --speaker-id s-001 \
  --runtime-mode "validated model" \
  --model-version 2026-07-05 \
  --artifact-uri research/artifacts/neurovoz-demo_baseline_model.json \
  --model-card-gate research/artifacts/neurovoz-demo_model_card_gate.json
```

## Convert Feature Tables

Feature-only datasets such as UCI Parkinson's Speech do not validate raw-audio embeddings, but they are useful for quick speaker-level sanity checks:

```bash
python3 research/speech_ml/run_experiment.py \
  --feature-table research/datasets/uci-parkinson/training_data.csv \
  --output-dir research/artifacts \
  --experiment-name uci-parkinson-feature \
  --dataset "UCI Parkinson Speech" \
  --language Turkish
```

The runner writes z-scored numeric feature vectors using `source_type=feature_table` provenance, then evaluates and trains through the same speaker-level baseline flow. Keep results labelled as feature-table experiments, and do not flip the model-card gate to validated without manual review.

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

## Build Personal Baselines

Estimate within-speaker speech drift thresholds from repeated samples:

```bash
python3 research/speech_ml/build_personal_baselines.py \
  --input research/artifacts/sample_embeddings.jsonl \
  --output research/artifacts/sample_personal_baselines.json \
  --min-samples 3
```

This is closer to EarlyCare's real product signal than a diagnosis classifier: it measures how far a speaker's future call may drift from their own stable pattern. Treat thresholds as draft research values until a model card is manually reviewed.

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
