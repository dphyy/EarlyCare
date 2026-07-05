# EarlyCare Speech ML Training Runbook

This runbook explains what we can train, how we find datasets, and what must stay out of the live app until validation is complete.

## Training Decision

We can train offline research models for **speech-deviation support**. We should not train or ship a direct Parkinson's detector or concussion detector for the product.

For EarlyCare, the right model target is:

1. Personal-baseline speech deviation across repeated calls.
2. Parkinson's watch validation on public Parkinson's speech datasets.
3. Post-fall/concussion support evidence only after symptom-led triage.

Concussion escalation remains rule-led: fall, head impact, body jolt, whiplash-like event, worsening headache, vomiting, confusion, slurred speech, weakness, numbness, unusual behaviour, drowsiness, or cannot wake.

## How We Find Datasets

Use primary sources first:

- Dataset papers with direct data links.
- Official repositories such as Zenodo, UCI, PhysioNet, Synapse, TalkBank, and institutional project pages.
- Dataset owner GitHub repositories when they link to the primary release.
- Hugging Face or Kaggle only as discovery or convenience mirrors, then trace back to the original source and license.

Screen every candidate before download:

- Does it have raw audio or only extracted features?
- Does it include speaker IDs so we can do speaker-level splits?
- Are labels clinically defined, self-reported, or proxy labels?
- Are language, task type, microphone/device, age, sex, and recording condition available?
- Are consent, license, and redistribution terms compatible with research use?
- Does it match EarlyCare's intended use: older adults, repeated check-ins, speech tasks, and human follow-up?

Keep data local only:

- Raw data: `research/datasets/`
- Derived embeddings/models/reports: `research/artifacts/`
- Never commit dataset files, embeddings, model artifacts, or subject exports.

## Dataset Shortlist

| Dataset | Action | Training Use | Notes |
| --- | --- | --- | --- |
| NeuroVoz | Download locally first and confirm license. | First raw-audio Parkinson's watch benchmark. | Best immediate dataset because it has PD/control labels, raw audio, DDK, vowels, repeat utterances, and monologues. |
| UCI Parkinson's Speech with Multiple Types of Sound Recordings | Use as feature-only sanity check. | Quick feature-table baseline, not raw-audio embedding validation. | Has 20 PD and 20 controls with multiple voice samples, but public release is feature-focused. |
| UCI Parkinsons Telemonitoring | Use for longitudinal scoring ideas. | Severity/progression analysis only. | Large feature table, no raw audio and no healthy controls. |
| mPower | Request Synapse access if we need scale. | Large mobile phonation validation after access review. | Self-report and privacy constraints require careful review. |
| PC-GITA | Request access if cross-dataset evaluation is needed. | Cross-dataset Parkinson's speech validation. | Strong benchmark, but access/licensing must be confirmed. |
| Bridge2AI Voice | Track for future broad voice-health features. | Public feature-level analysis; raw audio only after institutional sign-off. | Not a quick hackathon training source. |
| TBIBank / Coelho | Access-needed for language/discourse research. | Chronic TBI communication research only. | Not acute concussion detection. |
| Concussion speech studies | Literature-only for now. | Do not train product model yet. | Several papers report promising speech features, but app-ready public raw-audio datasets are not clearly available. |

## Training Commands

After downloading an approved raw-audio dataset locally, create a first-pass manifest under `research/datasets/`:

```bash
python3 research/speech_ml/prepare_manifest.py \
  --audio-root research/datasets/NeuroVoz \
  --dataset NeuroVoz \
  --language Spanish \
  --output research/datasets/neurovoz_manifest.csv
```

Open the manifest and manually review any row with `review_status=needs-review` before extraction or training. The script infers labels and speaker IDs from folder/file names; it does not prove clinical truth.

Run the full offline experiment:

```bash
python3 research/speech_ml/run_experiment.py \
  --manifest research/datasets/neurovoz_manifest.csv \
  --audio-root research/datasets/NeuroVoz \
  --output-dir research/artifacts \
  --experiment-name neurovoz-demo \
  --model demo
```

The runner writes embeddings, a speaker-level evaluation JSON file, a baseline model artifact, personal-baseline drift thresholds, a markdown experiment report, a draft model card, and a conservative `model_card_gate.json`. It refuses `needs-review` manifest rows unless `--allow-review-rows` is explicitly passed.

Manual commands are still available when a step needs to be inspected separately.

Build a backend payload from one offline row when you want to attach experiment output to a saved call:

```bash
python3 research/speech_ml/make_enrichment_payload.py \
  --input research/artifacts/neurovoz-demo_embeddings.jsonl \
  --output research/artifacts/neurovoz-demo_payload.json \
  --speaker-id s-001
```

Patch it into the app after the backend is running and the call exists:

```bash
CALL_ID=call-id-from-save-response
curl -X PATCH "http://localhost:8000/calls/${CALL_ID}/speech-enrichment" \
  -H "Content-Type: application/json" \
  --data @research/artifacts/neurovoz-demo_payload.json
```

This app handoff defaults to `offline embedding`. Do not use `validated model` unless the model-card gate has been manually completed and the payload includes a model version, stable artifact URI, and human follow-up action.

Feature-only UCI sanity check, after downloading and extracting the UCI table locally:

```bash
python3 research/speech_ml/run_experiment.py \
  --feature-table research/datasets/uci-parkinson/training_data.csv \
  --output-dir research/artifacts \
  --experiment-name uci-parkinson-feature \
  --dataset "UCI Parkinson Speech" \
  --language Turkish
```

This writes feature rows, a speaker-level evaluation JSON file, a baseline model artifact, personal-baseline drift thresholds when enough repeated samples exist, a markdown report, and a model-card draft. Manual commands are still available when a step needs to be inspected separately:

```bash
python3 research/speech_ml/convert_feature_table.py \
  --input research/datasets/uci-parkinson/training_data.csv \
  --output research/artifacts/uci_parkinson_feature_rows.jsonl \
  --dataset "UCI Parkinson Speech" \
  --language Turkish

python3 research/speech_ml/evaluate_baseline.py \
  --input research/artifacts/uci_parkinson_feature_rows.jsonl \
  --output research/artifacts/uci_parkinson_feature_eval.json \
  --positive-labels pd,parkinson,parkinsonian

python3 research/speech_ml/train_baseline.py \
  --input research/artifacts/uci_parkinson_feature_rows.jsonl \
  --output research/artifacts/uci_parkinson_feature_model.json \
  --positive-labels pd,parkinson,parkinsonian

python3 research/speech_ml/build_personal_baselines.py \
  --input research/artifacts/uci_parkinson_feature_rows.jsonl \
  --output research/artifacts/uci_parkinson_personal_baselines.json \
  --min-samples 3
```

Treat this as feature-table validation only. It does not prove raw-audio embedding performance.

Extract embeddings:

```bash
python3 research/speech_ml/extract_embeddings.py \
  --manifest research/datasets/neurovoz_manifest.csv \
  --audio-root research/datasets/NeuroVoz \
  --model demo \
  --output research/artifacts/neurovoz_embeddings.jsonl
```

Train the offline baseline:

```bash
python3 research/speech_ml/train_baseline.py \
  --input research/artifacts/neurovoz_embeddings.jsonl \
  --output research/artifacts/neurovoz_baseline_model.json \
  --positive-labels pd,parkinson,parkinsonian
```

Evaluate with speaker-level splits:

```bash
python3 research/speech_ml/evaluate_baseline.py \
  --input research/artifacts/neurovoz_embeddings.jsonl \
  --output research/artifacts/neurovoz_eval.json \
  --positive-labels pd,parkinson,parkinsonian
```

Cross-dataset evaluation, after access to a second dataset:

```bash
python3 research/speech_ml/evaluate_baseline.py \
  --input research/artifacts/combined_embeddings.jsonl \
  --train-dataset NeuroVoz \
  --test-dataset PC-GITA \
  --output research/artifacts/neurovoz_to_pcgita_eval.json
```

## Release Gate

A model can be labelled `validated model` only after:

- Dataset access and consent terms are documented.
- Speaker-level splits are verified.
- No speaker leakage exists.
- Balanced accuracy, ROC-AUC, sensitivity, specificity, calibration, false positives, and false negatives are reported.
- Subgroup checks cover age, sex, task type, language, and recording condition where metadata exists.
- Failure modes are documented.
- UI copy says speech deviation or possible watch signal, never diagnosis.
- Human follow-up action is defined.
- Rollback path is documented.

Until then, EarlyCare should keep using `demo metrics` or `offline embedding` provenance and keep escalation human-led.
