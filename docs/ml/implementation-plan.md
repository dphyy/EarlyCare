# EarlyCare Speech ML Implementation Plan

**Goal:** Add an optional, validated speech-deviation pipeline that supports earlier human follow-up for seniors living alone without claiming diagnosis.

**Architecture:** Keep the production app rule-first and safety-first. The core product works through scheduled check-ins, missed-call handling, structured conversation, and human escalation. Speech ML is a personal-baseline anomaly signal layered on top of that workflow, then combined with fall/head-impact reports, concussion danger signs, Parkinson's watch markers, and care-team escalation rules.

**Tech Stack:** FastAPI, Pydantic, local file storage for the hackathon demo, optional offline Python research jobs using PyTorch/Transformers, MERaLiON SpeechEncoder or WavLM/wav2vec-style embeddings, and the existing React Patient overview.

---

## Decision

EarlyCare does not need ML for the core MVP. The compulsory use case is regular living-alone senior check-ins plus escalation when someone misses a call or says something concerning.

ML is still useful as a differentiator, but not as a direct "Parkinson's detector" or "concussion detector" in the current product.

The better approach is:

- Use rules, structured check-in prompts, and volunteer/caregiver escalation as the foundation.
- Use ML only for **speech deviation from a senior's own stable baseline**.
- Treat Parkinson's as a **watch pattern** across repeated calls, not a single-call diagnosis.
- Treat concussion as **symptom-led triage after a fall, head impact, blow, jolt, or whiplash-like event**. Speech change can strengthen concern, but it should not be the primary concussion detector.
- Keep Red alerts rule-led: fall/head impact plus danger signs such as confusion, repeated vomiting, slurred speech, weakness, numbness, unusual behaviour, worsening headache, or inability to wake.
- Build an offline dataset and evaluation harness before any model affects user-facing risk levels.

This matches the current codebase. `backend/app/ml.py` scores demo speech metrics and compares them with `SpeechProfile.embedding`; `backend/app/risk.py` keeps symptom, category, and escalation logic separate from diagnosis.

## Dataset Findings

| Dataset | Relevance | What We Can Use It For | Constraints |
| --- | --- | --- | --- |
| [NeuroVoz](https://github.com/BYO-UPM/Neurovoz_Dababase) | Strong Parkinson's speech dataset. The repo describes 108 Castilian Spanish speakers, 55 controls and 53 Parkinsonian speakers, with sustained vowels, DDK, repeat utterances, and monologue tasks. | Offline PD speech feature extraction, embedding sanity checks, and task design for sustained vowels, `pa-ta-ka`, repeated phrases, and monologue. | Spanish, controlled clinical recording, patients in ON medication state. Good for research validation, not direct Singapore deployment. |
| [NeuroVoz Scientific Data paper](https://www.nature.com/articles/s41597-024-04186-z) | Primary dataset paper. It frames NeuroVoz as public PD/healthy-control speech data covering articulatory, phonatory, and prosodic tasks. | Citation and dataset understanding. | Not a concussion dataset and not multilingual Singapore data. |
| [PC-GITA](https://aclanthology.org/L14-1549/) | Classic Spanish PD corpus with 50 PD and 50 healthy controls, matched by age and gender, covering vowels, DDK, words, sentences, reading, and monologue. | Secondary offline benchmark if access is available. Helps compare generalization against NeuroVoz. | Access/licensing must be confirmed. Spanish and clinical lab context. |
| [UCI Parkinson's Speech with Multiple Types of Sound Recordings](https://archive.ics.uci.edu/dataset/301/parkinson%2Bspeech%2Bdataset%2Bwith%2Bmultiple%2Btypes%2Bof%2BAudio%2Brecordings) | Parkinson's feature dataset from 20 PD and 20 controls, with 26 voice samples and UPDRS metadata, plus an independent sustained-vowel test set. | Quick feature-level baselines and regression/classification smoke tests. | Mostly extracted features, not enough for validating raw-audio embedding quality. |
| [UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons%2Btelemonitoring) | Longitudinal feature dataset from 42 early-stage PD participants, 5,875 voice recordings, and UPDRS labels. | Severity/progression modeling ideas and longitudinal evaluation design. | Feature-level CSV, no healthy controls, no raw audio pipeline. |
| [mPower](https://www.synapse.org/mpower) | Large mobile Parkinson study with phonation, gait, tapping, cognition, and surveys. | Potential large-scale sustained-vowel baseline if access and consent terms fit. | Synapse access process, self-report label caveats, privacy and licensing review required. |
| [TBIBank](https://talkbank.org/tbi/) | TBI communication database with multimedia interactions. | Future language/discourse research for TBI communication changes. | Password-protected consortium access; not a plug-and-play public dataset. |
| [TBIBank Coelho corpus](https://talkbank.org/tbi/access/English/Coelho.html) | Closed-head-injury discourse/conversation samples from 55 CHI and 52 controls. | Possible language-feature research for chronic TBI communication differences. | Not acute concussion, access restrictions apply, and participants recovered high functional language. |
| Concussion speech pilot datasets | Some papers report concussion speech datasets, but the datasets are not clearly public or large enough for immediate product training. | Literature review only for now. | Do not build a supervised concussion detector without access, consent review, and validation. |

## Model Direction

Use foundation speech encoders as feature extractors first:

- Preferred Singapore-oriented path: [MERaLiON SpeechEncoder](https://arxiv.org/abs/2412.11538) or [MERaLiON-SpeechEncoder-2](https://huggingface.co/MERaLiON/MERaLiON-SpeechEncoder-2), because the product needs Singapore English, Singlish, Mandarin, Malay, Tamil, and code-switching coverage.
- Fallback research path: [WavLM](https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/) or [wav2vec 2.0](https://arxiv.org/abs/2006.11477) embeddings.
- Do not put heavy model inference inside the FastAPI request path until latency and memory are measured. For the hackathon path, run offline feature extraction and store `SpeechProfile.embedding`.

## Product Signal Design

### Parkinson's Watch

Track gradual changes across repeated calls:

- speech rate
- pause duration
- response latency
- pitch variability
- phrase accuracy
- optional sustained vowel
- optional `pa-ta-ka` DDK prompt
- embedding distance from personal baseline

Output: `Watch` or `Amber` follow-up only. Never output Parkinson's diagnosis.

### Post-Fall / Concussion Concern

Use symptom-led logic first:

- fall, head impact, whiplash-like jolt, or blow to head/body
- headache or worsening headache
- dizziness
- repeated vomiting
- confusion, unusual behaviour, agitation, drowsiness, or cannot wake
- slurred speech, weakness, numbness, decreased coordination

Speech ML can add support through slower response, large acute baseline deviation, or phrase failure, but it must not be a standalone concussion claim.

## Implementation Plan

### Current Status

| Task | Status | Evidence |
| --- | --- | --- |
| Data registry | Done | `docs/ml/dataset-registry.md` |
| Dataset manifest prep | Done | `research/speech_ml/prepare_manifest.py`, `research/speech_ml/test_research_tools.py` |
| Feature extractor interface | Done | `backend/app/speech_features.py`, `backend/tests/test_speech_features.py` |
| Offline embedding job | Done | `research/speech_ml/extract_embeddings.py`, `research/speech_ml/README.md` |
| Feature-table converter | Done | `research/speech_ml/convert_feature_table.py`, `research/speech_ml/run_experiment.py`, `research/speech_ml/test_research_tools.py` |
| Personal-baseline builder | Done | `research/speech_ml/build_personal_baselines.py`, `research/speech_ml/run_experiment.py`, `research/speech_ml/test_research_tools.py` |
| Offline baseline trainer | Done | `research/speech_ml/train_baseline.py`, `docs/ml/training-runbook.md` |
| Offline experiment runner | Done | `research/speech_ml/run_experiment.py`, `research/speech_ml/test_research_tools.py`; writes experiment report, model-card draft, and gate JSON. |
| Enrichment payload bridge | Done | `research/speech_ml/make_enrichment_payload.py`; `backend/tests/test_call_workflow.py` patches a generated payload into `/calls/{call_id}/speech-enrichment`. |
| Evaluation harness | Done | `research/speech_ml/evaluate_baseline.py`, `research/speech_ml/metrics.md` |
| App integration | Done | Saved calls store `speechModelProvenance`; `/calls/{call_id}/speech-enrichment` accepts offline embedding rows. |
| Safety gate | Done | Validated enrichment requires `modelCard`; `pnpm safety:copy` blocks diagnosis-style UI and README copy. |

### Task 1: Data Registry

**Files:**

- Create: `docs/ml/dataset-registry.md`
- Optional later: `research/datasets.yaml`

Steps:

- Document every candidate dataset, access status, license/terms, labels, language, task types, participant counts, and whether raw audio is available.
- Mark datasets as `usable-now`, `access-needed`, `feature-only`, or `literature-only`.
- Add a rule that no dataset files are committed to git.
- Acceptance: the registry can answer which datasets can be used for offline PD validation and why concussion training is deferred.

### Task 2: Feature Extractor Interface

**Files:**

- Create: `backend/app/speech_features.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_speech_features.py`

Steps:

- Define a small interface that accepts an audio path plus transcript segments and returns `SpeechProfile`.
- Implement `DemoSpeechFeatureExtractor` using the current timing metrics and optional stored embedding.
- Implement `NoopEmbeddingExtractor` as the default when no model is configured.
- Keep all output inside the existing `SpeechProfile` shape so the frontend does not need a data-contract rewrite.
- Acceptance: backend tests can produce deterministic `SpeechProfile` values without downloading a model.

### Task 3: Offline Embedding Job

**Files:**

- Create: `research/speech_ml/extract_embeddings.py`
- Create: `research/speech_ml/README.md`
- Modify: `.gitignore`

Steps:

- Add a local-only script that loads audio from an ignored dataset folder.
- Add one backend-compatible output format: JSONL rows containing `dataset`, `speaker_id`, `label`, `task`, `embedding`, `speech_metrics`, and `provenance`.
- Support one model at a time behind a `--model` flag: `meralion`, `wavlm`, or `wav2vec2`.
- Cache outputs under ignored `research/artifacts/`.
- Acceptance: the script can run on a tiny local sample without changing app runtime dependencies.

### Task 4: Evaluation Harness

**Files:**

- Create: `research/speech_ml/evaluate_baseline.py`
- Create: `research/speech_ml/metrics.md`

Steps:

- Evaluate speaker-level splits only; never let clips from the same speaker appear in both train and test.
- Report ROC-AUC, balanced accuracy, sensitivity, specificity, calibration, and false-positive cases.
- Run cross-dataset checks where possible, especially NeuroVoz to PC-GITA or feature-only comparisons where raw audio is unavailable.
- Include subgroup checks for sex, age range, task type, language, and recording condition when metadata exists.
- Acceptance: evaluation output makes it obvious when a model fails to generalize.

### Task 5: App Integration

**Files:**

- Modify: `backend/app/main.py`
- Modify: `backend/app/models.py`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/styles.css`

Steps:

- Keep the live call save path fast by using current timing metrics immediately.
- Add optional background/offline embedding enrichment for stored calls.
- Show model provenance beside speech timing: `demo metrics`, `offline embedding`, or `validated model`.
- Keep UI copy as "speech deviation" and "possible watch signal".
- Acceptance: the app remains useful without ML model weights, and model-backed fields are clearly labelled when present.

### Task 6: Safety Gate

**Files:**

- Modify: `PRODUCT_CONTEXT.md`
- Create: `docs/ml/model-card-template.md`

Steps:

- Add a model card template before any trained model is surfaced.
- Require dataset provenance, consent/licensing notes, intended use, excluded use, subgroup performance, and known failure modes.
- Block any UI or README text that says "detected Parkinson's", "detected concussion", or equivalent diagnosis language.
- Reject `validated model` enrichment unless all model-card release-gate checks are true and a human follow-up action is defined.
- Acceptance: every ML-backed score has safety wording and a human follow-up action.

## Build Order

1. Finish the data registry and model-card template.
2. Add deterministic feature-extractor tests around the current demo scoring.
3. Add offline embedding extraction without touching live request latency.
4. Train an offline baseline only after approved data is downloaded under `research/datasets/`.
5. Evaluate Parkinson's speech datasets offline.
6. Add optional model provenance to saved call records.
7. Only then consider a model-backed `speechDeviationScore`.

## Next Goal Prompt

```text
/goal Implement EarlyCare Speech ML Phase 1 on the dev branch only. Start by creating the dataset registry and model-card template under docs/ml, then add a deterministic backend speech feature extractor interface that preserves the existing SpeechProfile contract and keeps the current app working without model weights. Do not build a diagnostic Parkinson's or concussion classifier. Keep concussion logic symptom-led and use speech ML only as baseline deviation support. Run backend compile, backend smoke, frontend smoke, lint, and build before committing. Commit with Saai's configured identity and push origin dev.
```
