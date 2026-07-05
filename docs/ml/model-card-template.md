# EarlyCare Speech Model Card Template

Use this template before any ML-backed speech score is surfaced in the app. A model is not production-ready until every section is filled with evidence.

## Model Summary

- Model name:
- Version:
- Owner:
- Date:
- Runtime mode: `demo metrics`, `offline embedding`, or `validated model`
- Feature extractor: MERaLiON SpeechEncoder, WavLM, wav2vec 2.0, handcrafted features, or other
- Downstream model: none, anomaly threshold, logistic regression, random forest, neural classifier, or other

## Intended Use

- Intended EarlyCare use:
- Supported decision-support category:
- Human follow-up action:
- Expected users:
- Expected setting:

## Excluded Use

- Do not use this model to diagnose Parkinson's disease.
- Do not use this model to diagnose concussion or traumatic brain injury.
- Do not use this model for emergency dispatch without human review.
- Do not use this model outside the languages, tasks, age ranges, and recording conditions evaluated below.

## Dataset Provenance

| Dataset | Version / Date | Access Terms | Labels | Language | Task Types | Participants | Raw Audio | Used For |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |

## Consent And Privacy

- Consent basis:
- Data-use restrictions:
- Personal data handling:
- Retention period:
- Storage location:
- Derived artifacts created:

## Preprocessing

- Audio format:
- Resampling:
- Voice activity detection:
- Noise handling:
- Transcript dependency:
- Segment selection:
- Exclusions:

## Evaluation

Use speaker-level splits only.

| Metric | Value | Notes |
| --- | --- | --- |
| ROC-AUC |  |  |
| Balanced accuracy |  |  |
| Sensitivity |  |  |
| Specificity |  |  |
| Calibration |  |  |
| False-positive review |  |  |
| False-negative review |  |  |

## Subgroup Checks

| Subgroup | Result | Risk |
| --- | --- | --- |
| Sex / gender |  |  |
| Age band |  |  |
| Language / accent |  |  |
| Speech task |  |  |
| Recording condition |  |  |
| Known condition |  |  |

## Thresholds

- Green threshold:
- Watch threshold:
- Amber threshold:
- Red threshold:
- Rationale:
- Calibration dataset:

## Known Failure Modes

- Background noise:
- Poor microphone:
- Code-switching:
- Dialect/accent mismatch:
- Illness unrelated to target condition:
- Medication state:
- Fatigue, mood, dehydration, or anxiety:
- Missing personal baseline:

## Product Copy

Approved labels:

- speech deviation
- possible Parkinson's speech watch
- possible post-head-impact concern
- baseline change
- follow-up recommended

Blocked labels:

- Parkinson's detected
- concussion detected
- disease diagnosis
- medical certainty
- emergency confirmed

## Release Gate

- [ ] Dataset access and terms reviewed.
- [ ] Speaker-level split verified.
- [ ] Evaluation metrics recorded.
- [ ] Subgroup checks reviewed.
- [ ] Failure modes documented.
- [ ] UI copy reviewed for no-diagnosis language.
- [ ] Human follow-up action defined.
- [ ] Rollback path documented.

## API Gate

`validated model` enrichment is blocked unless the API receives completed release-gate evidence through `modelCard`, including a human follow-up action. Use `offline embedding` for research artifacts that have not passed this gate.
