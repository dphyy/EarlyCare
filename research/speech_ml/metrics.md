# EarlyCare Speech ML Metrics

Use this file as the reporting standard for offline speech experiments. These metrics are for research validation only. They must not be presented as diagnosis.

## Required Split Rule

All evaluation must use speaker-level splits. A speaker can appear in train or test, never both. Clip-level random splits are invalid because they leak speaker identity and inflate results.

## Required Metrics

| Metric | Why It Matters |
| --- | --- |
| ROC-AUC | Checks ranking quality before choosing a threshold. |
| Balanced accuracy | Handles uneven Parkinson's/control or positive/negative label counts. |
| Sensitivity | Shows how often positive cases are caught. |
| Specificity | Shows how often controls are not falsely flagged. |
| Calibration / Brier score | Shows whether score-like outputs behave like probabilities. |
| False-positive review | Lists speakers who would be escalated incorrectly. |
| False-negative review | Lists speakers who would be missed. |

## Minimum Report

Every experiment note should include:

- dataset names and versions
- access date and license/terms
- label definitions
- language and task types
- participant and speaker counts
- train/test split details
- model or feature extractor name
- preprocessing steps
- metrics table
- false-positive and false-negative examples
- subgroup checks when metadata exists

## Release Gate

Do not connect a model-backed score to the app unless the report proves speaker-level evaluation, no leakage, subgroup review, no-diagnosis copy, and a human follow-up action.
