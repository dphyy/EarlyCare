# EarlyCare ML Dataset Registry

This registry is the source of truth for datasets considered for EarlyCare speech ML. Do not commit dataset files, downloaded audio, derived embeddings, or subject-level exports to git.

## Status Labels

- `usable-now`: enough public information and access path for offline experimentation.
- `access-needed`: promising dataset, but access, licensing, or download terms must be completed before use.
- `feature-only`: useful for scoring ideas, but does not provide raw audio for embedding validation.
- `literature-only`: useful for product/research framing only; do not train on it now.

## Registry

| Dataset | Status | Target | Access / Terms | Labels | Language | Tasks | Participants / Size | Raw Audio | EarlyCare Use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [NeuroVoz](https://github.com/BYO-UPM/Neurovoz_Dababase) | `usable-now` | Parkinson's speech watch | Public repo and Zenodo citation path; confirm dataset license before redistribution or derived release. | PD vs healthy control. | Castilian Spanish. | Sustained vowels, DDK, listen-and-repeat utterances, monologue. | Repo describes 108 speakers: 53 Parkinsonian, 55 controls; 2,903 audio files. | Yes. | First offline raw-audio benchmark for embeddings, task design, and no-diagnosis watch scoring. |
| [NeuroVoz Scientific Data paper](https://www.nature.com/articles/s41597-024-04186-z) | `literature-only` | Parkinson's speech watch | Paper; cite when using NeuroVoz. | Dataset description. | Castilian Spanish. | Vowels, text-dependent utterances, DDK, spontaneous monologue. | Paper describes PD and healthy-control recordings. | N/A. | Citation, limitations, and methodology context. |
| [PC-GITA](https://aclanthology.org/L14-1549/) | `access-needed` | Parkinson's speech watch | Access/licensing must be confirmed before local use. | PD vs healthy control. | Colombian Spanish. | Sustained vowels, DDK, words, sentences, reading text, monologue. | 50 PD and 50 matched controls. | Usually yes, but access-dependent. | Secondary raw-audio benchmark and cross-dataset check after access is approved. |
| [UCI Parkinson's Speech with Multiple Types of Sound Recordings](https://archive.ics.uci.edu/dataset/301/parkinson%2Bspeech%2Bdataset%2Bwith%2Bmultiple%2Btypes%2Bof%2BAudio%2Brecordings) | `feature-only` | Parkinson's speech watch | UCI dataset terms. | PD vs healthy control; UPDRS available for PD. | Turkish. | Sustained vowels, numbers, words, short sentences. | 20 PD and 20 controls in training set; additional sustained-vowel test set. | No, extracted feature tables. | Feature-level smoke tests and scoring calibration ideas, not raw embedding validation. |
| [UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons%2Btelemonitoring) | `feature-only` | Parkinson's progression | UCI dataset terms. | Motor and total UPDRS. | English context, feature table only. | Home voice recordings represented as biomedical voice measures. | 42 early-stage PD participants; 5,875 rows. | No, extracted feature table. | Longitudinal scoring ideas for progression, not app model training. |
| [mPower](https://www.synapse.org/mpower) | `access-needed` | Parkinson's speech watch | Synapse account, data-use review, and privacy constraints required. | Self-reported PD/control plus surveys and mobile measures. | Mostly English mobile study context. | Phonation, gait, tapping, cognition, surveys. | Portal describes 8,320 participants. | Access-dependent. | Possible larger sustained-vowel validation after access and consent review. |
| [TBIBank](https://talkbank.org/tbi/) | `access-needed` | TBI communication research | Password-protected consortium access; faculty/clinical sponsorship may be required. | TBI communication samples. | Mostly English corpora plus other TalkBank languages. | Discourse and conversation. | Corpus-dependent. | Access-dependent. | Future discourse/language research only; not acute concussion model training. |
| [TBIBank Coelho corpus](https://talkbank.org/tbi/access/English/Coelho.html) | `access-needed` | Chronic closed-head-injury communication | TalkBank membership and corpus-specific terms. | Closed head injury vs control. | English. | Discourse and conversation. | 55 closed-head-injury speakers and 52 controls. | Access-dependent. | Future language-feature research; not acute concussion detection. |
| Concussion speech pilot datasets | `literature-only` | Acute concussion research | Public download path not verified; do not use until access and consent terms are explicit. | Concussed vs healthy/control in small studies. | Study-dependent. | Often short speech tasks or acoustic features. | Study-dependent; current evidence is too fragmented for product training. | Usually no public app-ready audio. | Literature context only. Keep EarlyCare concussion escalation symptom-led. |

## Use Rules

- Keep all raw data under ignored local folders such as `research/datasets/`.
- Keep derived embeddings, CSV exports, plots, and notebooks under ignored `research/artifacts/`.
- Use speaker-level splits only. Never place clips from the same speaker in both train and test.
- Record dataset version, access date, source URL, license/terms, label definitions, and preprocessing in any experiment note.
- Do not combine languages or tasks without reporting per-language and per-task results.
- Do not use any dataset to claim EarlyCare diagnoses Parkinson's disease, concussion, TBI, stroke, depression, or any other medical condition.

## Current Decision

Use NeuroVoz first for offline Parkinson's speech-watch validation because it has raw audio, matched control data, and task types that map to the EarlyCare prompts. Use UCI feature datasets only for quick scoring sanity checks. Defer supervised concussion ML until there is an approved acute concussion speech dataset with raw audio, consent terms, and enough validation coverage.
