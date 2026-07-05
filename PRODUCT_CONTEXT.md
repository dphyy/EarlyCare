# EarlyCare Product Context

EarlyCare is a hackathon MVP for elderly people living alone in Singapore. The current product direction is scheduled voice check-ins every 2-3 days, not clinic intake and not stroke-first screening.

The product turns completed calls, missed calls, and conversation evidence into earlier human follow-up. It is decision support for volunteers, caregivers, and care teams. It does not diagnose Parkinson's disease, concussion, stroke, depression, or any other condition.

## Current Wedge

EarlyCare focuses on:

- missed scheduled check-ins for seniors living alone
- fall, head impact, blow, jolt, or whiplash-like incidents
- possible concussion danger signs after head/body impact
- possible Parkinson's speech-watch changes against a personal baseline
- chronic illness check-ins for CKD, diabetes, and high blood pressure
- medication, food, water, loneliness, and explicit help requests
- escalation from notification to retry call, caregiver/neighbour contact, volunteer/social-service task, or emergency alert

Stroke is only a red-flag safety concern when the conversation indicates immediate emergency symptoms. It is not the main product pitch.

## Grounding

- Singapore's Ministry of Health said the number of Singapore residents aged 65+ living alone rose from 58,000 in 2018 to 79,000 in 2022. MOH also described Silver Generation Ambassador visits for at-risk seniors and monthly Active Ageing Centre befriender contact when social support is needed: <https://www.moh.gov.sg/newsroom/seniors-staying-alone/>
- Singapore Red Cross ElderAid describes a volunteer befriending and social-support model for vulnerable seniors, including regular visits: <https://redcross.sg/elderaid.html>
- News reports describe fear of dying unnoticed and lonely deaths among seniors living alone: <https://www.straitstimes.com/singapore/when-i-die-i-want-someone-to-know-fear-of-dying-alone-increases-among-elderly-folk> and <https://www.channelnewsasia.com/cna-insider/reasons-watch-series-dying-alone-singapore-elderly-lonely-deaths-4459596>
- Emergency button systems help when seniors press a device. EarlyCare complements that with scheduled calls and missed-check-in escalation: <https://www.straitstimes.com/life/new-emergency-alert-system-for-seniors-in-distress-helps-their-adult-children-too>
- Local lonely-death reporting reinforces the operational risk of unnoticed incidents: <https://www.straitstimes.com/singapore/elderly-couple-who-lived-alone-in-jurong-flat-found-dead-together>
- Falls among older adults are common and clinically important; the Singapore Medical Journal review is the local reference for falls framing: <http://www.smj.org.sg/article/approach-falls-among-elderly-community>
- Parkinson's speech research supports speech changes as a monitoring signal, but EarlyCare only frames this as possible watch evidence for follow-up: <https://www.nature.com/articles/s41531-025-00913-4>
- CDC traumatic brain injury guidance lists danger signs after bumps, blows, jolts, or head/body impacts, including worsening headache, repeated vomiting, confusion, unusual behaviour, slurred speech, weakness, numbness, and inability to wake: <https://www.cdc.gov/traumatic-brain-injury/signs-symptoms/index.html>
- NeuroVoz is future research and validation context only, not a production dependency: <https://github.com/BYO-UPM/Neurovoz_Dababase>

## Demo Flow

1. Run the backend and frontend.
2. Open **Demo runner**.
3. Run one of seven scripted scenarios: Stable check-in, Missed check-in, Parkinson's watch, Post-Fall Amber, Post-Fall Red, Chronic Illness Check-In, or Mental Wellbeing / Loneliness.
4. The backend persists a check-in record, generates the eight evidence categories, builds the escalation trail, updates the senior-level categorized record, and creates or updates volunteer tasks when follow-up is needed.
5. Open **Patient overview** to review next due time, due/overdue status, categorized history, risk scores, transcripts, escalation, and task actions.
6. Use **Agents call** for the live ElevenLabs path when provider credentials are configured in `backend/.env`.

## Safety Language

Use these phrases:

- possible Parkinson's speech watch
- possible post-head-impact concern
- possible concussion danger signs
- follow-up recommended
- decision support
- baseline/demo scoring

Avoid these phrases:

- diagnosed
- detected Parkinson's
- detected concussion
- medical certainty
- emergency dispatch unless the app is only describing a simulated Red escalation

## Demo Scope

The current speech scoring is demo baseline scoring from structured speech metrics. It is not a validated ML model. Saved calls now carry speech model provenance labels: `demo metrics`, `offline embedding`, or `validated model`. Offline embeddings can be attached to stored calls for research review, but unvalidated enrichment must not change emergency routing by itself.

The `validated model` label is gated by model-card evidence, including dataset access review, speaker-level split verification, evaluation metrics, subgroup checks, failure modes, UI copy review, rollback path, and a human follow-up action.

The current ML implementation direction is documented in `docs/ml/implementation-plan.md`. The short version: build speech-deviation ML as a personal-baseline anomaly signal, validate Parkinson's watch offline with public speech datasets, and keep post-fall/concussion escalation symptom-led until there is a licensed, validated acute concussion speech dataset.
