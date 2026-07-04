import re

from app.models import (
    ConversationCategory,
    ConversationCategoryId,
    EscalationStep,
    RiskAssessment,
    RiskLevel,
    RiskSignal,
    Senior,
    Symptoms,
)


CATEGORY_LABELS: dict[ConversationCategoryId, str] = {
    "mental_wellbeing": "Mental wellbeing / basic check-in",
    "fall_head_impact": "Fall / head impact / whiplash",
    "concussion_danger": "Possible concussion danger signs",
    "parkinsons_watch": "Possible Parkinson's speech watch",
    "chronic_illness": "Chronic illness check-in",
    "medication_food_water": "Medication / food / water",
    "social_isolation": "Social isolation / help request",
    "missed_checkin": "Missed check-in",
}

RISK_ORDER: dict[RiskLevel, int] = {"Green": 0, "Watch": 1, "Amber": 2, "Red": 3}


def _max_risk(*levels: RiskLevel) -> RiskLevel:
    return max(levels, key=lambda level: RISK_ORDER[level])


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _evidence(text: str, fallback: str, *patterns: str) -> list[str]:
    if _has(text, *patterns):
        return [fallback]
    return []


def detect_symptoms_from_text(text: str) -> Symptoms:
    normalized = text.lower()
    return Symptoms(
        fall=_has(normalized, r"\bfall\b", r"\bfell\b", r"\bslipped\b"),
        headImpact=_has(normalized, r"hit (my|the)? ?head", r"head (bump|impact|knock)", r"banged .*head"),
        whiplashOrJolt=_has(normalized, r"whiplash", r"\bjolt\b", r"jerked", r"bump, blow, or jolt"),
        headache=_has(normalized, r"headache", r"head pain", r"head hurts"),
        worseningHeadache=_has(normalized, r"worse.*head", r"headache.*worse", r"does not go away"),
        dizziness=_has(normalized, r"dizz"),
        vomiting=_has(normalized, r"vomit", r"throwing up", r"nausea"),
        confusion=_has(normalized, r"confus", r"cannot recognize", r"don't know where"),
        slurredSpeech=_has(normalized, r"slurred", r"trouble speaking", r"speech difficult"),
        weakness=_has(normalized, r"weak", r"no strength"),
        numbness=_has(normalized, r"numb"),
        unusualBehavior=_has(normalized, r"unusual behavior", r"restless", r"agitated"),
        drowsinessOrUnwakeable=_has(normalized, r"drowsy", r"cannot wake", r"unwakeable", r"very sleepy"),
        poorIntake=_has(normalized, r"not eat", r"no appetite", r"not drink", r"dehydrat"),
        asksForHelp=_has(normalized, r"need help", r"ask for help", r"please help"),
        missedCheckIn=_has(normalized, r"no answer", r"missed check", r"did not answer"),
        loneliness=_has(normalized, r"lonely", r"alone", r"talking to the wall", r"die alone"),
        lowMood=_has(normalized, r"sad", r"low mood", r"hopeless", r"anxious", r"afraid"),
        medicationMissed=_has(normalized, r"missed.*(med|medicine|medication)", r"forgot.*(med|medicine|medication)", r"did not take.*(med|medicine|medication)"),
        chronicConcern=_has(normalized, r"kidney", r"\bckd\b", r"diabetes", r"blood pressure", r"bp "),
        ckdConcern=_has(normalized, r"kidney", r"\bckd\b", r"dialysis"),
        diabetesConcern=_has(normalized, r"diabetes", r"blood sugar", r"glucose"),
        highBloodPressureConcern=_has(normalized, r"blood pressure", r"hypertension", r"bp "),
    )


def assessment_from_symptoms(symptoms: Symptoms, risk_level: RiskLevel, reasons: list[str]) -> RiskAssessment:
    danger_signs = sum(
        [
            symptoms.confusion,
            symptoms.vomiting,
            symptoms.slurredSpeech,
            symptoms.weakness,
            symptoms.numbness,
            symptoms.unusualBehavior,
            symptoms.drowsinessOrUnwakeable,
            symptoms.worseningHeadache,
        ]
    )
    post_head_event = symptoms.fall or symptoms.headImpact or symptoms.whiplashOrJolt
    post_fall_score = min(
        100,
        (28 if symptoms.fall else 0)
        + (28 if symptoms.headImpact else 0)
        + (18 if symptoms.whiplashOrJolt else 0)
        + (14 if symptoms.headache or symptoms.dizziness else 0)
        + danger_signs * 20,
    )
    missed_score = 100 if symptoms.missedCheckIn else 0

    inferred_level: RiskLevel = risk_level
    if post_head_event and danger_signs:
        inferred_level = "Red"
    elif symptoms.missedCheckIn or (post_head_event and (symptoms.headache or symptoms.dizziness)):
        inferred_level = _max_risk(inferred_level, "Amber")
    elif symptoms.loneliness or symptoms.lowMood or symptoms.chronicConcern or symptoms.medicationMissed:
        inferred_level = _max_risk(inferred_level, "Watch")

    return RiskAssessment(
        speechDeviationScore=0,
        parkinsonsWatchScore=0,
        postFallConcernScore=post_fall_score,
        missedCheckInScore=missed_score,
        riskLevel=inferred_level,
        reasons=reasons or ["Conversation reviewed for follow-up signals."],
    )


def _category(
    category_id: ConversationCategoryId,
    severity: RiskLevel,
    evidence: list[str],
    action: str,
) -> ConversationCategory:
    return ConversationCategory(
        id=category_id,
        label=CATEGORY_LABELS[category_id],
        severity=severity,
        evidence=evidence,
        recommendedAction=action,
    )


def build_conversation_categories(text: str, symptoms: Symptoms, assessment: RiskAssessment, senior: Senior | None = None) -> list[ConversationCategory]:
    _ = senior
    danger = any(
        [
            symptoms.confusion,
            symptoms.vomiting,
            symptoms.slurredSpeech,
            symptoms.weakness,
            symptoms.numbness,
            symptoms.unusualBehavior,
            symptoms.drowsinessOrUnwakeable,
            symptoms.worseningHeadache,
        ]
    )
    post_head_event = symptoms.fall or symptoms.headImpact or symptoms.whiplashOrJolt
    chronic_evidence: list[str] = []
    if symptoms.ckdConcern:
        chronic_evidence.append("CKD or kidney concern mentioned.")
    if symptoms.diabetesConcern:
        chronic_evidence.append("Diabetes or blood sugar concern mentioned.")
    if symptoms.highBloodPressureConcern:
        chronic_evidence.append("High blood pressure or hypertension concern mentioned.")
    chronic_evidence.extend(_evidence(text, "Chronic condition check-in details mentioned.", r"kidney", r"\bckd\b", r"diabetes", r"blood pressure", r"hypertension"))

    parkinsons_evidence = []
    if assessment.parkinsonsWatchScore >= 50:
        parkinsons_evidence.append("Demo baseline scoring shows slower rate, longer pauses, lower pitch variation, or lower phrase accuracy.")
    parkinsons_evidence.extend(_evidence(text, "Speech-watch markers mentioned in conversation.", r"slow", r"long pause", r"pa\W*ta\W*ka", r"stammer", r"tremor", r"word.finding"))

    categories = [
        _category(
            "mental_wellbeing",
            "Watch" if symptoms.lowMood or symptoms.loneliness else "Green",
            _evidence(text, "Mood or loneliness concern mentioned.", r"lonely", r"sad", r"afraid", r"anxious", r"talking to the wall") + (["Senior asked for help."] if symptoms.asksForHelp else []),
            "Ask a short wellbeing question and log whether the senior wants a call, visit, or social support.",
        ),
        _category(
            "fall_head_impact",
            "Amber" if post_head_event else "Green",
            (["Fall reported."] if symptoms.fall else [])
            + (["Head impact reported."] if symptoms.headImpact else [])
            + (["Whiplash-like jolt or blow reported."] if symptoms.whiplashOrJolt else []),
            "Ask what happened, whether the head or body was hit, and whether the senior can move safely.",
        ),
        _category(
            "concussion_danger",
            "Red" if post_head_event and danger else "Amber" if post_head_event and (symptoms.headache or symptoms.dizziness) else "Green",
            (["Worsening headache reported."] if symptoms.worseningHeadache else [])
            + (["Vomiting reported."] if symptoms.vomiting else [])
            + (["Confusion reported."] if symptoms.confusion else [])
            + (["Slurred speech or trouble speaking reported."] if symptoms.slurredSpeech else [])
            + (["Weakness or numbness reported."] if symptoms.weakness or symptoms.numbness else [])
            + (["Drowsiness or difficulty waking reported."] if symptoms.drowsinessOrUnwakeable else [])
            + (["Headache or dizziness after impact reported."] if post_head_event and (symptoms.headache or symptoms.dizziness) else []),
            "For red danger signs after a bump, blow, jolt, or fall, escalate for urgent medical help.",
        ),
        _category(
            "parkinsons_watch",
            "Watch" if parkinsons_evidence else "Green",
            parkinsons_evidence,
            "Compare against baseline over repeated check-ins and schedule caregiver or clinician follow-up if the pattern persists.",
        ),
        _category(
            "chronic_illness",
            "Watch" if chronic_evidence else "Green",
            chronic_evidence,
            "Ask condition-specific follow-up for CKD, diabetes, or high blood pressure and confirm medication or appointment issues.",
        ),
        _category(
            "medication_food_water",
            "Watch" if symptoms.medicationMissed or symptoms.poorIntake else "Green",
            (["Medication missed or forgotten."] if symptoms.medicationMissed else [])
            + (["Poor food or water intake reported."] if symptoms.poorIntake else [])
            + _evidence(text, "Medication, food, or water status mentioned.", r"medicine", r"medication", r"eat", r"drink", r"water"),
            "Confirm medication, food, and water intake. Escalate if intake is poor or medication has been missed.",
        ),
        _category(
            "social_isolation",
            "Watch" if symptoms.loneliness or symptoms.asksForHelp else "Green",
            (["Loneliness or fear of being unnoticed mentioned."] if symptoms.loneliness else [])
            + (["Senior explicitly asked for help."] if symptoms.asksForHelp else []),
            "Offer a befriender call or visit and record whether the senior accepts help.",
        ),
        _category(
            "missed_checkin",
            "Amber" if symptoms.missedCheckIn else "Green",
            ["Scheduled check-in missed after retry."] if symptoms.missedCheckIn else [],
            "Retry the call, notify the caregiver or neighbour, and create a volunteer follow-up task if still unanswered.",
        ),
    ]
    return categories


def recommended_action_for(assessment: RiskAssessment, categories: list[ConversationCategory], senior: Senior) -> str:
    if assessment.riskLevel == "Red":
        return f"Trigger emergency escalation and notify {senior.caregiverContact}; include neighbour contact if reachable."
    if any(category.id == "missed_checkin" and category.severity == "Amber" for category in categories):
        return f"Retry call, notify {senior.caregiverContact}, then assign a same-day volunteer visit if no response."
    if assessment.riskLevel == "Amber":
        return f"Notify {senior.caregiverContact} and assign a same-day volunteer or social-service follow-up."
    if assessment.riskLevel == "Watch":
        return "Log the evidence, continue scheduled check-ins, and arrange a volunteer follow-up if the signal repeats."
    return "Record routine check-in and continue the next scheduled call."


def build_escalation_plan(assessment: RiskAssessment, categories: list[ConversationCategory], senior: Senior) -> list[EscalationStep]:
    missed = any(category.id == "missed_checkin" and category.severity != "Green" for category in categories)
    needs_follow_up = assessment.riskLevel in {"Watch", "Amber", "Red"} or missed
    needs_caregiver = assessment.riskLevel in {"Amber", "Red"} or missed
    neighbour = f" Neighbour contact: {senior.neighborContact}." if senior.neighborContact else ""

    return [
        EscalationStep(id="routine-note", label="Routine note", status="Complete", detail="Check-in evidence is recorded in the senior profile."),
        EscalationStep(
            id="app-notification",
            label="App notification",
            status="Triggered" if needs_follow_up else "Standby",
            detail="Show the care-team alert in EarlyCare when any Watch, Amber, Red, or missed-check-in signal appears.",
        ),
        EscalationStep(
            id="retry-call",
            label="Retry call",
            status="Triggered" if missed else "Standby",
            detail="Retry once when a scheduled check-in is missed before escalating to an in-person follow-up.",
        ),
        EscalationStep(
            id="notify-caregiver-neighbour",
            label="Notify caregiver / neighbour",
            status="Triggered" if needs_caregiver else "Standby",
            detail=f"Notify {senior.caregiverContact}.{neighbour}",
        ),
        EscalationStep(
            id="volunteer-social-task",
            label="Volunteer / social-service task",
            status="Triggered" if needs_follow_up and assessment.riskLevel != "Green" else "Standby",
            detail="Assign a befriender, volunteer, or Active Ageing Centre style follow-up task.",
        ),
        EscalationStep(
            id="emergency-alert",
            label="Emergency alert",
            status="Triggered" if assessment.riskLevel == "Red" else "Standby",
            detail="Use only for red danger signs such as confusion, repeated vomiting, slurred speech, weakness, numbness, or cannot be woken after head/body impact.",
        ),
    ]


def risk_signals_from_categories(categories: list[ConversationCategory]) -> list[RiskSignal]:
    signals: list[RiskSignal] = []
    for category in categories:
        if category.severity == "Green" or not category.evidence:
            continue
        signals.append(
            RiskSignal(
                id=category.id,
                label=category.label,
                severity=category.severity,
                quotedText=" ".join(category.evidence[:2]),
                reason=category.recommendedAction,
            )
        )
    return signals
