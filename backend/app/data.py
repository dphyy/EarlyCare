from app.models import CheckInSession, RiskAssessment, Senior, SpeechProfile, VolunteerTask


SENIORS = [
    Senior(
        id="s-001",
        name="Mdm Tan Bee Hoon",
        age=78,
        preferredLanguage="Mandarin",
        livingAlone=True,
        addressZone="Toa Payoh",
        caregiverContact="Daughter: Mei Ling",
        checkInFrequencyDays=2,
        baselineSpeechProfile=SpeechProfile(
            speechRate=122,
            avgPauseMs=620,
            responseLatencyMs=980,
            pitchVariability=0.64,
            phraseAccuracy=0.96,
            embedding=[0.12, 0.28, 0.45, 0.61, 0.33, 0.51],
            updatedAt="2026-07-01T09:00:00+08:00",
        ),
    ),
    Senior(
        id="s-002",
        name="Mr Raman Pillai",
        age=82,
        preferredLanguage="Tamil",
        livingAlone=True,
        addressZone="Jurong West",
        caregiverContact="Nephew: Arjun",
        checkInFrequencyDays=3,
        baselineSpeechProfile=SpeechProfile(
            speechRate=116,
            avgPauseMs=740,
            responseLatencyMs=1200,
            pitchVariability=0.58,
            phraseAccuracy=0.94,
            embedding=[0.31, 0.22, 0.54, 0.42, 0.27, 0.64],
            updatedAt="2026-06-30T10:00:00+08:00",
        ),
    ),
    Senior(
        id="s-003",
        name="Encik Ahmad Rahman",
        age=75,
        preferredLanguage="Malay",
        livingAlone=True,
        addressZone="Bedok",
        caregiverContact="Son: Hafiz",
        checkInFrequencyDays=2,
        baselineSpeechProfile=SpeechProfile(
            speechRate=128,
            avgPauseMs=580,
            responseLatencyMs=860,
            pitchVariability=0.69,
            phraseAccuracy=0.98,
            embedding=[0.18, 0.36, 0.29, 0.57, 0.48, 0.39],
            updatedAt="2026-07-02T09:30:00+08:00",
        ),
    ),
]


CHECKINS = [
    CheckInSession(
        id="c-101",
        seniorId="s-003",
        scheduledAt="2026-07-04T09:00:00+08:00",
        completedAt="2026-07-04T09:04:00+08:00",
        status="Checked in",
        language="Malay",
        riskLevel="Green",
        summary="Stable check-in. No falls, symptoms, or adherence concerns.",
        originalTranscript="Saya okay. Sudah makan dan makan ubat.",
        englishTranscript="I am okay. I ate and took my medication.",
        riskAssessment=RiskAssessment(
            speechDeviationScore=8,
            parkinsonsWatchScore=4,
            postFallConcernScore=0,
            missedCheckInScore=0,
            riskLevel="Green",
            reasons=["Speech remains close to baseline", "No fall or head-impact symptoms reported"],
        ),
    ),
    CheckInSession(
        id="c-102",
        seniorId="s-001",
        scheduledAt="2026-07-04T09:00:00+08:00",
        status="Urgent",
        language="Mandarin",
        riskLevel="Red",
        summary="Reported fall with head impact, worsening headache, confusion, and left-hand weakness.",
        originalTranscript="我昨晚跌倒，撞到头。头很痛，有点乱，左手没有力。",
        englishTranscript="I fell last night and hit my head. My head hurts, I feel confused, and my left hand is weak.",
        riskAssessment=RiskAssessment(
            speechDeviationScore=82,
            parkinsonsWatchScore=28,
            postFallConcernScore=96,
            missedCheckInScore=0,
            riskLevel="Red",
            reasons=["Fall with head impact", "Confusion and weakness reported", "Large speech deviation from baseline"],
        ),
    ),
]


VOLUNTEER_TASKS: list[VolunteerTask] = []
