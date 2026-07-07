import type { CallRecord, CheckInSession, Scenario, Senior, VolunteerTask } from "./types";

export const seniors: Senior[] = [
  {
    id: "s-001",
    name: "Mdm Tan Bee Hoon",
    age: 78,
    preferredLanguage: "Mandarin",
    livingAlone: true,
    addressZone: "Toa Payoh",
    caregiverContact: "Daughter: Mei Ling",
    checkInFrequencyDays: 2,
    baselineSpeechProfile: {
      speechRate: 122,
      avgPauseMs: 620,
      responseLatencyMs: 980,
      pitchVariability: 0.64,
      phraseAccuracy: 0.96,
      embedding: [0.12, 0.28, 0.45, 0.61, 0.33, 0.51],
      updatedAt: "2026-07-01T09:00:00+08:00"
    }
  },
  {
    id: "s-002",
    name: "Mr Raman Pillai",
    age: 82,
    preferredLanguage: "Tamil",
    livingAlone: true,
    addressZone: "Jurong West",
    caregiverContact: "Nephew: Arjun",
    checkInFrequencyDays: 3,
    baselineSpeechProfile: {
      speechRate: 116,
      avgPauseMs: 740,
      responseLatencyMs: 1200,
      pitchVariability: 0.58,
      phraseAccuracy: 0.94,
      embedding: [0.31, 0.22, 0.54, 0.42, 0.27, 0.64],
      updatedAt: "2026-06-30T10:00:00+08:00"
    }
  },
  {
    id: "s-003",
    name: "Encik Ahmad Rahman",
    age: 75,
    preferredLanguage: "Malay",
    livingAlone: true,
    addressZone: "Bedok",
    caregiverContact: "Son: Hafiz",
    checkInFrequencyDays: 2,
    baselineSpeechProfile: {
      speechRate: 128,
      avgPauseMs: 580,
      responseLatencyMs: 860,
      pitchVariability: 0.69,
      phraseAccuracy: 0.98,
      embedding: [0.18, 0.36, 0.29, 0.57, 0.48, 0.39],
      updatedAt: "2026-07-02T09:30:00+08:00"
    }
  }
];

export const scenarios: Scenario[] = [
  {
    id: "stable",
    name: "Stable check-in",
    label: "Routine call completed",
    seniorId: "s-003",
    script: [
      "Hello Encik Ahmad, this is EarlyCare. Are you feeling okay today?",
      "Yes, I am okay. I ate breakfast and took my medicine.",
      "Any falls, dizziness, headache, or blurred vision since our last call?",
      "No falls. No headache. I feel normal.",
      "Can you ask someone nearby for help if you need it?",
      "Yes, I can call my son if I need help."
    ],
    speechMetrics: {
      speechRate: 126,
      avgPauseMs: 610,
      responseLatencyMs: 920,
      pitchVariability: 0.66,
      phraseAccuracy: 0.97,
      embedding: [0.19, 0.35, 0.3, 0.55, 0.47, 0.4]
    },
    symptoms: {
      fall: false,
      headImpact: false,
      headache: false,
      dizziness: false,
      vomiting: false,
      confusion: false,
      slurredSpeech: false,
      weakness: false,
      missedCheckIn: false
    }
  },
  {
    id: "missed",
    name: "Missed check-in",
    label: "No answer after retry",
    seniorId: "s-001",
    script: [
      "EarlyCare attempted the scheduled call at 9:00 AM.",
      "No answer.",
      "EarlyCare retried at 9:20 AM.",
      "No answer after retry. Volunteer follow-up task created."
    ],
    speechMetrics: {
      speechRate: 0,
      avgPauseMs: 0,
      responseLatencyMs: 0,
      pitchVariability: 0,
      phraseAccuracy: 0,
      embedding: [0, 0, 0, 0, 0, 0]
    },
    symptoms: {
      fall: false,
      headImpact: false,
      headache: false,
      dizziness: false,
      vomiting: false,
      confusion: false,
      slurredSpeech: false,
      weakness: false,
      missedCheckIn: true
    }
  },
  {
    id: "parkinsons-watch",
    name: "Parkinson's watch",
    label: "Gradual speech drift",
    seniorId: "s-002",
    script: [
      "Hello Mr Raman, how are you today?",
      "I am okay... a bit slow today, but no fall.",
      "Can you say pa-ta-ka three times?",
      "Pa... ta... ka... pa... ta... ka...",
      "Please tell me what you had for breakfast.",
      "I had... tea. Toast. I think... yes, toast."
    ],
    speechMetrics: {
      speechRate: 84,
      avgPauseMs: 1450,
      responseLatencyMs: 2300,
      pitchVariability: 0.31,
      phraseAccuracy: 0.78,
      embedding: [0.46, 0.12, 0.61, 0.31, 0.19, 0.71]
    },
    symptoms: {
      fall: false,
      headImpact: false,
      headache: false,
      dizziness: false,
      vomiting: false,
      confusion: false,
      slurredSpeech: false,
      weakness: false,
      missedCheckIn: false
    }
  },
  {
    id: "post-fall-red",
    name: "Post-fall red",
    label: "Fall with danger signs",
    seniorId: "s-001",
    script: [
      "Hello Mdm Tan, this is EarlyCare. Are you okay today?",
      "I fell last night... hit my head near the kitchen.",
      "Do you have headache, vomiting, confusion, weakness, or trouble speaking?",
      "My head pain is worse. I feel confused and my left hand feels weak.",
      "I am going to alert your volunteer coordinator and caregiver now."
    ],
    speechMetrics: {
      speechRate: 68,
      avgPauseMs: 1900,
      responseLatencyMs: 3100,
      pitchVariability: 0.27,
      phraseAccuracy: 0.62,
      embedding: [0.59, 0.09, 0.68, 0.21, 0.12, 0.78]
    },
    symptoms: {
      fall: true,
      headImpact: true,
      headache: true,
      dizziness: false,
      vomiting: false,
      confusion: true,
      slurredSpeech: true,
      weakness: true,
      missedCheckIn: false
    }
  }
];

export const sessions: CheckInSession[] = [
  {
    id: "c-101",
    seniorId: "s-003",
    scheduledAt: "2026-07-04T09:00:00+08:00",
    completedAt: "2026-07-04T09:04:00+08:00",
    status: "Checked in",
    language: "Malay",
    riskLevel: "Green",
    summary: "Stable check-in. No falls, symptoms, or adherence concerns.",
    originalTranscript: "Saya okay. Sudah makan dan makan ubat.",
    englishTranscript: "I am okay. I ate and took my medication.",
    riskAssessment: {
      speechDeviationScore: 8,
      parkinsonsWatchScore: 4,
      postFallConcernScore: 0,
      missedCheckInScore: 0,
      riskLevel: "Green",
      reasons: ["Speech remains close to baseline", "No fall or head-impact symptoms reported"]
    }
  },
  {
    id: "c-102",
    seniorId: "s-001",
    scheduledAt: "2026-07-04T09:00:00+08:00",
    status: "Urgent",
    language: "Mandarin",
    riskLevel: "Red",
    summary: "Reported fall with head impact, worsening headache, confusion, and left-hand weakness.",
    originalTranscript: "我昨晚跌倒，撞到头。头很痛，有点乱，左手没有力。",
    englishTranscript: "I fell last night and hit my head. My head hurts, I feel confused, and my left hand is weak.",
    riskAssessment: {
      speechDeviationScore: 82,
      parkinsonsWatchScore: 28,
      postFallConcernScore: 96,
      missedCheckInScore: 0,
      riskLevel: "Red",
      reasons: ["Fall with head impact", "Confusion and weakness reported", "Large speech deviation from baseline"]
    }
  }
];

export const volunteerTasks: VolunteerTask[] = [
  {
    id: "t-001",
    seniorId: "s-001",
    priority: "Urgent",
    reason: "Post-fall danger signs",
    recommendedAction: "Call caregiver and coordinate urgent medical assessment.",
    assignedTo: "SGO-style volunteer team A",
    status: "Open",
    createdAt: "2026-07-04T09:05:00+08:00"
  },
  {
    id: "t-002",
    seniorId: "s-002",
    priority: "Today",
    reason: "Gradual speech drift over recent check-ins",
    recommendedAction: "Schedule in-person wellbeing visit and ask caregiver about changes.",
    assignedTo: "Community volunteer team B",
    status: "In progress",
    createdAt: "2026-07-04T10:15:00+08:00"
  }
];

export const demoCalls: CallRecord[] = [
  {
    id: "demo-call-fall-001",
    seniorId: "s-001",
    seniorName: "Mdm Tan Bee Hoon",
    startedAt: "2026-07-07T09:00:00+08:00",
    completedAt: "2026-07-07T09:07:00+08:00",
    status: "Complete",
    riskLevel: "Amber",
    originalTranscript:
      "Agent: Mdm Tan, how are you feeling today?\nPatient: I feel dizzy this morning, a bit weak.\nAgent: Have you eaten, had water, and taken your medicine?\nPatient: I ate bread only. I drank some water and took my blood pressure medicine.\nAgent: Did you fall or nearly fall today?\nPatient: I almost fell in the kitchen this morning.\nAgent: Did you hit your head?\nPatient: No, I did not hit my head.\nAgent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?\nPatient: No vomiting and no blurred vision, but I feel weak and a bit sleepy.\nAgent: Do you need help now?\nPatient: Please ask my daughter to check on me.\nAgent: Thank you for telling me. This may be a possible concern, so EarlyCare will recommend urgent volunteer or caregiver follow-up.\nPatient: Okay, thank you.",
    englishTranscript:
      "Agent: Mdm Tan, how are you feeling today?\nPatient: I feel dizzy this morning, a bit weak.\nAgent: Have you eaten, had water, and taken your medicine?\nPatient: I ate bread only. I drank some water and took my blood pressure medicine.\nAgent: Did you fall or nearly fall today?\nPatient: I almost fell in the kitchen this morning.\nAgent: Did you hit your head?\nPatient: No, I did not hit my head.\nAgent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?\nPatient: No vomiting and no blurred vision, but I feel weak and a bit sleepy.\nAgent: Do you need help now?\nPatient: Please ask my daughter to check on me.\nAgent: Thank you for telling me. This may be a possible concern, so EarlyCare will recommend urgent volunteer or caregiver follow-up.\nPatient: Okay, thank you.",
    transcriptMessages: [
      { role: "Agent", text: "Mdm Tan, how are you feeling today?", timestamp: "2026-07-07T09:00:10+08:00" },
      { role: "Senior", text: "I feel dizzy this morning, a bit weak.", timestamp: "2026-07-07T09:00:22+08:00" },
      { role: "Agent", text: "Have you eaten, had water, and taken your medicine?", timestamp: "2026-07-07T09:00:38+08:00" },
      { role: "Senior", text: "I ate bread only. I drank some water and took my blood pressure medicine.", timestamp: "2026-07-07T09:00:52+08:00" },
      { role: "Agent", text: "Did you fall or nearly fall today?", timestamp: "2026-07-07T09:01:10+08:00" },
      { role: "Senior", text: "I almost fell in the kitchen this morning.", timestamp: "2026-07-07T09:01:21+08:00" },
      { role: "Agent", text: "Did you hit your head?", timestamp: "2026-07-07T09:01:33+08:00" },
      { role: "Senior", text: "No, I did not hit my head.", timestamp: "2026-07-07T09:01:43+08:00" },
      { role: "Agent", text: "Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", timestamp: "2026-07-07T09:01:55+08:00" },
      { role: "Senior", text: "No vomiting and no blurred vision, but I feel weak and a bit sleepy.", timestamp: "2026-07-07T09:02:13+08:00" },
      { role: "Agent", text: "Do you need help now?", timestamp: "2026-07-07T09:02:30+08:00" },
      { role: "Senior", text: "Please ask my daughter to check on me.", timestamp: "2026-07-07T09:02:39+08:00" },
      { role: "Agent", text: "Thank you for telling me. This may be a possible concern, so EarlyCare will recommend urgent volunteer or caregiver follow-up.", timestamp: "2026-07-07T09:02:55+08:00" },
      { role: "Senior", text: "Okay, thank you.", timestamp: "2026-07-07T09:03:13+08:00" }
    ],
    translationProvider: "demo-record",
    translationFallbackUsed: false,
    audioAvailable: false,
    patientAudioAvailable: false,
    patientSpeechAudioAvailable: false,
    agentAudioCaptured: false,
    currentSpeechProfile: { speechRate: 86, avgPauseMs: 1420, responseLatencyMs: 2380, pitchVariability: 0.38, phraseAccuracy: 0.9, updatedAt: "2026-07-07T09:07:00+08:00" },
    transcriptSegments: [
      { text: "Agent: Mdm Tan, how are you feeling today?", englishText: "Agent: Mdm Tan, how are you feeling today?", role: "Agent", speaker: "Agent", startTimeSeconds: 0, endTimeSeconds: 4 },
      { text: "Patient: I feel dizzy this morning, a bit weak.", englishText: "Patient: I feel dizzy this morning, a bit weak.", role: "Patient", speaker: "Patient", startTimeSeconds: 5, endTimeSeconds: 10 },
      { text: "Agent: Have you eaten, had water, and taken your medicine?", englishText: "Agent: Have you eaten, had water, and taken your medicine?", role: "Agent", speaker: "Agent", startTimeSeconds: 11, endTimeSeconds: 16 },
      { text: "Patient: I ate bread only. I drank some water and took my blood pressure medicine.", englishText: "Patient: I ate bread only. I drank some water and took my blood pressure medicine.", role: "Patient", speaker: "Patient", startTimeSeconds: 17, endTimeSeconds: 26 },
      { text: "Agent: Did you fall or nearly fall today?", englishText: "Agent: Did you fall or nearly fall today?", role: "Agent", speaker: "Agent", startTimeSeconds: 27, endTimeSeconds: 30 },
      { text: "Patient: I almost fell in the kitchen this morning.", englishText: "Patient: I almost fell in the kitchen this morning.", role: "Patient", speaker: "Patient", startTimeSeconds: 31, endTimeSeconds: 36 },
      { text: "Agent: Did you hit your head?", englishText: "Agent: Did you hit your head?", role: "Agent", speaker: "Agent", startTimeSeconds: 37, endTimeSeconds: 39 },
      { text: "Patient: No, I did not hit my head.", englishText: "Patient: No, I did not hit my head.", role: "Patient", speaker: "Patient", startTimeSeconds: 40, endTimeSeconds: 43 },
      { text: "Agent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", englishText: "Agent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", role: "Agent", speaker: "Agent", startTimeSeconds: 44, endTimeSeconds: 52 },
      { text: "Patient: No vomiting and no blurred vision, but I feel weak and a bit sleepy.", englishText: "Patient: No vomiting and no blurred vision, but I feel weak and a bit sleepy.", role: "Patient", speaker: "Patient", startTimeSeconds: 53, endTimeSeconds: 62 },
      { text: "Agent: Do you need help now?", englishText: "Agent: Do you need help now?", role: "Agent", speaker: "Agent", startTimeSeconds: 63, endTimeSeconds: 65 },
      { text: "Patient: Please ask my daughter to check on me.", englishText: "Patient: Please ask my daughter to check on me.", role: "Patient", speaker: "Patient", startTimeSeconds: 66, endTimeSeconds: 70 },
      { text: "Agent: Thank you for telling me. This may be a possible concern, so EarlyCare will recommend urgent volunteer or caregiver follow-up.", englishText: "Agent: Thank you for telling me. This may be a possible concern, so EarlyCare will recommend urgent volunteer or caregiver follow-up.", role: "Agent", speaker: "Agent", startTimeSeconds: 71, endTimeSeconds: 80 },
      { text: "Patient: Okay, thank you.", englishText: "Patient: Okay, thank you.", role: "Patient", speaker: "Patient", startTimeSeconds: 81, endTimeSeconds: 83 }
    ],
    riskSignals: [
      {
        id: "demo-risk-fall",
        label: "Near fall with dizziness",
        severity: "Amber",
        quotedText: "I almost fell in the kitchen this morning.",
        highlightText: "I almost fell in the kitchen this morning",
        reason: "Patient reported a near fall and dizziness.",
        sentenceIndex: 2,
        startTimeSeconds: 31,
        endTimeSeconds: 36
      },
      {
        id: "demo-risk-help",
        label: "Help requested",
        severity: "Watch",
        quotedText: "Please ask my daughter to check on me.",
        highlightText: "Please ask my daughter to check on me",
        reason: "Patient requested caregiver follow-up.",
        sentenceIndex: 5,
        startTimeSeconds: 66,
        endTimeSeconds: 70
      }
    ],
    safeguardReviewAvailable: true,
    safeguardLevel: "None",
    safeguardCategory: null,
    safeguardEvidence: [],
    safeguardRecommendedAction: null,
    safeguardResources: [],
    consultationMemory: [
      {
        id: "demo-memory-fall",
        seniorId: "s-001",
        callId: "demo-call-fall-001",
        recordedAt: "2026-07-07T09:07:00+08:00",
        category: "fall",
        summary: "Patient reported a near fall in the kitchen with dizziness.",
        exactQuote: "I almost fell in the kitchen this morning.",
        startTimeSeconds: 31,
        endTimeSeconds: 36,
        severity: "watch",
        status: "new"
      },
      {
        id: "demo-memory-help",
        seniorId: "s-001",
        callId: "demo-call-fall-001",
        recordedAt: "2026-07-07T09:07:00+08:00",
        category: "help_needed",
        summary: "Patient asked for daughter to check on her.",
        exactQuote: "Please ask my daughter to check on me.",
        startTimeSeconds: 66,
        endTimeSeconds: 70,
        severity: "watch",
        status: "new"
      }
    ],
    parkinsonsSpeechReview: {
      modelVersion: "demo-parkinsons-marker",
      probability: 0.31,
      warnings: [],
      featuresSummary: { "MDVP:Jitter(%)": 0.0121, HNR: 18.4, "MDVP:Fhi(Hz)": 282.2, patientSpeechDurationSeconds: 30.2, rawPatientAudioDurationSeconds: 83.0 },
      qualityOk: true,
      warning: "Research-only Parkinson voice-feature signal. This is not a Parkinson's diagnosis or medical device output.",
      explanations: [
        { label: "Jitter stability", value: "0.0121", status: "watch", explanation: "Cycle-to-cycle pitch stability in voiced speech." },
        { label: "Harmonic-noise clarity", value: "18.4 dB", status: "normal", explanation: "Voice clarity versus noise in voiced speech." },
        { label: "Pitch range", value: "282.2 Hz", status: "normal", explanation: "Fundamental-frequency range in patient speech." }
      ]
    },
    concussionSpeechReview: {
      applicability: "applicable",
      modelVersion: "demo-concussion-speech",
      predictedLabel: "normal",
      probabilities: { normal: 0.7, dysarthria_like: 0.22, dysphonia_like: 0.08 },
      qualityOk: true,
      durationSec: 8.8,
      rms: 0.032,
      clippingFraction: 0,
      riskContribution: "Green",
      warning: "Research-only speech abnormality signal. This is not a concussion diagnosis, dysarthria diagnosis, dysphonia diagnosis, or medical device output.",
      explanations: [
        { label: "Predicted speech pattern", value: "normal", status: "normal", explanation: "The fall-related call made concussion speech review applicable, and the model did not flag abnormal speech." },
        { label: "Abnormal-class probability", value: "22%", status: "normal", explanation: "The strongest abnormal class was below the normal probability." },
        { label: "Audio quality", value: "8.8s, RMS 0.032, clipping 0%", status: "normal", explanation: "The demo clip passed duration, loudness, and clipping checks." }
      ]
    },
    riskAssessment: {
      speechDeviationScore: 34,
      parkinsonsWatchScore: 18,
      postFallConcernScore: 70,
      missedCheckInScore: 0,
      riskLevel: "Amber",
      reasons: ["Patient reported a near fall with dizziness.", "Human follow-up is recommended because she lives alone."]
    },
    recommendedAction: "Call caregiver and arrange same-day human follow-up.",
    consentCaptured: true,
    consentVersion: "earlycare-demo-v1",
    recordingNoticeShownAt: "2026-07-07T09:00:00+08:00",
    retentionPolicy: "local-demo-delete-after-hackathon",
    operatorId: "demo-operator",
    demoRecord: true
  },
  {
    id: "demo-call-parkinsons-001",
    seniorId: "s-002",
    seniorName: "Mr Raman Pillai",
    startedAt: "2026-07-07T09:30:00+08:00",
    completedAt: "2026-07-07T09:39:00+08:00",
    status: "Complete",
    riskLevel: "Watch",
    originalTranscript:
      "Agent: Mr Raman, how are you feeling today?\nPatient: I am okay lah, but my hands feel stiff.\nAgent: Have you eaten, had water, and taken your medicine?\nPatient: I ate idli, drank water, but I took my tablets late.\nAgent: Did you fall or nearly fall today?\nPatient: No fall today. I did not slip or trip.\nAgent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?\nPatient: No headache or vomiting. My speech feels slower, and I am walking slower today.\nAgent: Do you need help now?\nPatient: Not urgent. I just feel very lonely. Some days I feel like giving up, but I am safe now.\nAgent: Thank you for telling me. Are you safe right now?\nPatient: Yes, I am safe. Please ask Arjun to call me later.\nAgent: I will recommend caregiver or volunteer follow-up and share crisis support if you need someone now.",
    englishTranscript:
      "Agent: Mr Raman, how are you feeling today?\nPatient: I am okay lah, but my hands feel stiff.\nAgent: Have you eaten, had water, and taken your medicine?\nPatient: I ate idli, drank water, but I took my tablets late.\nAgent: Did you fall or nearly fall today?\nPatient: No fall today. I did not slip or trip.\nAgent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?\nPatient: No headache or vomiting. My speech feels slower, and I am walking slower today.\nAgent: Do you need help now?\nPatient: Not urgent. I just feel very lonely. Some days I feel like giving up, but I am safe now.\nAgent: Thank you for telling me. Are you safe right now?\nPatient: Yes, I am safe. Please ask Arjun to call me later.\nAgent: I will recommend caregiver or volunteer follow-up and share crisis support if you need someone now.",
    transcriptMessages: [
      { role: "Agent", text: "Mr Raman, how are you feeling today?", timestamp: "2026-07-07T09:30:10+08:00" },
      { role: "Senior", text: "I am okay lah, but my hands feel stiff.", timestamp: "2026-07-07T09:30:23+08:00" },
      { role: "Agent", text: "Have you eaten, had water, and taken your medicine?", timestamp: "2026-07-07T09:30:40+08:00" },
      { role: "Senior", text: "I ate idli, drank water, but I took my tablets late.", timestamp: "2026-07-07T09:30:54+08:00" },
      { role: "Agent", text: "Did you fall or nearly fall today?", timestamp: "2026-07-07T09:31:11+08:00" },
      { role: "Senior", text: "No fall today. I did not slip or trip.", timestamp: "2026-07-07T09:31:23+08:00" },
      { role: "Agent", text: "Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", timestamp: "2026-07-07T09:31:39+08:00" },
      { role: "Senior", text: "No headache or vomiting. My speech feels slower, and I am walking slower today.", timestamp: "2026-07-07T09:31:58+08:00" },
      { role: "Agent", text: "Do you need help now?", timestamp: "2026-07-07T09:32:18+08:00" },
      { role: "Senior", text: "Not urgent. I just feel very lonely. Some days I feel like giving up, but I am safe now.", timestamp: "2026-07-07T09:32:31+08:00" },
      { role: "Agent", text: "Thank you for telling me. Are you safe right now?", timestamp: "2026-07-07T09:32:55+08:00" },
      { role: "Senior", text: "Yes, I am safe. Please ask Arjun to call me later.", timestamp: "2026-07-07T09:33:08+08:00" },
      { role: "Agent", text: "I will recommend caregiver or volunteer follow-up and share crisis support if you need someone now.", timestamp: "2026-07-07T09:33:25+08:00" }
    ],
    translationProvider: "demo-record",
    translationFallbackUsed: false,
    audioAvailable: false,
    patientAudioAvailable: false,
    patientSpeechAudioAvailable: true,
    agentAudioCaptured: false,
    currentSpeechProfile: { speechRate: 76, avgPauseMs: 1820, responseLatencyMs: 2920, pitchVariability: 0.24, phraseAccuracy: 0.82, updatedAt: "2026-07-07T09:39:00+08:00" },
    transcriptSegments: [
      { text: "Agent: Mr Raman, how are you feeling today?", englishText: "Agent: Mr Raman, how are you feeling today?", role: "Agent", speaker: "Agent", startTimeSeconds: 0, endTimeSeconds: 4 },
      { text: "Patient: I am okay lah, but my hands feel stiff.", englishText: "Patient: I am okay lah, but my hands feel stiff.", role: "Patient", speaker: "Patient", startTimeSeconds: 5, endTimeSeconds: 10 },
      { text: "Agent: Have you eaten, had water, and taken your medicine?", englishText: "Agent: Have you eaten, had water, and taken your medicine?", role: "Agent", speaker: "Agent", startTimeSeconds: 11, endTimeSeconds: 16 },
      { text: "Patient: I ate idli, drank water, but I took my tablets late.", englishText: "Patient: I ate idli, drank water, but I took my tablets late.", role: "Patient", speaker: "Patient", startTimeSeconds: 17, endTimeSeconds: 24 },
      { text: "Agent: Did you fall or nearly fall today?", englishText: "Agent: Did you fall or nearly fall today?", role: "Agent", speaker: "Agent", startTimeSeconds: 25, endTimeSeconds: 28 },
      { text: "Patient: No fall today. I did not slip or trip.", englishText: "Patient: No fall today. I did not slip or trip.", role: "Patient", speaker: "Patient", startTimeSeconds: 29, endTimeSeconds: 34 },
      { text: "Agent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", englishText: "Agent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", role: "Agent", speaker: "Agent", startTimeSeconds: 35, endTimeSeconds: 43 },
      { text: "Patient: No headache or vomiting. My speech feels slower, and I am walking slower today.", englishText: "Patient: No headache or vomiting. My speech feels slower, and I am walking slower today.", role: "Patient", speaker: "Patient", startTimeSeconds: 44, endTimeSeconds: 53 },
      { text: "Agent: Do you need help now?", englishText: "Agent: Do you need help now?", role: "Agent", speaker: "Agent", startTimeSeconds: 54, endTimeSeconds: 56 },
      { text: "Patient: Not urgent. I just feel very lonely. Some days I feel like giving up, but I am safe now.", englishText: "Patient: Not urgent. I just feel very lonely. Some days I feel like giving up, but I am safe now.", role: "Patient", speaker: "Patient", startTimeSeconds: 57, endTimeSeconds: 70 },
      { text: "Agent: Thank you for telling me. Are you safe right now?", englishText: "Agent: Thank you for telling me. Are you safe right now?", role: "Agent", speaker: "Agent", startTimeSeconds: 71, endTimeSeconds: 75 },
      { text: "Patient: Yes, I am safe. Please ask Arjun to call me later.", englishText: "Patient: Yes, I am safe. Please ask Arjun to call me later.", role: "Patient", speaker: "Patient", startTimeSeconds: 76, endTimeSeconds: 82 },
      { text: "Agent: I will recommend caregiver or volunteer follow-up and share crisis support if you need someone now.", englishText: "Agent: I will recommend caregiver or volunteer follow-up and share crisis support if you need someone now.", role: "Agent", speaker: "Agent", startTimeSeconds: 83, endTimeSeconds: 90 }
    ],
    riskSignals: [
      {
        id: "demo-risk-mobility",
        label: "Mobility slowing",
        severity: "Watch",
        quotedText: "My hands feel stiff and I am walking slower today.",
        highlightText: "walking slower today",
        reason: "Patient reported stiffness and slower walking.",
        sentenceIndex: 3,
        startTimeSeconds: 44,
        endTimeSeconds: 53
      },
      {
        id: "demo-risk-medication",
        label: "Medication timing",
        severity: "Watch",
        quotedText: "I took my tablets late.",
        highlightText: "I took my tablets late",
        reason: "Patient reported delayed tablets.",
        sentenceIndex: 1,
        startTimeSeconds: 17,
        endTimeSeconds: 24
      }
    ],
    dominantPatientEmotion: "lonely",
    emotionReviewAvailable: true,
    emotionProvider: "demo-tone",
    emotionConcernLevel: "Review",
    emotionSegments: [
      {
        id: "demo-tone-lonely",
        label: "lonely",
        confidence: 0.84,
        startTimeSeconds: 57,
        endTimeSeconds: 70,
        transcriptSegmentIndex: 9,
        evidenceText: "Not urgent. I just feel very lonely. Some days I feel like giving up, but I am safe now."
      }
    ],
    safeguardReviewAvailable: true,
    safeguardLevel: "Support",
    safeguardCategory: "emotional_distress",
    safeguardEvidence: ["Some days I feel like giving up, but I am safe now."],
    safeguardRecommendedAction: "Patient-stated loneliness and giving-up language should receive same-day human support, while respecting that he stated he is safe now.",
    safeguardResources: [
      {
        name: "Samaritans of Singapore hotline",
        phone: "1767",
        description: "24-hour emotional support and crisis hotline in Singapore."
      },
      {
        name: "Samaritans of Singapore CareText",
        text: "WhatsApp 9151 1767",
        url: "https://www.sos.org.sg/",
        description: "24-hour WhatsApp text support for emotional support or crisis-related concerns."
      }
    ],
    consultationMemory: [
      {
        id: "demo-memory-medication-raman",
        seniorId: "s-002",
        callId: "demo-call-parkinsons-001",
        recordedAt: "2026-07-07T09:39:00+08:00",
        category: "medication",
        summary: "Patient took tablets late.",
        exactQuote: "I took my tablets late.",
        startTimeSeconds: 17,
        endTimeSeconds: 24,
        severity: "watch",
        status: "new"
      },
      {
        id: "demo-memory-mobility-raman",
        seniorId: "s-002",
        callId: "demo-call-parkinsons-001",
        recordedAt: "2026-07-07T09:39:00+08:00",
        category: "mobility",
        summary: "Patient reported stiff hands and slower walking.",
        exactQuote: "My hands feel stiff and I am walking slower today.",
        startTimeSeconds: 44,
        endTimeSeconds: 53,
        severity: "watch",
        status: "new"
      },
      {
        id: "demo-memory-mood-raman",
        seniorId: "s-002",
        callId: "demo-call-parkinsons-001",
        recordedAt: "2026-07-07T09:39:00+08:00",
        category: "mood",
        summary: "Patient reported loneliness and giving-up language while saying he is safe now.",
        exactQuote: "Some days I feel like giving up, but I am safe now.",
        startTimeSeconds: 57,
        endTimeSeconds: 70,
        severity: "watch",
        status: "new"
      }
    ],
    parkinsonsSpeechReview: {
      modelVersion: "demo-parkinsons-marker",
      probability: 0.68,
      warnings: ["Speech marker low confidence: MDVP:Jitter(%) outside the UCI/Kaggle training range."],
      featuresSummary: { "MDVP:Jitter(%)": 0.029, "MDVP:RAP": 0.018, HNR: 9.4, "MDVP:Fo(Hz)": 116.3, patientSpeechDurationSeconds: 32.6, rawPatientAudioDurationSeconds: 90.0 },
      qualityOk: true,
      warning: "Research-only Parkinson voice-feature signal. This is not a Parkinson's diagnosis or medical device output.",
      explanations: [
        { label: "Jitter stability", value: "0.0290", status: "watch", explanation: "Cycle-to-cycle pitch stability in voiced speech." },
        { label: "Harmonic-noise clarity", value: "9.4 dB", status: "watch", explanation: "Voice clarity versus noise in voiced speech." },
        { label: "Pitch range", value: "116.3 Hz", status: "normal", explanation: "Fundamental-frequency range in patient speech." }
      ]
    },
    concussionSpeechReview: {
      applicability: "not_applicable",
      modelVersion: null,
      predictedLabel: null,
      probabilities: {},
      qualityOk: false,
      riskContribution: "Green",
      failureReason: null,
      warning: "Research-only speech abnormality signal. This is not a concussion diagnosis, dysarthria diagnosis, dysphonia diagnosis, or medical device output.",
      explanations: [
        { label: "Applicability", value: "Not applicable", status: "normal", explanation: "No patient-stated fall or near-fall was found, so concussion speech review was not applied." }
      ]
    },
    riskAssessment: {
      speechDeviationScore: 52,
      parkinsonsWatchScore: 68,
      postFallConcernScore: 0,
      missedCheckInScore: 0,
      riskLevel: "Watch",
      reasons: ["Patient reported stiffness, slower walking, and delayed medication.", "Safeguard review flagged emotional distress support.", "Tone review suggested loneliness."]
    },
    recommendedAction: "Arrange a volunteer check-in today and ask caregiver about medication timing and mobility changes.",
    consentCaptured: true,
    consentVersion: "earlycare-demo-v1",
    recordingNoticeShownAt: "2026-07-07T09:30:00+08:00",
    retentionPolicy: "local-demo-delete-after-hackathon",
    operatorId: "demo-operator",
    demoRecord: true
  },
  {
    id: "demo-call-routine-001",
    seniorId: "s-003",
    seniorName: "Encik Ahmad Rahman",
    startedAt: "2026-07-07T10:00:00+08:00",
    completedAt: "2026-07-07T10:03:00+08:00",
    status: "Complete",
    riskLevel: "Green",
    originalTranscript:
      "Agent: Encik Ahmad, apa khabar hari ini?\nPatient: Saya okay, Alhamdulillah.\nAgent: Sudah makan, minum air, dan ambil ubat?\nPatient: Sudah makan roti, sudah minum air, dan sudah makan ubat pagi.\nAgent: Ada jatuh atau hampir jatuh hari ini?\nPatient: Tidak ada jatuh. Tidak hampir jatuh.\nAgent: Ada sakit kepala, muntah, pandangan kabur, terlalu mengantuk, lemah, kebas, atau susah bercakap?\nPatient: Tidak ada. Saya rasa biasa sahaja.\nAgent: Perlu bantuan sekarang?\nPatient: Tidak perlu. Anak saya Hafiz akan telefon malam nanti.\nAgent: Ada apa-apa lagi mahu kongsi?\nPatient: Tidak, terima kasih.",
    englishTranscript:
      "Agent: Encik Ahmad, how are you today?\nPatient: I am okay, thank God.\nAgent: Have you eaten, had water, and taken your medicine?\nPatient: I ate bread, drank water, and took my morning medicine.\nAgent: Did you fall or nearly fall today?\nPatient: No fall. No near fall.\nAgent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?\nPatient: None. I feel normal.\nAgent: Do you need help now?\nPatient: No need. My son Hafiz will call tonight.\nAgent: Is there anything more you would like to share?\nPatient: No, thank you.",
    transcriptMessages: [
      { role: "Agent", text: "Encik Ahmad, apa khabar hari ini?", timestamp: "2026-07-07T10:00:10+08:00" },
      { role: "Senior", text: "Saya okay, Alhamdulillah.", timestamp: "2026-07-07T10:00:19+08:00" },
      { role: "Agent", text: "Sudah makan, minum air, dan ambil ubat?", timestamp: "2026-07-07T10:00:32+08:00" },
      { role: "Senior", text: "Sudah makan roti, sudah minum air, dan sudah makan ubat pagi.", timestamp: "2026-07-07T10:00:44+08:00" },
      { role: "Agent", text: "Ada jatuh atau hampir jatuh hari ini?", timestamp: "2026-07-07T10:00:59+08:00" },
      { role: "Senior", text: "Tidak ada jatuh. Tidak hampir jatuh.", timestamp: "2026-07-07T10:01:10+08:00" },
      { role: "Agent", text: "Ada sakit kepala, muntah, pandangan kabur, terlalu mengantuk, lemah, kebas, atau susah bercakap?", timestamp: "2026-07-07T10:01:26+08:00" },
      { role: "Senior", text: "Tidak ada. Saya rasa biasa sahaja.", timestamp: "2026-07-07T10:01:43+08:00" },
      { role: "Agent", text: "Perlu bantuan sekarang?", timestamp: "2026-07-07T10:01:55+08:00" },
      { role: "Senior", text: "Tidak perlu. Anak saya Hafiz akan telefon malam nanti.", timestamp: "2026-07-07T10:02:04+08:00" },
      { role: "Agent", text: "Ada apa-apa lagi mahu kongsi?", timestamp: "2026-07-07T10:02:19+08:00" },
      { role: "Senior", text: "Tidak, terima kasih.", timestamp: "2026-07-07T10:02:27+08:00" }
    ],
    translationProvider: "demo-record",
    translationFallbackUsed: false,
    audioAvailable: false,
    patientAudioAvailable: false,
    patientSpeechAudioAvailable: true,
    agentAudioCaptured: false,
    currentSpeechProfile: { speechRate: 126, avgPauseMs: 610, responseLatencyMs: 920, pitchVariability: 0.66, phraseAccuracy: 0.97, updatedAt: "2026-07-07T10:03:00+08:00" },
    transcriptSegments: [
      { text: "Agent: Encik Ahmad, how are you today?", originalText: "Agent: Encik Ahmad, apa khabar hari ini?", englishText: "Agent: Encik Ahmad, how are you today?", role: "Agent", speaker: "Agent", startTimeSeconds: 0, endTimeSeconds: 3 },
      { text: "Patient: I am okay, thank God.", originalText: "Patient: Saya okay, Alhamdulillah.", englishText: "Patient: I am okay, thank God.", role: "Patient", speaker: "Patient", startTimeSeconds: 4, endTimeSeconds: 7 },
      { text: "Agent: Have you eaten, had water, and taken your medicine?", originalText: "Agent: Sudah makan, minum air, dan ambil ubat?", englishText: "Agent: Have you eaten, had water, and taken your medicine?", role: "Agent", speaker: "Agent", startTimeSeconds: 8, endTimeSeconds: 13 },
      { text: "Patient: I ate bread, drank water, and took my morning medicine.", originalText: "Patient: Sudah makan roti, sudah minum air, dan sudah makan ubat pagi.", englishText: "Patient: I ate bread, drank water, and took my morning medicine.", role: "Patient", speaker: "Patient", startTimeSeconds: 14, endTimeSeconds: 21 },
      { text: "Agent: Did you fall or nearly fall today?", originalText: "Agent: Ada jatuh atau hampir jatuh hari ini?", englishText: "Agent: Did you fall or nearly fall today?", role: "Agent", speaker: "Agent", startTimeSeconds: 22, endTimeSeconds: 25 },
      { text: "Patient: No fall. No near fall.", originalText: "Patient: Tidak ada jatuh. Tidak hampir jatuh.", englishText: "Patient: No fall. No near fall.", role: "Patient", speaker: "Patient", startTimeSeconds: 26, endTimeSeconds: 30 },
      { text: "Agent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", originalText: "Agent: Ada sakit kepala, muntah, pandangan kabur, terlalu mengantuk, lemah, kebas, atau susah bercakap?", englishText: "Agent: Any headache, vomiting, blurred vision, unusual sleepiness, weakness, numbness, or trouble speaking?", role: "Agent", speaker: "Agent", startTimeSeconds: 31, endTimeSeconds: 39 },
      { text: "Patient: None. I feel normal.", originalText: "Patient: Tidak ada. Saya rasa biasa sahaja.", englishText: "Patient: None. I feel normal.", role: "Patient", speaker: "Patient", startTimeSeconds: 40, endTimeSeconds: 44 },
      { text: "Agent: Do you need help now?", originalText: "Agent: Perlu bantuan sekarang?", englishText: "Agent: Do you need help now?", role: "Agent", speaker: "Agent", startTimeSeconds: 45, endTimeSeconds: 47 },
      { text: "Patient: No need. My son Hafiz will call tonight.", originalText: "Patient: Tidak perlu. Anak saya Hafiz akan telefon malam nanti.", englishText: "Patient: No need. My son Hafiz will call tonight.", role: "Patient", speaker: "Patient", startTimeSeconds: 48, endTimeSeconds: 54 },
      { text: "Agent: Is there anything more you would like to share?", originalText: "Agent: Ada apa-apa lagi mahu kongsi?", englishText: "Agent: Is there anything more you would like to share?", role: "Agent", speaker: "Agent", startTimeSeconds: 55, endTimeSeconds: 59 },
      { text: "Patient: No, thank you.", originalText: "Patient: Tidak, terima kasih.", englishText: "Patient: No, thank you.", role: "Patient", speaker: "Patient", startTimeSeconds: 60, endTimeSeconds: 62 }
    ],
    riskSignals: [],
    safeguardReviewAvailable: true,
    safeguardLevel: "None",
    safeguardCategory: null,
    safeguardEvidence: [],
    safeguardRecommendedAction: null,
    safeguardResources: [],
    consultationMemory: [
      {
        id: "demo-memory-meal-med",
        seniorId: "s-003",
        callId: "demo-call-routine-001",
        recordedAt: "2026-07-07T10:03:00+08:00",
        category: "medication",
        summary: "Patient reported taking medicine after breakfast.",
        exactQuote: "I ate bread, drank water, and took my morning medicine.",
        startTimeSeconds: 14,
        endTimeSeconds: 21,
        severity: "info",
        status: "new"
      },
      {
        id: "demo-memory-support-ahmad",
        seniorId: "s-003",
        callId: "demo-call-routine-001",
        recordedAt: "2026-07-07T10:03:00+08:00",
        category: "help_needed",
        summary: "Patient expects son to call tonight.",
        exactQuote: "My son will call tonight.",
        startTimeSeconds: 48,
        endTimeSeconds: 54,
        severity: "info",
        status: "new"
      }
    ],
    parkinsonsSpeechReview: {
      modelVersion: "demo-parkinsons-marker",
      probability: 0.12,
      warnings: [],
      featuresSummary: { "MDVP:Jitter(%)": 0.0042, HNR: 24.5, "MDVP:Fo(Hz)": 128.1, patientSpeechDurationSeconds: 24.4, rawPatientAudioDurationSeconds: 62.0 },
      qualityOk: true,
      warning: "Research-only Parkinson voice-feature signal. This is not a Parkinson's diagnosis or medical device output.",
      explanations: [
        { label: "Harmonic-noise clarity", value: "24.5 dB", status: "normal", explanation: "Voice clarity versus noise in voiced speech." },
        { label: "Jitter stability", value: "0.0042", status: "normal", explanation: "Cycle-to-cycle pitch stability in voiced speech." },
        { label: "Pitch range", value: "128.1 Hz", status: "normal", explanation: "Fundamental-frequency range in patient speech." }
      ]
    },
    concussionSpeechReview: {
      applicability: "not_applicable",
      modelVersion: null,
      predictedLabel: null,
      probabilities: {},
      qualityOk: false,
      riskContribution: "Green",
      failureReason: null,
      warning: "Research-only speech abnormality signal. This is not a concussion diagnosis, dysarthria diagnosis, dysphonia diagnosis, or medical device output.",
      explanations: [
        { label: "Applicability", value: "Not applicable", status: "normal", explanation: "No patient-stated fall or near-fall was found, so concussion speech review was not applied." }
      ]
    },
    riskAssessment: {
      speechDeviationScore: 8,
      parkinsonsWatchScore: 4,
      postFallConcernScore: 0,
      missedCheckInScore: 0,
      riskLevel: "Green",
      reasons: ["No urgent patient-stated cue found in this demo call.", "Meal, medication, and family support were confirmed."]
    },
    recommendedAction: "Continue routine check-in schedule.",
    consentCaptured: true,
    consentVersion: "earlycare-demo-v1",
    recordingNoticeShownAt: "2026-07-07T10:00:00+08:00",
    retentionPolicy: "local-demo-delete-after-hackathon",
    operatorId: "demo-operator",
    demoRecord: true
  }
];
