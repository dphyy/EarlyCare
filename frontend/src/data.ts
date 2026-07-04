import type { CheckInSession, Scenario, Senior, VolunteerTask } from "./types";

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
      "Please repeat: Today I am safe at home and I can ask for help.",
      "Today I am safe at home and I can ask for help."
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
