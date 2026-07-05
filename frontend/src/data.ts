import type { CheckInScheduleItem, CheckInSession, Scenario, Senior, SeniorRecord, VolunteerTask } from "./types";

export const seniors: Senior[] = [
  {
    id: "s-001",
    name: "Mdm Tan Bee Hoon",
    age: 78,
    preferredLanguage: "Mandarin",
    livingAlone: true,
    addressZone: "Toa Payoh",
    caregiverContact: "Daughter: Mei Ling",
    neighborContact: "Neighbour: Mr Lee, unit 08-112",
    knownConditions: ["high blood pressure", "arthritis", "fall risk"],
    promptFocus: ["fall/head impact", "medication", "blood pressure", "food and water"],
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
    neighborContact: "Neighbour: Mdm Koh, unit 05-241",
    knownConditions: ["early Parkinson's watch", "diabetes", "lives alone"],
    promptFocus: ["speech pace", "pa-ta-ka phrase", "diabetes medication", "loneliness"],
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
    neighborContact: "Neighbour: Encik Salleh, unit 03-018",
    knownConditions: ["CKD", "hypertension", "needs hydration reminders"],
    promptFocus: ["kidney symptoms", "water intake", "blood pressure", "mood"],
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
    description: "Routine 2-day wellbeing call. No symptoms, food and water are okay, speech remains close to baseline.",
    script: [
      { role: "Agent", text: "Hello Encik Ahmad, this is EarlyCare. Are you feeling okay today?" },
      { role: "Senior", text: "Yes, I am okay. I ate breakfast, drank water, and took my medicine." },
      { role: "Agent", text: "Any falls, head bumps, dizziness, headache, or blurred vision since our last call?" },
      { role: "Senior", text: "No falls. No headache. I feel normal." },
      { role: "Agent", text: "Please repeat: Today I am safe at home and I can ask for help." },
      { role: "Senior", text: "Today I am safe at home and I can ask for help." }
    ],
    speechMetrics: {
      speechRate: 126,
      avgPauseMs: 610,
      responseLatencyMs: 920,
      pitchVariability: 0.66,
      phraseAccuracy: 0.97,
      embedding: [0.19, 0.35, 0.3, 0.55, 0.47, 0.4],
      updatedAt: "2026-07-04T09:04:00+08:00"
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
    },
    originalTranscript: "Saya okay. Sudah makan, minum air dan makan ubat. Tiada jatuh atau sakit kepala.",
    englishTranscript: "I am okay. I ate, drank water, and took my medication. No fall or headache."
  },
  {
    id: "missed-checkin",
    name: "Missed check-in",
    label: "No answer after retry",
    seniorId: "s-001",
    description: "Scheduled call and retry are both unanswered, creating a same-day volunteer follow-up.",
    script: [
      { role: "System", text: "EarlyCare attempted the scheduled call at 9:00 AM." },
      { role: "System", text: "No answer." },
      { role: "System", text: "EarlyCare retried at 9:20 AM." },
      { role: "System", text: "No answer after retry. Volunteer follow-up task created." }
    ],
    speechMetrics: {
      speechRate: 0,
      avgPauseMs: 0,
      responseLatencyMs: 0,
      pitchVariability: 0,
      phraseAccuracy: 0,
      embedding: [0, 0, 0, 0, 0, 0],
      updatedAt: "2026-07-04T09:20:00+08:00"
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
    },
    originalTranscript: "No answer after scheduled call and retry.",
    englishTranscript: "No answer after scheduled call and retry."
  },
  {
    id: "parkinsons-watch",
    name: "Parkinson's watch",
    label: "Gradual speech drift",
    seniorId: "s-002",
    description: "No acute fall, but slower speech, longer pauses, lower pitch variation, and reduced phrase clarity compared with personal baseline.",
    script: [
      { role: "Agent", text: "Hello Mr Raman, how are you today?" },
      { role: "Senior", text: "I am okay... a bit slow today, but no fall." },
      { role: "Agent", text: "Can you say pa-ta-ka three times?" },
      { role: "Senior", text: "Pa... ta... ka... pa... ta... ka..." },
      { role: "Agent", text: "Please tell me what you had for breakfast." },
      { role: "Senior", text: "I had... tea. Toast. I think... yes, toast." }
    ],
    speechMetrics: {
      speechRate: 84,
      avgPauseMs: 1450,
      responseLatencyMs: 2300,
      pitchVariability: 0.31,
      phraseAccuracy: 0.78,
      embedding: [0.46, 0.12, 0.61, 0.31, 0.19, 0.71],
      updatedAt: "2026-07-04T10:15:00+08:00"
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
    },
    originalTranscript: "I am okay... a bit slow today. Pa... ta... ka... I had tea and toast.",
    englishTranscript: "I am okay, a bit slow today. Pa-ta-ka was slower with long pauses. I had tea and toast."
  },
  {
    id: "post-fall-amber",
    name: "Post-Fall Amber",
    label: "Fall with headache, coherent",
    seniorId: "s-001",
    description: "Fall and head bump with headache/dizziness, but no confusion, weakness, vomiting, slurred speech, or drowsiness.",
    script: [
      { role: "Agent", text: "Mdm Tan, did you fall, bump your head, or feel dizzy since our last call?" },
      { role: "Senior", text: "I slipped in the kitchen and bumped my head. I have a headache and feel a bit dizzy, but I know where I am." },
      { role: "Agent", text: "Any vomiting, slurred speech, confusion, weakness, numbness, or difficulty waking?" },
      { role: "Senior", text: "No vomiting. I can speak clearly and my arms feel normal." }
    ],
    speechMetrics: {
      speechRate: 104,
      avgPauseMs: 900,
      responseLatencyMs: 1320,
      pitchVariability: 0.54,
      phraseAccuracy: 0.9,
      embedding: [0.2, 0.29, 0.5, 0.56, 0.31, 0.5],
      updatedAt: "2026-07-04T10:30:00+08:00"
    },
    symptoms: {
      fall: true,
      headImpact: true,
      headache: true,
      dizziness: true,
      vomiting: false,
      confusion: false,
      slurredSpeech: false,
      weakness: false,
      missedCheckIn: false
    },
    originalTranscript: "I slipped in the kitchen and bumped my head. I have a headache and feel a bit dizzy. No vomiting or weakness.",
    englishTranscript: "I slipped in the kitchen and bumped my head. I have a headache and feel a bit dizzy. No vomiting, confusion, slurred speech, weakness, numbness, or drowsiness."
  },
  {
    id: "post-fall-red",
    name: "Post-fall red",
    label: "Fall with danger signs",
    seniorId: "s-001",
    description: "Fall/head impact plus red danger signs: worsening headache, confusion, slurred speech, and weakness.",
    script: [
      { role: "Agent", text: "Mdm Tan, are you okay today?" },
      { role: "Senior", text: "I fell last night and hit my head near the kitchen." },
      { role: "Agent", text: "Do you have headache, vomiting, confusion, weakness, numbness, or trouble speaking?" },
      { role: "Senior", text: "My head pain is getting worse. I feel confused and my left hand feels weak. My speech feels slurred." },
      { role: "Agent", text: "I am going to alert your volunteer coordinator and caregiver now." }
    ],
    speechMetrics: {
      speechRate: 68,
      avgPauseMs: 1900,
      responseLatencyMs: 3100,
      pitchVariability: 0.27,
      phraseAccuracy: 0.62,
      embedding: [0.59, 0.09, 0.68, 0.21, 0.12, 0.78],
      updatedAt: "2026-07-04T10:45:00+08:00"
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
    },
    originalTranscript: "I fell last night and hit my head. My head pain is getting worse. I feel confused and my left hand feels weak. My speech feels slurred.",
    englishTranscript: "I fell last night and hit my head. My headache is getting worse. I feel confused, my speech feels slurred, and my left hand feels weak."
  },
  {
    id: "chronic-illness",
    name: "Chronic Illness Check-In",
    label: "CKD / diabetes / blood pressure",
    seniorId: "s-003",
    description: "Condition-specific check-in covering CKD hydration, blood pressure, diabetes-style medication adherence, food, water, and appointments.",
    script: [
      { role: "Agent", text: "Encik Ahmad, did you drink enough water today and take your blood pressure medicine?" },
      { role: "Senior", text: "I drank only a little water. I took blood pressure medicine, but I am not sure about my kidney appointment." },
      { role: "Agent", text: "Any dizziness, swelling, missed medicine, or poor appetite?" },
      { role: "Senior", text: "No swelling. Appetite is lower today, and I want Hafiz to remind me about the appointment." }
    ],
    speechMetrics: {
      speechRate: 116,
      avgPauseMs: 760,
      responseLatencyMs: 1180,
      pitchVariability: 0.57,
      phraseAccuracy: 0.92,
      embedding: [0.22, 0.33, 0.31, 0.52, 0.45, 0.42],
      updatedAt: "2026-07-04T11:00:00+08:00"
    },
    symptoms: {
      poorIntake: true,
      chronicConcern: true,
      ckdConcern: true,
      highBloodPressureConcern: true
    },
    originalTranscript: "I drank only a little water. I took blood pressure medicine. I am not sure about my kidney appointment. Appetite is lower today.",
    englishTranscript: "I drank only a little water. I took my blood pressure medicine. I am not sure about my kidney appointment. My appetite is lower today."
  },
  {
    id: "mental-wellbeing",
    name: "Mental Wellbeing / Loneliness",
    label: "Fear of dying unnoticed",
    seniorId: "s-002",
    description: "Basic wellbeing call where the senior is physically stable but lonely and afraid nobody will know if something happens.",
    script: [
      { role: "Agent", text: "Mr Raman, how are you feeling today?" },
      { role: "Senior", text: "I am physically okay, but I feel lonely. Sometimes I worry I will die alone and nobody will know." },
      { role: "Agent", text: "Would you like a befriender call or a neighbour check-in this week?" },
      { role: "Senior", text: "Yes, please. I would like someone to call me." }
    ],
    speechMetrics: {
      speechRate: 108,
      avgPauseMs: 900,
      responseLatencyMs: 1500,
      pitchVariability: 0.5,
      phraseAccuracy: 0.9,
      embedding: [0.34, 0.2, 0.55, 0.39, 0.25, 0.66],
      updatedAt: "2026-07-04T11:15:00+08:00"
    },
    symptoms: {
      loneliness: true,
      lowMood: true,
      asksForHelp: true
    },
    originalTranscript: "I am physically okay, but I feel lonely. Sometimes I worry I will die alone and nobody will know. Yes, please call me.",
    englishTranscript: "I am physically okay, but I feel lonely. Sometimes I worry I will die alone and nobody will know. I would like someone to call me."
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
    recommendedAction: "Record routine check-in and continue the next scheduled call.",
    originalTranscript: "Saya okay. Sudah makan dan makan ubat.",
    englishTranscript: "I am okay. I ate and took my medication.",
    riskAssessment: {
      speechDeviationScore: 8,
      parkinsonsWatchScore: 4,
      postFallConcernScore: 0,
      missedCheckInScore: 0,
      riskLevel: "Green",
      reasons: ["Speech remains close to baseline", "No fall or head-impact symptoms reported"]
    },
    categories: [],
    escalationPlan: []
  },
  {
    id: "c-102",
    seniorId: "s-001",
    scheduledAt: "2026-07-04T09:00:00+08:00",
    status: "Urgent",
    language: "Mandarin",
    riskLevel: "Red",
    summary: "Reported fall with head impact, worsening headache, confusion, and left-hand weakness.",
    recommendedAction: "Trigger emergency escalation and notify caregiver.",
    originalTranscript: "我昨晚跌倒，撞到头。头很痛，有点乱，左手没有力。",
    englishTranscript: "I fell last night and hit my head. My head hurts, I feel confused, and my left hand is weak.",
    riskAssessment: {
      speechDeviationScore: 82,
      parkinsonsWatchScore: 28,
      postFallConcernScore: 96,
      missedCheckInScore: 0,
      riskLevel: "Red",
      reasons: ["Fall with head impact", "Confusion and weakness reported", "Large speech deviation from baseline"]
    },
    categories: [],
    escalationPlan: []
  }
];

export const scheduleItems: CheckInScheduleItem[] = [
  {
    seniorId: "s-002",
    seniorName: "Mr Raman Pillai",
    checkInFrequencyDays: 3,
    lastContactAt: null,
    lastContactKind: "none",
    lastAttemptAt: null,
    lastAttemptStatus: null,
    nextDueAt: "2026-07-05T10:00:00+08:00",
    status: "Due now",
    hoursUntilDue: 0,
    overdueHours: 0,
    recommendedAction: "Start the scheduled check-in now and retry once if there is no answer."
  },
  {
    seniorId: "s-001",
    seniorName: "Mdm Tan Bee Hoon",
    checkInFrequencyDays: 2,
    lastContactAt: "2026-07-04T09:00:00+08:00",
    lastContactKind: "check-in",
    lastAttemptAt: "2026-07-04T09:00:00+08:00",
    lastAttemptStatus: "Urgent",
    nextDueAt: "2026-07-06T09:00:00+08:00",
    status: "Due soon",
    hoursUntilDue: 23,
    overdueHours: 0,
    recommendedAction: "Prepare the next scheduled call within 23 hours."
  },
  {
    seniorId: "s-003",
    seniorName: "Encik Ahmad Rahman",
    checkInFrequencyDays: 2,
    lastContactAt: "2026-07-04T09:04:00+08:00",
    lastContactKind: "check-in",
    lastAttemptAt: "2026-07-04T09:04:00+08:00",
    lastAttemptStatus: "Checked in",
    nextDueAt: "2026-07-06T09:04:00+08:00",
    status: "Due soon",
    hoursUntilDue: 23.1,
    overdueHours: 0,
    recommendedAction: "Prepare the next scheduled call within 23 hours."
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

export const seniorRecords: SeniorRecord[] = [
  {
    seniorId: "s-001",
    seniorName: "Mdm Tan Bee Hoon",
    livingAlone: true,
    checkInFrequencyDays: 2,
    totalRecords: 1,
    openTaskCount: 1,
    highestRiskLevel: "Red",
    latestRecordAt: "2026-07-04T09:00:00+08:00",
    categories: [
      {
        id: "concussion_danger",
        label: "Possible concussion danger signs",
        highestSeverity: "Red",
        recordCount: 1,
        latestAt: "2026-07-04T09:00:00+08:00",
        latestEvidence: ["Confusion reported.", "Weakness or numbness reported.", "Headache or dizziness after impact reported."],
        recommendedAction: "For red danger signs after a bump, blow, jolt, or fall, escalate for urgent medical help."
      },
      {
        id: "fall_head_impact",
        label: "Fall / head impact / whiplash",
        highestSeverity: "Amber",
        recordCount: 1,
        latestAt: "2026-07-04T09:00:00+08:00",
        latestEvidence: ["Fall reported.", "Head impact reported."],
        recommendedAction: "Ask what happened, whether the head or body was hit, and whether the senior can move safely."
      }
    ],
    timeline: [
      {
        id: "c-102",
        source: "check-in",
        occurredAt: "2026-07-04T09:00:00+08:00",
        riskLevel: "Red",
        status: "Urgent",
        summary: "Reported fall with head impact, worsening headache, confusion, and left-hand weakness.",
        recommendedAction: "Trigger emergency escalation and notify caregiver.",
        categories: [
          {
            id: "concussion_danger",
            label: "Possible concussion danger signs",
            severity: "Red",
            evidence: ["Confusion reported.", "Weakness or numbness reported.", "Headache or dizziness after impact reported."],
            recommendedAction: "For red danger signs after a bump, blow, jolt, or fall, escalate for urgent medical help."
          },
          {
            id: "fall_head_impact",
            label: "Fall / head impact / whiplash",
            severity: "Amber",
            evidence: ["Fall reported.", "Head impact reported."],
            recommendedAction: "Ask what happened, whether the head or body was hit, and whether the senior can move safely."
          }
        ]
      }
    ]
  },
  {
    seniorId: "s-002",
    seniorName: "Mr Raman Pillai",
    livingAlone: true,
    checkInFrequencyDays: 3,
    totalRecords: 0,
    openTaskCount: 1,
    highestRiskLevel: "Green",
    latestRecordAt: null,
    categories: [],
    timeline: []
  },
  {
    seniorId: "s-003",
    seniorName: "Encik Ahmad Rahman",
    livingAlone: true,
    checkInFrequencyDays: 2,
    totalRecords: 1,
    openTaskCount: 0,
    highestRiskLevel: "Green",
    latestRecordAt: "2026-07-04T09:04:00+08:00",
    categories: [
      {
        id: "medication_food_water",
        label: "Medication / food / water",
        highestSeverity: "Green",
        recordCount: 1,
        latestAt: "2026-07-04T09:04:00+08:00",
        latestEvidence: ["Medication, food, or water status mentioned."],
        recommendedAction: "Confirm medication, food, and water intake. Escalate if intake is poor or medication has been missed."
      }
    ],
    timeline: [
      {
        id: "c-101",
        source: "check-in",
        occurredAt: "2026-07-04T09:04:00+08:00",
        riskLevel: "Green",
        status: "Checked in",
        summary: "Stable check-in. No falls, symptoms, or adherence concerns.",
        recommendedAction: "Record routine check-in and continue the next scheduled call.",
        categories: [
          {
            id: "medication_food_water",
            label: "Medication / food / water",
            severity: "Green",
            evidence: ["Medication, food, or water status mentioned."],
            recommendedAction: "Confirm medication, food, and water intake. Escalate if intake is poor or medication has been missed."
          }
        ]
      }
    ]
  }
];
