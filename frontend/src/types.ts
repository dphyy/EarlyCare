export type RiskLevel = "Green" | "Watch" | "Amber" | "Red";
export type CheckInStatus = "Checked in" | "Missed" | "Needs follow-up" | "Urgent";
export type CheckInScheduleStatus = "On track" | "Due soon" | "Due now" | "Overdue";
export type Language = "English" | "Mandarin" | "Malay" | "Tamil" | "Singlish/Dialect";
export type SpeechModelMode = "demo metrics" | "offline embedding" | "validated model";
export type CallPlanPriority = "Routine" | "Watch" | "Urgent";
export type ConversationCategoryId =
  | "mental_wellbeing"
  | "fall_head_impact"
  | "concussion_danger"
  | "parkinsons_watch"
  | "chronic_illness"
  | "medication_food_water"
  | "social_isolation"
  | "missed_checkin";

export interface SpeechProfile {
  speechRate: number;
  avgPauseMs: number;
  responseLatencyMs: number;
  pitchVariability: number;
  phraseAccuracy: number;
  embedding?: number[];
  updatedAt: string;
}

export interface SpeechModelProvenance {
  runtimeMode: SpeechModelMode;
  featureExtractor: string;
  modelName: string;
  modelVersion?: string | null;
  artifactUri?: string | null;
  generatedAt: string;
  validated: boolean;
  notes: string[];
}

export interface Senior {
  id: string;
  name: string;
  age: number;
  preferredLanguage: Language;
  livingAlone: boolean;
  addressZone: string;
  caregiverContact: string;
  neighborContact?: string | null;
  knownConditions: string[];
  promptFocus: string[];
  checkInFrequencyDays: number;
  baselineSpeechProfile: SpeechProfile;
}

export interface CheckInScheduleItem {
  seniorId: string;
  seniorName: string;
  checkInFrequencyDays: number;
  lastContactAt?: string | null;
  lastContactKind: "call" | "check-in" | "none";
  lastAttemptAt?: string | null;
  lastAttemptStatus?: string | null;
  nextDueAt: string;
  status: CheckInScheduleStatus;
  hoursUntilDue: number;
  overdueHours: number;
  recommendedAction: string;
}

export interface RiskAssessment {
  speechDeviationScore: number;
  parkinsonsWatchScore: number;
  postFallConcernScore: number;
  missedCheckInScore: number;
  riskLevel: RiskLevel;
  reasons: string[];
}

export interface ConversationCategory {
  id: ConversationCategoryId;
  label: string;
  severity: RiskLevel;
  evidence: string[];
  recommendedAction: string;
}

export interface SeniorRecordCategory {
  id: ConversationCategoryId;
  label: string;
  highestSeverity: RiskLevel;
  recordCount: number;
  latestAt?: string | null;
  latestEvidence: string[];
  recommendedAction: string;
}

export interface EscalationStep {
  id: string;
  label: string;
  status: "Standby" | "Triggered" | "Complete";
  detail: string;
}

export interface CheckInSession {
  id: string;
  seniorId: string;
  scenarioId?: string | null;
  scenarioName?: string | null;
  scheduledAt: string;
  completedAt?: string;
  status: CheckInStatus;
  language: Language;
  riskLevel: RiskLevel;
  summary: string;
  recommendedAction: string;
  originalTranscript: string;
  englishTranscript: string;
  riskAssessment: RiskAssessment;
  categories: ConversationCategory[];
  escalationPlan: EscalationStep[];
  modelNote?: string | null;
}

export interface VolunteerTask {
  id: string;
  seniorId: string;
  priority: "Routine" | "Today" | "Urgent";
  reason: string;
  recommendedAction: string;
  assignedTo: string;
  status: "Open" | "In progress" | "Closed";
  createdAt: string;
  sourceSessionId?: string | null;
  sourceCallId?: string | null;
  escalationStep?: string | null;
}

export interface SeniorRecordEvent {
  id: string;
  source: "call" | "check-in";
  occurredAt: string;
  riskLevel: RiskLevel;
  status: string;
  summary: string;
  recommendedAction: string;
  categories: ConversationCategory[];
}

export interface SeniorRecord {
  seniorId: string;
  seniorName: string;
  livingAlone: boolean;
  checkInFrequencyDays: number;
  totalRecords: number;
  openTaskCount: number;
  highestRiskLevel: RiskLevel;
  latestRecordAt?: string | null;
  categories: SeniorRecordCategory[];
  timeline: SeniorRecordEvent[];
}

export interface CallPlanQuestion {
  id: string;
  priority: CallPlanPriority;
  topic: string;
  prompt: string;
  rationale: string;
}

export interface CallPlan {
  seniorId: string;
  seniorName: string;
  preferredLanguage: Language;
  generatedAt: string;
  scheduleStatus: CheckInScheduleStatus;
  openingScript: string;
  questions: CallPlanQuestion[];
  escalationReminder: string;
}

export interface TranscriptMessage {
  role: "Agent" | "Senior" | "System";
  text: string;
  timestamp?: string;
}

export interface TranscriptSegment {
  text: string;
  originalText?: string | null;
  englishText?: string | null;
  startTimeSeconds?: number | null;
  endTimeSeconds?: number | null;
  role?: string | null;
  speaker?: string | null;
}

export interface RiskSignal {
  id: string;
  label: string;
  severity: RiskLevel;
  quotedText: string;
  highlightText?: string | null;
  reason: string;
  sentenceIndex?: number | null;
  startTimeSeconds?: number | null;
  endTimeSeconds?: number | null;
}

export interface CallRecord {
  id: string;
  seniorId: string;
  seniorName: string;
  startedAt: string;
  completedAt: string;
  status: "Complete" | "Failed" | "Saved";
  riskLevel: RiskLevel;
  originalTranscript: string;
  englishTranscript: string;
  transcriptMessages: TranscriptMessage[];
  translationProvider: string;
  translationFallbackUsed: boolean;
  audioFilePath?: string | null;
  audioUrl?: string | null;
  audioAvailable: boolean;
  agentAudioCaptured?: boolean;
  currentSpeechProfile?: SpeechProfile | null;
  speechModelProvenance?: SpeechModelProvenance | null;
  transcriptSegments?: TranscriptSegment[];
  riskSignals?: RiskSignal[];
  aiRiskFallbackUsed?: boolean;
  riskAssessment: RiskAssessment;
  recommendedAction: string;
  categories: ConversationCategory[];
  escalationPlan: EscalationStep[];
}

export interface Scenario {
  id: string;
  name: string;
  label: string;
  seniorId: string;
  description: string;
  script: TranscriptMessage[];
  speechMetrics: SpeechProfile;
  symptoms: Record<string, boolean>;
  originalTranscript: string;
  englishTranscript: string;
}

export interface ScenarioRunResponse {
  session: CheckInSession;
  tasks: VolunteerTask[];
}
