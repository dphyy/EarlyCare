export type RiskLevel = "Green" | "Watch" | "Amber" | "Red";
export type CheckInStatus = "Checked in" | "Missed" | "Needs follow-up" | "Urgent";
export type Language = "English" | "Mandarin" | "Malay" | "Tamil" | "Singlish/Dialect";

export interface SpeechProfile {
  speechRate: number;
  avgPauseMs: number;
  responseLatencyMs: number;
  pitchVariability: number;
  phraseAccuracy: number;
  embedding?: number[];
  updatedAt: string;
}

export interface Senior {
  id: string;
  name: string;
  age: number;
  preferredLanguage: Language;
  livingAlone: boolean;
  addressZone: string;
  caregiverContact: string;
  checkInFrequencyDays: number;
  baselineSpeechProfile: SpeechProfile;
}

export interface RiskAssessment {
  speechDeviationScore: number;
  parkinsonsWatchScore: number;
  postFallConcernScore: number;
  missedCheckInScore: number;
  riskLevel: RiskLevel;
  reasons: string[];
}

export interface CheckInSession {
  id: string;
  seniorId: string;
  scheduledAt: string;
  completedAt?: string;
  status: CheckInStatus;
  language: Language;
  riskLevel: RiskLevel;
  summary: string;
  originalTranscript: string;
  englishTranscript: string;
  riskAssessment: RiskAssessment;
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
  transcriptSegments?: TranscriptSegment[];
  riskSignals?: RiskSignal[];
  aiRiskFallbackUsed?: boolean;
  riskAssessment: RiskAssessment;
  recommendedAction: string;
}

export interface Scenario {
  id: string;
  name: string;
  label: string;
  seniorId: string;
  script: string[];
  speechMetrics: Omit<SpeechProfile, "updatedAt">;
  symptoms: {
    fall: boolean;
    headImpact: boolean;
    headache: boolean;
    dizziness: boolean;
    vomiting: boolean;
    confusion: boolean;
    slurredSpeech: boolean;
    weakness: boolean;
    missedCheckIn: boolean;
  };
}
