export type RiskLevel = "Green" | "Watch" | "Amber" | "Red";
export type SafeguardLevel = "None" | "Support" | "Urgent" | "Emergency";
export type EmotionConcernLevel = "None" | "Watch" | "Review";
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

export interface TranscriptionAttempt {
  provider: string;
  status: "success" | "failed" | "skipped";
  reason?: string | null;
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

export interface EmotionSegment {
  id: string;
  label: string;
  confidence: number;
  startTimeSeconds?: number | null;
  endTimeSeconds?: number | null;
  transcriptSegmentIndex?: number | null;
  evidenceText: string;
  valence?: number | null;
  arousal?: number | null;
  dominance?: number | null;
}

export interface CrisisResource {
  name: string;
  phone?: string | null;
  text?: string | null;
  url?: string | null;
  description: string;
}

export interface ConcussionSpeechReview {
  modelVersion?: string | null;
  predictedLabel?: string | null;
  probabilities: Record<string, number>;
  qualityOk: boolean;
  qualityReason?: string | null;
  durationSec?: number | null;
  sampleRate?: number | null;
  rms?: number | null;
  clippingFraction?: number | null;
  riskContribution: RiskLevel;
  riskReason?: string | null;
  warning: string;
  failureReason?: string | null;
}

export interface ParkinsonsSpeechReview {
  modelVersion?: string | null;
  probability?: number | null;
  warnings: string[];
  featuresSummary?: Record<string, number | string | null> | null;
  qualityOk: boolean;
  riskReason?: string | null;
  warning: string;
  failureReason?: string | null;
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
  elevenLabsConversationId?: string | null;
  translationProvider: string;
  translationFallbackUsed: boolean;
  transcriptionAttempts?: TranscriptionAttempt[];
  audioFilePath?: string | null;
  audioUrl?: string | null;
  audioAvailable: boolean;
  patientAudioFilePath?: string | null;
  patientAudioUrl?: string | null;
  patientAudioAvailable?: boolean;
  patientSpeechAudioFilePath?: string | null;
  patientSpeechAudioUrl?: string | null;
  patientSpeechAudioAvailable?: boolean;
  agentAudioCaptured?: boolean;
  parkinsonsSpeechReview?: ParkinsonsSpeechReview | null;
  speechModelVersion?: string | null;
  speechModelProbability?: number | null;
  speechModelWarnings?: string[];
  speechModelFeaturesSummary?: Record<string, number | string | null>;
  concussionSpeechReview?: ConcussionSpeechReview | null;
  currentSpeechProfile?: SpeechProfile | null;
  transcriptSegments?: TranscriptSegment[];
  transcriptAlignmentWarnings?: string[];
  riskSignals?: RiskSignal[];
  aiRiskFallbackUsed?: boolean;
  aiRiskFailureReason?: string | null;
  emotionReviewAvailable?: boolean;
  emotionProvider?: string | null;
  emotionFallbackUsed?: boolean;
  emotionFailureReason?: string | null;
  dominantPatientEmotion?: string | null;
  emotionConcernLevel?: EmotionConcernLevel;
  emotionSegments?: EmotionSegment[];
  emotionAttempts?: TranscriptionAttempt[];
  safeguardReviewAvailable?: boolean;
  safeguardLevel?: SafeguardLevel;
  safeguardCategory?: string | null;
  safeguardEvidence?: string[];
  safeguardRecommendedAction?: string | null;
  safeguardResources?: CrisisResource[];
  safeguardFailureReason?: string | null;
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
