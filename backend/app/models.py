from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["Green", "Watch", "Amber", "Red"]
Language = Literal["English", "Mandarin", "Malay", "Tamil", "Singlish/Dialect"]
SpeechModelMode = Literal["demo metrics", "offline embedding", "validated model"]
ConversationCategoryId = Literal[
    "mental_wellbeing",
    "fall_head_impact",
    "concussion_danger",
    "parkinsons_watch",
    "chronic_illness",
    "medication_food_water",
    "social_isolation",
    "missed_checkin",
]
EscalationStepStatus = Literal["Standby", "Triggered", "Complete"]
CheckInScheduleStatus = Literal["On track", "Due soon", "Due now", "Overdue"]
CheckInContactKind = Literal["call", "check-in", "none"]
SeniorRecordSource = Literal["call", "check-in"]
CallPlanPriority = Literal["Routine", "Watch", "Urgent"]


class SpeechProfile(BaseModel):
    speechRate: float
    avgPauseMs: float
    responseLatencyMs: float
    pitchVariability: float
    phraseAccuracy: float
    embedding: list[float] | None = None
    updatedAt: str | None = None


class SpeechModelCardGate(BaseModel):
    datasetAccessReviewed: bool = False
    speakerSplitVerified: bool = False
    evaluationMetricsRecorded: bool = False
    subgroupChecksReviewed: bool = False
    failureModesDocumented: bool = False
    uiCopyReviewed: bool = False
    humanFollowUpActionDefined: bool = False
    rollbackPathDocumented: bool = False
    humanFollowUpAction: str | None = None


class SpeechModelProvenance(BaseModel):
    runtimeMode: SpeechModelMode
    featureExtractor: str
    modelName: str
    modelVersion: str | None = None
    artifactUri: str | None = None
    generatedAt: str
    validated: bool = False
    modelCard: SpeechModelCardGate | None = None
    notes: list[str] = Field(default_factory=list)


class Senior(BaseModel):
    id: str
    name: str
    age: int
    preferredLanguage: Language
    livingAlone: bool
    addressZone: str
    caregiverContact: str
    neighborContact: str | None = None
    knownConditions: list[str] = Field(default_factory=list)
    promptFocus: list[str] = Field(default_factory=list)
    checkInFrequencyDays: int
    baselineSpeechProfile: SpeechProfile


class CheckInScheduleItem(BaseModel):
    seniorId: str
    seniorName: str
    checkInFrequencyDays: int
    lastContactAt: str | None = None
    lastContactKind: CheckInContactKind = "none"
    lastAttemptAt: str | None = None
    lastAttemptStatus: str | None = None
    nextDueAt: str
    status: CheckInScheduleStatus
    hoursUntilDue: float
    overdueHours: float
    recommendedAction: str


class Symptoms(BaseModel):
    fall: bool = False
    headImpact: bool = False
    whiplashOrJolt: bool = False
    headache: bool = False
    worseningHeadache: bool = False
    dizziness: bool = False
    vomiting: bool = False
    confusion: bool = False
    slurredSpeech: bool = False
    weakness: bool = False
    numbness: bool = False
    unusualBehavior: bool = False
    drowsinessOrUnwakeable: bool = False
    poorIntake: bool = False
    asksForHelp: bool = False
    missedCheckIn: bool = False
    loneliness: bool = False
    lowMood: bool = False
    medicationMissed: bool = False
    chronicConcern: bool = False
    ckdConcern: bool = False
    diabetesConcern: bool = False
    highBloodPressureConcern: bool = False


class RiskAssessment(BaseModel):
    speechDeviationScore: int
    parkinsonsWatchScore: int
    postFallConcernScore: int
    missedCheckInScore: int
    riskLevel: RiskLevel
    reasons: list[str]


class ConversationCategory(BaseModel):
    id: ConversationCategoryId
    label: str
    severity: RiskLevel
    evidence: list[str] = Field(default_factory=list)
    recommendedAction: str


class SeniorRecordCategory(BaseModel):
    id: ConversationCategoryId
    label: str
    highestSeverity: RiskLevel
    recordCount: int
    latestAt: str | None = None
    latestEvidence: list[str] = Field(default_factory=list)
    recommendedAction: str


class EscalationStep(BaseModel):
    id: str
    label: str
    status: EscalationStepStatus
    detail: str


class CheckInSession(BaseModel):
    id: str
    seniorId: str
    scenarioId: str | None = None
    scenarioName: str | None = None
    scheduledAt: str
    completedAt: str | None = None
    status: Literal["In progress", "Checked in", "Missed", "Needs follow-up", "Urgent"]
    language: Language
    riskLevel: RiskLevel
    summary: str
    recommendedAction: str = "Continue routine scheduled check-ins."
    originalTranscript: str
    englishTranscript: str
    riskAssessment: RiskAssessment
    categories: list[ConversationCategory] = Field(default_factory=list)
    escalationPlan: list[EscalationStep] = Field(default_factory=list)
    modelNote: str | None = None


class VolunteerTask(BaseModel):
    id: str
    seniorId: str
    priority: Literal["Routine", "Today", "Urgent"]
    reason: str
    recommendedAction: str
    assignedTo: str
    status: Literal["Open", "In progress", "Closed"]
    createdAt: str
    sourceSessionId: str | None = None
    sourceCallId: str | None = None
    escalationStep: str | None = None


class SeniorRecordEvent(BaseModel):
    id: str
    source: SeniorRecordSource
    occurredAt: str
    riskLevel: RiskLevel
    status: str
    summary: str
    recommendedAction: str
    categories: list[ConversationCategory] = Field(default_factory=list)


class SeniorRecord(BaseModel):
    seniorId: str
    seniorName: str
    livingAlone: bool
    checkInFrequencyDays: int
    totalRecords: int
    openTaskCount: int
    highestRiskLevel: RiskLevel
    latestRecordAt: str | None = None
    categories: list[SeniorRecordCategory] = Field(default_factory=list)
    timeline: list[SeniorRecordEvent] = Field(default_factory=list)


class CallPlanQuestion(BaseModel):
    id: str
    priority: CallPlanPriority
    topic: str
    prompt: str
    rationale: str


class CallPlan(BaseModel):
    seniorId: str
    seniorName: str
    preferredLanguage: Language
    generatedAt: str
    scheduleStatus: CheckInScheduleStatus
    openingScript: str
    questions: list[CallPlanQuestion] = Field(default_factory=list)
    escalationReminder: str


class SpeechDeviationRequest(BaseModel):
    seniorId: str
    currentSpeechProfile: SpeechProfile
    symptoms: Symptoms = Symptoms()


class SpeechEnrichmentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    runtimeMode: SpeechModelMode = "offline embedding"
    featureExtractor: str | None = None
    modelName: str | None = None
    modelVersion: str | None = None
    artifactUri: str | None = None
    embedding: list[float] | None = None
    speechMetrics: SpeechProfile | None = Field(default=None, alias="speech_metrics")
    provenance: dict[str, object] = Field(default_factory=dict)
    modelCard: SpeechModelCardGate | None = None


class MissedCheckInRequest(BaseModel):
    seniorId: str
    scheduledAt: str | None = None
    retryAt: str | None = None
    attemptCount: int = Field(default=2, ge=1, le=5)
    note: str | None = None


class CompleteCheckInRequest(BaseModel):
    completedAt: str | None = None
    originalTranscript: str | None = None
    englishTranscript: str | None = None
    summary: str | None = None


class ProviderResult(BaseModel):
    provider: str
    language: str
    transcript: str
    translation: str
    confidence: float
    fallbackUsed: bool
    segments: list["TranscriptSegment"] = []


class TranscriptSegment(BaseModel):
    text: str
    originalText: str | None = None
    englishText: str | None = None
    startTimeSeconds: float | None = None
    endTimeSeconds: float | None = None
    role: str | None = None
    speaker: str | None = None


class TranscriptMessage(BaseModel):
    role: Literal["Agent", "Senior", "System"]
    text: str
    timestamp: str | None = None


class RiskSignal(BaseModel):
    id: str
    label: str
    severity: RiskLevel
    quotedText: str
    highlightText: str | None = None
    reason: str
    sentenceIndex: int | None = None
    startTimeSeconds: float | None = None
    endTimeSeconds: float | None = None


class CallRecord(BaseModel):
    id: str
    seniorId: str
    seniorName: str
    startedAt: str
    completedAt: str
    status: Literal["Complete", "Failed", "Saved"]
    riskLevel: RiskLevel
    originalTranscript: str
    englishTranscript: str
    transcriptMessages: list[TranscriptMessage]
    translationProvider: str
    translationFallbackUsed: bool
    audioFilePath: str | None = None
    audioUrl: str | None = None
    audioAvailable: bool
    agentAudioCaptured: bool = False
    currentSpeechProfile: SpeechProfile | None = None
    speechModelProvenance: SpeechModelProvenance | None = None
    transcriptSegments: list[TranscriptSegment] = []
    riskSignals: list[RiskSignal] = []
    aiRiskFallbackUsed: bool = False
    riskAssessment: RiskAssessment
    recommendedAction: str
    categories: list[ConversationCategory] = Field(default_factory=list)
    escalationPlan: list[EscalationStep] = Field(default_factory=list)


class SavedCallResponse(BaseModel):
    call: CallRecord


class Scenario(BaseModel):
    id: str
    name: str
    label: str
    seniorId: str
    description: str
    script: list[TranscriptMessage]
    speechMetrics: SpeechProfile
    symptoms: Symptoms = Symptoms()
    originalTranscript: str
    englishTranscript: str


class ScenarioRunResponse(BaseModel):
    session: CheckInSession
    tasks: list[VolunteerTask]
