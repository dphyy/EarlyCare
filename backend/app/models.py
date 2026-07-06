from typing import Literal

from pydantic import BaseModel


RiskLevel = Literal["Green", "Watch", "Amber", "Red"]
Language = Literal["English", "Mandarin", "Malay", "Tamil", "Singlish/Dialect"]
VolunteerTaskStatus = Literal["Open", "In progress", "Closed"]


class SpeechProfile(BaseModel):
    speechRate: float
    avgPauseMs: float
    responseLatencyMs: float
    pitchVariability: float
    phraseAccuracy: float
    embedding: list[float] | None = None
    updatedAt: str | None = None


class Senior(BaseModel):
    id: str
    name: str
    age: int
    preferredLanguage: Language
    livingAlone: bool
    addressZone: str
    caregiverContact: str
    checkInFrequencyDays: int
    baselineSpeechProfile: SpeechProfile


class Symptoms(BaseModel):
    fall: bool = False
    headImpact: bool = False
    headache: bool = False
    dizziness: bool = False
    vomiting: bool = False
    confusion: bool = False
    slurredSpeech: bool = False
    weakness: bool = False
    poorIntake: bool = False
    asksForHelp: bool = False
    missedCheckIn: bool = False


class RiskAssessment(BaseModel):
    speechDeviationScore: int
    parkinsonsWatchScore: int
    postFallConcernScore: int
    missedCheckInScore: int
    riskLevel: RiskLevel
    reasons: list[str]


class CheckInSession(BaseModel):
    id: str
    seniorId: str
    scheduledAt: str
    completedAt: str | None = None
    status: Literal["Checked in", "Missed", "Needs follow-up", "Urgent"]
    language: Language
    riskLevel: RiskLevel
    summary: str
    originalTranscript: str
    englishTranscript: str
    riskAssessment: RiskAssessment


class VolunteerTask(BaseModel):
    id: str
    seniorId: str
    priority: Literal["Routine", "Today", "Urgent"]
    reason: str
    recommendedAction: str
    assignedTo: str
    status: VolunteerTaskStatus
    createdAt: str


class SpeechDeviationRequest(BaseModel):
    seniorId: str
    currentSpeechProfile: SpeechProfile
    symptoms: Symptoms = Symptoms()


class ProviderResult(BaseModel):
    provider: str
    language: str
    transcript: str
    translation: str
    confidence: float
    fallbackUsed: bool
    segments: list["TranscriptSegment"] = []
    attempts: list["TranscriptionAttempt"] = []


class TranscriptionAttempt(BaseModel):
    provider: str
    status: Literal["success", "failed", "skipped"]
    reason: str | None = None


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
    transcriptionAttempts: list[TranscriptionAttempt] = []
    audioFilePath: str | None = None
    audioUrl: str | None = None
    audioAvailable: bool
    patientAudioFilePath: str | None = None
    patientAudioUrl: str | None = None
    patientAudioAvailable: bool = False
    patientSpeechAudioFilePath: str | None = None
    patientSpeechAudioUrl: str | None = None
    patientSpeechAudioAvailable: bool = False
    agentAudioCaptured: bool = False
    currentSpeechProfile: SpeechProfile | None = None
    transcriptSegments: list[TranscriptSegment] = []
    transcriptAlignmentWarnings: list[str] = []
    riskSignals: list[RiskSignal] = []
    aiRiskFallbackUsed: bool = False
    aiRiskFailureReason: str | None = None
    speechModelVersion: str | None = None
    speechModelProbability: float | None = None
    speechModelWarnings: list[str] = []
    speechModelFeaturesSummary: dict[str, float | int | str | None] | None = None
    riskAssessment: RiskAssessment
    recommendedAction: str


class SavedCallResponse(BaseModel):
    call: CallRecord


class VolunteerTaskUpdate(BaseModel):
    status: VolunteerTaskStatus
