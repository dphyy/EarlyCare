import { callPlans, operationQueueItems, scenarios, scheduleItems, seniorRecords, seniors, sessions, volunteerTasks } from "./data";
import type { CallPlan, CallRecord, CheckInScheduleItem, CheckInSession, OperationsQueueItem, Scenario, ScenarioRunResponse, Senior, SeniorRecord, SpeechModelMode, SpeechProfile, VolunteerTask } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;

async function getJson<T>(path: string, fallback: T): Promise<T> {
  if (!API_BASE_URL) return fallback;

  try {
    const response = await fetch(`${API_BASE_URL}${path}`);
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as T;
  } catch {
    return fallback;
  }
}

export function fetchSeniors(): Promise<Senior[]> {
  return getJson<Senior[]>("/seniors", seniors);
}

export function fetchSessions(): Promise<CheckInSession[]> {
  return getJson<CheckInSession[]>("/checkins", sessions);
}

export function fetchSchedule(): Promise<CheckInScheduleItem[]> {
  return getJson<CheckInScheduleItem[]>("/schedule", scheduleItems);
}

export function fetchOperationsQueue(): Promise<OperationsQueueItem[]> {
  return getJson<OperationsQueueItem[]>("/operations-queue", operationQueueItems);
}

export function fetchSeniorRecords(): Promise<SeniorRecord[]> {
  return getJson<SeniorRecord[]>("/senior-records", seniorRecords);
}

export function fetchCallPlans(): Promise<CallPlan[]> {
  return getJson<CallPlan[]>("/call-plans", callPlans);
}

export function fetchVolunteerTasks(): Promise<VolunteerTask[]> {
  return getJson<VolunteerTask[]>("/volunteer-tasks", volunteerTasks);
}

export function fetchCalls(): Promise<CallRecord[]> {
  return getJson<CallRecord[]>("/calls", []);
}

export function fetchScenarios(): Promise<Scenario[]> {
  return getJson<Scenario[]>("/scenarios", scenarios);
}

export function getCallAudioUrl(call: CallRecord): string | null {
  if (!API_BASE_URL || !call.audioAvailable) return null;
  const path = call.audioUrl ?? `/calls/${call.id}/audio`;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
}

export interface ElevenLabsSessionRequest {
  seniorId: string;
  seniorName: string;
  preferredLanguage: string;
  caregiverContact: string;
  checkInReason: string;
}

export interface ElevenLabsSessionResponse {
  configured: boolean;
  signedUrl?: string;
  agentId?: string;
  message: string;
}

export async function createElevenLabsSession(payload: ElevenLabsSessionRequest): Promise<ElevenLabsSessionResponse> {
  if (!API_BASE_URL) {
    return {
      configured: false,
      message: "Backend API URL is not configured. Using scripted fallback."
    };
  }

  try {
    const response = await fetch(`${API_BASE_URL}/elevenlabs/signed-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as ElevenLabsSessionResponse;
  } catch (error) {
    return {
      configured: false,
      message: error instanceof Error ? error.message : "Unable to create ElevenLabs session."
    };
  }
}

export async function saveCall(formData: FormData): Promise<CallRecord | null> {
  if (!API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/calls`, {
      method: "POST",
      body: formData
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    const payload = (await response.json()) as { call: CallRecord };
    return payload.call;
  } catch {
    return null;
  }
}

export interface SpeechEnrichmentPayload {
  runtimeMode?: SpeechModelMode;
  featureExtractor?: string;
  modelName?: string;
  modelVersion?: string;
  artifactUri?: string;
  embedding?: number[];
  speech_metrics?: SpeechProfile;
  provenance?: Record<string, unknown>;
}

export async function enrichCallSpeech(callId: string, payload: SpeechEnrichmentPayload): Promise<CallRecord | null> {
  if (!API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/calls/${callId}/speech-enrichment`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as CallRecord;
  } catch {
    return null;
  }
}

export async function runScenario(scenarioId: string): Promise<ScenarioRunResponse | null> {
  if (!API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/scenarios/${scenarioId}/run`, {
      method: "POST"
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as ScenarioRunResponse;
  } catch {
    return null;
  }
}

export async function startCheckIn(seniorId: string): Promise<CheckInSession | null> {
  if (!API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/checkins/start?senior_id=${encodeURIComponent(seniorId)}`, {
      method: "POST"
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as CheckInSession;
  } catch {
    return null;
  }
}

export interface CompleteCheckInRequest {
  completedAt?: string | null;
  originalTranscript?: string | null;
  englishTranscript?: string | null;
  summary?: string | null;
}

export async function completeCheckIn(checkInId: string, payload: CompleteCheckInRequest = {}): Promise<CheckInSession | null> {
  if (!API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/checkins/${encodeURIComponent(checkInId)}/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as CheckInSession;
  } catch {
    return null;
  }
}

export interface MissedCheckInRequest {
  seniorId: string;
  scheduledAt?: string | null;
  retryAt?: string | null;
  attemptCount?: number;
  note?: string | null;
}

export async function recordMissedCheckIn(payload: MissedCheckInRequest): Promise<ScenarioRunResponse | null> {
  if (!API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/checkins/missed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as ScenarioRunResponse;
  } catch {
    return null;
  }
}

export async function updateVolunteerTask(taskId: string, status: VolunteerTask["status"]): Promise<VolunteerTask | null> {
  if (!API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/volunteer-tasks/${taskId}?status=${encodeURIComponent(status)}`, {
      method: "PATCH"
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return (await response.json()) as VolunteerTask;
  } catch {
    return null;
  }
}
