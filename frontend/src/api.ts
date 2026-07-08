import { seniors, sessions } from "./data";
import type { CallRecord, CheckInSession, ConsultationMemoryItem, ReadinessReport, Senior, VolunteerTask } from "./types";

const API_BASE_URL = ((import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "").replace(/\/$/, "");

async function getJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { credentials: "include" });
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

export function fetchVolunteerTasks(): Promise<VolunteerTask[]> {
  return getJson<VolunteerTask[]>("/volunteer-tasks", []);
}

export function fetchCalls(): Promise<CallRecord[]> {
  return getJson<CallRecord[]>("/calls", []);
}

export function fetchReadiness(): Promise<ReadinessReport> {
  return getJson<ReadinessReport>("/readiness", {
    status: "blocked",
    message: "Backend readiness is unavailable.",
    components: [{ name: "Backend API", status: "blocked", detail: "Frontend could not reach the FastAPI readiness endpoint." }]
  });
}

export function fetchConsultationMemory(seniorId: string): Promise<ConsultationMemoryItem[]> {
  return getJson<ConsultationMemoryItem[]>(`/seniors/${seniorId}/consultation-memory`, []);
}

export function getCallAudioUrl(call: CallRecord): string | null {
  if (!call.audioAvailable) return null;
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

export type SaveCallResult =
  | { ok: true; call: CallRecord }
  | { ok: false; message: string; status?: number };

export interface AuthStatus {
  authEnabled: boolean;
  authenticated: boolean;
  username?: string | null;
  message?: string | null;
}

async function responseErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (typeof item === "string") return item;
          if (item && typeof item === "object" && "msg" in item) return String((item as { msg: unknown }).msg);
          return JSON.stringify(item);
        })
        .join("; ");
    }
    if (detail) return JSON.stringify(detail);
  } catch {
    // Fall through to the status text below.
  }
  return response.statusText || `Request failed: ${response.status}`;
}

export async function fetchAuthStatus(): Promise<AuthStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, { credentials: "include" });
    if (!response.ok) {
      return { authEnabled: true, authenticated: false };
    }
    return (await response.json()) as AuthStatus;
  } catch {
    return { authEnabled: false, authenticated: true, username: "demo-operator" };
  }
}

export async function loginOperator(username: string, password: string): Promise<AuthStatus> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ username, password })
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
  return (await response.json()) as AuthStatus;
}

export async function logoutOperator(): Promise<void> {
  await fetch(`${API_BASE_URL}/auth/logout`, {
    method: "POST",
    credentials: "include"
  });
}

export async function createElevenLabsSession(payload: ElevenLabsSessionRequest): Promise<ElevenLabsSessionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/elevenlabs/signed-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
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

export async function saveCall(formData: FormData): Promise<SaveCallResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/calls`, {
      method: "POST",
      credentials: "include",
      body: formData
    });
    if (!response.ok) {
      return { ok: false, status: response.status, message: await responseErrorMessage(response) };
    }
    const payload = (await response.json()) as { call: CallRecord };
    return { ok: true, call: payload.call };
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : "Unable to save call."
    };
  }
}
