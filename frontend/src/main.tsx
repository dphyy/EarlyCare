import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ConversationProvider, useConversation } from "@elevenlabs/react";
import {
  Activity,
  AlertTriangle,
  Bell,
  Brain,
  CheckCircle2,
  Headphones,
  HeartPulse,
  Languages,
  PhoneCall,
  ShieldCheck,
  UserRoundCheck
} from "lucide-react";
import { createElevenLabsSession, fetchCalls, fetchSeniors, fetchSessions, fetchVolunteerTasks, getCallAudioUrl, saveCall } from "./api";
import type { CallRecord, CheckInSession, CrisisResource, RiskLevel, SafeguardLevel, Senior, TranscriptMessage, TranscriptSegment, VolunteerTask } from "./types";
import "./styles.css";

const riskOrder: Record<RiskLevel, number> = { Green: 0, Watch: 1, Amber: 2, Red: 3 };
type AppView = "call" | "dashboard";
type CallState = "Ready" | "Connecting" | "In call" | "Saving" | "Analysing" | "Complete" | "Failed";
type AgentAudioFormat = "pcm_8000" | "pcm_16000" | "pcm_22050" | "pcm_24000" | "pcm_44100" | "pcm_48000" | "ulaw_8000";
const singaporeCrisisResources: CrisisResource[] = [
  {
    name: "Emergency medical services",
    phone: "995",
    description: "Call for immediate medical danger or urgent ambulance support in Singapore."
  },
  {
    name: "Police emergency",
    phone: "999",
    description: "Call if there is immediate danger, violence, or urgent police assistance is needed in Singapore."
  },
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
];
interface WavRecorder {
  chunks: Float32Array[];
  input: AudioNode;
  processor: ScriptProcessorNode;
  silentOutput: GainNode;
  sampleRate: number;
}
interface PreparedMicStream {
  stream: MediaStream;
  warning: string;
}
interface TranscriptHighlight {
  id: string;
  kind: "risk" | "safeguard" | "emotion";
  label: string;
  text: string;
  severity: RiskLevel;
  startTimeSeconds?: number | null;
  endTimeSeconds?: number | null;
  sentenceIndex?: number | null;
  transcriptSegmentIndex?: number | null;
}

function StatCard({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <section className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
      </div>
    </section>
  );
}

function RiskBadge({ level }: { level: RiskLevel }) {
  return <span className={`risk-badge risk-${level.toLowerCase()}`}>{level}</span>;
}

function cleanTranscriptText(text: string): string {
  return text
    .replace(/\s*\[[^\]\r\n]{1,40}\]\s*/g, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function writeAscii(view: DataView, offset: number, text: string) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function encodeWav(chunks: Float32Array[], sampleRate: number): Blob | null {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
  if (!sampleCount) return null;

  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, sampleCount * 2, true);

  let offset = 44;
  chunks.forEach((chunk) => {
    for (let index = 0; index < chunk.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, chunk[index]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  });
  return new Blob([view], { type: "audio/wav" });
}

function createWavRecorder(audioContext: AudioContext, input: AudioNode): WavRecorder {
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const silentOutput = audioContext.createGain();
  const chunks: Float32Array[] = [];
  silentOutput.gain.value = 0;
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  input.connect(processor);
  processor.connect(silentOutput);
  silentOutput.connect(audioContext.destination);
  return { chunks, input, processor, silentOutput, sampleRate: audioContext.sampleRate };
}

async function requestCleanMicrophoneStream(): Promise<PreparedMicStream> {
  const getUserMedia = navigator.mediaDevices?.getUserMedia?.bind(navigator.mediaDevices);
  if (!getUserMedia) {
    throw new Error(
      "Microphone access is unavailable in this browser context. Open EarlyCare from http://127.0.0.1:5173 or http://localhost:5173 in a supported browser and allow microphone permission."
    );
  }

  const enhancedConstraints: MediaStreamConstraints = {
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
      sampleRate: 16_000
    }
  };
  try {
    const stream = await getUserMedia(enhancedConstraints);
    const settings = stream.getAudioTracks()[0]?.getSettings?.() ?? {};
    const unavailable = [
      settings.echoCancellation === false ? "echo cancellation" : "",
      settings.noiseSuppression === false ? "noise suppression" : "",
      settings.autoGainControl === false ? "auto gain control" : ""
    ].filter(Boolean);
    return {
      stream,
      warning: unavailable.length ? `Browser microphone cleanup is limited: ${unavailable.join(", ")} unavailable.` : ""
    };
  } catch (error) {
    const fallbackStream = await getUserMedia({ audio: true });
    const reason = error instanceof Error ? error.message : "enhanced constraints were rejected";
    return {
      stream: fallbackStream,
      warning: `Browser rejected enhanced microphone cleanup (${reason}); using basic microphone capture.`
    };
  }
}

function stopRecorder(recorder: WavRecorder | null): Promise<Blob | null> {
  if (!recorder) return Promise.resolve(null);
  recorder.processor.onaudioprocess = null;
  recorder.input.disconnect(recorder.processor);
  recorder.processor.disconnect();
  recorder.silentOutput.disconnect();
  return Promise.resolve(encodeWav(recorder.chunks, recorder.sampleRate));
}

function base64ToArrayBuffer(base64Audio: string): ArrayBuffer {
  const binary = window.atob(base64Audio);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes.buffer;
}

function sampleRateFromFormat(format: AgentAudioFormat): number {
  const rate = Number(format.split("_")[1]);
  return Number.isFinite(rate) ? rate : 16000;
}

function decodeMuLawSample(value: number): number {
  const inverted = ~value & 0xff;
  const sign = inverted & 0x80;
  const exponent = (inverted >> 4) & 0x07;
  const mantissa = inverted & 0x0f;
  let sample = ((mantissa << 3) + 0x84) << exponent;
  sample -= 0x84;
  return (sign ? -sample : sample) / 32768;
}

function createAgentAudioBuffer(audioContext: AudioContext, base64Audio: string, format: AgentAudioFormat): AudioBuffer {
  const bytes = new Uint8Array(base64ToArrayBuffer(base64Audio));
  const sampleRate = sampleRateFromFormat(format);
  const sampleCount = format.startsWith("pcm_") ? Math.floor(bytes.length / 2) : bytes.length;
  const audioBuffer = audioContext.createBuffer(1, sampleCount, sampleRate);
  const channel = audioBuffer.getChannelData(0);

  if (format.startsWith("pcm_")) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let index = 0; index < sampleCount; index += 1) {
      channel[index] = view.getInt16(index * 2, true) / 32768;
    }
    return audioBuffer;
  }

  if (format === "ulaw_8000") {
    for (let index = 0; index < sampleCount; index += 1) {
      channel[index] = decodeMuLawSample(bytes[index]);
    }
    return audioBuffer;
  }

  throw new Error(`Unsupported agent audio format: ${format}`);
}

function multilingualAgentPrompt(senior: Senior): string {
  const resourceText = singaporeCrisisResources
    .map((resource) => `${resource.name}: ${resource.phone || resource.text || resource.url || ""}`)
    .join("; ");
  return [
    `You are EarlyCare calling ${senior.name} for a routine wellbeing check-in.`,
    `The patient profile says their preferred language is ${senior.preferredLanguage}, but they may speak English, Mandarin, Malay, Tamil, Singlish, or a mix.`,
    "The live call transcript must stay in the original spoken language. Do not translate the patient's message before responding.",
    "For each agent reply, use the language or dialect that the patient used the most in their immediately previous response. If the patient code-switches, follow the dominant language from that one previous response.",
    "If the patient asks to speak Chinese or any other language, switch immediately.",
    "Ask concise turn-by-turn questions about falls, head impact, headache, dizziness, vomiting, confusion, weakness, speech difficulty, food and water, and whether they can ask for help.",
    "If the patient expresses immediate danger, intent to self-harm, inability to stay safe, abuse, neglect, or severe emotional distress, stay calm, do not diagnose or provide counselling, encourage them to contact trusted nearby help, and share the relevant Singapore emergency or crisis support resource.",
    `Singapore support resources available to mention when relevant: ${resourceText}.`,
    "Do not add bracketed emotional cues such as [concerned] or [happy]."
  ].join(" ");
}

function nextReplyLanguageInstruction(patientText: string): string {
  return [
    "For your next reply only, respond in the language or dialect used most in this patient's immediately previous response.",
    "Do not translate their response into English before replying.",
    `Previous patient response: ${patientText}`
  ].join(" ");
}

function looksNonEnglish(text: string): boolean {
  return /[\u4e00-\u9fff\u3040-\u30ff\u0b80-\u0bff\u0d00-\u0d7f\u0600-\u06ff]/.test(text);
}

function splitSentences(text: string): string[] {
  const cleaned = cleanTranscriptText(text);
  if (!cleaned) return [];
  return cleaned.split(/(?<=[.!?。？！])\s+/).map((line) => line.trim()).filter(Boolean);
}

function relativeSeconds(timestamp: string | undefined, startedAt: string): number | null {
  if (!timestamp || !startedAt) return null;
  const current = new Date(timestamp).getTime();
  const start = new Date(startedAt).getTime();
  if (!Number.isFinite(current) || !Number.isFinite(start)) return null;
  return Math.max(0, (current - start) / 1000);
}

function sentenceTimeFromMessages(call: CallRecord, sentenceIndex: number, sentenceCount: number): number | null {
  const timedMessages = call.transcriptMessages
    .map((message, index) => ({ message, index, start: relativeSeconds(message.timestamp, call.startedAt) }))
    .filter((item): item is { message: TranscriptMessage; index: number; start: number } => item.start !== null);
  if (!timedMessages.length) return null;
  const mappedIndex = Math.min(timedMessages.length - 1, Math.round(sentenceIndex * (timedMessages.length - 1) / Math.max(1, sentenceCount - 1)));
  return timedMessages[mappedIndex].start;
}

function isPatientSegment(segment: TranscriptSegment): boolean {
  const text = cleanTranscriptText(segment.englishText || segment.text);
  return segment.role === "Patient" || segment.speaker === "Patient" || text.startsWith("Patient:");
}

function textWithoutSpeakerLabel(text: string): string {
  let cleaned = cleanTranscriptText(text);
  while (/^(Agent|Patient|Senior):\s*/i.test(cleaned)) {
    cleaned = cleaned.replace(/^(Agent|Patient|Senior):\s*/i, "").trim();
  }
  return cleaned;
}

function displaySegmentText(segment: TranscriptSegment): string {
  const role = segment.role === "Senior" ? "Patient" : segment.role || segment.speaker;
  const text = textWithoutSpeakerLabel(segment.englishText || segment.text);
  return role === "Agent" || role === "Patient" ? `${role}: ${text}` : cleanTranscriptText(segment.englishText || segment.text);
}

function wordCount(text: string): number {
  const matches = textWithoutSpeakerLabel(text).match(/\w+|[\u4e00-\u9fff]/g);
  return Math.max(1, matches?.length ?? 1);
}

function estimatedUtteranceSeconds(text: string): number {
  return Math.min(12, Math.max(0.9, wordCount(text) / 2.4));
}

function isAgentSegment(segment: TranscriptSegment): boolean {
  const text = cleanTranscriptText(segment.englishText || segment.text);
  return segment.role === "Agent" || segment.speaker === "Agent" || text.startsWith("Agent:");
}

function agentEstimatedEnd(segment: TranscriptSegment): number | null {
  if (typeof segment.startTimeSeconds !== "number") return null;
  const estimatedEnd = segment.startTimeSeconds + estimatedUtteranceSeconds(segment.englishText || segment.text);
  if (typeof segment.endTimeSeconds === "number" && segment.endTimeSeconds > segment.startTimeSeconds) {
    return Math.min(segment.endTimeSeconds, estimatedEnd);
  }
  return estimatedEnd;
}

function previousAgentReplyEnd(segments: TranscriptSegment[], patientSegmentIndex: number): number | null {
  for (let index = patientSegmentIndex - 1; index >= 0; index -= 1) {
    const candidate = segments[index];
    if (isAgentSegment(candidate)) {
      return agentEstimatedEnd(candidate);
    }
  }
  return null;
}

function getEnglishTranscriptSegments(call: CallRecord): TranscriptSegment[] {
  const segments = call.transcriptSegments ?? [];
  const joinedSegmentEnglish = segments.map((segment) => cleanTranscriptText(segment.englishText || segment.text)).join("\n").trim();
  const hasLiveRoles = segments.some((segment) => segment.role === "Agent" || segment.role === "Patient" || segment.speaker === "Agent" || segment.speaker === "Patient");
  const shouldUseCallTranscript =
    !hasLiveRoles &&
    (!segments.length ||
      !joinedSegmentEnglish ||
      joinedSegmentEnglish === cleanTranscriptText(call.originalTranscript) ||
      (looksNonEnglish(joinedSegmentEnglish) && !looksNonEnglish(call.englishTranscript)));

  if (shouldUseCallTranscript) {
    const sentences = splitSentences(call.englishTranscript);
    if (!sentences.length) return [{ text: call.englishTranscript, englishText: call.englishTranscript }];
    if (segments.length === 1) {
      return sentences.map((sentence, index) => ({
        text: sentence,
        englishText: sentence,
        originalText: index === 0 ? segments[0].originalText : undefined,
        startTimeSeconds: index === 0 ? segments[0].startTimeSeconds ?? sentenceTimeFromMessages(call, index, sentences.length) : sentenceTimeFromMessages(call, index, sentences.length),
        endTimeSeconds: index === sentences.length - 1 ? segments[0].endTimeSeconds : undefined
      }));
    }
    return sentences.map((sentence, index) => ({
      text: sentence,
      englishText: sentence,
      startTimeSeconds: segments[index]?.startTimeSeconds ?? sentenceTimeFromMessages(call, index, sentences.length),
      endTimeSeconds: segments[index]?.endTimeSeconds
    }));
  }

  if (segments.length === 1) {
    const text = cleanTranscriptText(segments[0].englishText || segments[0].text);
    const sentences = splitSentences(text);
    if (sentences.length > 1) {
      return sentences.map((sentence, index) => ({
        ...segments[0],
        text: sentence,
        englishText: sentence,
        startTimeSeconds: index === 0 ? segments[0].startTimeSeconds ?? sentenceTimeFromMessages(call, index, sentences.length) : sentenceTimeFromMessages(call, index, sentences.length),
        endTimeSeconds: index === sentences.length - 1 ? segments[0].endTimeSeconds : undefined
      }));
    }
  }

  return segments;
}

function buildTranscriptText(call: CallRecord, language: "original" | "english"): string {
  if (language === "english") {
    return getEnglishTranscriptSegments(call).map(displaySegmentText).filter(Boolean).join("\n") || call.englishTranscript;
  }
  return call.originalTranscript.replace(/\bSenior:/g, "Patient:");
}

function evidenceMatchesPatientText(evidence: string, patientText: string): boolean {
  const normalizedEvidence = cleanTranscriptText(evidence).toLowerCase();
  const normalizedPatientText = cleanTranscriptText(patientText).toLowerCase();
  return Boolean(normalizedEvidence) && (normalizedPatientText.includes(normalizedEvidence) || normalizedEvidence.includes(normalizedPatientText));
}

function safeguardHighlightsForCall(call: CallRecord, segments: TranscriptSegment[], patientSegments: { segment: TranscriptSegment; segmentIndex: number }[]): TranscriptHighlight[] {
  const level = safeguardLevel(call);
  if (level === "None") return [];
  const severity: RiskLevel = level === "Emergency" ? "Red" : level === "Urgent" ? "Amber" : "Watch";
  return (call.safeguardEvidence ?? []).flatMap((evidence, index) => {
    const match = patientSegments.find(({ segment }) => evidenceMatchesPatientText(evidence, textWithoutSpeakerLabel(segment.englishText || segment.text)));
    if (!match) return [];
    return [
      {
        id: `safeguard-${index}`,
        kind: "safeguard",
        label: safeguardTitle(call),
        text: evidence,
        severity,
        startTimeSeconds: previousAgentReplyEnd(segments, match.segmentIndex) ?? match.segment.startTimeSeconds,
        endTimeSeconds: match.segment.endTimeSeconds,
        sentenceIndex: patientSegments.findIndex((item) => item.segmentIndex === match.segmentIndex),
        transcriptSegmentIndex: match.segmentIndex
      }
    ];
  });
}

function emotionHighlightsForCall(call: CallRecord): TranscriptHighlight[] {
  return (call.emotionSegments ?? []).map((emotion) => ({
    id: emotion.id,
    kind: "emotion",
    label: `Tone: ${emotion.label}`,
    text: emotion.evidenceText || emotion.label,
    severity: call.emotionConcernLevel === "Review" ? "Watch" : "Green",
    startTimeSeconds: emotion.startTimeSeconds,
    endTimeSeconds: emotion.endTimeSeconds,
    transcriptSegmentIndex: emotion.transcriptSegmentIndex
  }));
}

function highlightChipText(highlight: TranscriptHighlight): string | null {
  const timestamp = typeof highlight.startTimeSeconds === "number" ? formatTimestamp(highlight.startTimeSeconds) : null;
  if (highlight.kind === "emotion") {
    return timestamp ? `${highlight.label} ${timestamp}` : highlight.label;
  }
  return timestamp;
}

function HighlightedEnglishTranscript({
  call,
  highlightedSignalId,
  onSelectHighlight
}: {
  call: CallRecord;
  highlightedSignalId: string | null;
  onSelectHighlight: (highlight: TranscriptHighlight) => void;
}) {
  const segments = getEnglishTranscriptSegments(call);
  const patientSegments = segments
    .map((segment, segmentIndex) => ({ segment, segmentIndex }))
    .filter(({ segment }) => isPatientSegment(segment));
  const riskHighlights: TranscriptHighlight[] = (call.riskSignals ?? []).map((signal) => ({
    id: signal.id,
    kind: "risk",
    label: signal.label,
    text: signal.highlightText || signal.quotedText,
    severity: signal.severity,
    startTimeSeconds: signal.startTimeSeconds,
    endTimeSeconds: signal.endTimeSeconds,
    sentenceIndex: signal.sentenceIndex
  }));
  const emotionHighlights = emotionHighlightsForCall(call);
  const highlights = [...safeguardHighlightsForCall(call, segments, patientSegments), ...emotionHighlights, ...riskHighlights];

  return (
    <div className="highlighted-transcript">
      {segments.map((segment, segmentIndex) => {
        const text = displaySegmentText(segment);
        const patientIndex = patientSegments.findIndex((item) => item.segmentIndex === segmentIndex);
        const patientOnly = patientIndex >= 0;
        const matches = highlights.filter((highlightItem) => {
          const highlight = cleanTranscriptText(highlightItem.text);
          if (!patientOnly || !highlight) return false;
          const lowerHighlight = highlight.toLowerCase();
          const patientText = textWithoutSpeakerLabel(text).toLowerCase();
          const exactPatientMatchExists = patientSegments.some((item) =>
            textWithoutSpeakerLabel(item.segment.englishText || item.segment.text).toLowerCase().includes(lowerHighlight)
          );
          return (
            patientText.includes(lowerHighlight) ||
            highlightItem.transcriptSegmentIndex === segmentIndex ||
            (!exactPatientMatchExists && highlightItem.sentenceIndex === patientIndex)
          );
        });

        if (!matches.length || !text) {
          return <p key={`${call.id}-segment-${segmentIndex}`}>{text}</p>;
        }

        const highlightItem = matches[0];
        const highlightWithSentenceTime: TranscriptHighlight = {
          ...highlightItem,
          startTimeSeconds: highlightItem.startTimeSeconds ?? previousAgentReplyEnd(segments, segmentIndex) ?? segment.startTimeSeconds,
          endTimeSeconds: highlightItem.endTimeSeconds ?? segment.endTimeSeconds,
          sentenceIndex: patientIndex >= 0 ? patientIndex : highlightItem.sentenceIndex
        };
        const highlight = cleanTranscriptText(highlightItem.text);
        const patientText = textWithoutSpeakerLabel(text);
        const patientTextStart = text.indexOf(patientText);
        const matchIndexInPatientText = patientText.toLowerCase().indexOf(highlight.toLowerCase());
        const matchIndex = matchIndexInPatientText >= 0 ? Math.max(0, patientTextStart) + matchIndexInPatientText : -1;
        const activeId = `${call.id}-${highlightItem.id}`;
        if (matchIndex < 0) {
          return (
            <p key={`${call.id}-segment-${segmentIndex}`}>
              <button
                className={`transcript-highlight transcript-highlight-${highlightItem.kind} ${highlightedSignalId === activeId ? "active" : ""}`}
                id={`signal-${activeId}`}
                onClick={() => onSelectHighlight(highlightWithSentenceTime)}
              >
                {text}
                {highlightChipText(highlightWithSentenceTime) ? <span>{highlightChipText(highlightWithSentenceTime)}</span> : null}
              </button>
            </p>
          );
        }

        const before = text.slice(0, matchIndex);
        const matched = text.slice(matchIndex, matchIndex + highlight.length);
        const after = text.slice(matchIndex + highlight.length);
        return (
          <p key={`${call.id}-segment-${segmentIndex}`}>
            {before}
            <button
              className={`transcript-highlight transcript-highlight-${highlightItem.kind} ${highlightedSignalId === activeId ? "active" : ""}`}
              id={`signal-${activeId}`}
              onClick={() => onSelectHighlight(highlightWithSentenceTime)}
            >
              {matched}
              {highlightChipText(highlightWithSentenceTime) ? <span>{highlightChipText(highlightWithSentenceTime)}</span> : null}
            </button>
            {after}
          </p>
        );
      })}
    </div>
  );
}

function formatMetric(value: number | undefined | null, suffix = ""): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return "Not enough data";
  return `${Math.round(value)}${suffix}`;
}

function formatSeconds(value: number | undefined | null): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return "Not enough data";
  return value >= 10 ? `${Math.round(value)}s` : `${value.toFixed(1)}s`;
}

function formatTimestamp(value: number): string {
  const totalSeconds = Math.max(0, Math.floor(value));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function formatPercent(value: number | undefined | null): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "Not enough data";
  return `${Math.round(value * 100)}%`;
}

function parkinsonsFeaturesSummary(call: CallRecord | null): Record<string, number | string | null> | null | undefined {
  return call?.parkinsonsSpeechReview?.featuresSummary ?? call?.speechModelFeaturesSummary;
}

function parkinsonsWarnings(call: CallRecord | null): string[] {
  return call?.parkinsonsSpeechReview?.warnings ?? call?.speechModelWarnings ?? [];
}

function parkinsonsProbability(call: CallRecord | null): number | null | undefined {
  return call?.parkinsonsSpeechReview?.probability ?? call?.speechModelProbability;
}

function parkinsonsModelVersion(call: CallRecord): string | null | undefined {
  return call.parkinsonsSpeechReview?.modelVersion ?? call.speechModelVersion;
}

function modelFeatureNumber(call: CallRecord | null, key: string): number | null {
  const value = parkinsonsFeaturesSummary(call)?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function averageProfiles(profiles: NonNullable<CallRecord["currentSpeechProfile"]>[]): NonNullable<CallRecord["currentSpeechProfile"]> | null {
  if (!profiles.length) return null;
  const sum = profiles.reduce(
    (totals, profile) => ({
      speechRate: totals.speechRate + (profile.speechRate || 0),
      avgPauseMs: totals.avgPauseMs + (profile.avgPauseMs || 0),
      responseLatencyMs: totals.responseLatencyMs + (profile.responseLatencyMs || 0),
      pitchVariability: totals.pitchVariability + (profile.pitchVariability || 0),
      phraseAccuracy: totals.phraseAccuracy + (profile.phraseAccuracy || 0)
    }),
    { speechRate: 0, avgPauseMs: 0, responseLatencyMs: 0, pitchVariability: 0, phraseAccuracy: 0 }
  );
  return {
    speechRate: sum.speechRate / profiles.length,
    avgPauseMs: sum.avgPauseMs / profiles.length,
    responseLatencyMs: sum.responseLatencyMs / profiles.length,
    pitchVariability: sum.pitchVariability / profiles.length,
    phraseAccuracy: sum.phraseAccuracy / profiles.length,
    updatedAt: profiles[0]?.updatedAt ?? new Date().toISOString()
  };
}

function baselineFromCalls(senior: Senior, calls: CallRecord[]): { profile: Senior["baselineSpeechProfile"]; source: string } {
  const profiles = calls
    .filter((call) => call.currentSpeechProfile)
    .slice(0, 5)
    .map((call) => call.currentSpeechProfile)
    .filter((profile): profile is NonNullable<CallRecord["currentSpeechProfile"]> => Boolean(profile));
  const averaged = averageProfiles(profiles);
  if (averaged) {
    return {
      profile: { ...senior.baselineSpeechProfile, ...averaged },
      source: profiles.length >= 5 ? "Average of latest 5 recordings" : `Average of ${profiles.length} recording${profiles.length === 1 ? "" : "s"}`
    };
  }
  return { profile: senior.baselineSpeechProfile, source: "Default baseline until recordings are available" };
}

function averageModelFeature(calls: CallRecord[], key: string): number | null {
  const values = calls
    .slice(0, 5)
    .map((call) => modelFeatureNumber(call, key))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0);
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function averageSpeechCoverage(calls: CallRecord[]): number | null {
  const values = calls
    .slice(0, 5)
    .map((item) => {
      const speechSeconds = modelFeatureNumber(item, "patientSpeechDurationSeconds");
      const rawSeconds = modelFeatureNumber(item, "rawPatientAudioDurationSeconds");
      return speechSeconds && rawSeconds ? speechSeconds / rawSeconds : null;
    })
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0);
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function modelReadiness(call: CallRecord | null): { value: string; note: string } {
  if (!call) return { value: "No call", note: "Complete a call first" };
  const summary = parkinsonsFeaturesSummary(call);
  const warnings = parkinsonsWarnings(call);
  const warningText = warnings.join(" ").toLowerCase();
  const usable = summary?.speechModelUsable;
  if (warningText.includes("low confidence")) {
    return { value: "Low confidence", note: warnings.find((warning) => warning.toLowerCase().includes("low confidence")) ?? "Review audio quality" };
  }
  const probability = parkinsonsProbability(call);
  if (probability !== null && probability !== undefined && usable !== "false") {
    return { value: "Usable", note: `${Math.round(probability * 100)}% Parkinson marker probability` };
  }
  if (usable === "false" || warningText.includes("unavailable")) {
    return { value: "Unavailable", note: warnings.find((warning) => warning.toLowerCase().includes("unavailable")) ?? "Not enough patient speech" };
  }
  if (call.patientSpeechAudioAvailable) {
    return { value: "Ready", note: "Saved model will run after call save" };
  }
  return { value: "Missing", note: "Not enough patient speech" };
}

function concussionReviewReadiness(call: CallRecord | null): { value: string; note: string } {
  if (!call?.concussionSpeechReview) return { value: "Not run", note: "No concussion speech review result" };
  const review = call.concussionSpeechReview;
  const abnormalProbability = Math.max(
    review.probabilities?.dysarthria_like ?? 0,
    review.probabilities?.dysphonia_like ?? 0
  );
  const probabilityNote =
    abnormalProbability > 0
      ? `${Math.round(abnormalProbability * 100)}% abnormal-class probability`
      : typeof review.probabilities?.normal === "number"
        ? `${Math.round(review.probabilities.normal * 100)}% normal probability`
        : null;

  if (review.failureReason) {
    return { value: "Unavailable", note: review.failureReason };
  }
  if (!review.qualityOk) {
    return { value: "Low audio quality", note: review.qualityReason || probabilityNote || "Patient speech did not pass quality checks" };
  }
  return { value: "Usable", note: probabilityNote || review.riskReason || "Patient speech passed concussion review quality checks" };
}

function SpeechTimingPanel({ senior, call, calls }: { senior: Senior; call: CallRecord | null; calls: CallRecord[] }) {
  const priorCalls = call ? calls.filter((item) => item.id !== call.id) : calls;
  const baselineState = baselineFromCalls(senior, priorCalls);
  const baseline = baselineState.profile;
  const current = call?.currentSpeechProfile ?? null;
  const patientSpeechSeconds = modelFeatureNumber(call, "patientSpeechDurationSeconds");
  const rawPatientSeconds = modelFeatureNumber(call, "rawPatientAudioDurationSeconds");
  const speechCoverage = patientSpeechSeconds && rawPatientSeconds ? patientSpeechSeconds / rawPatientSeconds : null;
  const baselinePatientSpeech = averageModelFeature(priorCalls, "patientSpeechDurationSeconds");
  const baselineSpeechCoverage = averageSpeechCoverage(priorCalls);
  const readiness = modelReadiness(call);
  const concussionReadiness = concussionReviewReadiness(call);
  const rows = [
    { label: "Patient speech", baseline: baselinePatientSpeech ? formatSeconds(baselinePatientSpeech) : "No baseline", current: formatSeconds(patientSpeechSeconds) },
    {
      label: "Speech coverage",
      baseline: baselineSpeechCoverage !== null ? `${Math.round(baselineSpeechCoverage * 100)}% avg` : "No baseline",
      current: formatPercent(speechCoverage)
    },
    { label: "Response latency", baseline: `${Math.round(baseline.responseLatencyMs)} ms`, current: formatMetric(current?.responseLatencyMs, " ms") },
    { label: "Speaking rate", baseline: `${Math.round(baseline.speechRate)} wpm`, current: formatMetric(current?.speechRate, " wpm") },
    { label: "Parkinson model", baseline: readiness.note, current: readiness.value },
    { label: "Concussion review", baseline: concussionReadiness.note, current: concussionReadiness.value }
  ];

  return (
    <section className="speech-timing-panel">
      <div className="panel-heading compact-heading">
        <h3>Patient speech quality</h3>
        <span>{baselineState.source}</span>
      </div>
      <div className="speech-metric-grid">
        {rows.map((row) => (
          <div className="speech-metric" key={row.label}>
            <span>{row.label}</span>
            <strong>{row.current}</strong>
            <small>Baseline {row.baseline}</small>
          </div>
        ))}
      </div>
      {!parkinsonsFeaturesSummary(call) && !call?.patientSpeechAudioAvailable ? (
        <p className="metric-note">
          Not enough patient speech yet. Keep patient-only recording enabled and let the agent ask short turn-by-turn questions so patient answers can be isolated cleanly.
        </p>
      ) : null}
    </section>
  );
}

function parkinsonsSpeechTitle(call: CallRecord): string {
  const probability = parkinsonsProbability(call);
  if (probability !== null && probability !== undefined) {
    return `Parkinson marker probability: ${Math.round(probability * 100)}%`;
  }
  const review = call.parkinsonsSpeechReview;
  const warnings = parkinsonsWarnings(call).join(" ").toLowerCase();
  if (review?.failureReason) return "Parkinson model unavailable";
  if (warnings.includes("low confidence")) return "Parkinson model low confidence";
  if (warnings.includes("unavailable")) return "Parkinson model unavailable";
  return call.patientAudioAvailable ? "Parkinson model ready" : "Parkinson model unavailable";
}

function parkinsonsSpeechDescription(call: CallRecord): string {
  const probability = parkinsonsProbability(call);
  if (probability !== null && probability !== undefined) {
    return `${parkinsonsModelVersion(call) ?? "saved Parkinson voice-feature model"} scored patient-only pitch, jitter, and noise features. This is saved-model inference, not a diagnosis.`;
  }
  if (call.parkinsonsSpeechReview?.failureReason) {
    return call.parkinsonsSpeechReview.failureReason;
  }
  const warnings = parkinsonsWarnings(call);
  if (warnings.length) {
    return warnings.join(" ");
  }
  return call.patientAudioAvailable
    ? "Patient-only audio is saved; the saved Parkinson model scores derived patient speech after call save."
    : "Patient-only audio is not available for Parkinson voice-feature scoring.";
}

function concussionSpeechTitle(call: CallRecord): string {
  const review = call.concussionSpeechReview;
  if (!review) return "Not run";
  if (review.failureReason) return "Unavailable";
  if (!review.qualityOk) return "Low audio quality";
  if (review.predictedLabel && review.predictedLabel !== "normal") {
    return `Speech review: ${review.predictedLabel.replace(/_/g, " ")}`;
  }
  return "No abnormal speech flag";
}

function concussionSpeechDescription(call: CallRecord): string {
  const review = call.concussionSpeechReview;
  if (!review) return "Concussion speech-abnormality review has not returned a result for this call.";
  if (review.failureReason) return review.failureReason;
  if (!review.qualityOk) return review.qualityReason || "Patient speech audio did not pass quality checks.";
  if (review.riskReason) return `${review.riskReason} Research-only; not a diagnosis.`;
  const normalProbability = review.probabilities?.normal;
  if (typeof normalProbability === "number") {
    return `${Math.round(normalProbability * 100)}% normal probability. Research-only; not a diagnosis.`;
  }
  return review.warning;
}

function aiReviewNote(call: CallRecord): string | null {
  if (!call.aiRiskFallbackUsed) return null;
  if (call.aiRiskFailureReason) return `Manual review required: ${call.aiRiskFailureReason}`;
  return call.riskSignals?.length
    ? "AI review used fallback logic; highlights may be incomplete."
    : "Manual review required; AI risk highlights are unavailable for this call.";
}

function safeguardLevel(call: CallRecord): SafeguardLevel {
  return call.safeguardLevel ?? "None";
}

function safeguardTitle(call: CallRecord): string {
  const level = safeguardLevel(call);
  if (call.safeguardFailureReason) return "Safeguard review unavailable";
  if (!call.safeguardReviewAvailable) return "Safeguard review unavailable";
  if (level === "None") return "No distress safeguard flagged";
  return `${level} safeguard flagged`;
}

function safeguardDescription(call: CallRecord): string {
  if (call.safeguardFailureReason) return `Manual review required: ${call.safeguardFailureReason}`;
  if (!call.safeguardReviewAvailable) return "Manual review required; safeguard classification is unavailable for this call.";
  const level = safeguardLevel(call);
  if (level === "None") return "OpenAI safeguard review did not find patient-stated distress requiring hotline guidance.";
  const category = call.safeguardCategory ? call.safeguardCategory.replace(/_/g, " ") : "distress";
  return call.safeguardRecommendedAction || `Patient-stated ${category} should receive human follow-up.`;
}

function safeguardResourceText(call: CallRecord): string | null {
  const resources = call.safeguardResources ?? [];
  if (!resources.length) return null;
  return resources
    .map((resource) => `${resource.name}: ${resource.phone || resource.text || resource.url || resource.description}`)
    .join("; ");
}

function visibleAssessmentReasons(call: CallRecord): string[] {
  return call.riskAssessment.reasons.filter((reason) => !reason.toLowerCase().startsWith("safeguard review flagged"));
}

function callReviewStatus(call: CallRecord): string {
  if (call.aiRiskFallbackUsed || call.safeguardFailureReason) return "Needs manual review";
  if (safeguardLevel(call) !== "None") return safeguardTitle(call);
  return call.riskSignals?.length ? "Clinical risk cues found" : "No urgent cue found";
}

function callRecordingSummary(call: CallRecord): string {
  const parts = [
    call.audioAvailable ? "recording saved" : "no recording",
    call.patientAudioAvailable ? "patient audio saved" : "patient audio missing",
    call.agentAudioCaptured ? "agent voice captured" : "agent voice not confirmed"
  ];
  return parts.join(" · ");
}

function emotionToneSummary(call: CallRecord): string | null {
  if (call.emotionReviewAvailable && call.dominantPatientEmotion) {
    const provider = call.emotionProvider ? ` via ${call.emotionProvider}` : "";
    const tagNote = call.emotionSegments?.length ? "" : " No per-response tone tags were returned.";
    return `Tone: ${call.dominantPatientEmotion}${call.emotionConcernLevel && call.emotionConcernLevel !== "None" ? ` (${call.emotionConcernLevel.toLowerCase()})` : ""}${provider}.${tagNote}`;
  }
  if (call.emotionFailureReason) {
    return `Tone unavailable: ${call.emotionFailureReason}`;
  }
  return null;
}

function AgentsCall({
  seniors,
  selectedSeniorId,
  onSelectSenior,
  onSavedCall
}: {
  seniors: Senior[];
  selectedSeniorId: string;
  onSelectSenior: (id: string) => void;
  onSavedCall: (call: CallRecord) => void;
}) {
  const selectedSenior = seniors.find((senior) => senior.id === selectedSeniorId) ?? seniors[0];
  const [callState, setCallState] = useState<CallState>("Ready");
  const [callMessage, setCallMessage] = useState("Ready to simulate a phone call through the website.");
  const [debugMessage, setDebugMessage] = useState("");
  const [audioCleanupWarning, setAudioCleanupWarning] = useState("");
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const recorderRef = useRef<WavRecorder | null>(null);
  const patientRecorderRef = useRef<WavRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const recordingInputRef = useRef<GainNode | null>(null);
  const agentAudioFormatRef = useRef<AgentAudioFormat>("pcm_16000");
  const agentPlaybackTimeRef = useRef(0);
  const agentAudioCapturedRef = useRef(false);
  const agentAudioWarningShownRef = useRef(false);
  const elevenLabsConversationIdRef = useRef<string | null>(null);
  const transcriptRef = useRef<TranscriptMessage[]>([]);

  const conversation = useConversation({
    onConnect: () => {
      setCallState("In call");
      setCallMessage("Agent connected. Waiting for the first agent message...");
      conversation.sendContextualUpdate(multilingualAgentPrompt(selectedSenior));
      window.setTimeout(() => {
        const hasAgentMessage = transcriptRef.current.some((line) => line.role === "Agent");
        if (!hasAgentMessage) {
          conversation.sendUserMessage(
            `Begin the EarlyCare check-in for ${selectedSenior.name}. The patient may answer in ${selectedSenior.preferredLanguage}, English, Mandarin, Malay, Tamil, Singlish, or a mix. Detect their language and continue in the language they use.`
          );
          setCallMessage("Agent was silent, so EarlyCare nudged the session to begin.");
        }
      }, 2200);
    },
    onDisconnect: (details) => {
      const reason = JSON.stringify(details);
      setDebugMessage(reason);
      setCallState((current) => (current === "In call" || current === "Connecting" ? "Failed" : current));
      setCallMessage(`Agents session ended. ${reason}`);
    },
    onError: (message, context) => {
      setCallState("Failed");
      setCallMessage(message);
      setDebugMessage(context ? JSON.stringify(context) : "");
    },
    onStatusChange: ({ status }) => {
      setDebugMessage((previous) => `SDK status changed to ${status}${previous ? ` | ${previous}` : ""}`);
    },
    onDebug: (debug) => {
      setDebugMessage(JSON.stringify(debug));
    },
    onConversationMetadata: (metadata) => {
      agentAudioFormatRef.current = metadata.agent_output_audio_format as AgentAudioFormat;
      const conversationId = (metadata as { conversation_id?: string; conversationId?: string }).conversation_id ?? (metadata as { conversationId?: string }).conversationId;
      elevenLabsConversationIdRef.current = conversationId ?? elevenLabsConversationIdRef.current;
    },
    onAudio: (base64Audio) => {
      const audioContext = audioContextRef.current;
      const recordingInput = recordingInputRef.current;
      if (!audioContext || !recordingInput) return;
      try {
        const audioBuffer = createAgentAudioBuffer(audioContext, base64Audio, agentAudioFormatRef.current);
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(recordingInput);
        const startAt = Math.max(audioContext.currentTime, agentPlaybackTimeRef.current);
        source.start(startAt);
        agentPlaybackTimeRef.current = startAt + audioBuffer.duration;
        agentAudioCapturedRef.current = true;
      } catch (error) {
        if (!agentAudioWarningShownRef.current) {
          agentAudioWarningShownRef.current = true;
          setDebugMessage(error instanceof Error ? error.message : "Unable to mix agent audio into recording.");
        }
      }
    },
    onMessage: (message) => {
      const line: TranscriptMessage = {
        role: message.role === "agent" ? "Agent" : "Senior",
        text: cleanTranscriptText(message.message),
        timestamp: new Date().toISOString()
      };
      transcriptRef.current = [...transcriptRef.current, line];
      if (line.role === "Senior" && line.text) {
        conversation.sendContextualUpdate(nextReplyLanguageInstruction(line.text));
      }
    }
  });

  const startCall = async () => {
    setCallState("Connecting");
    setCallMessage("Connecting to Agents...");
    setDebugMessage("");
    setAudioCleanupWarning("");
    transcriptRef.current = [];
    elevenLabsConversationIdRef.current = null;
    const callStartedAt = new Date().toISOString();
    setStartedAt(callStartedAt);

    try {
      const preparedMic = await requestCleanMicrophoneStream();
      const stream = preparedMic.stream;
      streamRef.current = stream;
      if (preparedMic.warning) {
        setAudioCleanupWarning(preparedMic.warning);
      }
      const AudioContextConstructor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextConstructor) throw new Error("Browser audio recording is not supported.");
      const audioContext = new AudioContextConstructor();
      const recordingInput = audioContext.createGain();
      const microphoneSource = audioContext.createMediaStreamSource(stream);
      const patientOnlyInput = audioContext.createGain();
      microphoneSource.connect(recordingInput);
      microphoneSource.connect(patientOnlyInput);
      audioContextRef.current = audioContext;
      recordingInputRef.current = recordingInput;
      agentPlaybackTimeRef.current = audioContext.currentTime;
      agentAudioCapturedRef.current = false;
      agentAudioWarningShownRef.current = false;
      agentAudioFormatRef.current = "pcm_16000";
      recorderRef.current = createWavRecorder(audioContext, recordingInput);
      patientRecorderRef.current = createWavRecorder(audioContext, patientOnlyInput);

      const session = await createElevenLabsSession({
        seniorId: selectedSenior.id,
        seniorName: selectedSenior.name,
        preferredLanguage: selectedSenior.preferredLanguage,
        caregiverContact: selectedSenior.caregiverContact,
        checkInReason: "Routine living-alone wellbeing check-in"
      });

      if (!session.configured || !session.signedUrl) {
        const fallbackLine: TranscriptMessage = {
          role: "System",
          text: `${session.message} No Agents session was started.`,
          timestamp: new Date().toISOString()
        };
        transcriptRef.current = [fallbackLine];
        setCallState("Failed");
        setCallMessage("Agents unavailable. Transcript/audio can still be saved if needed.");
        return;
      }

      await conversation.startSession({
        signedUrl: session.signedUrl,
        onConnect: ({ conversationId }) => {
          elevenLabsConversationIdRef.current = conversationId;
        },
        dynamicVariables: {
          senior_name: selectedSenior.name,
          preferred_language: selectedSenior.preferredLanguage,
          living_alone: selectedSenior.livingAlone,
          caregiver_contact: selectedSenior.caregiverContact,
          check_in_reason: "Routine living-alone wellbeing check-in",
          baseline_reminder: "Use EarlyCare's stored personal baseline context without saying exact baseline metrics aloud.",
          crisis_resources: singaporeCrisisResources.map((resource) => `${resource.name}: ${resource.phone || resource.text || resource.url}`).join("; ")
        }
      });
    } catch (error) {
      setCallState("Failed");
      setCallMessage(error instanceof Error ? error.message : "Unable to start call.");
    }
  };

  const endAndSaveCall = async () => {
    setCallState("Saving");
    setCallMessage("Ending call and saving transcript/audio...");
    if (conversation.status === "connected" || conversation.status === "connecting") {
      conversation.endSession();
    }
    const audioBlob = await stopRecorder(recorderRef.current);
    const patientAudioBlob = await stopRecorder(patientRecorderRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
    patientRecorderRef.current = null;

    const formData = new FormData();
    formData.append("seniorId", selectedSenior.id);
    formData.append("status", "Complete");
    formData.append("startedAt", startedAt ?? new Date().toISOString());
    formData.append("completedAt", new Date().toISOString());
    formData.append("transcriptMessages", JSON.stringify(transcriptRef.current));
    if (elevenLabsConversationIdRef.current) {
      formData.append("elevenLabsConversationId", elevenLabsConversationIdRef.current);
    }
    formData.append("agentAudioCaptured", String(agentAudioCapturedRef.current));
    audioContextRef.current?.close();
    audioContextRef.current = null;
    recordingInputRef.current = null;
    if (audioBlob) formData.append("audio", audioBlob, "full-call.wav");
    if (patientAudioBlob) formData.append("patientAudio", patientAudioBlob, "patient-audio.wav");

    setCallState("Analysing");
    const saved = await saveCall(formData);
    if (saved.ok) {
      onSavedCall(saved.call);
      setCallState("Complete");
      setCallMessage("Call saved. Patient overview now shows original and English transcripts.");
    } else {
      setCallState("Failed");
      setCallMessage(`Call ended, but saving to backend failed: ${saved.message}`);
    }
  };

  const busy = callState === "Connecting" || callState === "Saving" || callState === "Analysing";
  const canEndAndSave = callState === "In call" || callState === "Failed";
  const callVisualState = conversation.isSpeaking ? "speaking" : conversation.isListening ? "listening" : callState.toLowerCase().replace(/\s+/g, "-");

  return (
    <main className="call-only-grid">
      <section className="panel call-panel">
        <div className="panel-heading">
          <h2>Agents Website Call</h2>
          <span>Simulated phone call</span>
        </div>

        <div className="senior-selector">
          {seniors.map((senior) => (
            <button className={senior.id === selectedSenior.id ? "active" : ""} key={senior.id} onClick={() => onSelectSenior(senior.id)}>
              {senior.name}
            </button>
          ))}
        </div>

        <div className="phone-shell">
          <div className="phone-top">
            <div>
              <strong>{selectedSenior.name}</strong>
              <small>
                {selectedSenior.age} · {selectedSenior.addressZone} · {selectedSenior.preferredLanguage}
              </small>
            </div>
            <span className="live-dot">{callState}</span>
          </div>

          <div className={`voice-visual ${callVisualState}`} aria-label="Live call voice visual">
            <div className="voice-orb">
              <span />
              <span />
              <span />
            </div>
            <button
              aria-label={canEndAndSave ? "End and save call" : "Start call"}
              className="call-orb-button"
              disabled={busy}
              onClick={() => void (canEndAndSave ? endAndSaveCall() : startCall())}
              type="button"
            >
              <PhoneCall size={28} />
            </button>
          </div>

          <div className="answer-box">
            <p>{callMessage}</p>
            {debugMessage ? <small>Debug: {debugMessage}</small> : null}
            {audioCleanupWarning ? <small>{audioCleanupWarning}</small> : null}
            <small>
              SDK status: {conversation.status}. Mode: {conversation.mode}. {conversation.isSpeaking ? "Agent speaking." : conversation.isListening ? "Listening." : ""}
            </small>
          </div>

          <p className="call-action-hint">{canEndAndSave ? "Tap the call button to end and save." : busy ? "EarlyCare is preparing the call." : "Tap the call button to start."}</p>
        </div>
      </section>
    </main>
  );
}

function OfficerDashboard({
  seniors,
  sessions,
  tasks,
  calls,
  selectedSeniorId,
  setSelectedSeniorId,
  onStartCall
}: {
  seniors: Senior[];
  sessions: CheckInSession[];
  tasks: VolunteerTask[];
  calls: CallRecord[];
  selectedSeniorId: string;
  setSelectedSeniorId: (id: string) => void;
  onStartCall: (id: string) => void;
}) {
  const selectedSenior = seniors.find((senior) => senior.id === selectedSeniorId) ?? seniors[0];
  const selectedTasks = tasks.filter((task) => task.seniorId === selectedSenior.id);
  const selectedCalls = calls.filter((call) => call.seniorId === selectedSenior.id);
  const latestSignal = selectedCalls[0] ?? null;
  const latestAssessment = latestSignal?.riskAssessment ?? null;
  const audioRefs = useRef<Record<string, HTMLAudioElement | null>>({});
  const [highlightedSignalId, setHighlightedSignalId] = useState<string | null>(null);
  const highestRisk = selectedCalls.map((call) => call.riskLevel).reduce<RiskLevel>(
    (risk, level) => (riskOrder[level] > riskOrder[risk] ? level : risk),
    "Green"
  );
  const playTranscriptHighlight = (call: CallRecord, highlight: TranscriptHighlight) => {
    setHighlightedSignalId(`${call.id}-${highlight.id}`);
    const audio = audioRefs.current[call.id];
    if (audio && typeof highlight.startTimeSeconds === "number") {
      audio.currentTime = Math.max(0, highlight.startTimeSeconds);
      void audio.play();
      return;
    }
    document.getElementById(`signal-${call.id}-${highlight.id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <main className="dashboard-grid">
      <section className="panel roster-panel">
        <div className="panel-heading">
          <h2>Living-Alone Roster</h2>
          <span>{seniors.length} seniors</span>
        </div>
        <div className="senior-list">
          {seniors.map((senior) => (
            <button
              className={`senior-row ${senior.id === selectedSenior.id ? "active" : ""}`}
              key={senior.id}
              onClick={() => setSelectedSeniorId(senior.id)}
            >
              <span>
                <strong>{senior.name}</strong>
                <small>
                  {senior.age} · {senior.addressZone} · {senior.preferredLanguage}
                </small>
              </span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel detail-panel">
        <div className="profile-header">
          <div>
            <span className="eyebrow">Patient overview</span>
            <h2>{selectedSenior.name}</h2>
            <p>
              Lives alone in {selectedSenior.addressZone}. Check-in every {selectedSenior.checkInFrequencyDays} days.
            </p>
          </div>
          <div className="profile-actions">
            <RiskBadge level={highestRisk} />
            <button className="primary-action" onClick={() => onStartCall(selectedSenior.id)}>
              <PhoneCall size={18} />
              Start new call
            </button>
          </div>
        </div>

        <div className="metric-grid">
          <StatCard label="Language" value={selectedSenior.preferredLanguage} icon={<Languages size={20} />} />
          <StatCard label="Caregiver" value={selectedSenior.caregiverContact} icon={<UserRoundCheck size={20} />} />
        </div>

        <SpeechTimingPanel senior={selectedSenior} call={latestSignal} calls={selectedCalls} />

        <section className="analysis-panel dashboard-analysis safety-snapshot">
          <div className="analysis-header">
            <div>
              <span className="eyebrow">Latest call</span>
              <h2>Safety Snapshot</h2>
            </div>
            <RiskBadge level={latestSignal?.riskLevel ?? "Green"} />
          </div>
          {latestSignal && latestAssessment ? (
            <>
              <div className="snapshot-lead">
                <div>
                  <strong>{callReviewStatus(latestSignal)}</strong>
                  <p>{latestSignal.recommendedAction}</p>
                </div>
                <small>{new Date(latestSignal.completedAt).toLocaleString()}</small>
              </div>
              <div className="snapshot-grid">
                <div className="snapshot-card">
                  <Brain size={22} />
                  <div>
                    <span>Clinical review</span>
                    <strong>{latestSignal.aiRiskFallbackUsed ? "Manual review" : `${latestSignal.riskSignals?.length ?? 0} cue${(latestSignal.riskSignals?.length ?? 0) === 1 ? "" : "s"}`}</strong>
                    {aiReviewNote(latestSignal) ? <small>{aiReviewNote(latestSignal)}</small> : null}
                  </div>
                </div>
                <div className={`snapshot-card safeguard-card safeguard-${safeguardLevel(latestSignal).toLowerCase()}`}>
                  <ShieldCheck size={22} />
                  <div>
                    <span>Distress safeguard</span>
                    <strong>{safeguardTitle(latestSignal)}</strong>
                    <small>{safeguardDescription(latestSignal)}</small>
                  </div>
                </div>
                <div className="snapshot-card">
                  <Activity size={22} />
                  <div>
                    <span>Parkinson speech marker</span>
                    <strong>{parkinsonsSpeechTitle(latestSignal)}</strong>
                    <small>{parkinsonsSpeechDescription(latestSignal)}</small>
                  </div>
                </div>
                <div className="snapshot-card">
                  <AlertTriangle size={22} />
                  <div>
                    <span>Concussion speech review</span>
                    <strong>{concussionSpeechTitle(latestSignal)}</strong>
                    <small>{concussionSpeechDescription(latestSignal)}</small>
                  </div>
                </div>
              </div>
              {visibleAssessmentReasons(latestSignal).length ? (
                <div className="reason-box compact-reasons">
                  {visibleAssessmentReasons(latestSignal).slice(0, 3).map((reason) => (
                    <p key={reason}>{reason}</p>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <p className="empty-state">No risk assessment is available for this senior yet.</p>
          )}
        </section>

        <section className="saved-calls">
          <div className="section-title-row">
            <div>
              <span className="eyebrow">Call archive</span>
              <h3>Saved Agent Calls</h3>
            </div>
            <small>{selectedCalls.length} saved</small>
          </div>
          {selectedCalls.length ? (
            <div className="call-record-list">
              {selectedCalls.map((call) => {
                const audioUrl = getCallAudioUrl(call);
                return (
                  <article className="call-record" key={call.id}>
                    <div className="call-record-header">
                      <div>
                        <RiskBadge level={call.riskLevel} />
                        <strong>{new Date(call.completedAt).toLocaleString()}</strong>
                        <small>{callReviewStatus(call)}</small>
                      </div>
                      <small>{callRecordingSummary(call)}</small>
                    </div>
                    <p className="call-action-summary">{call.recommendedAction}</p>
                    {emotionToneSummary(call) ? <p className="metric-note">{emotionToneSummary(call)}</p> : null}
                    {aiReviewNote(call) ? <p className="metric-note">{aiReviewNote(call)}</p> : null}
                    {safeguardLevel(call) !== "None" || call.safeguardFailureReason ? (
                      <div className={`safeguard-summary safeguard-${safeguardLevel(call).toLowerCase()}`}>
                        <strong>{safeguardTitle(call)}</strong>
                        <p>{safeguardDescription(call)}</p>
                        {safeguardResourceText(call) ? <small>{safeguardResourceText(call)}</small> : null}
                      </div>
                    ) : null}

                    <div className="recording-player">
                      <h4>Original recording</h4>
                      {audioUrl ? (
                        <audio
                          controls
                          ref={(node) => {
                            audioRefs.current[call.id] = node;
                          }}
                          src={audioUrl}
                        />
                      ) : (
                        <p className="empty-state">No recording is available for this call.</p>
                      )}
                    </div>

                    <div className="transcript-stack">
                      <div>
                        <h4>English transcript</h4>
                        <HighlightedEnglishTranscript
                          call={call}
                          highlightedSignalId={highlightedSignalId}
                          onSelectHighlight={(highlight) => playTranscriptHighlight(call, highlight)}
                        />
                      </div>
                      <div>
                        <h4>Original transcript</h4>
                        <pre>{buildTranscriptText(call, "original")}</pre>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          ) : (
            <p className="empty-state">No saved Agents calls for this senior yet.</p>
          )}
        </section>

        <section className="volunteer-task-section">
          <h3>Volunteer Tasks</h3>
          <div className="task-list">
            {selectedTasks.length ? (
              selectedTasks.map((task) => (
                <article key={task.id} className="task-card">
                  <span className={`priority priority-${task.priority.toLowerCase()}`}>{task.priority}</span>
                  <strong>{task.reason}</strong>
                  <p>{task.recommendedAction}</p>
                  <small>
                    {task.assignedTo} · {task.status}
                  </small>
                </article>
              ))
            ) : (
              <p className="empty-state">No open task for this senior.</p>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

function App() {
  const [view, setView] = useState<AppView>("call");
  const [loadedSeniors, setLoadedSeniors] = useState<Senior[]>([]);
  const [loadedSessions, setLoadedSessions] = useState<CheckInSession[]>([]);
  const [loadedTasks, setLoadedTasks] = useState<VolunteerTask[]>([]);
  const [loadedCalls, setLoadedCalls] = useState<CallRecord[]>([]);
  const [selectedSeniorId, setSelectedSeniorId] = useState("s-001");

  useEffect(() => {
    void Promise.all([fetchSeniors(), fetchSessions(), fetchVolunteerTasks(), fetchCalls()]).then(([nextSeniors, nextSessions, nextTasks, nextCalls]) => {
      setLoadedSeniors(nextSeniors);
      setLoadedSessions(nextSessions);
      setLoadedTasks(nextTasks);
      setLoadedCalls(nextCalls);
      setSelectedSeniorId(nextSeniors[0]?.id ?? "s-001");
    });
  }, []);

  if (!loadedSeniors.length) {
    return <div className="loading">Loading EarlyCare...</div>;
  }

  const urgentTasks = loadedTasks.filter((task) => task.priority === "Urgent").length;
  const openTasks = loadedTasks.filter((task) => task.status !== "Closed").length;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <HeartPulse size={28} />
          </div>
          <div>
            <strong>EarlyCare</strong>
            <span>Living-alone elderly check-ins</span>
          </div>
        </div>
        <nav>
          <button className={view === "call" ? "active" : ""} onClick={() => setView("call")}>
            <Headphones size={18} />
            Agents call
          </button>
          <button className={view === "dashboard" ? "active" : ""} onClick={() => setView("dashboard")}>
            <Activity size={18} />
            Patient overview
          </button>
        </nav>
      </header>

      <section className="hero-band">
        <div>
          <span className="eyebrow">Preventive care + patient engagement</span>
          <h1>Regular check-ins that turn silence and speech change into earlier volunteer action.</h1>
        </div>
        <div className="hero-stats">
          <StatCard label="Open tasks" value={`${openTasks}`} icon={<Bell size={20} />} />
          <StatCard label="Urgent" value={`${urgentTasks}`} icon={<AlertTriangle size={20} />} />
          <StatCard label="Safety stance" value="No diagnosis" icon={<ShieldCheck size={20} />} />
          <StatCard label="Saved calls" value={`${loadedCalls.length}`} icon={<CheckCircle2 size={20} />} />
        </div>
      </section>

      {view === "call" ? (
        <ConversationProvider>
          <AgentsCall
            seniors={loadedSeniors}
            selectedSeniorId={selectedSeniorId}
            onSelectSenior={setSelectedSeniorId}
            onSavedCall={(call) => setLoadedCalls((calls) => [call, ...calls.filter((item) => item.id !== call.id)])}
          />
        </ConversationProvider>
      ) : (
        <OfficerDashboard
          seniors={loadedSeniors}
          sessions={loadedSessions}
          tasks={loadedTasks}
          calls={loadedCalls}
          selectedSeniorId={selectedSeniorId}
          setSelectedSeniorId={setSelectedSeniorId}
          onStartCall={(id) => {
            setSelectedSeniorId(id);
            setView("call");
          }}
        />
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
