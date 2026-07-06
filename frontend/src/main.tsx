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
import type { CallRecord, CheckInSession, RiskLevel, RiskSignal, Senior, TranscriptMessage, TranscriptSegment, VolunteerTask } from "./types";
import "./styles.css";

const riskOrder: Record<RiskLevel, number> = { Green: 0, Watch: 1, Amber: 2, Red: 3 };
type AppView = "call" | "dashboard";
type CallState = "Ready" | "Connecting" | "In call" | "Saving" | "Analysing" | "Complete" | "Failed";
type AgentAudioFormat = "pcm_8000" | "pcm_16000" | "pcm_22050" | "pcm_24000" | "pcm_44100" | "pcm_48000" | "ulaw_8000";
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
    const stream = await navigator.mediaDevices.getUserMedia(enhancedConstraints);
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
    const fallbackStream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
  return [
    `You are EarlyCare calling ${senior.name} for a routine wellbeing check-in.`,
    `The patient profile says their preferred language is ${senior.preferredLanguage}, but they may speak English, Mandarin, Malay, Tamil, Singlish, or a mix.`,
    "The live call transcript must stay in the original spoken language. Do not translate the patient's message before responding.",
    "For each agent reply, use the language or dialect that the patient used the most in their immediately previous response. If the patient code-switches, follow the dominant language from that one previous response.",
    "If the patient asks to speak Chinese or any other language, switch immediately.",
    "Ask concise turn-by-turn questions about falls, head impact, headache, dizziness, vomiting, confusion, weakness, speech difficulty, food and water, and whether they can ask for help.",
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

function HighlightedEnglishTranscript({
  call,
  highlightedSignalId,
  onSelectSignal
}: {
  call: CallRecord;
  highlightedSignalId: string | null;
  onSelectSignal: (signal: RiskSignal) => void;
}) {
  const segments = getEnglishTranscriptSegments(call);
  const patientSegments = segments
    .map((segment, segmentIndex) => ({ segment, segmentIndex }))
    .filter(({ segment }) => isPatientSegment(segment));
  const signals = call.riskSignals ?? [];

  return (
    <div className="highlighted-transcript">
      {segments.map((segment, segmentIndex) => {
        const text = displaySegmentText(segment);
        const patientIndex = patientSegments.findIndex((item) => item.segmentIndex === segmentIndex);
        const patientOnly = patientIndex >= 0;
        const matches = signals.filter((signal) => {
          const highlight = cleanTranscriptText(signal.highlightText || signal.quotedText);
          if (!patientOnly || !highlight) return false;
          const lowerHighlight = highlight.toLowerCase();
          const patientText = textWithoutSpeakerLabel(text).toLowerCase();
          const exactPatientMatchExists = patientSegments.some((item) =>
            textWithoutSpeakerLabel(item.segment.englishText || item.segment.text).toLowerCase().includes(lowerHighlight)
          );
          return patientText.includes(lowerHighlight) || (!exactPatientMatchExists && signal.sentenceIndex === patientIndex);
        });

        if (!matches.length || !text) {
          return <p key={`${call.id}-segment-${segmentIndex}`}>{text}</p>;
        }

        const signal = matches[0];
        const signalWithSentenceTime: RiskSignal = {
          ...signal,
          startTimeSeconds: previousAgentReplyEnd(segments, segmentIndex) ?? segment.startTimeSeconds ?? signal.startTimeSeconds,
          endTimeSeconds: segment.endTimeSeconds ?? signal.endTimeSeconds,
          sentenceIndex: patientIndex >= 0 ? patientIndex : signal.sentenceIndex
        };
        const highlight = cleanTranscriptText(signal.highlightText || signal.quotedText);
        const patientText = textWithoutSpeakerLabel(text);
        const patientTextStart = text.indexOf(patientText);
        const matchIndexInPatientText = patientText.toLowerCase().indexOf(highlight.toLowerCase());
        const matchIndex = matchIndexInPatientText >= 0 ? Math.max(0, patientTextStart) + matchIndexInPatientText : -1;
        if (matchIndex < 0) {
          return (
            <p key={`${call.id}-segment-${segmentIndex}`}>
              <button
                className={`transcript-highlight ${highlightedSignalId === `${call.id}-${signal.id}` ? "active" : ""}`}
                id={`signal-${call.id}-${signal.id}`}
                onClick={() => onSelectSignal(signalWithSentenceTime)}
              >
                {text}
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
              className={`transcript-highlight ${highlightedSignalId === `${call.id}-${signal.id}` ? "active" : ""}`}
              id={`signal-${call.id}-${signal.id}`}
              onClick={() => onSelectSignal(signalWithSentenceTime)}
            >
              {matched}
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

function formatPercent(value: number | undefined | null): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "Not enough data";
  return `${Math.round(value * 100)}%`;
}

function modelFeatureNumber(call: CallRecord | null, key: string): number | null {
  const value = call?.speechModelFeaturesSummary?.[key];
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
  const summary = call.speechModelFeaturesSummary;
  const warnings = call.speechModelWarnings ?? [];
  const warningText = warnings.join(" ").toLowerCase();
  const usable = summary?.speechModelUsable;
  if (warningText.includes("low confidence")) {
    return { value: "Low confidence", note: warnings.find((warning) => warning.toLowerCase().includes("low confidence")) ?? "Review audio quality" };
  }
  if (call.speechModelProbability !== null && call.speechModelProbability !== undefined && usable !== "false") {
    return { value: "Usable", note: `${Math.round(call.speechModelProbability * 100)}% marker score` };
  }
  if (usable === "false" || warningText.includes("unavailable")) {
    return { value: "Unavailable", note: warnings.find((warning) => warning.toLowerCase().includes("unavailable")) ?? "Not enough patient speech" };
  }
  if (call.patientSpeechAudioAvailable) {
    return { value: "Ready", note: "Ready when enabled" };
  }
  return { value: "Missing", note: "Not enough patient speech" };
}

function SpeechTimingPanel({ senior, call, calls }: { senior: Senior; call: CallRecord | null; calls: CallRecord[] }) {
  const baselineState = baselineFromCalls(senior, calls);
  const baseline = baselineState.profile;
  const current = call?.currentSpeechProfile ?? null;
  const patientSpeechSeconds = modelFeatureNumber(call, "patientSpeechDurationSeconds");
  const rawPatientSeconds = modelFeatureNumber(call, "rawPatientAudioDurationSeconds");
  const speechCoverage = patientSpeechSeconds && rawPatientSeconds ? patientSpeechSeconds / rawPatientSeconds : null;
  const baselinePatientSpeech = averageModelFeature(calls, "patientSpeechDurationSeconds");
  const baselineSpeechCoverage = averageSpeechCoverage(calls);
  const readiness = modelReadiness(call);
  const rows = [
    { label: "Patient speech", baseline: baselinePatientSpeech ? formatSeconds(baselinePatientSpeech) : "No baseline", current: formatSeconds(patientSpeechSeconds) },
    {
      label: "Speech coverage",
      baseline: baselineSpeechCoverage !== null ? `${Math.round(baselineSpeechCoverage * 100)}% avg` : "No baseline",
      current: formatPercent(speechCoverage)
    },
    { label: "Response latency", baseline: `${Math.round(baseline.responseLatencyMs)} ms`, current: formatMetric(current?.responseLatencyMs, " ms") },
    { label: "Speaking rate", baseline: `${Math.round(baseline.speechRate)} wpm`, current: formatMetric(current?.speechRate, " wpm") },
    { label: "Model readiness", baseline: readiness.note, current: readiness.value }
  ];

  return (
    <section className="speech-timing-panel">
      <div className="panel-heading compact-heading">
        <h3>Speech signal quality</h3>
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
      {!call?.speechModelFeaturesSummary && !call?.patientSpeechAudioAvailable ? (
        <p className="metric-note">
          Not enough patient speech yet. Keep patient-only recording enabled and let the agent ask short turn-by-turn questions so patient answers can be isolated cleanly.
        </p>
      ) : null}
    </section>
  );
}

function speechMarkerTitle(call: CallRecord): string {
  if (call.speechModelProbability !== null && call.speechModelProbability !== undefined) {
    return `Speech marker watch: ${Math.round(call.speechModelProbability * 100)}%`;
  }
  const warnings = (call.speechModelWarnings ?? []).join(" ").toLowerCase();
  if (warnings.includes("low confidence")) return "Speech marker low confidence";
  if (warnings.includes("unavailable")) return "Speech marker unavailable";
  return call.patientAudioAvailable ? "Speech marker model ready" : "Speech marker unavailable";
}

function speechMarkerDescription(call: CallRecord): string {
  if (call.speechModelProbability !== null && call.speechModelProbability !== undefined) {
    return `${call.speechModelVersion ?? "UCI/Kaggle tabular model"} scored the patient-only audio. This is a research screening signal, not a diagnosis.`;
  }
  if (call.speechModelWarnings?.length) {
    return call.speechModelWarnings.join(" ");
  }
  return call.patientAudioAvailable
    ? "Patient-only audio is saved. Enable backend speech scoring to populate the high-risk speech marker."
    : "Patient-only audio is not available for speech-marker scoring.";
}

function transcriptionAttemptSummary(call: CallRecord): string {
  const attempts = call.transcriptionAttempts ?? [];
  if (!attempts.length) {
    return `${call.translationProvider}${call.translationFallbackUsed ? " fallback" : ""}`;
  }
  return attempts
    .map((attempt) => {
      const reason = attempt.reason ? ` (${attempt.reason})` : "";
      return `${attempt.provider} ${attempt.status}${reason}`;
    })
    .join(" -> ");
}

function aiReviewNote(call: CallRecord): string | null {
  if (!call.aiRiskFallbackUsed) return null;
  return call.riskSignals?.length
    ? "AI review used fallback logic; highlights may be incomplete."
    : "Manual review required; AI risk highlights are unavailable for this call.";
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

      conversation.startSession({
        signedUrl: session.signedUrl,
        dynamicVariables: {
          senior_name: selectedSenior.name,
          preferred_language: selectedSenior.preferredLanguage,
          living_alone: selectedSenior.livingAlone,
          caregiver_contact: selectedSenior.caregiverContact,
          check_in_reason: "Routine living-alone wellbeing check-in",
          baseline_reminder: "Use EarlyCare's stored personal baseline context without saying exact baseline metrics aloud."
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
    formData.append("agentAudioCaptured", String(agentAudioCapturedRef.current));
    audioContextRef.current?.close();
    audioContextRef.current = null;
    recordingInputRef.current = null;
    if (audioBlob) formData.append("audio", audioBlob, "full-call.wav");
    if (patientAudioBlob) formData.append("patientAudio", patientAudioBlob, "patient-audio.wav");

    setCallState("Analysing");
    const saved = await saveCall(formData);
    if (saved) {
      onSavedCall(saved);
      setCallState("Complete");
      setCallMessage("Call saved. Patient overview now shows original and English transcripts.");
    } else {
      setCallState("Failed");
      setCallMessage("Call ended, but saving to backend failed.");
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
  const playRiskSignal = (call: CallRecord, signal: RiskSignal) => {
    setHighlightedSignalId(`${call.id}-${signal.id}`);
    const audio = audioRefs.current[call.id];
    if (audio && typeof signal.startTimeSeconds === "number") {
      audio.currentTime = Math.max(0, signal.startTimeSeconds);
      void audio.play();
      return;
    }
    document.getElementById(`signal-${call.id}-${signal.id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
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

        <section className="analysis-panel dashboard-analysis">
          <div className="analysis-header">
            <div>
              <span className="eyebrow">Deviation from baseline</span>
              <h2>AI Risk Review</h2>
            </div>
            <RiskBadge level={latestSignal?.riskLevel ?? "Green"} />
          </div>
          {latestAssessment ? (
            <>
              <div className="reason-box">
                <h3>Reasons</h3>
                <ul>
                  {latestAssessment.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>
              {"translationProvider" in latestSignal ? (
                <div className="model-card">
                  <Brain size={22} />
                  <div>
                    <strong>{latestSignal.aiRiskFallbackUsed ? "Manual review required" : "AI review completed"}</strong>
                    <p>
                      Transcript pipeline: {transcriptionAttemptSummary(latestSignal)}. Audio recording:{" "}
                      {latestSignal.audioAvailable ? "saved" : "not available"}.
                    </p>
                    {aiReviewNote(latestSignal) ? <p>{aiReviewNote(latestSignal)}</p> : null}
                  </div>
                </div>
              ) : null}
              {latestSignal.patientAudioAvailable || latestSignal.speechModelWarnings?.length || latestSignal.speechModelProbability !== null ? (
                <div className="model-card">
                  <Activity size={22} />
                  <div>
                    <strong>{speechMarkerTitle(latestSignal)}</strong>
                    <p>{speechMarkerDescription(latestSignal)}</p>
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <p className="empty-state">No risk assessment is available for this senior yet.</p>
          )}
        </section>

        <section className="saved-calls">
          <h3>Saved Agents Calls</h3>
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
                      </div>
                      <small>
                        {transcriptionAttemptSummary(call)} · audio {call.audioAvailable ? "saved" : "missing"} · patient audio{" "}
                        {call.patientAudioAvailable ? "saved" : "missing"} · agent voice {call.agentAudioCaptured ? "captured" : "not confirmed"}
                      </small>
                    </div>
                    <p>
                      <strong>Recommended action:</strong> {call.recommendedAction}
                    </p>
                    {call.patientAudioAvailable || call.speechModelWarnings?.length || call.speechModelProbability !== null ? (
                      <p>
                        <strong>{speechMarkerTitle(call)}:</strong> {speechMarkerDescription(call)}
                      </p>
                    ) : null}
                    {aiReviewNote(call) ? <p className="metric-note">{aiReviewNote(call)}</p> : null}

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
                          onSelectSignal={(signal) => playRiskSignal(call, signal)}
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
