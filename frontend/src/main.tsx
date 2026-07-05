import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ConversationProvider, useConversation } from "@elevenlabs/react";
import {
  Activity,
  AlertTriangle,
  Bell,
  Brain,
  CalendarClock,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  ClipboardList,
  FileClock,
  Headphones,
  HeartPulse,
  Languages,
  ListChecks,
  MapPin,
  Mic,
  PhoneCall,
  PlayCircle,
  RadioTower,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Stethoscope,
  Timer,
  UserRoundCheck,
  UsersRound
} from "lucide-react";
import {
  completeCheckIn,
  createElevenLabsSession,
  fetchCalls,
  fetchCallPlans,
  fetchSchedule,
  fetchScenarios,
  fetchSeniors,
  fetchSeniorRecords,
  fetchSessions,
  fetchVolunteerTasks,
  getCallAudioUrl,
  recordMissedCheckIn,
  runScenario,
  saveCall,
  startCheckIn,
  updateVolunteerTask
} from "./api";
import type {
  CallRecord,
  CallPlan,
  CheckInScheduleItem,
  CheckInSession,
  ConversationCategory,
  EscalationStep,
  RiskLevel,
  RiskSignal,
  Scenario,
  Senior,
  SeniorRecord,
  SpeechModelProvenance,
  TranscriptMessage,
  VolunteerTask,
  TranscriptSegment
} from "./types";
import "./styles.css";

const riskOrder: Record<RiskLevel, number> = { Green: 0, Watch: 1, Amber: 2, Red: 3 };
type AppView = "demo" | "call" | "dashboard";
type CallState = "Ready" | "Connecting" | "In call" | "Saving" | "Analysing" | "Complete" | "Failed";
type ScenarioTone = { label: string; risk: RiskLevel; detail: string };
type AgentAudioFormat = "pcm_8000" | "pcm_16000" | "pcm_22050" | "pcm_24000" | "pcm_44100" | "pcm_48000" | "ulaw_8000";
type RosterFilter = "all" | "due" | "tasks" | "risk";

function scenarioToneFor(scenario: Scenario): ScenarioTone {
  if (scenario.id.includes("red")) return { label: "Emergency path", risk: "Red", detail: "Escalates to urgent medical help" };
  if (scenario.id.includes("amber") || scenario.id.includes("missed")) return { label: "Same-day path", risk: "Amber", detail: "Creates caregiver or volunteer follow-up" };
  if (scenario.id.includes("parkinsons") || scenario.id.includes("chronic") || scenario.id.includes("mental")) {
    return { label: "Watch path", risk: "Watch", detail: "Logs evidence for repeated-pattern follow-up" };
  }
  return { label: "Routine path", risk: "Green", detail: "Records a completed baseline check-in" };
}

function countsByRisk(categories: ConversationCategory[]): Record<RiskLevel, number> {
  return categories.reduce<Record<RiskLevel, number>>(
    (counts, category) => ({ ...counts, [category.severity]: counts[category.severity] + 1 }),
    { Green: 0, Watch: 0, Amber: 0, Red: 0 }
  );
}

function sortByRisk<T extends { severity: RiskLevel }>(items: T[]): T[] {
  return [...items].sort((a, b) => riskOrder[b.severity] - riskOrder[a.severity]);
}

function StatCard({ label, value, icon, meta }: { label: string; value: string; icon: React.ReactNode; meta?: string }) {
  return (
    <section className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        {meta ? <small>{meta}</small> : null}
      </div>
    </section>
  );
}

function RiskBadge({ level }: { level: RiskLevel }) {
  return <span className={`risk-badge risk-${level.toLowerCase()}`}>{level}</span>;
}

function ScheduleBadge({ status }: { status: CheckInScheduleItem["status"] }) {
  return <span className={`schedule-badge schedule-${status.toLowerCase().replaceAll(" ", "-")}`}>{status}</span>;
}

function SectionHeading({
  eyebrow,
  title,
  meta
}: {
  eyebrow?: string;
  title: string;
  meta?: React.ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow ? <span className="eyebrow">{eyebrow}</span> : null}
        <h3>{title}</h3>
      </div>
      {meta ? <div className="section-meta">{meta}</div> : null}
    </div>
  );
}

function cleanTranscriptText(text: string): string {
  return text
    .replace(/\s*\[[^\]\r\n]{1,40}\]\s*/g, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function formatDate(value?: string | null): string {
  return value ? new Date(value).toLocaleString() : "Not completed";
}

function formatHours(value: number): string {
  if (!Number.isFinite(value)) return "Not set";
  if (Math.abs(value) < 0.1) return "now";
  const rounded = Math.round(Math.abs(value));
  return `${rounded} hour${rounded === 1 ? "" : "s"}`;
}

function highestRiskLevel(levels: RiskLevel[]): RiskLevel {
  return levels.reduce<RiskLevel>((risk, level) => (riskOrder[level] > riskOrder[risk] ? level : risk), "Green");
}

function stopRecorder(recorder: MediaRecorder | null): Promise<Blob | null> {
  return new Promise((resolve) => {
    if (!recorder || recorder.state === "inactive") {
      resolve(null);
      return;
    }
    const chunks: BlobPart[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = () => resolve(chunks.length ? new Blob(chunks, { type: "audio/webm" }) : null);
    recorder.stop();
  });
}

function CategoryList({ categories }: { categories: ConversationCategory[] }) {
  if (!categories.length) return <p className="empty-state">No categorized evidence is available for this record.</p>;

  const counts = countsByRisk(categories);
  const elevated = sortByRisk(categories.filter((category) => category.severity !== "Green"));
  const clear = categories.filter((category) => category.severity === "Green");
  const primaryCategories = elevated.length ? elevated : categories;

  const renderCategory = (category: ConversationCategory) => (
    <article className={`category-card category-${category.severity.toLowerCase()}`} key={category.id}>
      <div className="category-card-header">
        <strong>{category.label}</strong>
        <RiskBadge level={category.severity} />
      </div>
      {category.evidence.length ? (
        <ul>
          {category.evidence.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p>No evidence in this check-in.</p>
      )}
      <small>{category.recommendedAction}</small>
    </article>
  );

  return (
    <>
      <div className="category-summary">
        <span className="summary-pill summary-red">{counts.Red} Red</span>
        <span className="summary-pill summary-amber">{counts.Amber} Amber</span>
        <span className="summary-pill summary-watch">{counts.Watch} Watch</span>
        <span className="summary-pill summary-green">{counts.Green} Clear</span>
      </div>
      <div className="category-grid">{primaryCategories.map(renderCategory)}</div>
      {elevated.length > 0 && clear.length > 0 ? (
        <details className="clear-category-drawer">
          <summary>{clear.length} clear categories recorded</summary>
          <div className="category-grid compact-category-grid">{clear.map(renderCategory)}</div>
        </details>
      ) : null}
    </>
  );
}

function EscalationTrail({ steps }: { steps: EscalationStep[] }) {
  if (!steps.length) return null;

  return (
    <div className="escalation-list">
      {steps.map((step, index) => (
        <article className={`escalation-step status-${step.status.toLowerCase()}`} key={step.id}>
          <div className="step-index">{index + 1}</div>
          <div>
            <span>{step.status}</span>
            <strong>{step.label}</strong>
            <p>{step.detail}</p>
          </div>
        </article>
      ))}
    </div>
  );
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

function multilingualAgentPrompt(senior: Senior, callPlan?: CallPlan | null): string {
  const plannedQuestions = callPlan?.questions.length
    ? `Next-call plan questions: ${callPlan.questions.map((question, index) => `${index + 1}. ${question.prompt}`).join(" ")}`
    : "Use the general EarlyCare check-in question set.";
  return [
    `You are EarlyCare calling ${senior.name} for a routine wellbeing check-in.`,
    `The patient profile says their preferred language is ${senior.preferredLanguage}, but they may speak English, Mandarin, Malay, Tamil, Singlish, or a mix.`,
    "The live call transcript must stay in the original spoken language. Do not translate the patient's message before responding.",
    "For each agent reply, use the language or dialect that the patient used the most in their immediately previous response. If the patient code-switches, follow the dominant language from that one previous response.",
    "If the patient asks to speak Chinese or any other language, switch immediately.",
    callPlan ? `Opening script: ${callPlan.openingScript}` : "",
    plannedQuestions,
    callPlan ? `Escalation reminder: ${callPlan.escalationReminder}` : "",
    "Ask concise turn-by-turn questions. Do not ask several safety questions in one long block.",
    "Do not add bracketed emotional cues such as [concerned] or [happy]."
  ].filter(Boolean).join(" ");
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
  return cleanTranscriptText(text).replace(/^(Agent|Patient):\s*/i, "");
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
  const shouldUseCallTranscript =
    !segments.length ||
    !joinedSegmentEnglish ||
    joinedSegmentEnglish === cleanTranscriptText(call.originalTranscript) ||
    (looksNonEnglish(joinedSegmentEnglish) && !looksNonEnglish(call.englishTranscript));

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
    return getEnglishTranscriptSegments(call).map((segment) => cleanTranscriptText(segment.englishText || segment.text)).filter(Boolean).join("\n") || call.englishTranscript;
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
        const text = cleanTranscriptText(segment.englishText || segment.text);
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

function ScoreBars({ assessment }: { assessment: CheckInSession["riskAssessment"] }) {
  const scores = [
    ["Speech deviation", assessment.speechDeviationScore],
    ["Parkinson's watch", assessment.parkinsonsWatchScore],
    ["Post-head impact", assessment.postFallConcernScore],
    ["Missed check-in", assessment.missedCheckInScore]
  ] as const;

  return (
    <div className="score-grid">
      {scores.map(([label, value]) => (
        <div className="score-row" key={label}>
          <div className="score-label">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
          <div className="score-track">
            <span style={{ width: `${Math.min(100, value)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function TranscriptBubbleList({ messages }: { messages: TranscriptMessage[] }) {
  return (
    <div className="transcript">
      {messages.map((line, index) => (
        <p className={line.role === "Senior" ? "senior-line" : line.role === "System" ? "system-line" : "agent-line"} key={`${line.role}-${line.text}-${index}`}>
          {line.role === "Senior" ? "Patient" : line.role}: {line.text}
        </p>
      ))}
    </div>
  );
}

function formatMetric(value: number | undefined | null, suffix = ""): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return "Not enough data";
  return `${Math.round(value)}${suffix}`;
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

function speechProvenanceFor(call: CallRecord | null): SpeechModelProvenance | null {
  if (call?.speechModelProvenance) return call.speechModelProvenance;
  if (!call?.currentSpeechProfile) return null;
  return {
    runtimeMode: "demo metrics",
    featureExtractor: "transcript timing metrics",
    modelName: "EarlyCare demo speech metrics",
    generatedAt: call.currentSpeechProfile.updatedAt ?? call.completedAt,
    validated: false,
    notes: ["No diagnostic classifier or model weights were used."]
  };
}

function provenanceClassName(provenance: SpeechModelProvenance | null): string {
  if (!provenance) return "provenance-none";
  return `provenance-${provenance.runtimeMode.replaceAll(" ", "-")}`;
}

function speechModelModeLabel(mode: SpeechModelProvenance["runtimeMode"]): string {
  if (mode === "demo metrics") return "demo metrics";
  if (mode === "offline embedding") return "offline embedding";
  return "validated model";
}

function SpeechProvenanceSummary({ provenance, compact = false }: { provenance: SpeechModelProvenance | null; compact?: boolean }) {
  if (!provenance) {
    return (
      <div className={`speech-provenance-card ${compact ? "compact" : ""} provenance-none`}>
        <Brain size={17} />
        <div>
          <strong>speech model not available</strong>
          <small>No speech timing profile has been saved for this call.</small>
        </div>
      </div>
    );
  }
  return (
    <div className={`speech-provenance-card ${compact ? "compact" : ""} ${provenanceClassName(provenance)}`}>
      <Brain size={17} />
      <div>
        <strong>
          {speechModelModeLabel(provenance.runtimeMode)} · {provenance.modelName}
        </strong>
        <small>
          {provenance.featureExtractor}
          {provenance.validated ? " · validated model card" : " · no diagnosis"}
        </small>
      </div>
    </div>
  );
}

function SpeechTimingPanel({ senior, call, calls }: { senior: Senior; call: CallRecord | null; calls: CallRecord[] }) {
  const baselineState = baselineFromCalls(senior, calls);
  const baseline = baselineState.profile;
  const current = call?.currentSpeechProfile ?? null;
  const provenance = speechProvenanceFor(call);
  const rows = [
    { label: "Speech rate", baseline: `${Math.round(baseline.speechRate)} wpm`, current: formatMetric(current?.speechRate, " wpm") },
    { label: "Average pause", baseline: `${Math.round(baseline.avgPauseMs)} ms`, current: formatMetric(current?.avgPauseMs, " ms") },
    { label: "Response latency", baseline: `${Math.round(baseline.responseLatencyMs)} ms`, current: formatMetric(current?.responseLatencyMs, " ms") },
    { label: "Pitch variability", baseline: baseline.pitchVariability.toFixed(2), current: formatMetric(current?.pitchVariability) },
    { label: "Phrase accuracy", baseline: `${Math.round(baseline.phraseAccuracy * 100)}%`, current: formatMetric(current ? current.phraseAccuracy * 100 : null, "%") }
  ];

  return (
    <section className="speech-timing-panel">
      <div className="panel-heading compact-heading">
        <h3>Speech timing</h3>
        <span className="speech-heading-meta">
          <span>{baselineState.source}</span>
          <span className={`provenance-chip ${provenanceClassName(provenance)}`}>{provenance ? speechModelModeLabel(provenance.runtimeMode) : "not available"}</span>
        </span>
      </div>
      <SpeechProvenanceSummary provenance={provenance} compact />
      <div className="speech-metric-grid">
        {rows.map((row) => (
          <div className="speech-metric" key={row.label}>
            <span>{row.label}</span>
            <strong>{row.current}</strong>
            <small>Baseline {row.baseline}</small>
          </div>
        ))}
      </div>
      {!current || !current.avgPauseMs || !current.responseLatencyMs ? (
        <p className="metric-note">
          Not enough timing data yet. Improve this by keeping full-call recording enabled, ensuring Meralion returns timestamps/diarization, and letting the agent ask short turn-by-turn questions so patient responses can be timed cleanly.
        </p>
      ) : null}
    </section>
  );
}

function AgentsCall({
  seniors,
  callPlan,
  selectedSeniorId,
  onSelectSenior,
  onSavedCall
}: {
  seniors: Senior[];
  callPlan?: CallPlan | null;
  selectedSeniorId: string;
  onSelectSenior: (id: string) => void;
  onSavedCall: (call: CallRecord) => void | Promise<void>;
}) {
  const selectedSenior = seniors.find((senior) => senior.id === selectedSeniorId) ?? seniors[0];
  const [callState, setCallState] = useState<CallState>("Ready");
  const [callMessage, setCallMessage] = useState("Ready for a scheduled living-alone check-in.");
  const [debugMessage, setDebugMessage] = useState("");
  const [transcriptMessages, setTranscriptMessages] = useState<TranscriptMessage[]>([]);
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mixedDestinationRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const agentAudioFormatRef = useRef<AgentAudioFormat>("pcm_16000");
  const agentPlaybackTimeRef = useRef(0);
  const agentAudioCapturedRef = useRef(false);
  const agentAudioWarningShownRef = useRef(false);
  const transcriptRef = useRef<TranscriptMessage[]>([]);

  const conversation = useConversation({
    onConnect: () => {
      setCallState("In call");
      setCallMessage("Agent connected. Waiting for the first check-in question...");
      conversation.sendContextualUpdate(multilingualAgentPrompt(selectedSenior, callPlan));
      window.setTimeout(() => {
        const hasAgentMessage = transcriptRef.current.some((line) => line.role === "Agent");
        if (!hasAgentMessage) {
          conversation.sendUserMessage(
            [
              `Begin the EarlyCare check-in for ${selectedSenior.name}.`,
              `Preferred language: ${selectedSenior.preferredLanguage}.`,
              "Continue in the patient's dominant language or dialect from their previous response.",
              `Known conditions: ${selectedSenior.knownConditions.join(", ") || "none listed"}.`,
              `Focus areas: ${selectedSenior.promptFocus.join(", ") || "basic wellbeing"}.`,
              callPlan ? `Opening script: ${callPlan.openingScript}.` : "Ask for consent before continuing.",
              callPlan?.questions.length
                ? `Ask these planned questions one by one: ${callPlan.questions.map((question) => question.prompt).join(" ")}`
                : "Cover wellbeing, falls, head impact, whiplash or jolts, headache, dizziness, vomiting, confusion, weakness, numbness, slurred speech, food, water, medication, loneliness, and the repeat phrase.",
              callPlan ? `Escalation reminder: ${callPlan.escalationReminder}.` : "",
              "Do not diagnose. Escalate only as follow-up guidance."
            ].filter(Boolean).join(" ")
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
      const destination = mixedDestinationRef.current;
      if (!audioContext || !destination) return;
      try {
        const audioBuffer = createAgentAudioBuffer(audioContext, base64Audio, agentAudioFormatRef.current);
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(destination);
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
      setTranscriptMessages(transcriptRef.current);
      if (line.role === "Senior" && line.text) {
        conversation.sendContextualUpdate(nextReplyLanguageInstruction(line.text));
      }
    }
  });

  const startCall = async () => {
    setCallState("Connecting");
    setCallMessage("Connecting to Agents...");
    setDebugMessage("");
    setTranscriptMessages([]);
    transcriptRef.current = [];
    const callStartedAt = new Date().toISOString();
    setStartedAt(callStartedAt);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const AudioContextConstructor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextConstructor) throw new Error("Browser audio recording is not supported.");
      const audioContext = new AudioContextConstructor();
      const destination = audioContext.createMediaStreamDestination();
      audioContext.createMediaStreamSource(stream).connect(destination);
      audioContextRef.current = audioContext;
      mixedDestinationRef.current = destination;
      agentPlaybackTimeRef.current = audioContext.currentTime;
      agentAudioCapturedRef.current = false;
      agentAudioWarningShownRef.current = false;
      agentAudioFormatRef.current = "pcm_16000";
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
      const recorder = new MediaRecorder(destination.stream, { mimeType });
      recorder.start();
      recorderRef.current = recorder;

      const session = await createElevenLabsSession({
        seniorId: selectedSenior.id,
        seniorName: selectedSenior.name,
        preferredLanguage: selectedSenior.preferredLanguage,
        caregiverContact: selectedSenior.caregiverContact,
        checkInReason: `Scheduled ${selectedSenior.checkInFrequencyDays}-day living-alone wellbeing check-in`
      });

      if (!session.configured || !session.signedUrl) {
        const fallbackLine: TranscriptMessage = {
          role: "System",
          text: `${session.message} No Agents session was started.`,
          timestamp: new Date().toISOString()
        };
        transcriptRef.current = [fallbackLine];
        setTranscriptMessages(transcriptRef.current);
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
          neighbor_contact: selectedSenior.neighborContact ?? "",
          known_conditions: selectedSenior.knownConditions.join(", "),
          check_in_reason: `Scheduled ${selectedSenior.checkInFrequencyDays}-day living-alone wellbeing check-in`,
          prompt_focus: selectedSenior.promptFocus.join(", "),
          next_call_opening: callPlan?.openingScript ?? "",
          next_call_questions: callPlan?.questions.map((question) => question.prompt).join(" | ") ?? "",
          escalation_reminder: callPlan?.escalationReminder ?? "",
          repeat_phrase: "Today I am safe at home and I can ask for help."
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
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;

    const formData = new FormData();
    formData.append("seniorId", selectedSenior.id);
    formData.append("status", "Complete");
    formData.append("startedAt", startedAt ?? new Date().toISOString());
    formData.append("completedAt", new Date().toISOString());
    formData.append("transcriptMessages", JSON.stringify(transcriptRef.current));
    formData.append("agentAudioCaptured", String(agentAudioCapturedRef.current));
    audioContextRef.current?.close();
    audioContextRef.current = null;
    mixedDestinationRef.current = null;
    if (audioBlob) formData.append("audio", audioBlob, "full-call.webm");

    setCallState("Analysing");
    const saved = await saveCall(formData);
    if (saved) {
      await onSavedCall(saved);
      setCallState("Complete");
      setCallMessage("Call saved. Patient overview now shows categorized evidence and escalation.");
    } else {
      setCallState("Failed");
      setCallMessage("Call ended, but saving to backend failed.");
    }
  };

  return (
    <main className="call-only-grid">
      <section className="panel call-panel">
        <div className="panel-heading">
          <h2>Agents Website Call</h2>
          <span>Live provider path</span>
        </div>

        <div className="senior-selector">
          {seniors.map((senior) => (
            <button className={senior.id === selectedSenior.id ? "active" : ""} key={senior.id} onClick={() => onSelectSenior(senior.id)}>
              {senior.name}
            </button>
          ))}
        </div>

        <div className="profile-strip">
          <span>{selectedSenior.knownConditions.join(" · ")}</span>
          <span>{selectedSenior.neighborContact}</span>
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

          <div className="waveform" aria-label="voice waveform">
            {Array.from({ length: 36 }).map((_, index) => (
              <span key={index} style={{ height: `${18 + ((index * 17) % 58)}px` }} />
            ))}
          </div>

          <TranscriptBubbleList messages={transcriptMessages.length ? transcriptMessages : [{ role: "System", text: "Click Start call. The agent will begin once connected." }]} />

          <div className="answer-box">
            <p>{callMessage}</p>
            {debugMessage ? <small>Session note: {debugMessage}</small> : null}
            <small>
              SDK status: {conversation.status}. Mode: {conversation.mode}. {conversation.isSpeaking ? "Agent speaking." : conversation.isListening ? "Listening." : ""}
            </small>
          </div>

          <div className="call-actions">
            <button onClick={() => void startCall()} disabled={callState === "Connecting" || callState === "In call" || callState === "Saving" || callState === "Analysing"}>
              <Mic size={18} />
              Start call
            </button>
            <button onClick={() => void endAndSaveCall()} disabled={callState !== "In call" && callState !== "Failed"}>
              <PhoneCall size={18} />
              End & save
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}

function ScenarioRunner({
  scenarios,
  seniors,
  selectedSeniorId,
  onSelectSenior,
  onScenarioRun,
  onOpenDashboard
}: {
  scenarios: Scenario[];
  seniors: Senior[];
  selectedSeniorId: string;
  onSelectSenior: (id: string) => void;
  onScenarioRun: (session: CheckInSession, tasks: VolunteerTask[]) => void | Promise<void>;
  onOpenDashboard: () => void;
}) {
  const selectedScenario = scenarios.find((scenario) => scenario.seniorId === selectedSeniorId) ?? scenarios[0];
  const [scenarioId, setScenarioId] = useState(selectedScenario?.id ?? "");
  const [lastSession, setLastSession] = useState<CheckInSession | null>(null);
  const [status, setStatus] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const activeScenario = scenarios.find((scenario) => scenario.id === scenarioId) ?? selectedScenario;
  const activeSenior = seniors.find((senior) => senior.id === activeScenario?.seniorId) ?? seniors[0];
  const activeTone = activeScenario ? scenarioToneFor(activeScenario) : null;

  useEffect(() => {
    if (activeScenario && activeScenario.seniorId !== selectedSeniorId) {
      onSelectSenior(activeScenario.seniorId);
    }
  }, [activeScenario, onSelectSenior, selectedSeniorId]);

  const executeScenario = async () => {
    if (!activeScenario || isRunning) return;
    setIsRunning(true);
    setStatus("Running scenario and saving check-in...");
    try {
      const response = await runScenario(activeScenario.id);
      if (!response) {
        setStatus("Backend API is required to save scenario history.");
        return;
      }
      setLastSession(response.session);
      await onScenarioRun(response.session, response.tasks);
      setStatus("Scenario saved. Review the result here or open Patient overview.");
    } finally {
      setIsRunning(false);
    }
  };

  if (!activeScenario) return <p className="empty-state">No scenarios are configured.</p>;

  return (
    <main className="simulator-grid">
      <section className="panel scenario-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Demo control room</span>
            <h2>Scenario Runner</h2>
          </div>
          <span>{scenarios.length} demo cases</span>
        </div>
        <div className="scenario-library">
          {scenarios.map((scenario) => (
            <button
              className={`scenario-card scenario-${scenarioToneFor(scenario).risk.toLowerCase()} ${scenario.id === activeScenario.id ? "active" : ""}`}
              key={scenario.id}
              onClick={() => setScenarioId(scenario.id)}
            >
              <span>{scenarioToneFor(scenario).label}</span>
              <strong>{scenario.name}</strong>
              <small>{scenario.label}</small>
            </button>
          ))}
        </div>

        <article className="scenario-brief">
          <div>
            <div className="scenario-brief-header">
              {activeTone ? <RiskBadge level={activeTone.risk} /> : null}
              {activeTone ? <small>{activeTone.detail}</small> : null}
            </div>
            <h3>{activeScenario.label}</h3>
            <p>{activeScenario.description}</p>
            <div className="scenario-facts">
              <span>
                <MapPin size={15} />
                {activeSenior.name} · {activeSenior.addressZone}
              </span>
              <span>
                <Languages size={15} />
                {activeSenior.preferredLanguage}
              </span>
              <span>
                <CalendarClock size={15} />
                Every {activeSenior.checkInFrequencyDays} days
              </span>
              <span>
                <Stethoscope size={15} />
                {activeSenior.knownConditions.join(", ")}
              </span>
            </div>
          </div>
          <button className="primary-action" onClick={() => void executeScenario()} disabled={isRunning}>
            <PlayCircle size={18} />
            {isRunning ? "Running..." : "Run scenario"}
          </button>
        </article>

        <div className="script-and-model">
          <TranscriptBubbleList messages={activeScenario.script} />
          <div className="model-card">
            <Brain size={22} />
            <div>
              <strong>Demo baseline scoring</strong>
              <p>
                Speech scores use synthetic baseline metrics. Real wav2vec, WavLM, or MERaLiON SpeechEncoder embeddings are future validation work, not live diagnosis.
              </p>
            </div>
          </div>
        </div>
        {status ? <p className="status-note">{status}</p> : null}
      </section>

      <section className="panel result-panel">
        <div className="panel-heading">
          <h2>Saved Result</h2>
          <span>{lastSession ? lastSession.status : "Run a scenario"}</span>
        </div>
        {lastSession ? (
          <>
            <div className={`handoff-card handoff-${lastSession.riskLevel.toLowerCase()}`}>
              <div>
                <span className="eyebrow">Care-team handoff</span>
                <h3>{lastSession.recommendedAction}</h3>
                <p>{lastSession.summary}</p>
              </div>
              <RiskBadge level={lastSession.riskLevel} />
            </div>
            <ScoreBars assessment={lastSession.riskAssessment} />
            <CategoryList categories={lastSession.categories} />
            <SectionHeading title="Escalation path" meta={<span>{lastSession.escalationPlan.filter((step) => step.status === "Triggered").length} triggered</span>} />
            <EscalationTrail steps={lastSession.escalationPlan} />
            <div className="result-actions">
              <button className="primary-action" onClick={onOpenDashboard}>
                <Activity size={18} />
                Open Patient overview
              </button>
              <button onClick={() => setLastSession(null)}>Run another scenario</button>
            </div>
          </>
        ) : (
          <p className="empty-state">Run any scenario to create a persisted check-in, categorized evidence, and follow-up task when needed.</p>
        )}
      </section>
    </main>
  );
}

function SeniorRecordPanel({ record }: { record: SeniorRecord | null }) {
  if (!record) {
    return (
      <section className="record-panel">
        <SectionHeading eyebrow="Longitudinal record" title="Categorized History" meta={<span>No record</span>} />
        <p className="empty-state">No categorized record is available for this senior yet.</p>
      </section>
    );
  }

  const topCategories = record.categories.slice(0, 5);
  const latestEvents = record.timeline.slice(0, 4);

  return (
    <section className="record-panel">
      <SectionHeading
        eyebrow="Longitudinal record"
        title="Categorized History"
        meta={<span>{record.totalRecords} record{record.totalRecords === 1 ? "" : "s"}</span>}
      />
      <div className="record-summary-grid">
        <div>
          <span>Latest record</span>
          <strong>{record.latestRecordAt ? formatDate(record.latestRecordAt) : "No check-in yet"}</strong>
        </div>
        <div>
          <span>Highest signal</span>
          <RiskBadge level={record.highestRiskLevel} />
        </div>
        <div>
          <span>Open follow-up</span>
          <strong>{record.openTaskCount}</strong>
        </div>
        <div>
          <span>Cadence</span>
          <strong>Every {record.checkInFrequencyDays} days</strong>
        </div>
      </div>

      {topCategories.length ? (
        <div className="record-category-grid">
          {topCategories.map((category) => (
            <article className="record-category" key={category.id}>
              <div>
                <RiskBadge level={category.highestSeverity} />
                <strong>{category.label}</strong>
              </div>
              <small>
                {category.recordCount} mention{category.recordCount === 1 ? "" : "s"}
                {category.latestAt ? ` · latest ${formatDate(category.latestAt)}` : ""}
              </small>
              {category.latestEvidence.length ? <p>{category.latestEvidence.slice(0, 2).join(" ")}</p> : <p>{category.recommendedAction}</p>}
            </article>
          ))}
        </div>
      ) : (
        <p className="empty-state">No repeated category signal has been recorded yet.</p>
      )}

      {latestEvents.length ? (
        <div className="record-timeline">
          {latestEvents.map((event) => (
            <article className="record-event" key={event.id}>
              <div>
                <RiskBadge level={event.riskLevel} />
                <strong>{event.source === "call" ? "Agents call" : "Check-in"} · {event.status}</strong>
                <small>{formatDate(event.occurredAt)}</small>
              </div>
              <p>{event.summary}</p>
              {event.categories.length ? (
                <div className="record-event-tags">
                  {event.categories.slice(0, 4).map((category) => (
                    <span key={`${event.id}-${category.id}`}>{category.label}</span>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function CallPlanPanel({ callPlan, onStartCall }: { callPlan: CallPlan | null; onStartCall: () => void }) {
  return (
    <section className="call-plan-panel">
      <SectionHeading
        eyebrow="Personalized prompts"
        title="Next Call Plan"
        meta={callPlan ? <ScheduleBadge status={callPlan.scheduleStatus} /> : <span>No plan</span>}
      />
      {callPlan ? (
        <>
          <div className="call-plan-opening">
            <span>Opening</span>
            <strong>{callPlan.openingScript}</strong>
          </div>
          <div className="call-plan-question-list">
            {callPlan.questions.map((question) => (
              <article className="call-plan-question" key={question.id}>
                <div>
                  <span className={`priority priority-${question.priority.toLowerCase()}`}>{question.priority}</span>
                  <strong>{question.topic}</strong>
                </div>
                <p>{question.prompt}</p>
                <small>{question.rationale}</small>
              </article>
            ))}
          </div>
          <div className="call-plan-footer">
            <p>{callPlan.escalationReminder}</p>
            <button className="primary-action" onClick={onStartCall}>
              <PhoneCall size={18} />
              Use plan in call
            </button>
          </div>
        </>
      ) : (
        <p className="empty-state">No personalized call plan is available for this senior.</p>
      )}
    </section>
  );
}

function OfficerDashboard({
  seniors,
  sessions,
  tasks,
  calls,
  schedule,
  records,
  callPlans,
  selectedSeniorId,
  setSelectedSeniorId,
  onStartCall,
  onRecordCompletedCheckIn,
  onRecordMissedCheckIn,
  onTaskStatus
}: {
  seniors: Senior[];
  sessions: CheckInSession[];
  tasks: VolunteerTask[];
  calls: CallRecord[];
  schedule: CheckInScheduleItem[];
  records: SeniorRecord[];
  callPlans: CallPlan[];
  selectedSeniorId: string;
  setSelectedSeniorId: (id: string) => void;
  onStartCall: (id: string) => void;
  onRecordCompletedCheckIn: (id: string) => Promise<boolean>;
  onRecordMissedCheckIn: (id: string) => Promise<boolean>;
  onTaskStatus: (taskId: string, status: VolunteerTask["status"]) => void;
}) {
  const selectedSenior = seniors.find((senior) => senior.id === selectedSeniorId) ?? seniors[0];
  const [rosterQuery, setRosterQuery] = useState("");
  const [rosterFilter, setRosterFilter] = useState<RosterFilter>("all");
  const selectedTasks = tasks
    .filter((task) => task.seniorId === selectedSenior.id)
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
  const selectedCalls = calls.filter((call) => call.seniorId === selectedSenior.id);
  const selectedSessions = sessions.filter((session) => session.seniorId === selectedSenior.id);
  const selectedSchedule = schedule.find((item) => item.seniorId === selectedSenior.id) ?? null;
  const selectedSeniorRecord = records.find((record) => record.seniorId === selectedSenior.id) ?? null;
  const selectedCallPlan = callPlans.find((plan) => plan.seniorId === selectedSenior.id) ?? null;
  const selectedRecords = [
    ...selectedCalls.map((call) => ({ kind: "call" as const, date: call.completedAt, record: call })),
    ...selectedSessions.map((session) => ({ kind: "session" as const, date: session.completedAt ?? session.scheduledAt, record: session }))
  ].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  const latestRecord = selectedRecords[0]?.record ?? null;
  const latestAssessment = latestRecord?.riskAssessment ?? null;
  const audioRefs = useRef<Record<string, HTMLAudioElement | null>>({});
  const [highlightedSignalId, setHighlightedSignalId] = useState<string | null>(null);
  const highestRisk = highestRiskLevel(selectedRecords.map((item) => item.record.riskLevel));
  const selectedOpenTasks = selectedTasks.filter((task) => task.status !== "Closed");
  const latestRecordKind = selectedRecords[0]?.kind === "call" ? "Agents call" : selectedRecords[0]?.kind === "session" ? "Check-in" : "No record";
  const latestRecordTime = selectedRecords[0]?.date ? formatDate(selectedRecords[0].date) : "No check-in yet";
  const nextAction = selectedOpenTasks[0]?.recommendedAction ?? latestRecord?.recommendedAction ?? "Continue routine scheduled check-ins.";
  const [scheduleLogStatus, setScheduleLogStatus] = useState("");
  const [isLoggingAnswered, setIsLoggingAnswered] = useState(false);
  const [isLoggingMissed, setIsLoggingMissed] = useState(false);
  const rosterRows = seniors.map((senior) => {
    const seniorRecords = [...calls, ...sessions].filter((record) => record.seniorId === senior.id);
    const seniorRisk = highestRiskLevel(seniorRecords.map((record) => record.riskLevel));
    const seniorOpenTasks = tasks.filter((task) => task.seniorId === senior.id && task.status !== "Closed").length;
    const seniorSchedule = schedule.find((item) => item.seniorId === senior.id) ?? null;
    return { senior, risk: seniorRisk, openTaskCount: seniorOpenTasks, schedule: seniorSchedule };
  });
  const rosterFilterOptions: Array<{ id: RosterFilter; label: string; count: number }> = [
    { id: "all", label: "All", count: rosterRows.length },
    { id: "due", label: "Due", count: rosterRows.filter((row) => row.schedule ? row.schedule.status !== "On track" : false).length },
    { id: "tasks", label: "Open", count: rosterRows.filter((row) => row.openTaskCount > 0).length },
    { id: "risk", label: "Elevated", count: rosterRows.filter((row) => row.risk !== "Green").length }
  ];
  const normalizedRosterQuery = rosterQuery.trim().toLowerCase();
  const filteredRosterRows = rosterRows.filter((row) => {
    const searchable = [
      row.senior.name,
      row.senior.addressZone,
      row.senior.preferredLanguage,
      row.senior.caregiverContact,
      row.senior.neighborContact ?? "",
      ...row.senior.knownConditions,
      ...row.senior.promptFocus
    ]
      .join(" ")
      .toLowerCase();
    const matchesQuery = !normalizedRosterQuery || searchable.includes(normalizedRosterQuery);
    const matchesFilter =
      rosterFilter === "all" ||
      (rosterFilter === "due" && row.schedule !== null && row.schedule.status !== "On track") ||
      (rosterFilter === "tasks" && row.openTaskCount > 0) ||
      (rosterFilter === "risk" && row.risk !== "Green");
    return matchesQuery && matchesFilter;
  });
  const scheduleTimingSummary = selectedSchedule
    ? selectedSchedule.status === "Overdue"
      ? `${formatHours(selectedSchedule.overdueHours)} overdue`
      : selectedSchedule.status === "Due now"
        ? "due now"
        : `${formatHours(selectedSchedule.hoursUntilDue)} left`
    : "No active cadence";

  useEffect(() => {
    setScheduleLogStatus("");
  }, [selectedSenior.id]);

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

  const logCompletedCheckIn = async () => {
    if (isLoggingAnswered) return;
    setIsLoggingAnswered(true);
    setScheduleLogStatus("Logging answered check-in...");
    try {
      const logged = await onRecordCompletedCheckIn(selectedSenior.id);
      setScheduleLogStatus(logged ? "Answered check-in logged. Schedule and senior record refreshed." : "Live service connection is required to log an answered check-in.");
    } catch {
      setScheduleLogStatus("Unable to log answered check-in. Try again after checking the service connection.");
    } finally {
      setIsLoggingAnswered(false);
    }
  };

  const logMissedCheckIn = async () => {
    if (isLoggingMissed) return;
    setIsLoggingMissed(true);
    setScheduleLogStatus("Logging missed check-in...");
    try {
      const logged = await onRecordMissedCheckIn(selectedSenior.id);
      setScheduleLogStatus(logged ? "Missed check-in logged. Schedule and volunteer tasks refreshed." : "Live service connection is required to log a missed check-in.");
    } catch {
      setScheduleLogStatus("Unable to log missed check-in. Try again after checking the service connection.");
    } finally {
      setIsLoggingMissed(false);
    }
  };

  return (
    <main className="dashboard-grid">
      <section className="panel roster-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Care roster</span>
            <h2>Living-Alone Roster</h2>
          </div>
          <span>{filteredRosterRows.length} of {seniors.length}</span>
        </div>
        <div className="roster-toolbar">
          <label className="search-field">
            <Search size={16} />
            <span className="sr-only">Search roster</span>
            <input
              aria-label="Search roster"
              onChange={(event) => setRosterQuery(event.target.value)}
              placeholder="Search name, zone, language"
              type="search"
              value={rosterQuery}
            />
          </label>
          <div className="filter-group">
            <span className="filter-label">
              <SlidersHorizontal size={14} />
              Triage
            </span>
            <div className="filter-segments" aria-label="Roster filters">
              {rosterFilterOptions.map((option) => (
                <button
                  className={rosterFilter === option.id ? "active" : ""}
                  key={option.id}
                  onClick={() => setRosterFilter(option.id)}
                  type="button"
                >
                  <span>{option.label}</span>
                  <strong>{option.count}</strong>
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="senior-list">
          {filteredRosterRows.length ? (
            filteredRosterRows.map(({ senior, risk, openTaskCount, schedule: seniorSchedule }) => (
              <button
                className={`senior-row ${senior.id === selectedSenior.id ? "active" : ""}`}
                key={senior.id}
                onClick={() => setSelectedSeniorId(senior.id)}
              >
                <span className="senior-row-text">
                  <strong>{senior.name}</strong>
                  <small>
                    {senior.age} · {senior.addressZone} · {senior.preferredLanguage}
                  </small>
                </span>
                <span className="senior-row-meta">
                  <RiskBadge level={risk} />
                  {seniorSchedule ? <ScheduleBadge status={seniorSchedule.status} /> : openTaskCount ? <small>{openTaskCount} task{openTaskCount === 1 ? "" : "s"}</small> : <small>clear</small>}
                </span>
              </button>
            ))
          ) : (
            <p className="empty-state roster-empty">No seniors match this roster view.</p>
          )}
        </div>
      </section>

      <section className="panel detail-panel">
        <div className="profile-header">
          <div className="profile-copy">
            <div>
              <span className="eyebrow">Patient overview</span>
              <h2>{selectedSenior.name}</h2>
              <p>
                Lives alone in {selectedSenior.addressZone}. Check-in every {selectedSenior.checkInFrequencyDays} days.
              </p>
            </div>
            <div className="profile-summary-grid" aria-label="Selected patient snapshot">
              <div>
                <span>Schedule</span>
                <strong>{selectedSchedule?.status ?? "Not scheduled"}</strong>
                <small>{scheduleTimingSummary}</small>
              </div>
              <div>
                <span>Latest record</span>
                <strong>{latestRecordKind}</strong>
                <small>{latestRecordTime}</small>
              </div>
              <div>
                <span>Open work</span>
                <strong>{selectedOpenTasks.length} task{selectedOpenTasks.length === 1 ? "" : "s"}</strong>
                <small>{selectedOpenTasks[0]?.priority ?? "No follow-up"}</small>
              </div>
              <div>
                <span>Language</span>
                <strong>{selectedSenior.preferredLanguage}</strong>
                <small>{selectedSenior.caregiverContact}</small>
              </div>
            </div>
          </div>
          <div className="profile-actions">
            <RiskBadge level={highestRisk} />
            <button className="primary-action" onClick={() => onStartCall(selectedSenior.id)}>
              <PhoneCall size={18} />
              Start new call
            </button>
          </div>
        </div>

        <section className={`profile-handoff handoff-${highestRisk.toLowerCase()}`}>
          <div className="handoff-icon">
            <RadioTower size={22} />
          </div>
          <div>
            <span className="eyebrow">Current handoff</span>
            <h3>{nextAction}</h3>
            <p>
              {latestRecordKind} · {latestRecordTime}. {selectedOpenTasks.length ? `${selectedOpenTasks.length} open follow-up task${selectedOpenTasks.length === 1 ? "" : "s"}.` : "No open follow-up task."}
            </p>
          </div>
          <div className="handoff-metrics">
            <span>
              <ListChecks size={16} />
              {selectedOpenTasks.length} open
            </span>
            <span>
              <FileClock size={16} />
              {selectedSessions.length} check-ins
            </span>
          </div>
        </section>

        <section className={`schedule-panel schedule-panel-${selectedSchedule?.status.toLowerCase().replaceAll(" ", "-") ?? "none"}`}>
          <SectionHeading
            eyebrow="2-3 day cadence"
            title="Check-in Schedule"
            meta={selectedSchedule ? <ScheduleBadge status={selectedSchedule.status} /> : <span>Not scheduled</span>}
          />
          {selectedSchedule ? (
            <>
              <div className="schedule-grid">
                <div>
                  <span>Cadence</span>
                  <strong>Every {selectedSchedule.checkInFrequencyDays} days</strong>
                </div>
                <div>
                  <span>Next due</span>
                  <strong>{formatDate(selectedSchedule.nextDueAt)}</strong>
                  <small>
                    {selectedSchedule.status === "Overdue"
                      ? `${formatHours(selectedSchedule.overdueHours)} overdue`
                      : selectedSchedule.status === "Due now"
                        ? "due now"
                        : `${formatHours(selectedSchedule.hoursUntilDue)} left`}
                  </small>
                </div>
                <div>
                  <span>Last contact</span>
                  <strong>{selectedSchedule.lastContactAt ? formatDate(selectedSchedule.lastContactAt) : "No completed contact"}</strong>
                  <small>{selectedSchedule.lastContactKind}</small>
                </div>
                <div>
                  <span>Last attempt</span>
                  <strong>{selectedSchedule.lastAttemptAt ? formatDate(selectedSchedule.lastAttemptAt) : "No attempt logged"}</strong>
                  <small>{selectedSchedule.lastAttemptStatus ?? "none"}</small>
                </div>
              </div>
              <div className="schedule-action-row">
                <p>{selectedSchedule.recommendedAction}</p>
                <div className="schedule-actions">
                  <button className="primary-action" onClick={() => onStartCall(selectedSenior.id)}>
                    <PhoneCall size={18} />
                    Start scheduled call
                  </button>
                  <button className="secondary-action" onClick={() => void logCompletedCheckIn()} disabled={isLoggingAnswered}>
                    <CheckCircle2 size={18} />
                    {isLoggingAnswered ? "Logging..." : "Log answered"}
                  </button>
                  <button className="secondary-action" onClick={() => void logMissedCheckIn()} disabled={isLoggingMissed}>
                    <FileClock size={18} />
                    {isLoggingMissed ? "Logging..." : "Log missed"}
                  </button>
                </div>
              </div>
              {scheduleLogStatus ? <p className="status-note schedule-status-note" aria-live="polite">{scheduleLogStatus}</p> : null}
            </>
          ) : (
            <p className="empty-state">No schedule is available for this senior.</p>
          )}
        </section>

        <CallPlanPanel callPlan={selectedCallPlan} onStartCall={() => onStartCall(selectedSenior.id)} />

        <SeniorRecordPanel record={selectedSeniorRecord} />

        <section className="volunteer-task-section priority-section">
          <SectionHeading title="Volunteer Tasks" meta={<span>{selectedOpenTasks.length} open</span>} />
          <div className="task-list">
            {selectedTasks.length ? (
              selectedTasks.map((task) => (
                <article key={task.id} className={`task-card task-${task.status.toLowerCase().replace(" ", "-")}`}>
                  <div className="task-card-top">
                    <span className={`priority priority-${task.priority.toLowerCase()}`}>{task.priority}</span>
                    <small>{task.status}</small>
                  </div>
                  <strong>{task.reason}</strong>
                  <p>{task.recommendedAction}</p>
                  <small>
                    {task.assignedTo} · {formatDate(task.createdAt)}
                  </small>
                  <div className="task-actions">
                    <button onClick={() => onTaskStatus(task.id, "In progress")} disabled={task.status === "In progress"}>
                      <CheckCircle2 size={16} />
                      Acknowledge
                    </button>
                    <button onClick={() => onTaskStatus(task.id, "Closed")} disabled={task.status === "Closed"}>
                      <ShieldCheck size={16} />
                      Close
                    </button>
                  </div>
                </article>
              ))
            ) : (
              <p className="empty-state">No open task for this senior.</p>
            )}
          </div>
        </section>

        <div className="metric-grid">
          <StatCard label="Language" value={selectedSenior.preferredLanguage} icon={<Languages size={20} />} />
          <StatCard label="Caregiver" value={selectedSenior.caregiverContact} icon={<UserRoundCheck size={20} />} />
          <StatCard label="Neighbour" value={selectedSenior.neighborContact ?? "Not listed"} icon={<UsersRound size={20} />} />
          <StatCard label="Known conditions" value={selectedSenior.knownConditions.join(", ")} icon={<Stethoscope size={20} />} />
        </div>

        <div className="focus-strip">
          {selectedSenior.promptFocus.map((item) => (
            <span key={item}>
              <CircleDot size={13} />
              {item}
            </span>
          ))}
        </div>

        <SpeechTimingPanel senior={selectedSenior} call={selectedCalls[0] ?? null} calls={selectedCalls} />

        <section className="analysis-panel dashboard-analysis">
          <div className="analysis-header">
            <div>
              <span className="eyebrow">Decision support</span>
              <h2>Latest Risk Review</h2>
            </div>
            <RiskBadge level={latestRecord?.riskLevel ?? "Green"} />
          </div>
          {latestAssessment && latestRecord ? (
            <>
              <ScoreBars assessment={latestAssessment} />
              <div className="reason-box">
                <SectionHeading title="Reasons" meta={<span>{latestAssessment.reasons.length} signals</span>} />
                <ul>
                  {latestAssessment.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>
              <CategoryList categories={latestRecord.categories ?? []} />
              <SectionHeading title="Escalation path" meta={<span>{(latestRecord.escalationPlan ?? []).filter((step) => step.status === "Triggered").length} triggered</span>} />
              <EscalationTrail steps={latestRecord.escalationPlan ?? []} />
            </>
          ) : (
            <p className="empty-state">No risk assessment is available for this senior yet.</p>
          )}
        </section>

        <section className="history-section">
          <SectionHeading title="Check-In History" meta={<span>{selectedSessions.length} records</span>} />
          {selectedSessions.length ? (
            <div className="call-record-list">
              {selectedSessions.map((session) => (
                <article className="call-record" key={session.id}>
                  <div className="call-record-header">
                    <div>
                      <RiskBadge level={session.riskLevel} />
                      <strong>{session.scenarioName ?? session.status}</strong>
                    </div>
                    <small>{formatDate(session.completedAt ?? session.scheduledAt)}</small>
                  </div>
                  <p>{session.summary}</p>
                  <p>
                    <strong>Recommended action:</strong> {session.recommendedAction}
                  </p>
                  <CategoryList categories={session.categories ?? []} />
                  <div className="transcript-columns">
                    <div>
                      <h4>Original transcript</h4>
                      <pre>{session.originalTranscript}</pre>
                    </div>
                    <div>
                      <h4>English transcript</h4>
                      <pre>{session.englishTranscript}</pre>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <p className="empty-state">No historical check-ins for this senior yet.</p>
          )}
        </section>

        <section className="saved-calls">
          <SectionHeading title="Saved Agents Calls" meta={<span>{selectedCalls.length} calls</span>} />
          {selectedCalls.length ? (
            <div className="call-record-list">
              {selectedCalls.map((call) => {
                const audioUrl = getCallAudioUrl(call);
                const riskSignals = call.riskSignals ?? [];
                const provenance = speechProvenanceFor(call);
                return (
                  <article className="call-record" key={call.id}>
                    <div className="call-record-header">
                      <div>
                        <RiskBadge level={call.riskLevel} />
                        <strong>{formatDate(call.completedAt)}</strong>
                      </div>
                      <small>
                        {call.translationProvider}
                        {call.translationFallbackUsed ? " fallback" : ""} · audio {call.audioAvailable ? "saved" : "missing"} · agent voice{" "}
                        {call.agentAudioCaptured ? "captured" : "not confirmed"}
                      </small>
                    </div>
                    <p>
                      <strong>Recommended action:</strong> {call.recommendedAction}
                    </p>
                    <SpeechProvenanceSummary provenance={provenance} compact />

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

                    {riskSignals.length ? (
                      <div className="risk-signal-list">
                        <h4>Risk signals</h4>
                        {riskSignals.map((signal) => (
                          <button
                            className={`risk-signal ${highlightedSignalId === `${call.id}-${signal.id}` ? "active" : ""}`}
                            id={`signal-${call.id}-${signal.id}`}
                            key={signal.id}
                            onClick={() => playRiskSignal(call, signal)}
                          >
                            <span>
                              <RiskBadge level={signal.severity} />
                              <strong>{signal.label}</strong>
                            </span>
                            <p>{signal.quotedText}</p>
                            <small>{signal.reason}</small>
                          </button>
                        ))}
                      </div>
                    ) : null}

                    <CategoryList categories={call.categories ?? []} />
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

      </section>
    </main>
  );
}

function App() {
  const [view, setView] = useState<AppView>("dashboard");
  const [loadedSeniors, setLoadedSeniors] = useState<Senior[]>([]);
  const [loadedSessions, setLoadedSessions] = useState<CheckInSession[]>([]);
  const [loadedTasks, setLoadedTasks] = useState<VolunteerTask[]>([]);
  const [loadedCalls, setLoadedCalls] = useState<CallRecord[]>([]);
  const [loadedSchedule, setLoadedSchedule] = useState<CheckInScheduleItem[]>([]);
  const [loadedScenarios, setLoadedScenarios] = useState<Scenario[]>([]);
  const [loadedSeniorRecords, setLoadedSeniorRecords] = useState<SeniorRecord[]>([]);
  const [loadedCallPlans, setLoadedCallPlans] = useState<CallPlan[]>([]);
  const [selectedSeniorId, setSelectedSeniorId] = useState("s-001");

  const refreshSchedule = async () => {
    const nextSchedule = await fetchSchedule();
    setLoadedSchedule(nextSchedule);
  };

  const refreshTasks = async () => {
    const nextTasks = await fetchVolunteerTasks();
    setLoadedTasks(nextTasks);
  };

  const refreshSeniorRecords = async () => {
    const nextRecords = await fetchSeniorRecords();
    setLoadedSeniorRecords(nextRecords);
  };

  const refreshCallPlans = async () => {
    const nextPlans = await fetchCallPlans();
    setLoadedCallPlans(nextPlans);
  };

  useEffect(() => {
    void Promise.all([fetchSeniors(), fetchSessions(), fetchVolunteerTasks(), fetchCalls(), fetchSchedule(), fetchScenarios(), fetchSeniorRecords(), fetchCallPlans()]).then(
      ([nextSeniors, nextSessions, nextTasks, nextCalls, nextSchedule, nextScenarios, nextRecords, nextCallPlans]) => {
        setLoadedSeniors(nextSeniors);
        setLoadedSessions(nextSessions);
        setLoadedTasks(nextTasks);
        setLoadedCalls(nextCalls);
        setLoadedSchedule(nextSchedule);
        setLoadedScenarios(nextScenarios);
        setLoadedSeniorRecords(nextRecords);
        setLoadedCallPlans(nextCallPlans);
        setSelectedSeniorId(nextSeniors[0]?.id ?? "s-001");
      }
    );
  }, []);

  if (!loadedSeniors.length) {
    return <div className="loading">Loading EarlyCare...</div>;
  }

  const urgentTasks = loadedTasks.filter((task) => task.priority === "Urgent" && task.status !== "Closed").length;
  const openTasks = loadedTasks.filter((task) => task.status !== "Closed").length;
  const dueNow = loadedSchedule.filter((item) => item.status === "Due now" || item.status === "Overdue").length;

  const handleTaskStatus = async (taskId: string, status: VolunteerTask["status"]) => {
    const updated = await updateVolunteerTask(taskId, status);
    if (updated) {
      setLoadedTasks((tasks) => tasks.map((task) => (task.id === updated.id ? updated : task)));
      await refreshSeniorRecords();
      await refreshCallPlans();
      return;
    }
    setLoadedTasks((tasks) => tasks.map((task) => (task.id === taskId ? { ...task, status } : task)));
  };

  const handleRecordMissedCheckIn = async (seniorId: string) => {
    const scheduleItem = loadedSchedule.find((item) => item.seniorId === seniorId);
    const now = new Date().toISOString();
    const response = await recordMissedCheckIn({
      seniorId,
      scheduledAt: scheduleItem?.nextDueAt ?? now,
      retryAt: now,
      attemptCount: 2,
      note: "Logged from Patient overview after scheduled call and retry were unanswered."
    });

    if (!response) return false;

    setLoadedSessions((sessions) => [response.session, ...sessions.filter((session) => session.id !== response.session.id)]);
    setLoadedTasks(response.tasks);
    setSelectedSeniorId(response.session.seniorId);
    await refreshSchedule();
    await refreshSeniorRecords();
    await refreshCallPlans();
    return true;
  };

  const handleRecordCompletedCheckIn = async (seniorId: string) => {
    const started = await startCheckIn(seniorId);
    if (!started) return false;

    const transcript = "Patient answered the scheduled check-in and reported no immediate concern.";
    const completed = await completeCheckIn(started.id, {
      completedAt: new Date().toISOString(),
      originalTranscript: transcript,
      englishTranscript: transcript,
      summary: "Completed scheduled check-in from Patient overview."
    });
    if (!completed) return false;

    setLoadedSessions((sessions) => [completed, ...sessions.filter((session) => session.id !== completed.id && session.id !== started.id)]);
    setSelectedSeniorId(completed.seniorId);
    await refreshTasks();
    await refreshSchedule();
    await refreshSeniorRecords();
    await refreshCallPlans();
    return true;
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <HeartPulse size={28} />
          </div>
          <div>
            <strong>EarlyCare</strong>
            <span>2-3 day living-alone check-ins</span>
          </div>
        </div>
        <nav aria-label="Primary workspace">
          <button className={view === "demo" ? "active" : ""} onClick={() => setView("demo")}>
            <ClipboardList size={18} />
            Demo runner
          </button>
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
        <div className="hero-copy">
          <span className="eyebrow">Care operations command center</span>
          <h1>Care operations</h1>
          <p>Monitor due check-ins, unanswered attempts, and human follow-up before silence turns into a welfare risk.</p>
          <div className="routing-strip" aria-label="EarlyCare escalation route">
            <span>
              <Timer size={15} />
              2-3 day cadence
            </span>
            <ChevronRight size={16} />
            <span>
              <Bell size={15} />
              retry + notify
            </span>
            <ChevronRight size={16} />
            <span>
              <UsersRound size={15} />
              caregiver / neighbour
            </span>
            <ChevronRight size={16} />
            <span>
              <RadioTower size={15} />
              volunteer or emergency path
            </span>
          </div>
        </div>
        <div className="hero-stats">
          <StatCard label="Open tasks" value={`${openTasks}`} meta="follow-up queue" icon={<Bell size={20} />} />
          <StatCard label="Urgent" value={`${urgentTasks}`} meta="same-day attention" icon={<AlertTriangle size={20} />} />
          <StatCard label="Due / overdue" value={`${dueNow}`} meta="cadence risk" icon={<CalendarClock size={20} />} />
          <StatCard label="Safety stance" value="No diagnosis" meta="human review only" icon={<ShieldCheck size={20} />} />
        </div>
      </section>

      {view === "demo" ? (
        <ScenarioRunner
          scenarios={loadedScenarios}
          seniors={loadedSeniors}
          selectedSeniorId={selectedSeniorId}
          onSelectSenior={setSelectedSeniorId}
          onScenarioRun={async (session, tasks) => {
            setLoadedSessions((sessions) => [session, ...sessions.filter((item) => item.id !== session.id)]);
            setLoadedTasks(tasks);
            setSelectedSeniorId(session.seniorId);
            await refreshSchedule();
            await refreshSeniorRecords();
            await refreshCallPlans();
          }}
          onOpenDashboard={() => setView("dashboard")}
        />
      ) : view === "call" ? (
        <ConversationProvider>
          <AgentsCall
            seniors={loadedSeniors}
            selectedSeniorId={selectedSeniorId}
            onSelectSenior={setSelectedSeniorId}
            callPlan={loadedCallPlans.find((plan) => plan.seniorId === selectedSeniorId) ?? null}
            onSavedCall={async (call) => {
              setLoadedCalls((calls) => [call, ...calls.filter((item) => item.id !== call.id)]);
              await refreshTasks();
              await refreshSchedule();
              await refreshSeniorRecords();
              await refreshCallPlans();
            }}
          />
        </ConversationProvider>
      ) : (
        <OfficerDashboard
          seniors={loadedSeniors}
          sessions={loadedSessions}
          tasks={loadedTasks}
          calls={loadedCalls}
          schedule={loadedSchedule}
          records={loadedSeniorRecords}
          callPlans={loadedCallPlans}
          selectedSeniorId={selectedSeniorId}
          setSelectedSeniorId={setSelectedSeniorId}
          onStartCall={(id) => {
            setSelectedSeniorId(id);
            setView("call");
          }}
          onRecordCompletedCheckIn={handleRecordCompletedCheckIn}
          onRecordMissedCheckIn={handleRecordMissedCheckIn}
          onTaskStatus={(taskId, status) => void handleTaskStatus(taskId, status)}
        />
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
