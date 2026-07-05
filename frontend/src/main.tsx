import React, { lazy, Suspense, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bell,
  Brain,
  CalendarClock,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  CircleDot,
  Clock3,
  ClipboardList,
  FileClock,
  Gauge,
  Headphones,
  HeartPulse,
  History,
  Languages,
  ListChecks,
  MapPin,
  PhoneCall,
  PhoneForwarded,
  PlayCircle,
  RadioTower,
  RefreshCw,
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
  fetchCalls,
  fetchCallPlans,
  fetchOperationsQueue,
  fetchSchedule,
  fetchScenarios,
  fetchSeniors,
  fetchSeniorRecords,
  fetchServiceStatus,
  fetchSessions,
  fetchVolunteerTasks,
  getCallAudioUrl,
  recordMissedCheckIn,
  runScenario,
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
  OperationsQueueItem,
  RiskLevel,
  RiskSignal,
  Scenario,
  Senior,
  SeniorRecord,
  ServiceStatus,
  SpeechModelProvenance,
  TranscriptMessage,
  VolunteerTask,
  TranscriptSegment
} from "./types";
import "./styles.css";

const riskOrder: Record<RiskLevel, number> = { Green: 0, Watch: 1, Amber: 2, Red: 3 };
type AppView = "demo" | "call" | "dashboard";
type ScenarioTone = { label: string; risk: RiskLevel; detail: string };
type RosterFilter = "all" | "due" | "tasks" | "risk";
type TaskLane = VolunteerTask["status"];

const AgentsCall = lazy(() => import("./AgentsCall"));

const initialServiceStatus: ServiceStatus = {
  mode: "checking",
  configured: false,
  reachable: false,
  message: "Checking service connection."
};

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

function ServiceStatusIndicator({ status }: { status: ServiceStatus }) {
  const hasStorageWarnings = Boolean(status.storageWarnings?.length);
  const label = status.mode === "live" && hasStorageWarnings ? "Storage warning" : status.mode === "live" ? "Live API" : status.mode === "checking" ? "Checking" : "Demo data";
  return (
    <div className={`service-status service-${status.mode} ${hasStorageWarnings ? "service-warning" : ""}`} aria-label={`Service status: ${status.message}`} title={status.message}>
      <span className="service-dot" aria-hidden="true" />
      <strong>{label}</strong>
    </div>
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

function formatSyncTime(value?: string | null): string {
  if (!value) return "Not synced";
  return `Synced ${new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function formatShortDate(value?: string | null): string {
  if (!value) return "Not available";
  return new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
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

function FollowUpPipeline({
  tasks,
  onTaskStatus
}: {
  tasks: VolunteerTask[];
  onTaskStatus: (taskId: string, status: VolunteerTask["status"]) => void;
}) {
  const lanes: Array<{ status: TaskLane; title: string; icon: React.ReactNode; empty: string }> = [
    { status: "Open", title: "Needs owner", icon: <Bell size={16} />, empty: "No unowned work." },
    { status: "In progress", title: "Being handled", icon: <PhoneForwarded size={16} />, empty: "No active handoff." },
    { status: "Closed", title: "Closed", icon: <ClipboardCheck size={16} />, empty: "No closed tasks yet." }
  ];
  const groupedTasks = lanes.map((lane) => ({
    ...lane,
    tasks: tasks.filter((task) => task.status === lane.status)
  }));
  const openCount = tasks.filter((task) => task.status !== "Closed").length;
  const urgentCount = tasks.filter((task) => task.priority === "Urgent" && task.status !== "Closed").length;

  return (
    <section className="task-pipeline-section">
      <SectionHeading
        eyebrow="Follow-up control"
        title="Action Pipeline"
        meta={
          <span className="pipeline-meta">
            <strong>{openCount}</strong> open · <strong>{urgentCount}</strong> urgent
          </span>
        }
      />
      <div className="task-lane-grid" aria-label="Volunteer task pipeline">
        {groupedTasks.map((lane) => (
          <div className={`task-lane lane-${lane.status.toLowerCase().replace(" ", "-")}`} key={lane.status}>
            <div className="task-lane-header">
              <span>
                {lane.icon}
                {lane.title}
              </span>
              <strong>{lane.tasks.length}</strong>
            </div>
            <div className="task-lane-body">
              {lane.tasks.length ? (
                lane.tasks.map((task) => (
                  <article className={`task-workflow-card task-${task.status.toLowerCase().replace(" ", "-")}`} key={task.id}>
                    <div className="task-card-top">
                      <span className={`priority priority-${task.priority.toLowerCase()}`}>{task.priority}</span>
                      <small>{formatShortDate(task.createdAt)}</small>
                    </div>
                    <strong>{task.reason}</strong>
                    <p>{task.recommendedAction}</p>
                    <div className="task-source">
                      <span>{task.assignedTo}</span>
                      {task.escalationStep ? <span>{task.escalationStep.replaceAll("-", " ")}</span> : null}
                    </div>
                    <div className="task-actions">
                      <button onClick={() => onTaskStatus(task.id, "In progress")} disabled={task.status !== "Open"}>
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
                <p className="task-empty">{lane.empty}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
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
      setStatus("Scenario saved. Review the result here or open Care desk.");
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
                Open Care desk
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

function OperationsQueuePanel({
  queue,
  selectedSeniorId,
  onOpenSenior,
  onStartCall
}: {
  queue: OperationsQueueItem[];
  selectedSeniorId: string;
  onOpenSenior: (id: string) => void;
  onStartCall: (id: string) => void;
}) {
  const activeItems = queue.filter((item) => item.priority !== "Routine").slice(0, 5);

  if (!activeItems.length) {
    return (
      <section className="operations-queue-panel">
        <SectionHeading eyebrow="Care desk" title="Operations Queue" meta={<span>clear</span>} />
        <p className="empty-state">No due, elevated-risk, or open follow-up items are in the queue.</p>
      </section>
    );
  }

  return (
    <section className="operations-queue-panel">
      <SectionHeading eyebrow="Care desk" title="Operations Queue" meta={<span>{activeItems.length} active</span>} />
      <div className="queue-list">
        {activeItems.map((item) => (
          <article className={`queue-card queue-${item.priority.toLowerCase()} ${item.seniorId === selectedSeniorId ? "active" : ""}`} key={`${item.seniorId}-${item.queueRank}`}>
            <div className="queue-rank">{item.queueRank}</div>
            <div className="queue-body">
              <div className="queue-card-header">
                <div>
                  <span className="eyebrow">{item.priority}</span>
                  <strong>{item.seniorName}</strong>
                </div>
                <div className="queue-badges">
                  <ScheduleBadge status={item.scheduleStatus} />
                  <RiskBadge level={item.riskLevel} />
                </div>
              </div>
              <p>{item.reason}</p>
              <div className="queue-meta">
                <span>
                  <CalendarClock size={14} />
                  {item.scheduleStatus === "Overdue"
                    ? `${formatHours(item.dueInHours)} overdue`
                    : item.scheduleStatus === "Due now"
                      ? "due now"
                      : `${formatHours(item.dueInHours)} left`}
                </span>
                <span>
                  <ListChecks size={14} />
                  {item.openTaskCount} open
                </span>
                {item.assignedTo ? (
                  <span>
                    <UserRoundCheck size={14} />
                    {item.assignedTo}
                  </span>
                ) : null}
              </div>
            </div>
            <div className="queue-actions">
              <button className="secondary-action" onClick={() => onOpenSenior(item.seniorId)} type="button">
                Open patient
              </button>
              <button className="primary-action" onClick={() => onStartCall(item.seniorId)} type="button">
                <PhoneCall size={16} />
                Start call
              </button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function OfficerDashboard({
  seniors,
  sessions,
  tasks,
  calls,
  schedule,
  operationsQueue,
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
  operationsQueue: OperationsQueueItem[];
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
  const [handoffCopyStatus, setHandoffCopyStatus] = useState("");
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
    setHandoffCopyStatus("");
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

  const buildCareHandoffText = () => {
    const elevatedCategories = (latestRecord?.categories ?? []).filter((category) => category.severity !== "Green").slice(0, 4);
    const taskLines = selectedOpenTasks.slice(0, 3).map((task) => `- ${task.priority}: ${task.recommendedAction} (${task.assignedTo})`);
    const categoryLines = elevatedCategories.map((category) => {
      const evidence = category.evidence.slice(0, 2).join(" ");
      return `- ${category.severity}: ${category.label}${evidence ? ` - ${evidence}` : ""}`;
    });

    return [
      "EarlyCare care-team handoff",
      `Senior: ${selectedSenior.name}, ${selectedSenior.age}, ${selectedSenior.addressZone}`,
      `Language: ${selectedSenior.preferredLanguage}`,
      `Known conditions: ${selectedSenior.knownConditions.join(", ") || "none listed"}`,
      `Schedule: ${selectedSchedule ? `${selectedSchedule.status}; next due ${formatDate(selectedSchedule.nextDueAt)} (${scheduleTimingSummary})` : "not scheduled"}`,
      `Highest risk: ${highestRisk}`,
      `Latest record: ${latestRecordKind} at ${latestRecordTime}`,
      `Recommended action: ${nextAction}`,
      selectedSchedule ? `Schedule guidance: ${selectedSchedule.recommendedAction}` : "",
      selectedOpenTasks.length ? `Open follow-up tasks:\n${taskLines.join("\n")}` : "Open follow-up tasks: none",
      categoryLines.length ? `Elevated evidence:\n${categoryLines.join("\n")}` : "Elevated evidence: none recorded",
      `Caregiver: ${selectedSenior.caregiverContact}`,
      `Neighbour: ${selectedSenior.neighborContact ?? "not listed"}`,
      "Safety note: EarlyCare is decision support only. A human should assess and escalate urgent danger signs."
    ].filter(Boolean).join("\n");
  };

  const copyCareHandoff = async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable.");
      await navigator.clipboard.writeText(buildCareHandoffText());
      setHandoffCopyStatus("Handoff copied.");
    } catch {
      setHandoffCopyStatus("Unable to copy. Use the visible handoff details.");
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
        <div className={`profile-header profile-risk-${highestRisk.toLowerCase()}`}>
          <div className="profile-copy">
            <div className="profile-title-block">
              <div>
                <span className="eyebrow">Care desk profile</span>
                <h2>{selectedSenior.name}</h2>
              </div>
              <RiskBadge level={highestRisk} />
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
            <div className="contact-route-grid" aria-label="Care contact route">
              <span>
                <UserRoundCheck size={15} />
                {selectedSenior.caregiverContact}
              </span>
              <span>
                <UsersRound size={15} />
                {selectedSenior.neighborContact ?? "Neighbour not listed"}
              </span>
              <span>
                <Stethoscope size={15} />
                {selectedSenior.knownConditions.join(", ")}
              </span>
            </div>
          </div>
          <div className="profile-actions">
            <span className="profile-action-label">Next action</span>
            <div className="profile-action-buttons">
              <button className="secondary-action" onClick={() => void copyCareHandoff()} type="button">
                <ClipboardList size={18} />
                Copy handoff
              </button>
              <button className="primary-action" onClick={() => onStartCall(selectedSenior.id)}>
                <PhoneCall size={18} />
                Start new call
              </button>
            </div>
            {handoffCopyStatus ? <small className="copy-status" aria-live="polite">{handoffCopyStatus}</small> : null}
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

        <section className="patient-command-strip" aria-label="Selected patient command summary">
          <span>
            <Clock3 size={16} />
            {selectedSchedule ? `${selectedSchedule.status} · ${scheduleTimingSummary}` : "No active cadence"}
          </span>
          <span>
            <Gauge size={16} />
            Highest risk {highestRisk}
          </span>
          <span>
            <History size={16} />
            {selectedRecords.length} record{selectedRecords.length === 1 ? "" : "s"}
          </span>
          <span>
            <ArrowRight size={16} />
            {selectedOpenTasks.length ? selectedOpenTasks[0].assignedTo : "Routine monitoring"}
          </span>
        </section>

        <div className="operator-workbench">
          <FollowUpPipeline tasks={selectedTasks} onTaskStatus={onTaskStatus} />
          <CallPlanPanel callPlan={selectedCallPlan} onStartCall={() => onStartCall(selectedSenior.id)} />
        </div>

        <OperationsQueuePanel
          queue={operationsQueue}
          selectedSeniorId={selectedSenior.id}
          onOpenSenior={setSelectedSeniorId}
          onStartCall={onStartCall}
        />

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

        <SeniorRecordPanel record={selectedSeniorRecord} />

        <div className="focus-strip">
          {selectedSenior.promptFocus.map((item) => (
            <span key={item}>
              <CircleDot size={13} />
              {item}
            </span>
          ))}
        </div>

        <div className="decision-grid">
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
        </div>

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
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>(initialServiceStatus);
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadedSeniors, setLoadedSeniors] = useState<Senior[]>([]);
  const [loadedSessions, setLoadedSessions] = useState<CheckInSession[]>([]);
  const [loadedTasks, setLoadedTasks] = useState<VolunteerTask[]>([]);
  const [loadedCalls, setLoadedCalls] = useState<CallRecord[]>([]);
  const [loadedSchedule, setLoadedSchedule] = useState<CheckInScheduleItem[]>([]);
  const [loadedOperationsQueue, setLoadedOperationsQueue] = useState<OperationsQueueItem[]>([]);
  const [loadedScenarios, setLoadedScenarios] = useState<Scenario[]>([]);
  const [loadedSeniorRecords, setLoadedSeniorRecords] = useState<SeniorRecord[]>([]);
  const [loadedCallPlans, setLoadedCallPlans] = useState<CallPlan[]>([]);
  const [selectedSeniorId, setSelectedSeniorId] = useState("s-001");

  const refreshSchedule = async () => {
    const nextSchedule = await fetchSchedule();
    setLoadedSchedule(nextSchedule);
  };

  const refreshOperationsQueue = async () => {
    const nextQueue = await fetchOperationsQueue();
    setLoadedOperationsQueue(nextQueue);
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

  const refreshServiceStatus = async () => {
    const nextStatus = await fetchServiceStatus();
    setServiceStatus(nextStatus);
  };

  const refreshWorkspace = async () => {
    if (isRefreshing) return;
    setIsRefreshing(true);
    try {
      const [nextStatus, nextSeniors, nextSessions, nextTasks, nextCalls, nextSchedule, nextQueue, nextScenarios, nextRecords, nextCallPlans] = await Promise.all([
        fetchServiceStatus(),
        fetchSeniors(),
        fetchSessions(),
        fetchVolunteerTasks(),
        fetchCalls(),
        fetchSchedule(),
        fetchOperationsQueue(),
        fetchScenarios(),
        fetchSeniorRecords(),
        fetchCallPlans()
      ]);
      setServiceStatus(nextStatus);
      setLoadedSeniors(nextSeniors);
      setLoadedSessions(nextSessions);
      setLoadedTasks(nextTasks);
      setLoadedCalls(nextCalls);
      setLoadedSchedule(nextSchedule);
      setLoadedOperationsQueue(nextQueue);
      setLoadedScenarios(nextScenarios);
      setLoadedSeniorRecords(nextRecords);
      setLoadedCallPlans(nextCallPlans);
      setSelectedSeniorId((current) => (nextSeniors.some((senior) => senior.id === current) ? current : nextSeniors[0]?.id ?? "s-001"));
      setLastSyncedAt(new Date().toISOString());
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    void Promise.all([fetchServiceStatus(), fetchSeniors(), fetchSessions(), fetchVolunteerTasks(), fetchCalls(), fetchSchedule(), fetchOperationsQueue(), fetchScenarios(), fetchSeniorRecords(), fetchCallPlans()]).then(
      ([nextStatus, nextSeniors, nextSessions, nextTasks, nextCalls, nextSchedule, nextQueue, nextScenarios, nextRecords, nextCallPlans]) => {
        setServiceStatus(nextStatus);
        setLoadedSeniors(nextSeniors);
        setLoadedSessions(nextSessions);
        setLoadedTasks(nextTasks);
        setLoadedCalls(nextCalls);
        setLoadedSchedule(nextSchedule);
        setLoadedOperationsQueue(nextQueue);
        setLoadedScenarios(nextScenarios);
        setLoadedSeniorRecords(nextRecords);
        setLoadedCallPlans(nextCallPlans);
        setSelectedSeniorId(nextSeniors[0]?.id ?? "s-001");
        setLastSyncedAt(new Date().toISOString());
      }
    );
  }, []);

  if (!loadedSeniors.length) {
    return <div className="loading">Loading EarlyCare...</div>;
  }

  const urgentTasks = loadedTasks.filter((task) => task.priority === "Urgent" && task.status !== "Closed").length;
  const openTasks = loadedTasks.filter((task) => task.status !== "Closed").length;
  const dueNow = loadedSchedule.filter((item) => item.status === "Due now" || item.status === "Overdue").length;
  const activeQueueItems = loadedOperationsQueue.filter((item) => item.priority !== "Routine").sort((a, b) => a.queueRank - b.queueRank);
  const topQueueItem = activeQueueItems[0] ?? null;
  const queuePriorityCounts = ["Emergency", "Today", "Due", "Routine"].map((priority) => ({
    priority,
    count: loadedOperationsQueue.filter((item) => item.priority === priority).length
  }));

  const handleTaskStatus = async (taskId: string, status: VolunteerTask["status"]) => {
    const updated = await updateVolunteerTask(taskId, status);
    if (updated) {
      setLoadedTasks((tasks) => tasks.map((task) => (task.id === updated.id ? updated : task)));
      await refreshServiceStatus();
      await refreshSeniorRecords();
      await refreshCallPlans();
      await refreshOperationsQueue();
      return;
    }
    setLoadedTasks((tasks) => tasks.map((task) => (task.id === taskId ? { ...task, status } : task)));
    await refreshServiceStatus();
    await refreshOperationsQueue();
  };

  const handleRecordMissedCheckIn = async (seniorId: string) => {
    const scheduleItem = loadedSchedule.find((item) => item.seniorId === seniorId);
    const now = new Date().toISOString();
    const response = await recordMissedCheckIn({
      seniorId,
      scheduledAt: scheduleItem?.nextDueAt ?? now,
      retryAt: now,
      attemptCount: 2,
      note: "Logged from Care desk after scheduled call and retry were unanswered."
    });

    if (!response) return false;

    setLoadedSessions((sessions) => [response.session, ...sessions.filter((session) => session.id !== response.session.id)]);
    setLoadedTasks(response.tasks);
    setSelectedSeniorId(response.session.seniorId);
    await refreshServiceStatus();
    await refreshSchedule();
    await refreshOperationsQueue();
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
      summary: "Completed scheduled check-in from Care desk."
    });
    if (!completed) return false;

    setLoadedSessions((sessions) => [completed, ...sessions.filter((session) => session.id !== completed.id && session.id !== started.id)]);
    setSelectedSeniorId(completed.seniorId);
    await refreshServiceStatus();
    await refreshTasks();
    await refreshSchedule();
    await refreshOperationsQueue();
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
        <div className="topbar-actions">
          <ServiceStatusIndicator status={serviceStatus} />
          <nav aria-label="Primary workspace">
            <button className={view === "demo" ? "active" : ""} onClick={() => setView("demo")}>
              <ClipboardList size={18} />
              Demo runner
            </button>
            <button className={view === "call" ? "active" : ""} onClick={() => setView("call")}>
              <Headphones size={18} />
              Live call
            </button>
            <button className={view === "dashboard" ? "active" : ""} onClick={() => setView("dashboard")}>
              <Activity size={18} />
              Care desk
            </button>
          </nav>
        </div>
      </header>

      <section className="hero-band" aria-label="Care desk command center">
        <div className="hero-copy">
          <div className="ops-title-row">
            <span className="eyebrow">Care desk command center</span>
            <span className="sync-chip">
              <Activity size={14} />
              {formatSyncTime(lastSyncedAt)}
            </span>
          </div>
          <h1>Care desk</h1>
          <p>Triage scheduled voice check-ins, unanswered attempts, and human follow-up before silence turns into a welfare risk.</p>
          <div className="ops-command-panel">
            <div className="priority-brief">
              <span className={`priority priority-${topQueueItem?.priority.toLowerCase() ?? "routine"}`}>{topQueueItem?.priority ?? "Clear"}</span>
              <strong>{topQueueItem ? `${topQueueItem.seniorName}: ${topQueueItem.reason}` : "No elevated queue item right now."}</strong>
              <small>{topQueueItem ? topQueueItem.recommendedAction : "Continue scheduled monitoring and keep the service connection visible."}</small>
            </div>
            <div className="ops-command-actions">
              <button
                className="primary-action"
                disabled={!topQueueItem}
                onClick={() => {
                  if (!topQueueItem) return;
                  setSelectedSeniorId(topQueueItem.seniorId);
                  setView("dashboard");
                }}
                type="button"
              >
                <AlertTriangle size={18} />
                Open priority
              </button>
              <button className="secondary-action" disabled={isRefreshing} onClick={() => void refreshWorkspace()} type="button">
                <RefreshCw className={isRefreshing ? "spin" : undefined} size={18} />
                {isRefreshing ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>
          <div className="queue-pressure-strip" aria-label="Operations queue pressure">
            <div className="pressure-summary">
              <span className="eyebrow">Queue pressure</span>
              <strong>{activeQueueItems.length} active</strong>
              <small>{loadedSeniors.length} seniors covered</small>
            </div>
            <div className="pressure-segments">
              {queuePriorityCounts.map((item) => (
                <span className={`pressure-segment pressure-${item.priority.toLowerCase()}`} key={item.priority}>
                  <strong>{item.count}</strong>
                  <small>{item.priority}</small>
                </span>
              ))}
            </div>
          </div>
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

      {serviceStatus.mode === "demo" || serviceStatus.storageWarnings?.length ? (
        <section className={`service-banner ${serviceStatus.storageWarnings?.length ? "service-warning-banner" : ""}`}>
          {serviceStatus.storageWarnings?.length ? <AlertTriangle size={18} /> : <Activity size={18} />}
          <p>{serviceStatus.message}</p>
        </section>
      ) : null}

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
            await refreshServiceStatus();
            await refreshSchedule();
            await refreshOperationsQueue();
            await refreshSeniorRecords();
            await refreshCallPlans();
          }}
          onOpenDashboard={() => setView("dashboard")}
        />
      ) : view === "call" ? (
        <Suspense
          fallback={
            <main className="call-only-grid">
              <section className="panel call-panel">
                <p className="empty-state">Loading call workspace...</p>
              </section>
            </main>
          }
        >
          <AgentsCall
            seniors={loadedSeniors}
            selectedSeniorId={selectedSeniorId}
            onSelectSenior={setSelectedSeniorId}
            callPlan={loadedCallPlans.find((plan) => plan.seniorId === selectedSeniorId) ?? null}
            onSavedCall={async (call) => {
              setLoadedCalls((calls) => [call, ...calls.filter((item) => item.id !== call.id)]);
              await refreshServiceStatus();
              await refreshTasks();
              await refreshSchedule();
              await refreshOperationsQueue();
              await refreshSeniorRecords();
              await refreshCallPlans();
            }}
          />
        </Suspense>
      ) : (
        <OfficerDashboard
          seniors={loadedSeniors}
          sessions={loadedSessions}
          tasks={loadedTasks}
          calls={loadedCalls}
          schedule={loadedSchedule}
          operationsQueue={loadedOperationsQueue}
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
