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
  ShieldAlert,
  ShieldCheck,
  Stethoscope,
  Timer,
  UserRoundCheck,
  UsersRound
} from "lucide-react";
import {
  createElevenLabsSession,
  fetchCalls,
  fetchScenarios,
  fetchSeniors,
  fetchSessions,
  fetchVolunteerTasks,
  getCallAudioUrl,
  runScenario,
  saveCall,
  updateVolunteerTask
} from "./api";
import type {
  CallRecord,
  CheckInSession,
  ConversationCategory,
  EscalationStep,
  RiskLevel,
  RiskSignal,
  Scenario,
  Senior,
  TranscriptMessage,
  VolunteerTask
} from "./types";
import "./styles.css";

const riskOrder: Record<RiskLevel, number> = { Green: 0, Watch: 1, Amber: 2, Red: 3 };
type AppView = "demo" | "call" | "dashboard";
type CallState = "Ready" | "Connecting" | "In call" | "Saving" | "Analysing" | "Complete" | "Failed";
type ScenarioTone = { label: string; risk: RiskLevel; detail: string };

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
    .replace(/\s*\[(happy|relieved|good|sad|angry|calm|cheerful|concerned|empathetic|laughs?|sighs?|pause|thinking)\]\s*/gi, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function formatDate(value?: string | null): string {
  return value ? new Date(value).toLocaleString() : "Not completed";
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
          {line.role}: {line.text}
        </p>
      ))}
    </div>
  );
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
  const transcriptRef = useRef<TranscriptMessage[]>([]);

  const conversation = useConversation({
    onConnect: () => {
      setCallState("In call");
      setCallMessage("Agent connected. Waiting for the first check-in question...");
      window.setTimeout(() => {
        const hasAgentMessage = transcriptRef.current.some((line) => line.role === "Agent");
        if (!hasAgentMessage) {
          conversation.sendUserMessage(
            [
              `Begin the EarlyCare check-in for ${selectedSenior.name}.`,
              `Preferred language: ${selectedSenior.preferredLanguage}.`,
              `Known conditions: ${selectedSenior.knownConditions.join(", ") || "none listed"}.`,
              `Focus areas: ${selectedSenior.promptFocus.join(", ") || "basic wellbeing"}.`,
              "Ask for consent, then cover wellbeing, falls, head impact, whiplash or jolts, headache, dizziness, vomiting, confusion, weakness, numbness, slurred speech, food, water, medication, loneliness, and the repeat phrase.",
              "Do not diagnose. Escalate only as follow-up guidance."
            ].join(" ")
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
    onMessage: (message) => {
      const line: TranscriptMessage = {
        role: message.role === "agent" ? "Agent" : "Senior",
        text: cleanTranscriptText(message.message),
        timestamp: new Date().toISOString()
      };
      transcriptRef.current = [...transcriptRef.current, line];
      setTranscriptMessages(transcriptRef.current);
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
      const recorder = new MediaRecorder(stream);
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
    if (audioBlob) formData.append("audio", audioBlob, "mic-audio.webm");

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
  onScenarioRun
}: {
  scenarios: Scenario[];
  seniors: Senior[];
  selectedSeniorId: string;
  onSelectSenior: (id: string) => void;
  onScenarioRun: (session: CheckInSession, tasks: VolunteerTask[]) => void;
}) {
  const selectedScenario = scenarios.find((scenario) => scenario.seniorId === selectedSeniorId) ?? scenarios[0];
  const [scenarioId, setScenarioId] = useState(selectedScenario?.id ?? "");
  const [lastSession, setLastSession] = useState<CheckInSession | null>(null);
  const [status, setStatus] = useState("");
  const activeScenario = scenarios.find((scenario) => scenario.id === scenarioId) ?? selectedScenario;
  const activeSenior = seniors.find((senior) => senior.id === activeScenario?.seniorId) ?? seniors[0];
  const activeTone = activeScenario ? scenarioToneFor(activeScenario) : null;

  useEffect(() => {
    if (activeScenario && activeScenario.seniorId !== selectedSeniorId) {
      onSelectSenior(activeScenario.seniorId);
    }
  }, [activeScenario, onSelectSenior, selectedSeniorId]);

  const executeScenario = async () => {
    if (!activeScenario) return;
    setStatus("Running scenario and saving check-in...");
    const response = await runScenario(activeScenario.id);
    if (!response) {
      setStatus("Backend API is required to save scenario history.");
      return;
    }
    setLastSession(response.session);
    onScenarioRun(response.session, response.tasks);
    setStatus("Scenario saved to Patient overview.");
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
          <button className="primary-action" onClick={() => void executeScenario()}>
            <PlayCircle size={18} />
            Run scenario
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
          </>
        ) : (
          <p className="empty-state">Run any scenario to create a persisted check-in, categorized evidence, and follow-up task when needed.</p>
        )}
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
  onStartCall,
  onTaskStatus
}: {
  seniors: Senior[];
  sessions: CheckInSession[];
  tasks: VolunteerTask[];
  calls: CallRecord[];
  selectedSeniorId: string;
  setSelectedSeniorId: (id: string) => void;
  onStartCall: (id: string) => void;
  onTaskStatus: (taskId: string, status: VolunteerTask["status"]) => void;
}) {
  const selectedSenior = seniors.find((senior) => senior.id === selectedSeniorId) ?? seniors[0];
  const selectedTasks = tasks
    .filter((task) => task.seniorId === selectedSenior.id)
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
  const selectedCalls = calls.filter((call) => call.seniorId === selectedSenior.id);
  const selectedSessions = sessions.filter((session) => session.seniorId === selectedSenior.id);
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
          {seniors.map((senior) => {
            const seniorRecords = [...calls, ...sessions].filter((record) => record.seniorId === senior.id);
            const seniorRisk = highestRiskLevel(seniorRecords.map((record) => record.riskLevel));
            const seniorOpenTasks = tasks.filter((task) => task.seniorId === senior.id && task.status !== "Closed").length;
            return (
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
                <span className="senior-row-meta">
                  <RiskBadge level={seniorRisk} />
                  {seniorOpenTasks ? <small>{seniorOpenTasks} task{seniorOpenTasks === 1 ? "" : "s"}</small> : <small>clear</small>}
                </span>
              </button>
            );
          })}
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
                return (
                  <article className="call-record" key={call.id}>
                    <div className="call-record-header">
                      <div>
                        <RiskBadge level={call.riskLevel} />
                        <strong>{formatDate(call.completedAt)}</strong>
                      </div>
                      <small>
                        {call.translationProvider}
                        {call.translationFallbackUsed ? " fallback" : ""} · audio {call.audioAvailable ? "saved" : "missing"}
                      </small>
                    </div>
                    <p>
                      <strong>Recommended action:</strong> {call.recommendedAction}
                    </p>

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
                    <div className="transcript-columns">
                      <div>
                        <h4>Original transcript</h4>
                        <pre>{call.originalTranscript}</pre>
                      </div>
                      <div>
                        <h4>English transcript</h4>
                        <pre>{call.englishTranscript}</pre>
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
  const [view, setView] = useState<AppView>("demo");
  const [loadedSeniors, setLoadedSeniors] = useState<Senior[]>([]);
  const [loadedSessions, setLoadedSessions] = useState<CheckInSession[]>([]);
  const [loadedTasks, setLoadedTasks] = useState<VolunteerTask[]>([]);
  const [loadedCalls, setLoadedCalls] = useState<CallRecord[]>([]);
  const [loadedScenarios, setLoadedScenarios] = useState<Scenario[]>([]);
  const [selectedSeniorId, setSelectedSeniorId] = useState("s-001");

  const refreshTasks = async () => {
    const nextTasks = await fetchVolunteerTasks();
    setLoadedTasks(nextTasks);
  };

  useEffect(() => {
    void Promise.all([fetchSeniors(), fetchSessions(), fetchVolunteerTasks(), fetchCalls(), fetchScenarios()]).then(
      ([nextSeniors, nextSessions, nextTasks, nextCalls, nextScenarios]) => {
        setLoadedSeniors(nextSeniors);
        setLoadedSessions(nextSessions);
        setLoadedTasks(nextTasks);
        setLoadedCalls(nextCalls);
        setLoadedScenarios(nextScenarios);
        setSelectedSeniorId(nextSeniors[0]?.id ?? "s-001");
      }
    );
  }, []);

  if (!loadedSeniors.length) {
    return <div className="loading">Loading EarlyCare...</div>;
  }

  const urgentTasks = loadedTasks.filter((task) => task.priority === "Urgent" && task.status !== "Closed").length;
  const openTasks = loadedTasks.filter((task) => task.status !== "Closed").length;
  const redRecords = [...loadedCalls, ...loadedSessions].filter((record) => record.riskLevel === "Red").length;

  const handleTaskStatus = async (taskId: string, status: VolunteerTask["status"]) => {
    const updated = await updateVolunteerTask(taskId, status);
    if (updated) {
      setLoadedTasks((tasks) => tasks.map((task) => (task.id === updated.id ? updated : task)));
      return;
    }
    setLoadedTasks((tasks) => tasks.map((task) => (task.id === taskId ? { ...task, status } : task)));
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
        <nav>
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
        <div>
          <span className="eyebrow">Preventive care + volunteer escalation</span>
          <h1>Scheduled calls that turn silence, falls, and speech change into earlier human follow-up.</h1>
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
          <StatCard label="Open tasks" value={`${openTasks}`} icon={<Bell size={20} />} />
          <StatCard label="Urgent" value={`${urgentTasks}`} icon={<AlertTriangle size={20} />} />
          <StatCard label="Red records" value={`${redRecords}`} icon={<ShieldAlert size={20} />} />
          <StatCard label="Safety stance" value="No diagnosis" icon={<ShieldCheck size={20} />} />
        </div>
      </section>

      {view === "demo" ? (
        <ScenarioRunner
          scenarios={loadedScenarios}
          seniors={loadedSeniors}
          selectedSeniorId={selectedSeniorId}
          onSelectSenior={setSelectedSeniorId}
          onScenarioRun={(session, tasks) => {
            setLoadedSessions((sessions) => [session, ...sessions.filter((item) => item.id !== session.id)]);
            setLoadedTasks(tasks);
            setSelectedSeniorId(session.seniorId);
            setView("dashboard");
          }}
        />
      ) : view === "call" ? (
        <ConversationProvider>
          <AgentsCall
            seniors={loadedSeniors}
            selectedSeniorId={selectedSeniorId}
            onSelectSenior={setSelectedSeniorId}
            onSavedCall={async (call) => {
              setLoadedCalls((calls) => [call, ...calls.filter((item) => item.id !== call.id)]);
              await refreshTasks();
            }}
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
          onTaskStatus={(taskId, status) => void handleTaskStatus(taskId, status)}
        />
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
