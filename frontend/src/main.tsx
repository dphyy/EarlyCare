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
  Mic,
  PhoneCall,
  ShieldCheck,
  UserRoundCheck
} from "lucide-react";
import { createElevenLabsSession, fetchCalls, fetchSeniors, fetchSessions, fetchVolunteerTasks, getCallAudioUrl, saveCall } from "./api";
import type { CallRecord, CheckInSession, RiskLevel, RiskSignal, Senior, TranscriptMessage, VolunteerTask } from "./types";
import "./styles.css";

const riskOrder: Record<RiskLevel, number> = { Green: 0, Watch: 1, Amber: 2, Red: 3 };
type AppView = "call" | "dashboard";
type CallState = "Ready" | "Connecting" | "In call" | "Saving" | "Analysing" | "Complete" | "Failed";

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
    .replace(/\s*\[(happy|relieved|good|sad|angry|calm|cheerful|concerned|empathetic|laughs?|sighs?|pause|thinking)\]\s*/gi, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
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
  const [transcriptMessages, setTranscriptMessages] = useState<TranscriptMessage[]>([]);
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const transcriptRef = useRef<TranscriptMessage[]>([]);

  const conversation = useConversation({
    onConnect: () => {
      setCallState("In call");
      setCallMessage("Agent connected. Waiting for the first agent message...");
      window.setTimeout(() => {
        const hasAgentMessage = transcriptRef.current.some((line) => line.role === "Agent");
        if (!hasAgentMessage) {
          conversation.sendUserMessage(
            `Please begin the EarlyCare wellbeing check-in for ${selectedSenior.name}. Ask for consent, then ask about falls, head impact, headache, dizziness, vomiting, confusion, weakness, speech difficulty, food and water, and the repeat phrase.`
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
        checkInReason: "Routine living-alone wellbeing check-in"
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
          check_in_reason: "Routine living-alone wellbeing check-in",
          baseline_reminder: "Use EarlyCare's stored personal baseline context without saying exact baseline metrics aloud.",
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
      onSavedCall(saved);
      setCallState("Complete");
      setCallMessage("Call saved. Patient overview now shows original and English transcripts.");
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

          <div className="waveform" aria-label="voice waveform">
            {Array.from({ length: 36 }).map((_, index) => (
              <span key={index} style={{ height: `${18 + ((index * 17) % 58)}px` }} />
            ))}
          </div>

          <div className="transcript">
            {(transcriptMessages.length ? transcriptMessages : [{ role: "System", text: "Click Start call. The agent will begin talking once connected." }]).map(
              (line, index) => (
                <p
                  className={line.role === "Senior" ? "senior-line" : line.role === "System" ? "system-line" : "agent-line"}
                  key={`${line.role}-${line.text}-${index}`}
                >
                  {line.role}: {line.text}
                </p>
              )
            )}
          </div>

          <div className="answer-box">
            <p>{callMessage}</p>
            {debugMessage ? <small>Debug: {debugMessage}</small> : null}
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
                      Translation provider: {latestSignal.translationProvider}
                      {latestSignal.translationFallbackUsed ? " (fallback used)" : ""}. Audio recording:{" "}
                      {latestSignal.audioAvailable ? "saved" : "not available"}.
                    </p>
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
                const riskSignals = call.riskSignals ?? [];
                return (
                  <article className="call-record" key={call.id}>
                    <div className="call-record-header">
                      <div>
                        <RiskBadge level={call.riskLevel} />
                        <strong>{new Date(call.completedAt).toLocaleString()}</strong>
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
