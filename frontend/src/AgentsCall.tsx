import { useRef, useState } from "react";
import { ConversationProvider, useConversation } from "@elevenlabs/react";
import { Mic, PhoneCall } from "lucide-react";
import { createElevenLabsSession, saveCall } from "./api";
import type { CallPlan, CallRecord, Senior, TranscriptMessage } from "./types";

type CallState = "Ready" | "Connecting" | "In call" | "Saving" | "Analysing" | "Complete" | "Failed";
type AgentAudioFormat = "pcm_8000" | "pcm_16000" | "pcm_22050" | "pcm_24000" | "pcm_44100" | "pcm_48000" | "ulaw_8000";

export interface AgentsCallProps {
  seniors: Senior[];
  callPlan?: CallPlan | null;
  selectedSeniorId: string;
  onSelectSenior: (id: string) => void;
  onSavedCall: (call: CallRecord) => void | Promise<void>;
}

function cleanTranscriptText(text: string): string {
  return text
    .replace(/\s*\[[^\]\r\n]{1,40}\]\s*/g, " ")
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

function AgentsCallWorkspace({ seniors, callPlan, selectedSeniorId, onSelectSenior, onSavedCall }: AgentsCallProps) {
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

export default function AgentsCall(props: AgentsCallProps) {
  return (
    <ConversationProvider>
      <AgentsCallWorkspace {...props} />
    </ConversationProvider>
  );
}
