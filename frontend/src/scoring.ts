import type { RiskAssessment, Scenario, Senior } from "./types";

function cosineDistance(a: number[] = [], b: number[] = []): number {
  if (!a.length || !b.length || a.length !== b.length) return 0.3;
  const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
  const magA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
  const magB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
  if (!magA || !magB) return 0.3;
  return Math.max(0, 1 - dot / (magA * magB));
}

function clampScore(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value)));
}

export function assessScenario(senior: Senior, scenario: Scenario): RiskAssessment {
  if (scenario.symptoms.missedCheckIn) {
    return {
      speechDeviationScore: 0,
      parkinsonsWatchScore: 0,
      postFallConcernScore: 0,
      missedCheckInScore: 100,
      riskLevel: "Amber",
      reasons: ["Scheduled check-in missed after retry", "Volunteer follow-up needed because the senior lives alone"]
    };
  }

  const baseline = senior.baselineSpeechProfile;
  const current = scenario.speechMetrics;
  const embeddingDelta = cosineDistance(baseline.embedding, current.embedding) * 100;
  const rateDelta = Math.abs(current.speechRate - baseline.speechRate) / baseline.speechRate;
  const pauseDelta = Math.abs(current.avgPauseMs - baseline.avgPauseMs) / baseline.avgPauseMs;
  const latencyDelta = Math.abs(current.responseLatencyMs - baseline.responseLatencyMs) / baseline.responseLatencyMs;
  const pitchDelta = Math.abs(current.pitchVariability - baseline.pitchVariability) / Math.max(baseline.pitchVariability, 0.1);
  const phraseDelta = Math.max(0, baseline.phraseAccuracy - current.phraseAccuracy);

  const speechDeviationScore = clampScore(
    embeddingDelta * 0.45 + rateDelta * 35 + pauseDelta * 24 + latencyDelta * 18 + pitchDelta * 20 + phraseDelta * 35
  );

  const parkinsonsWatchScore = clampScore(
    (current.speechRate < baseline.speechRate * 0.8 ? 22 : 0) +
      (current.avgPauseMs > baseline.avgPauseMs * 1.55 ? 24 : 0) +
      (current.pitchVariability < baseline.pitchVariability * 0.7 ? 22 : 0) +
      (current.phraseAccuracy < baseline.phraseAccuracy * 0.88 ? 14 : 0) +
      Math.min(18, speechDeviationScore * 0.22)
  );

  const dangerSigns = [
    scenario.symptoms.confusion,
    scenario.symptoms.vomiting,
    scenario.symptoms.slurredSpeech,
    scenario.symptoms.weakness
  ].filter(Boolean).length;

  const postFallConcernScore = clampScore(
    (scenario.symptoms.fall ? 24 : 0) +
      (scenario.symptoms.headImpact ? 24 : 0) +
      (scenario.symptoms.headache ? 12 : 0) +
      (scenario.symptoms.dizziness ? 10 : 0) +
      dangerSigns * 18 +
      (scenario.symptoms.fall || scenario.symptoms.headImpact ? speechDeviationScore * 0.22 : 0)
  );

  const reasons: string[] = [];
  if (speechDeviationScore > 45) reasons.push("Speech differs meaningfully from personal baseline");
  if (parkinsonsWatchScore > 50) reasons.push("Gradual pattern resembles Parkinson's watch markers: slower rate, longer pauses, lower pitch variation");
  if (scenario.symptoms.fall) reasons.push("Fall reported during check-in");
  if (scenario.symptoms.headImpact) reasons.push("Head impact reported");
  if (dangerSigns) reasons.push("Danger signs reported: confusion, slurred speech, weakness, or vomiting");
  if (!reasons.length) reasons.push("No concerning symptoms and speech remains close to baseline");

  let riskLevel: RiskAssessment["riskLevel"] = "Green";
  if (postFallConcernScore >= 75) riskLevel = "Red";
  else if (postFallConcernScore >= 40 || speechDeviationScore >= 60) riskLevel = "Amber";
  else if (parkinsonsWatchScore >= 50 || speechDeviationScore >= 35) riskLevel = "Watch";

  return {
    speechDeviationScore,
    parkinsonsWatchScore,
    postFallConcernScore,
    missedCheckInScore: 0,
    riskLevel,
    reasons
  };
}
