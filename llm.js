// OpenRouter streaming client + prompt builder

const SYSTEM_PROMPT = `You are an elite fitness and sports coach AI. You receive real-time pose joint-angle data from a phone camera and give ONE brief coaching cue.

Rules:
- Max 20 words
- Present tense, active voice ("Keep your...", "Drive those...", "Lock out...")
- Focus on the single most critical form issue
- If form is great, give a power-up ("Beautiful form, push through!")
- If near failure, be urgent ("Don't stop — one more rep in you!")
- No greetings, no explanations — just the cue`;

export async function* streamCompletion(messages, model, apiKey) {
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : 'https://fitcoach.ai',
      'X-Title': 'FitCoach AI',
    },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
      max_tokens: 60,
      temperature: 0.8,
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`OpenRouter ${res.status}: ${txt}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') return;
      try {
        const chunk = JSON.parse(data).choices?.[0]?.delta?.content;
        if (chunk) yield chunk;
      } catch {}
    }
  }
}

export function buildPrompt(exercise, metrics, repCount, failureInfo, phase) {
  const m = metrics;
  const parts = [];

  if (m.rightElbow != null) parts.push(`R.Elbow: ${Math.round(m.rightElbow)}°`);
  if (m.leftElbow != null)  parts.push(`L.Elbow: ${Math.round(m.leftElbow)}°`);
  if (m.rightKnee != null)  parts.push(`R.Knee: ${Math.round(m.rightKnee)}°`);
  if (m.leftKnee != null)   parts.push(`L.Knee: ${Math.round(m.leftKnee)}°`);
  if (m.rightShoulder != null) parts.push(`R.Shoulder: ${Math.round(m.rightShoulder)}°`);
  if (m.leftShoulder != null)  parts.push(`L.Shoulder: ${Math.round(m.leftShoulder)}°`);
  if (m.rightHip != null)   parts.push(`R.Hip: ${Math.round(m.rightHip)}°`);
  if (m.spineTilt != null)  parts.push(`Spine tilt: ${Math.round(m.spineTilt)}°`);
  if (m.hipTilt != null)    parts.push(`Hip sway: ${m.hipTilt.toFixed(1)}%`);
  if (m.elbowAsymmetry != null) parts.push(`Arm diff: ${Math.round(m.elbowAsymmetry)}°`);

  const issues = exercise.formChecks(m).issues;
  const failureLine = failureInfo?.isNearFailure
    ? `\n⚠ NEAR FAILURE — speed +${Math.round((failureInfo.slowingFactor - 1) * 100)}%, ROM -${Math.round(failureInfo.rangeLoss * 100)}%`
    : '';

  const userContent = [
    `${exercise.name}${repCount > 0 ? ` | Rep #${repCount}` : ''}${phase ? ` | phase: ${phase}` : ''}`,
    `\nAngles:\n${parts.map(p => `- ${p}`).join('\n')}`,
    issues.length ? `\nDetected issues: ${issues.join(', ')}` : '',
    failureLine,
  ].join('');

  return [
    { role: 'system', content: `${SYSTEM_PROMPT}\n\nExercise context: ${exercise.coachingContext}` },
    { role: 'user', content: userContent },
  ];
}
