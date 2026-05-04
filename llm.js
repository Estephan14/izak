// OpenRouter streaming client + prompt builder

// Sonnet is intentionally used here for better judgment on noisy pose data.
export const MODEL = 'anthropic/claude-sonnet-4.5';

const SYSTEM_PROMPT = `You are a real-time fitness coach watching someone work out via a phone camera.

STRICT RULES:
1. Reply with ONE sentence, max 12 words.
2. The "Allowed cues" line is the source of truth.
3. If allowed cues are listed, choose one and use nearly the same wording.
4. Do NOT mention core, hips, breathing, bracing, balance, depth, or tempo unless those exact words appear in Allowed cues.
5. If "Allowed cues: NONE" give a neutral rep-count cue ("Rep logged.", "Good rep.").
6. If near failure be urgent ("One more, give it everything!").
7. No greetings, no prefixes, no explanations. Just the cue.`;

export async function* streamCompletion(messages, apiKey) {
  const useProxy = !apiKey;
  const res = await fetch(useProxy ? '/api/openrouter' : 'https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      'Content-Type': 'application/json',
      ...(apiKey ? {
        'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : 'https://fitcoach.ai',
        'X-Title': 'FitCoach AI',
      } : {}),
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      stream: true,
      max_tokens: 40,
      temperature: 0.4,
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

export function buildPrompt(exercise, metrics, repCount, failureInfo, phase, formSnapshot = null) {
  const m = metrics;
  const parts = [];

  if (m.rightElbow     != null) parts.push(`R.Elbow: ${Math.round(m.rightElbow)} deg`);
  if (m.leftElbow      != null) parts.push(`L.Elbow: ${Math.round(m.leftElbow)} deg`);
  if (m.rightKnee      != null) parts.push(`R.Knee: ${Math.round(m.rightKnee)} deg`);
  if (m.leftKnee       != null) parts.push(`L.Knee: ${Math.round(m.leftKnee)} deg`);
  if (m.rightShoulder  != null) parts.push(`R.Shoulder: ${Math.round(m.rightShoulder)} deg`);
  if (m.rightHip       != null) parts.push(`R.Hip: ${Math.round(m.rightHip)} deg`);
  if (m.spineTilt      != null) parts.push(`Spine tilt: ${Math.round(m.spineTilt)} deg`);
  if (m.hipTilt        != null) parts.push(`Hip sway: ${m.hipTilt.toFixed(1)}%`);
  if (m.elbowAsymmetry != null) parts.push(`Arm diff: ${Math.round(m.elbowAsymmetry)} deg`);

  const form = formSnapshot ?? exercise.formChecks(m);
  const issues = form.issues;
  const score = form.score;

  const issuesLine = issues.length ? issues.join(', ') : 'NONE';
  const allowedCues = buildAllowedCues(issues);
  const cuesLine = allowedCues.length ? allowedCues.join(' | ') : 'NONE';
  const failureLine = failureInfo?.isNearFailure
    ? `\nNEAR FAILURE: reps slowing ${Math.round((failureInfo.slowingFactor - 1) * 100)}%, ROM dropping ${Math.round(failureInfo.rangeLoss * 100)}%`
    : '';

  const userContent =
`${exercise.name}${repCount > 0 ? ` | Rep #${repCount}` : ''}${phase ? ` | ${phase}` : ''}
Form score: ${score}/100
Detected issues: ${issuesLine}
Allowed cues: ${cuesLine}${failureLine}

Angles:
${parts.map(p => `- ${p}`).join('\n')}`;

  return [
    { role: 'system', content: `${SYSTEM_PROMPT}\n\nExercise: ${exercise.coachingContext}` },
    { role: 'user', content: userContent },
  ];
}

function buildAllowedCues(issues) {
  const cues = [];
  for (const issue of issues) {
    const text = issue.toLowerCase();
    if (/incomplete arm extension/.test(text)) cues.push('Reach full arm extension before the next rep.');
    else if (/incomplete lockout/.test(text)) cues.push('Lock out fully at the top.');
    else if (/not high enough at top/.test(text)) cues.push('Pull higher before lowering.');
    else if (/not curled high enough/.test(text)) cues.push('Curl higher before lowering.');
    else if (/not low enough/.test(text)) cues.push('Lower closer to full depth.');
    else if (/arm imbalance|uneven curl|uneven press|uneven push|uneven extension/.test(text)) cues.push('Even out left and right side.');
    else if (/torso swinging|swinging|body swing|momentum/.test(text)) cues.push('Reduce swing before the next rep.');
    else if (/excessive lean|forward lean/.test(text)) cues.push('Bring your chest more upright.');
    else if (/hips not aligned|hip rotation|hip shift/.test(text)) cues.push('Keep your body line steady.');
    else if (/uneven depth/.test(text)) cues.push('Match depth on both sides.');
    else if (/back rounding/.test(text)) cues.push('Stop and reset your back position.');
    else if (/too much knee bend/.test(text)) cues.push('Use a smaller knee bend.');
    else if (/bend your knees/.test(text)) cues.push('Add a small knee bend.');
    else if (/extend fully/.test(text)) cues.push('Finish the extension fully.');
    else if (/guide hand/.test(text)) cues.push('Release with only the shooting hand.');
    else if (/off-balance/.test(text)) cues.push('Center your weight before release.');
  }
  return [...new Set(cues)].slice(0, 2);
}
