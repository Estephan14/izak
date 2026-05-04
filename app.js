import { EXERCISES, EXERCISE_GROUPS, extractMetrics, RepCounter } from './exercises.js';
import { streamCompletion, buildPrompt } from './llm.js';
import tts from './tts.js';

let PoseLandmarker = null;
let FilesetResolver = null;

// ─── DOM refs ────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const on = (el, event, handler, opts) => {
  if (el) el.addEventListener(event, handler, opts);
};
const video         = $('video');
const canvas        = $('canvas');
const ctx           = canvas.getContext('2d');
const repCountEl    = $('rep-count');
const exNameEl      = $('exercise-name');
const statusDot     = $('status-dot');
const feedbackEl    = $('feedback-text');
const phaseEl       = $('phase-indicator');
const formScoreEl   = $('form-score');
const startBtn      = $('start-btn');
const flipBtn       = $('flip-btn');
const settingsBtn   = $('settings-btn');
const settingsPanel = $('settings-panel');
const closeSettings = $('close-settings');
const saveSettings  = $('save-settings');
const resetReps     = $('reset-reps');
const loadingOv     = $('loading-overlay');
const loadingMsg    = $('loading-msg');

// ─── Config (localStorage) ───────────────────────────────────────────────────
const cfg = (() => {
  const defaults = {
    openrouterKey: '', elevenLabsKey: '', elevenLabsVoice: '',
    exercise: 'pullup', voiceEnabled: true,
  };
  let data = { ...defaults };
  return {
    load() {
      try { Object.assign(data, JSON.parse(localStorage.getItem('fc_cfg') || '{}')); } catch {}
    },
    save() { localStorage.setItem('fc_cfg', JSON.stringify(data)); },
    get(k)     { return data[k]; },
    set(k, v)  { data[k] = v; },
  };
})();

// ─── State ───────────────────────────────────────────────────────────────────
let poseLandmarker = null;
let stream         = null;
let facingMode     = 'environment';
let running        = false;
let animId         = null;
let lastVideoTime  = -1;
let lastLLMCall    = 0;
let llmInFlight    = false;
let repCounter     = null;
let exercise       = null;
let wakeLock       = null;
let videoMetrics   = { dW: 0, dH: 0, oX: 0, oY: 0 }; // cover-mode draw dimensions
let setState       = 'setup'; // setup | ready | active
let startHeldAt    = null;
let startLostAt    = null;
let readyMetrics   = null;
let activeStartedAt = null;
let inactiveSince  = null;
let lastAngle      = null;
let motionRangeMin = Infinity;
let motionRangeMax = -Infinity;
let lastIssueKey   = '';
let issueSeenAt    = null;
let lastFeedbackRep = 0;
let firstRepFeedbackDone = false;
let pendingRepFeedback = false;
let serverConfig = { openrouter: false, elevenlabs: false };

const READY_HOLD_MS = 250;       // brief stable-pose check before going 'ready'
const POSE_LOST_GRACE_MS = 800;  // tolerate dropped frames before falling back to setup
const ACTIVE_MOTION_DEG = 5;     // angle delta that promotes ready → active
const STOP_IDLE_MS = 6500;
const MIN_FEEDBACK_REPS = 1;
const MIN_CLEAN_FEEDBACK_REPS = 2;
const FIRST_REP_DELAY_MS = 900;
const ISSUE_PERSIST_MS = 1400;
const ISSUE_COOLDOWN_MS = 9000;
const CLEAN_COOLDOWN_REPS = 4;

// Sliding metric buffer (avg before sending to LLM)
const metricsBuf = { frames: [], push(m) { this.frames.push(m); if (this.frames.length > 30) this.frames.shift(); }, avg() {
  if (!this.frames.length) return null;
  const keys = Object.keys(this.frames[0]);
  const out = {};
  for (const k of keys) {
    const vals = this.frames.map(f => f[k]).filter(v => v != null && typeof v === 'number');
    out[k] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }
  return out;
} };

function getRepAngle(metrics) {
  if (!exercise?.repCounter) return null;
  return exercise.repCounter.getAngle(metrics);
}

function hasValidPose(metrics) {
  if (!exercise?.repCounter) return true;
  return getRepAngle(metrics) != null;
}

function resetSetState(next = 'setup') {
  setState = next;
  startHeldAt = null;
  startLostAt = null;
  readyMetrics = null;
  activeStartedAt = null;
  inactiveSince = null;
  lastAngle = null;
  motionRangeMin = Infinity;
  motionRangeMax = -Infinity;
  metricsBuf.frames.length = 0;
  if (repCounter) repCounter.reset();
  lastIssueKey = '';
  issueSeenAt = null;
  lastFeedbackRep = 0;
  firstRepFeedbackDone = false;
  pendingRepFeedback = false;
}

function updateSetState(metrics, now) {
  const angle = getRepAngle(metrics);
  const poseOk = hasValidPose(metrics);

  // Lost the pose: tolerate brief gaps, then fall back to setup.
  if (!poseOk) {
    if (setState !== 'setup') {
      startLostAt ??= now;
      if (now - startLostAt > POSE_LOST_GRACE_MS) {
        startHeldAt = null;
        readyMetrics = null;
        setState = 'setup';
      }
    }
    return setState;
  }
  startLostAt = null;

  // setup → ready: just confirm the pose stays valid for a brief moment.
  // We don't gate on a specific extreme — the rep counter auto-anchors on
  // whichever extreme the user reaches first.
  if (setState === 'setup') {
    startHeldAt ??= now;
    if (now - startHeldAt >= READY_HOLD_MS) {
      setState = 'ready';
      readyMetrics = metrics;
      lastAngle = angle;
    }
    return setState;
  }

  // ready → active: fire as soon as we see meaningful motion.
  if (setState === 'ready') {
    if (lastAngle == null) lastAngle = angle;
    if (angle != null && Math.abs(angle - lastAngle) >= ACTIVE_MOTION_DEG) {
      setState = 'active';
      activeStartedAt = now;
      inactiveSince = null;
      motionRangeMin = angle;
      motionRangeMax = angle;
      metricsBuf.frames.length = 0;
      // RepCounter anchors itself on the first observed extreme.
    }
    lastAngle = angle ?? lastAngle;
    return setState;
  }

  // active: track motion and detect rest/idle.
  if (angle != null) {
    motionRangeMin = Math.min(motionRangeMin, angle);
    motionRangeMax = Math.max(motionRangeMax, angle);
    const delta = lastAngle == null ? 0 : Math.abs(angle - lastAngle);
    lastAngle = angle;

    const meaningfulRange = motionRangeMax - motionRangeMin >= 10;
    if (delta > 0.8 || meaningfulRange) inactiveSince = null;
    else inactiveSince ??= now;
  }

  if (inactiveSince && now - inactiveSince > STOP_IDLE_MS) {
    resetSetState('ready');
  }

  return setState;
}

function setStatusText(text, color = '#888') {
  formScoreEl.textContent = text;
  formScoreEl.style.color = color;
}

function issueKey(form) {
  return form.issues?.length ? form.issues.join('|') : 'NONE';
}

function shouldAskForFeedback(form, now) {
  if (llmInFlight || (!cfg.get('openrouterKey') && !serverConfig.openrouter) || setState !== 'active') return false;

  const reps = repCounter?.count ?? 0;
  const fi = repCounter?.getFailureIndicators();
  if (fi?.isNearFailure && reps > lastFeedbackRep) return true;
  if (exercise?.repBased && reps < MIN_FEEDBACK_REPS) return false;

  const key = issueKey(form);
  if (key !== lastIssueKey) {
    lastIssueKey = key;
    issueSeenAt = now;
    return false;
  }

  if (key !== 'NONE') {
    return now - issueSeenAt > ISSUE_PERSIST_MS && now - lastLLMCall > ISSUE_COOLDOWN_MS;
  }

  if (pendingRepFeedback && !firstRepFeedbackDone && reps >= MIN_CLEAN_FEEDBACK_REPS && now - activeStartedAt > FIRST_REP_DELAY_MS) {
    return true;
  }

  return pendingRepFeedback && reps >= lastFeedbackRep + CLEAN_COOLDOWN_REPS && now - lastLLMCall > ISSUE_COOLDOWN_MS;
}

// ─── MediaPipe init ──────────────────────────────────────────────────────────
async function initPose() {
  setLoading('Loading AI model...');
  try {
    if (!PoseLandmarker || !FilesetResolver) {
      ({ PoseLandmarker, FilesetResolver } = await import(
        /* webpackIgnore: true */ 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm'
      ));
    }
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  } catch (e) {
    setLoading('Model load failed: ' + e.message);
    throw e;
  }
  hideLoading();
}

// ─── Camera ──────────────────────────────────────────────────────────────────
async function startCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    const constraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
    };
    if (!flipBtn.classList.contains('hidden')) constraints.facingMode = facingMode;
    stream = await navigator.mediaDevices.getUserMedia({
      video: constraints,
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    return true;
  } catch (e) {
    console.error('[camera]', e);
    return false;
  }
}

async function updateCameraControls() {
  try {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    flipBtn.classList.toggle('hidden', cameras.length < 2);
  } catch {}
}

// ─── Render loop ─────────────────────────────────────────────────────────────
function resizeCanvas() {
  const w = window.innerWidth, h = window.innerHeight;
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w;
  canvas.height = h;
}

function updateVideoMetrics() {
  const vW = video.videoWidth  || 640;
  const vH = video.videoHeight || 480;
  const cW = canvas.width, cH = canvas.height;
  const vAR = vW / vH, cAR = cW / cH;
  if (vAR > cAR) {
    videoMetrics.dH = cH; videoMetrics.dW = cH * vAR;
    videoMetrics.oX = (cW - videoMetrics.dW) / 2; videoMetrics.oY = 0;
  } else {
    videoMetrics.dW = cW; videoMetrics.dH = cW / vAR;
    videoMetrics.oX = 0; videoMetrics.oY = (cH - videoMetrics.dH) / 2;
  }
}

// Map normalized landmark [0-1] to canvas pixels using cover scaling
function lmPx(lm) {
  const { dW, dH, oX, oY } = videoMetrics;
  return { x: lm.x * dW + oX, y: lm.y * dH + oY };
}

function renderLoop() {
  if (!running) return;
  resizeCanvas();
  updateVideoMetrics();

  // Draw video frame
  const { dW, dH, oX, oY } = videoMetrics;
  ctx.drawImage(video, oX, oY, dW, dH);

  // Pose detection
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const results = poseLandmarker.detectForVideo(video, performance.now());
    processFrame(results);
  }

  animId = requestAnimationFrame(renderLoop);
}

// ─── Pose processing ──────────────────────────────────────────────────────────
// POSE_CONNECTIONS entries are {start: number, end: number} objects
function scoreToColor(score) {
  if (score >= 80) return '#00d4aa';
  if (score >= 50) return '#ffb800';
  return '#ff3333';
}

function drawSkeleton(lms, color) {
  const conns = PoseLandmarker.POSE_CONNECTIONS;
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  for (const conn of conns) {
    const i = conn.start ?? conn[0];
    const j = conn.end   ?? conn[1];
    const a = lms[i], b = lms[j];
    if (!a || !b || (a.visibility ?? 1) < 0.3 || (b.visibility ?? 1) < 0.3) continue;
    const pa = lmPx(a), pb = lmPx(b);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  // Joints
  for (const lm of lms) {
    if ((lm.visibility ?? 1) < 0.3) continue;
    const p = lmPx(lm);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function drawAngleLabel(lm, angle, isGood) {
  if (angle == null || !lm || (lm.visibility ?? 1) < 0.4) return;
  const p = lmPx(lm);
  const text = Math.round(angle) + '°';
  ctx.font = 'bold 13px monospace';
  ctx.textAlign = 'center';
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(0,0,0,0.85)';
  ctx.fillStyle = isGood ? '#00d4aa' : '#ff4444';
  ctx.strokeText(text, p.x, p.y - 10);
  ctx.fillText(text, p.x, p.y - 10);
}

function processFrame(results) {
  if (!results.landmarks?.length) {
    drawHint('Step into frame');
    return;
  }

  const lms = results.landmarks[0];
  const metrics = extractMetrics(lms);
  const now = performance.now();
  const currentState = updateSetState(metrics, now);

  const prevReps = repCounter?.count ?? 0;
  if (currentState === 'active') {
    metricsBuf.push(metrics);
  }

  // Rep counter auto-anchors on the first extreme it sees, so we just feed it
  // frames as soon as we're 'active'. No "hold the start position" gate.
  if (currentState === 'active' && repCounter && exercise?.repBased) {
    repCounter.update(metrics, now);
    if (repCounter.count > prevReps) onRepCompleted();
    repCountEl.textContent = repCounter.count;
    phaseEl.textContent = repCounter.phaseName.toUpperCase();
  } else if (currentState === 'ready') {
    phaseEl.textContent = (exercise?.startCue ?? 'BEGIN WHEN READY').toUpperCase();
  } else {
    phaseEl.textContent = 'STEP INTO FRAME';
  }

  // Form analysis
  const form = exercise?.formChecks(metrics, repCounter) ?? { issues: [], score: 80 };

  // Draw skeleton
  drawSkeleton(lms, scoreToColor(form.score));

  // Draw angles at key joints
  const noArm = form.issues.some(i => /arm|imbalance|asym/i.test(i));
  drawAngleLabel(lms[14], metrics.rightElbow, !noArm);
  drawAngleLabel(lms[13], metrics.leftElbow,  !noArm);
  drawAngleLabel(lms[26], metrics.rightKnee,  form.issues.every(i => !/knee/i.test(i)));
  drawAngleLabel(lms[12], metrics.rightShoulder, true);

  // Form score/status
  if (currentState === 'active') {
    setStatusText('FORM ' + form.score + '%', scoreToColor(form.score));
  } else {
    setStatusText(currentState === 'ready' ? 'READY' : 'SETUP', currentState === 'ready' ? '#00d4aa' : '#888');
  }

  if (shouldAskForFeedback(form, now)) {
    lastLLMCall = now;
    lastFeedbackRep = repCounter?.count ?? 0;
    if (!form.issues.length) firstRepFeedbackDone = true;
    pendingRepFeedback = false;
    callLLM(false, form);
  }
}

function drawHint(msg) {
  ctx.font = 'bold 15px sans-serif';
  ctx.textAlign = 'center';
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(0,0,0,0.7)';
  ctx.fillStyle = '#ffb800';
  ctx.strokeText(msg, canvas.width / 2, canvas.height * 0.88);
  ctx.fillText(msg, canvas.width / 2, canvas.height * 0.88);
}

// ─── LLM ─────────────────────────────────────────────────────────────────────
async function callLLM(urgent, formSnapshot = null) {
  const avg = metricsBuf.avg();
  if (!avg || !exercise) return;
  llmInFlight = true;
  statusDot.className = 'thinking';

  const failInfo = repCounter?.getFailureIndicators();
  const msgs = buildPrompt(exercise, avg, repCounter?.count ?? 0, failInfo, repCounter?.phaseName, formSnapshot);

  feedbackEl.textContent = '';
  let full = '';
  try {
    for await (const chunk of streamCompletion(msgs, cfg.get('openrouterKey'))) {
      full += chunk;
      feedbackEl.textContent = full;
    }
    if (cfg.get('voiceEnabled') && (cfg.get('elevenLabsKey') || serverConfig.elevenlabs) && full.trim()) {
      tts.apiKey   = cfg.get('elevenLabsKey');
      tts.voiceId  = cfg.get('elevenLabsVoice') || undefined;
      tts.speak(full, { urgent });
    }
  } catch (e) {
    feedbackEl.textContent = '⚠ ' + (e.message || 'AI error');
  } finally {
    llmInFlight = false;
    statusDot.className = running ? 'active' : '';
  }
}

function onRepCompleted() {
  pendingRepFeedback = true;
  // Flash rep counter
  repCountEl.style.transform = 'scale(1.4)';
  setTimeout(() => { repCountEl.style.transform = ''; }, 200);

  const fi = repCounter?.getFailureIndicators();
  if (fi?.isNearFailure) {
    callLLM(true); // urgent TTS
    lastLLMCall = performance.now();
    lastFeedbackRep = repCounter?.count ?? 0;
  }
}

// ─── Session start / stop ────────────────────────────────────────────────────
async function startSession() {
  if (running) return;
  const ok = await startCamera();
  if (!ok) { alert('Camera access denied. Please allow camera access.'); return; }
  updateCameraControls();
  running = true;
  resetSetState('setup');
  metricsBuf.frames.length = 0;
  lastLLMCall = 0;
  requestWakeLock();
  renderLoop();
  startBtn.textContent = 'STOP';
  startBtn.classList.add('active');
  statusDot.className = 'active';
}

function stopSession() {
  running = false;
  if (animId) cancelAnimationFrame(animId);
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  tts.stop();
  releaseWakeLock();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.textContent = 'GO';
  startBtn.classList.remove('active');
  statusDot.className = '';
  phaseEl.textContent = '';
  setStatusText('-', '#888');
}

// ─── Exercise setup ──────────────────────────────────────────────────────────
function applyExercise(key) {
  exercise = EXERCISES[key];
  if (!exercise) return;
  repCounter = exercise.repBased ? new RepCounter(exercise.repCounter) : null;
  exNameEl.textContent = exercise.name;
  repCountEl.textContent = '0';
  phaseEl.textContent = '';
  resetSetState('setup');
}

// ─── Wake lock ────────────────────────────────────────────────────────────────
async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen');
  } catch {}
}
function releaseWakeLock() { if (wakeLock) { wakeLock.release(); wakeLock = null; } }
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && running) requestWakeLock();
});

// ─── Settings panel ───────────────────────────────────────────────────────────
function populateExerciseSelect() {
  const sel = $('exercise-select');
  for (const group of EXERCISE_GROUPS) {
    const og = document.createElement('optgroup');
    og.label = group.label;
    for (const key of group.keys) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = EXERCISES[key].name;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
}

function openSettings() {
  $('inp-or-key').value   = cfg.get('openrouterKey')   || '';
  $('inp-el-key').value   = cfg.get('elevenLabsKey')   || '';
  $('inp-el-voice').value = cfg.get('elevenLabsVoice') || '';
  $('exercise-select').value = cfg.get('exercise');
  $('voice-toggle').checked  = cfg.get('voiceEnabled');
  settingsPanel.classList.add('visible');
}

function closeSettingsPanel() { settingsPanel.classList.remove('visible'); }

// ─── UI bindings ─────────────────────────────────────────────────────────────
on(startBtn, 'click', () => { running ? stopSession() : startSession(); });

on(flipBtn, 'click', () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  if (running) { stopSession(); startSession(); }
});

on(settingsBtn, 'click', openSettings);
on(closeSettings, 'click', closeSettingsPanel);

on(saveSettings, 'click', () => {
  cfg.set('openrouterKey',   $('inp-or-key').value.trim());
  cfg.set('elevenLabsKey',   $('inp-el-key').value.trim());
  cfg.set('elevenLabsVoice', $('inp-el-voice').value.trim());
  cfg.set('voiceEnabled',    $('voice-toggle').checked);
  const newEx = $('exercise-select').value;
  cfg.set('exercise', newEx);
  applyExercise(newEx);
  cfg.save();
  closeSettingsPanel();
});

on(resetReps, 'click', () => {
  resetSetState(setState === 'active' ? 'ready' : setState);
  repCountEl.textContent = '0';
});

// Swipe right to close settings on mobile
on(settingsPanel, 'touchstart', (e) => { settingsPanel._tx = e.touches[0].clientX; });
on(settingsPanel, 'touchend', (e) => {
  if (e.changedTouches[0].clientX - settingsPanel._tx > 60) closeSettingsPanel();
});

// ─── Loading helpers ──────────────────────────────────────────────────────────
function setLoading(msg) { loadingOv.classList.remove('hidden'); loadingMsg.textContent = msg; }
function hideLoading()   { loadingOv.classList.add('hidden'); }

async function loadServerConfig() {
  try {
    const res = await fetch('/api/config', { cache: 'no-store' });
    if (res.ok) serverConfig = await res.json();
  } catch {}
}

// ─── Boot ────────────────────────────────────────────────────────────────────
(async () => {
  cfg.load();
  await loadServerConfig();
  populateExerciseSelect();
  applyExercise(cfg.get('exercise') || 'pullup');
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // Unlock audio context on first touch (iOS Safari)
  document.addEventListener('touchend', () => {
    const ac = new AudioContext();
    ac.resume().then(() => ac.close());
  }, { once: true });

  await initPose();
  updateCameraControls();

  if (!cfg.get('openrouterKey') && !serverConfig.openrouter) openSettings();
})();
