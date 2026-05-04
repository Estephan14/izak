// MediaPipe landmark indices
export const LM = {
  NOSE: 0,
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13,    R_ELBOW: 14,
  L_WRIST: 15,    R_WRIST: 16,
  L_HIP: 23,      R_HIP: 24,
  L_KNEE: 25,     R_KNEE: 26,
  L_ANKLE: 27,    R_ANKLE: 28,
};

// Angle at joint B in the image plane (2D x/y only). MediaPipe's z is
// derived by fitting a generic body model and is too noisy for monocular
// rep-counting — it's the documented failure mode for front-view elbow
// angles. 2D image-plane angles are what working trackers use.
export function calcAngle(a, b, c) {
  if (!a || !b || !c) return null;
  const bax = a.x - b.x, bay = a.y - b.y;
  const bcx = c.x - b.x, bcy = c.y - b.y;
  const dot = bax * bcx + bay * bcy;
  const mag = Math.hypot(bax, bay) * Math.hypot(bcx, bcy);
  if (mag < 1e-6) return null;
  return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * (180 / Math.PI);
}

export function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

const VIS = 0.25; // minimum landmark visibility to trust an angle (loose enough for front-on views where one side self-occludes)

function ang(lm, i, j, k) {
  if ((lm[i].visibility ?? 1) < VIS || (lm[j].visibility ?? 1) < VIS || (lm[k].visibility ?? 1) < VIS) return null;
  return calcAngle(lm[i], lm[j], lm[k]);
}

// Sticky side picker. Picks an arm/leg side and stays on it across frames
// — switching only when that side disappears or another side becomes
// significantly more visible. Frame-to-frame swapping was making the
// driving angle look like noise (one arm curling, other not), which broke
// rep detection.
function makeSidePicker(rightVisKey, leftVisKey) {
  let lockedSide = null; // 'r' | 'l'
  return (rightVal, leftVal, m) => {
    const rVis = m[rightVisKey] ?? 0;
    const lVis = m[leftVisKey] ?? 0;
    const rOk = rightVal != null;
    const lOk = leftVal != null;
    if (!rOk && !lOk) return null;
    if (!rOk) { lockedSide = 'l'; return leftVal; }
    if (!lOk) { lockedSide = 'r'; return rightVal; }
    if (lockedSide == null) {
      lockedSide = rVis >= lVis ? 'r' : 'l';
    } else {
      // Sticky: only switch if the other side is meaningfully more visible.
      const other = lockedSide === 'r' ? lVis : rVis;
      const cur   = lockedSide === 'r' ? rVis : lVis;
      if (other - cur > 0.15) lockedSide = lockedSide === 'r' ? 'l' : 'r';
    }
    return lockedSide === 'r' ? rightVal : leftVal;
  };
}

// Extract useful metrics from the 33 MediaPipe landmarks
export function extractMetrics(lm) {
  const sMid = midpoint(lm[11], lm[12]);
  const hMid = midpoint(lm[23], lm[24]);

  // Spine tilt from vertical (small = upright)
  const spineVec = { x: sMid.x - hMid.x, y: sMid.y - hMid.y };
  const spineTilt = Math.atan2(Math.abs(spineVec.x), -spineVec.y) * (180 / Math.PI);

  const rightElbow    = ang(lm, 12, 14, 16);
  const leftElbow     = ang(lm, 11, 13, 15);
  const rightKnee     = ang(lm, 24, 26, 28);
  const leftKnee      = ang(lm, 23, 25, 27);
  const rightShoulder = ang(lm, 14, 12, 24);
  const leftShoulder  = ang(lm, 13, 11, 23);
  const rightHip      = ang(lm, 12, 24, 26);
  const leftHip       = ang(lm, 11, 23, 25);

  const hipTilt = Math.abs(lm[23].y - lm[24].y) * 100;
  const shoulderTilt = Math.abs(lm[11].y - lm[12].y) * 100;
  const elbowAsymmetry = (rightElbow != null && leftElbow != null)
    ? Math.abs(rightElbow - leftElbow) : null;

  // Worst visibility along each chain — used by getAngle pickers to choose
  // the more reliable side from the current camera angle.
  const visMin = (...idxs) => Math.min(...idxs.map(i => lm[i].visibility ?? 1));
  const rightArmVis  = visMin(12, 14, 16);
  const leftArmVis   = visMin(11, 13, 15);
  const rightLegVis  = visMin(24, 26, 28);
  const leftLegVis   = visMin(23, 25, 27);
  const rightHipVis  = visMin(12, 24, 26);
  const leftHipVis   = visMin(11, 23, 25);

  return {
    rightElbow, leftElbow,
    rightKnee, leftKnee,
    rightShoulder, leftShoulder,
    rightHip, leftHip,
    rightArmVis, leftArmVis,
    rightLegVis, leftLegVis,
    rightHipVis, leftHipVis,
    spineTilt, hipTilt, shoulderTilt, elbowAsymmetry,
    rightWristY: lm[16].y, leftWristY: lm[15].y,
    noseY: lm[0].y, shoulderY: sMid.y,
    wristAboveShoulder: Math.min(lm[15].y, lm[16].y) < sMid.y,
    shoulderHipDrop: Math.abs(sMid.y - hMid.y),
    hipKneeDrop: Math.abs(((lm[23].y + lm[24].y) / 2) - ((lm[25].y + lm[26].y) / 2)),
  };
}

// View-aware sticky pickers for two-sided exercises. One stateful picker
// per metric so left/right doesn't flip-flop frame to frame (which used to
// inject step-changes into the angle signal and prevent reps from
// counting). Each call to make…Picker() returns a fresh closure so each
// RepCounter instance gets its own sticky state.
function bestElbow(m, picker) { return picker(m.rightElbow, m.leftElbow, m); }
function bestKnee (m, picker) { return picker(m.rightKnee,  m.leftKnee,  m); }
function bestHip  (m, picker) { return picker(m.rightHip,   m.leftHip,   m); }
const elbowPicker = makeSidePicker('rightArmVis', 'leftArmVis');
const kneePicker  = makeSidePicker('rightLegVis', 'leftLegVis');
const hipPicker   = makeSidePicker('rightHipVis', 'leftHipVis');
const pickElbow = m => bestElbow(m, elbowPicker);
const pickKnee  = m => bestKnee(m,  kneePicker);
const pickHip   = m => bestHip(m,   hipPicker);

// Rep counter — bidirectional zone-transition counter.
// Anchors at the first extreme zone reached (top OR bottom), so the user can
// enter the frame already mid-rep, half-flexed, or in any position. A full
// rep = one complete cycle (anchor → opposite extreme → anchor) that spans at
// least `minRepRange` and lasts at least `minRepMs`.
export class RepCounter {
  constructor({
    getAngle,
    highThreshold,
    lowThreshold,
    invert = false,
    minRepRange = null,
    startTolerance = null,
    endTolerance = null,
    minRepMs = 450,
  }) {
    this.getAngle = getAngle;
    this.invert = invert;
    this._minRepRangeOverride = minRepRange;
    this._startToleranceOverride = startTolerance;
    this._endToleranceOverride = endTolerance;
    this.minRepMs = minRepMs;
    this.setThresholds(highThreshold, lowThreshold);
    this._init();
  }

  // Recompute zones / min-range from a (possibly user-calibrated) high/low
  // pair. Tolerances and minRepRange scale with the working span so a
  // shorter calibrated ROM still counts properly.
  setThresholds(high, low) {
    // The user might capture them in either order — accept either way.
    this.high = Math.max(high, low);
    this.low  = Math.min(high, low);
    const span = Math.abs(this.high - this.low);
    this.startTolerance = this._startToleranceOverride ?? Math.max(10, span * 0.25);
    this.endTolerance   = this._endToleranceOverride   ?? Math.max(10, span * 0.25);
    this.minRepRange    = this._minRepRangeOverride    ?? Math.max(15, span * 0.55);
  }

  _init() {
    this.count = 0;
    this._buf = [];
    this._lastAngle = null;
    this._lastZone = null;     // last extreme zone we were in ('top'|'bottom')
    this._anchorZone = null;   // which extreme the user first reached
    this._topExtreme = null;   // best (max) angle observed in top zone this cycle
    this._bottomExtreme = null;// best (min) angle observed in bottom zone this cycle
    this._currentRepStart = null;
    this.repDurations = [];
    this.repRanges = [];
    this.lastRep = null;
  }

  reset() { this._init(); }

  _smooth(raw) {
    this._buf.push(raw);
    if (this._buf.length > 6) this._buf.shift();
    return this._buf.reduce((s, v) => s + v, 0) / this._buf.length;
  }

  // Hysteresis-friendly zone classifier. Returns 'top', 'bottom', or 'mid'.
  _zoneOf(a) {
    if (a >= this.high - this.startTolerance) return 'top';
    if (a <= this.low + this.endTolerance) return 'bottom';
    return 'mid';
  }

  update(metrics, ts) {
    const raw = this.getAngle(metrics);
    if (raw == null) return;
    const a = this._smooth(raw);
    this._lastAngle = a;

    const zone = this._zoneOf(a);

    // Track running extremes so partial entries into a zone still register
    // their peak/valley before we leave the zone.
    if (zone === 'top') {
      this._topExtreme = this._topExtreme == null ? a : Math.max(this._topExtreme, a);
    } else if (zone === 'bottom') {
      this._bottomExtreme = this._bottomExtreme == null ? a : Math.min(this._bottomExtreme, a);
    }

    // Only act on transitions between extreme zones; mid-range motion is just
    // travel and the hysteresis prevents jittery toggling at the boundaries.
    if (zone === 'mid' || zone === this._lastZone) return;

    this._lastZone = zone;

    if (this._anchorZone == null) {
      // First extreme observed: anchor the cycle here. We discard whatever
      // partial range came before because we don't know if the user was
      // already mid-rep when they entered the frame.
      this._anchorZone = zone;
      this._currentRepStart = ts;
      if (zone === 'top') this._bottomExtreme = null;
      else this._topExtreme = null;
      return;
    }

    if (zone !== this._anchorZone) {
      // Half-cycle: user reached the opposite extreme. Wait for the return.
      return;
    }

    // Full cycle complete: anchor → opposite → anchor.
    const top = this._topExtreme;
    const bot = this._bottomExtreme;
    if (top == null || bot == null) {
      this._currentRepStart = ts;
      return;
    }

    const range = top - bot;
    const duration = this._currentRepStart == null ? null : ts - this._currentRepStart;
    const longEnough = duration == null || duration >= this.minRepMs;

    if (range >= this.minRepRange && longEnough) {
      this.count++;
      const startAng = this._anchorZone === 'top' ? top : bot;
      const endAng   = this._anchorZone === 'top' ? bot : top;
      this.lastRep = {
        range,
        duration,
        // Side-agnostic extremes: form checks should prefer these so they
        // stay correct regardless of which side the user anchored on.
        topAngle: top,
        bottomAngle: bot,
        // Anchor-relative aliases (kept for any caller that thinks in
        // start/end terms): startAngle = where the rep began, endAngle = the
        // far extreme reached.
        startAngle: startAng,
        endAngle: endAng,
        fullStart: this._anchorZone === 'top'
          ? top >= this.high - this.startTolerance
          : bot <= this.low + this.endTolerance,
        fullEnd: this._anchorZone === 'top'
          ? bot <= this.low + this.endTolerance
          : top >= this.high - this.startTolerance,
      };
      if (duration != null) {
        this.repDurations.push(duration);
        if (this.repDurations.length > 12) this.repDurations.shift();
      }
      this.repRanges.push(range);
      if (this.repRanges.length > 12) this.repRanges.shift();
    }

    // Reset for next rep — keep current anchor extreme, drop the opposite.
    this._currentRepStart = ts;
    if (this._anchorZone === 'top') this._bottomExtreme = null;
    else this._topExtreme = null;
  }

  // Kept for API compatibility with older callers. The new state machine
  // auto-anchors on the first observed extreme, so explicit locking is a no-op.
  lockStart(_metrics) { return true; }

  // Returns null if insufficient data, otherwise { slowingFactor, rangeLoss, isNearFailure }
  getFailureIndicators() {
    if (this.repDurations.length < 4) return null;
    const n = Math.max(2, Math.floor(this.repDurations.length / 2));
    const early = this.repDurations.slice(0, n);
    const late  = this.repDurations.slice(-n);
    const avgE = early.reduce((a, b) => a + b, 0) / early.length;
    const avgL = late.reduce((a, b) => a + b, 0) / late.length;
    const slowingFactor = avgL / avgE;

    let rangeLoss = 0;
    if (this.repRanges.length >= 4) {
      const earlyR = this.repRanges.slice(0, n);
      const lateR  = this.repRanges.slice(-n);
      const avgER = earlyR.reduce((a, b) => a + b, 0) / earlyR.length;
      const avgLR = lateR.reduce((a, b) => a + b, 0) / lateR.length;
      rangeLoss = avgER > 0 ? Math.max(0, (avgER - avgLR) / avgER) : 0;
    }

    return { slowingFactor, rangeLoss, isNearFailure: slowingFactor > 1.4 || rangeLoss > 0.2 };
  }

  get phaseName() {
    if (this._anchorZone == null) return 'ready';
    const oppositeZone = this._anchorZone === 'top' ? 'bottom' : 'top';
    if (this._lastZone === this._anchorZone) {
      return this._anchorZone === 'top' ? 'extended' : 'contracted';
    }
    if (this._lastZone === oppositeZone) {
      return this._anchorZone === 'top' ? 'contracted' : 'extended';
    }
    return 'moving';
  }
}

// ─── Exercise Library ────────────────────────────────────────────────────────

export const EXERCISES = {
  pullup: {
    name: 'Pull-Ups',
    icon: '🏋️',
    category: 'Upper Body',
    repBased: true,
    repCounter: {
      getAngle: pickElbow,
      highThreshold: 150, lowThreshold: 80, invert: false, minRepMs: 500,
    },
    startCue: 'Pull up to begin counting',
    calibration: {
      extendedPrompt: 'Hang from the bar with arms fully extended (dead hang)',
      contractedPrompt: 'Pull up — chin over the bar, hold',
    },
    formChecks(m, repCounter) {
      const issues = [];
      const score_base = 100;
      const lr = repCounter?.lastRep;
      if (lr && lr.topAngle < 150) issues.push('incomplete arm extension');
      if (lr && lr.bottomAngle > 85) issues.push('not high enough at top');
      if (m.elbowAsymmetry > 24) issues.push(`arm imbalance (${Math.round(m.elbowAsymmetry)}°)`);
      if (m.hipTilt > 12) issues.push('torso swinging');
      if (m.spineTilt > 28) issues.push('excessive lean');
      return { issues, score: Math.max(0, score_base - issues.length * 25) };
    },
    coachingContext: 'Pull-up. Full hang at bottom (dead hang). Drive elbows down to pull up. Chin clears bar at top. Minimize body swing. Scapular depression before pulling.',
  },

  chinup: {
    name: 'Chin-Ups',
    icon: '💪',
    category: 'Upper Body',
    repBased: true,
    repCounter: {
      getAngle: pickElbow,
      highThreshold: 150, lowThreshold: 80, invert: false, minRepMs: 500,
    },
    startCue: 'Pull up to begin counting',
    calibration: {
      extendedPrompt: 'Hang from the bar with arms fully extended (dead hang)',
      contractedPrompt: 'Pull up — chin over the bar, hold',
    },
    formChecks(m, repCounter) {
      const issues = [];
      const lr = repCounter?.lastRep;
      if (lr && lr.topAngle < 150) issues.push('incomplete arm extension');
      if (lr && lr.bottomAngle > 85) issues.push('not high enough at top');
      if (m.elbowAsymmetry > 24) issues.push(`arm imbalance (${Math.round(m.elbowAsymmetry)}°)`);
      if (m.hipTilt > 12) issues.push('swinging');
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Chin-up (underhand grip). Supinate grip engages biceps more. Full dead hang at bottom, chin above bar at top. Controlled negative.',
  },

  pushup: {
    name: 'Push-Ups',
    icon: '⬇️',
    category: 'Bodyweight',
    repBased: true,
    repCounter: {
      getAngle: pickElbow,
      highThreshold: 155, lowThreshold: 90, invert: false, minRepMs: 450,
    },
    startCue: 'Push up to begin counting',
    calibration: {
      extendedPrompt: 'Get into top push-up position — arms locked out, plank',
      contractedPrompt: 'Lower to your bottom position — chest near floor, hold',
    },
    formChecks(m, repCounter) {
      const issues = [];
      const lr = repCounter?.lastRep;
      if (lr && lr.topAngle < 150) issues.push('incomplete lockout');
      if (lr && lr.bottomAngle > 95) issues.push('not low enough');
      if (m.spineTilt < 45) issues.push('hips not aligned — plank position');
      if (m.elbowAsymmetry > 22) issues.push('uneven push');
      if (m.hipTilt > 14) issues.push('hip rotation');
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Push-up. Straight body from head to heels. Elbows 45° from torso (not flared). Full lockout at top. Chest touches or nearly touches floor.',
  },

  bicep_curl_barbell: {
    name: 'Barbell Curl',
    icon: '🏋️',
    category: 'Arms',
    repBased: true,
    repCounter: {
      getAngle: pickElbow,
      highThreshold: 155, lowThreshold: 65, invert: false, minRepMs: 420,
    },
    startCue: 'Curl to begin counting',
    calibration: {
      extendedPrompt: 'Hold the bar at your hips — arms fully straight',
      contractedPrompt: 'Curl all the way up — bar near shoulders, hold',
    },
    formChecks(m, repCounter) {
      const issues = [];
      const lr = repCounter?.lastRep;
      if (lr && lr.topAngle < 150) issues.push('incomplete arm extension');
      if (lr && lr.bottomAngle > 70) issues.push('not curled high enough');
      if (m.hipTilt > 14 || m.spineTilt > 26) issues.push('body swinging — strict form');
      if (m.elbowAsymmetry > 18) issues.push(`uneven curl (${Math.round(m.elbowAsymmetry)}°)`);
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Barbell bicep curl. Zero body momentum, elbows pinned at sides, full extension at bottom, peak squeeze at top. Slow negative.',
  },

  bicep_curl_dumbbell: {
    name: 'Dumbbell Curl',
    icon: '🏋️',
    category: 'Arms',
    repBased: true,
    repCounter: {
      getAngle: pickElbow,
      highThreshold: 155, lowThreshold: 65, invert: false, minRepMs: 420,
    },
    startCue: 'Curl to begin counting',
    calibration: {
      extendedPrompt: 'Hold the dumbbell at your side — arm fully straight',
      contractedPrompt: 'Curl all the way up — dumbbell near shoulder, hold',
    },
    formChecks(m, repCounter) {
      const issues = [];
      const lr = repCounter?.lastRep;
      if (lr && lr.topAngle < 150) issues.push('incomplete arm extension');
      if (lr && lr.bottomAngle > 70) issues.push('not curled high enough');
      if (m.hipTilt > 14 || m.spineTilt > 24) issues.push('too much body swing');
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Dumbbell curl. Supinate wrist as you curl up. Alternate or simultaneous. Elbow stays at side as pivot point.',
  },

  tricep_extension: {
    name: 'Tricep Extension',
    icon: '💪',
    category: 'Arms',
    repBased: true,
    repCounter: {
      // Overhead extension: starts with elbows bent (low angle), extends to high
      getAngle: pickElbow,
      highThreshold: 150, lowThreshold: 80, invert: true,
    },
    startCue: 'Extend overhead to begin counting',
    calibration: {
      extendedPrompt: 'Press arms straight overhead — full lockout, hold',
      contractedPrompt: 'Bend elbows behind head — bottom of the rep, hold',
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 28) issues.push('leaning too far back');
      if (m.elbowAsymmetry > 22) issues.push('uneven extension');
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Tricep overhead extension. Keep upper arms close to head. Full extension overhead, controlled bend down. Elbow position is key — they should not flare.',
  },

  squat: {
    name: 'Squat',
    icon: '🦵',
    category: 'Legs',
    repBased: true,
    repCounter: {
      getAngle: pickKnee,
      highThreshold: 150, lowThreshold: 105, invert: false,
    },
    startCue: 'Squat to begin counting',
    calibration: {
      extendedPrompt: 'Stand tall — knees fully extended, hold',
      contractedPrompt: 'Squat down to your bottom position — hold',
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 42) issues.push('excessive forward lean');
      if (m.hipTilt > 14) issues.push('hip shift');
      const kAsym = (m.rightKnee != null && m.leftKnee != null) ? Math.abs(m.rightKnee - m.leftKnee) : 0;
      if (kAsym > 22) issues.push('uneven depth');
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Squat. Feet shoulder-width, toes slightly out. Depth at least parallel. Knees track over toes. Chest tall. Core braced. Weight through whole foot.',
  },

  deadlift: {
    name: 'Deadlift',
    icon: '🏋️',
    category: 'Compound',
    repBased: true,
    repCounter: {
      getAngle: pickHip,
      highThreshold: 148, lowThreshold: 95, invert: false,
    },
    startCue: 'Stand up to begin counting',
    calibration: {
      extendedPrompt: 'Stand tall, fully locked out — hold',
      contractedPrompt: 'Hinge at hips, bar at floor — bottom position, hold',
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 45) issues.push('back rounding — STOP, reset form');
      else if (m.spineTilt > 35) issues.push('back rounding — lighten load');
      if (m.hipTilt > 14) issues.push('hip shift');
      return { issues, score: Math.max(0, 100 - issues.length * 35) }; // higher penalty — safety critical
    },
    coachingContext: 'Deadlift. Hip hinge pattern. Neutral spine throughout (critical). Bar close to body. Lats engaged before pulling. Drive floor away. Lock hips and shoulders out together.',
  },

  ohp: {
    name: 'Overhead Press',
    icon: '⬆️',
    category: 'Compound',
    repBased: true,
    repCounter: {
      // Starts at rack (~80° elbow), extends to lockout (~165°)
      getAngle: pickElbow,
      highThreshold: 145, lowThreshold: 95, invert: true,
    },
    startCue: 'Press up to begin counting',
    calibration: {
      extendedPrompt: 'Press the bar straight overhead — full lockout, hold',
      contractedPrompt: 'Bar at front rack (collarbone) — bottom position, hold',
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 30) issues.push('excessive back arch — lower weight');
      if (m.elbowAsymmetry > 22) issues.push('uneven press');
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Overhead press. Brace core, slight forward lean is OK. Press in slight forward arc around face. Full lockout overhead. Elbows slightly forward of bar in rack position.',
  },

  bench_press: {
    name: 'Bench Press',
    icon: '🏋️',
    category: 'Compound',
    repBased: true,
    repCounter: {
      getAngle: pickElbow,
      highThreshold: 125, lowThreshold: 95, invert: false,
    },
    startCue: 'Press up to begin counting',
    calibration: {
      extendedPrompt: 'Press the bar straight up — arms locked out, hold',
      contractedPrompt: 'Lower the bar to mid-chest — hold at the bottom',
    },
    formChecks(m) {
      const issues = [];
      if (m.elbowAsymmetry > 22) issues.push('uneven press');
      return { issues, score: Math.max(0, 100 - issues.length * 25) };
    },
    coachingContext: 'Bench press. (Camera angle is limited lying down.) Slight arch in lower back, feet flat, retract scapulae. Bar touches mid-chest, full lockout at top. 45° elbow angle from torso.',
  },

  basketball_freethrow: {
    name: 'Free Throw',
    icon: '🏀',
    category: 'Basketball',
    repBased: false,
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 12) issues.push('off-balance — center your weight');
      if (m.rightKnee != null && m.rightKnee > 175) issues.push('bend your knees for power');
      if (m.rightKnee != null && m.rightKnee < 130) issues.push('too much knee bend');
      const wristHigh = m.rightWristY < m.shoulderY; // wrist above shoulder = follow-through
      if (!wristHigh && m.rightWristY != null) issues.push('extend fully — wrist snap follow-through');
      return { issues, score: Math.max(0, 100 - issues.length * 20) };
    },
    coachingContext: 'Basketball free throw. BEEF: Balance (feet shoulder-width, slight knee bend), Eyes (back of rim), Elbow (under ball, in line with basket), Follow-through (hold wrist bent, fingers pointing down).',
  },

  basketball_jumpshot: {
    name: 'Jump Shot',
    icon: '🏀',
    category: 'Basketball',
    repBased: false,
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 15) issues.push('keep your balance on release');
      if (m.elbowAsymmetry > 20) issues.push('guide hand off — only shooting hand on ball at release');
      return { issues, score: Math.max(0, 100 - issues.length * 20) };
    },
    coachingContext: 'Basketball jump shot. Catch in triple threat. Square up, elevate straight (not into the shot). Release at peak. Consistent elbow alignment under ball. Follow-through toward target.',
  },

  basketball_layup: {
    name: 'Layup',
    icon: '🏀',
    category: 'Basketball',
    repBased: false,
    formChecks(m) {
      const issues = [];
      if (m.hipTilt > 10) issues.push('protect the ball — keep it high');
      return { issues, score: Math.max(0, 100 - issues.length * 20) };
    },
    coachingContext: 'Basketball layup. Right hand = right foot last step (left hand = left foot). Use backboard (aim at top corner of box). Soft touch, protect ball with off hand, extend fully.',
  },
};

// Ordered list for UI select
export const EXERCISE_GROUPS = [
  { label: 'Basketball', keys: ['basketball_freethrow', 'basketball_jumpshot', 'basketball_layup'] },
  { label: 'Upper Body', keys: ['pullup', 'chinup', 'pushup'] },
  { label: 'Arms',       keys: ['bicep_curl_barbell', 'bicep_curl_dumbbell', 'tricep_extension'] },
  { label: 'Compound',   keys: ['squat', 'deadlift', 'ohp', 'bench_press'] },
];
