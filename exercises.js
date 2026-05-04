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

// Angle at joint B using full 3D coordinates (x, y, z) for accuracy across camera angles
export function calcAngle(a, b, c) {
  if (!a || !b || !c) return null;
  const bax = a.x - b.x, bay = a.y - b.y, baz = (a.z || 0) - (b.z || 0);
  const bcx = c.x - b.x, bcy = c.y - b.y, bcz = (c.z || 0) - (b.z || 0);
  const dot = bax * bcx + bay * bcy + baz * bcz;
  const mag = Math.hypot(bax, bay, baz) * Math.hypot(bcx, bcy, bcz);
  if (mag < 1e-6) return null;
  return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * (180 / Math.PI);
}

export function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

const VIS = 0.35; // minimum landmark visibility to trust an angle

function ang(lm, i, j, k) {
  if ((lm[i].visibility ?? 1) < VIS || (lm[j].visibility ?? 1) < VIS || (lm[k].visibility ?? 1) < VIS) return null;
  return calcAngle(lm[i], lm[j], lm[k]);
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

  return {
    rightElbow, leftElbow,
    rightKnee, leftKnee,
    rightShoulder, leftShoulder,
    rightHip, leftHip,
    spineTilt, hipTilt, shoulderTilt, elbowAsymmetry,
    rightWristY: lm[16].y, leftWristY: lm[15].y,
    noseY: lm[0].y, shoulderY: sMid.y,
    wristAboveShoulder: Math.min(lm[15].y, lm[16].y) < sMid.y,
    shoulderHipDrop: Math.abs(sMid.y - hMid.y),
    hipKneeDrop: Math.abs(((lm[23].y + lm[24].y) / 2) - ((lm[25].y + lm[26].y) / 2)),
  };
}

// Rep counter — state machine with angle smoothing and auto-initialisation
export class RepCounter {
  constructor({
    getAngle,
    highThreshold,
    lowThreshold,
    invert = false,
    minRepRange = null,
    startTolerance = 8,
    endTolerance = 7,
    minRepMs = 450,
  }) {
    this.getAngle = getAngle;
    this.high = highThreshold;
    this.low  = lowThreshold;
    this.invert = invert;
    this.startTolerance = startTolerance;
    this.endTolerance = endTolerance;
    this.minRepRange = minRepRange ?? Math.max(26, Math.abs(highThreshold - lowThreshold) * 0.75);
    this.minRepMs = minRepMs;
    this.count = 0;
    this.phase = 'seeking_start'; // seeking_start | at_start | moving | at_end | returning
    this._buf = []; // smoothing buffer
    this._repStart = null;
    this.repDurations = [];
    this.repRanges = [];
    this._rangeMin = Infinity;
    this._rangeMax = -Infinity;
    this._lastAngle = null;
    this._direction = 0;
    this._startAngle = null;
    this._endExtreme = null;
    this._startHoldFrames = 0;
    this._endHoldFrames = 0;
    this._returnHoldFrames = 0;
    this.lastRep = null;
  }

  _smooth(raw) {
    this._buf.push(raw);
    if (this._buf.length > 4) this._buf.shift();
    return this._buf.reduce((s, v) => s + v, 0) / this._buf.length;
  }

  update(metrics, ts) {
    const raw = this.getAngle(metrics);
    if (raw == null) return;
    const a = this._smooth(raw);
    const delta = this._lastAngle == null ? 0 : a - this._lastAngle;
    if (Math.abs(delta) > 1.2) this._direction = Math.sign(delta);
    this._lastAngle = a;

    const atStart = this._isAtStart(a);
    const atEnd = this._isAtEnd(a);

    if (this.phase === 'seeking_start') {
      if (atStart) this._startHoldFrames++;
      else this._startHoldFrames = 0;
      if (this._startHoldFrames >= 2) this._lockStart(a);
      return;
    }

    this._rangeMin = Math.min(this._rangeMin, a);
    this._rangeMax = Math.max(this._rangeMax, a);

    if (this.phase === 'at_start' && this._movedAwayFromStart(a)) {
      this.phase = 'moving';
      this._repStart = ts;
      this.lastRep = null;
    }

    if (this.phase === 'moving') {
      this._endExtreme = this._endExtreme == null
        ? a
        : (this.invert ? Math.max(this._endExtreme, a) : Math.min(this._endExtreme, a));
      if (atEnd && this._rangeMax - this._rangeMin >= this.minRepRange) this._endHoldFrames++;
      else this._endHoldFrames = 0;
      if (this._endHoldFrames >= 1) this.phase = 'at_end';
    }

    if (this.phase === 'at_end' && this._returnedTowardStart(a)) {
      this.phase = 'returning';
    }

    if (this.phase === 'returning' && atStart) this._returnHoldFrames++;
    else if (this.phase === 'returning') this._returnHoldFrames = 0;

    if (this.phase === 'returning' && this._returnHoldFrames >= 1 && this._isLongEnough(ts)) {
      this._finishRep(ts);
      this._lockStart(a);
    }
  }

  _isAtStart(a) {
    return this.invert ? a <= this.low + this.startTolerance : a >= this.high - this.startTolerance;
  }

  _isAtEnd(a) {
    return this.invert ? a >= this.high - this.endTolerance : a <= this.low + this.endTolerance;
  }

  _lockStart(a) {
    this.phase = 'at_start';
    this._startAngle = a;
    this._rangeMin = a;
    this._rangeMax = a;
    this._endExtreme = null;
    this._repStart = null;
    this._startHoldFrames = 0;
    this._endHoldFrames = 0;
    this._returnHoldFrames = 0;
  }

  _movedAwayFromStart(a) {
    if (this._startAngle == null) return false;
    const minExit = Math.max(22, this.minRepRange * 0.4);
    return this.invert
      ? a - this._startAngle >= minExit
      : this._startAngle - a >= minExit;
  }

  _returnedTowardStart(a) {
    if (this._endExtreme == null) return false;
    return this.invert
      ? this._endExtreme - a >= Math.min(12, this.minRepRange * 0.4)
      : a - this._endExtreme >= Math.min(12, this.minRepRange * 0.4);
  }

  _isLongEnough(ts) {
    return this._repStart == null || ts - this._repStart >= this.minRepMs;
  }

  lockStart(metrics) {
    const raw = this.getAngle(metrics);
    if (raw == null) return false;
    const a = this._smooth(raw);
    if (!this._isAtStart(a)) return false;
    this._lockStart(a);
    this._lastAngle = a;
    return true;
  }

  _finishRep(ts) {
    const range = this._rangeMax - this._rangeMin;
    const duration = this._repStart == null ? null : ts - this._repStart;
    if (range < this.minRepRange || (duration != null && duration < this.minRepMs)) return;

    this.count++;
    if (this._repStart != null) {
      this.repDurations.push(duration);
      if (this.repDurations.length > 12) this.repDurations.shift();
    }
    this.lastRep = {
      range,
      duration,
      startAngle: this._startAngle,
      endAngle: this._endExtreme,
      fullStart: this._startAngle != null && this._isAtStart(this._startAngle),
      fullEnd: this._endExtreme != null && this._isAtEnd(this._endExtreme),
    };
    this.repRanges.push(range);
    if (this.repRanges.length > 12) this.repRanges.shift();
    this._rangeMin = Infinity;
    this._rangeMax = -Infinity;
    this._endExtreme = null;
    this._repStart = null;
  }

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
    if (this.phase === 'at_start') return this.invert ? 'rack' : 'extended';
    if (this.phase === 'at_end')   return this.invert ? 'lockout' : 'contracted';
    if (this.phase === 'seeking_start') return 'setup';
    return 'moving';
  }

  reset() {
    this.count = 0; this.phase = 'seeking_start'; this._buf = [];
    this._repStart = null; this.repDurations = []; this.repRanges = [];
    this._rangeMin = Infinity; this._rangeMax = -Infinity;
    this._lastAngle = null; this._direction = 0;
    this._startAngle = null; this._endExtreme = null;
    this._startHoldFrames = 0; this._endHoldFrames = 0; this._returnHoldFrames = 0;
    this.lastRep = null;
  }
}

// ─── Exercise Library ────────────────────────────────────────────────────────

function avg2(a, b) { return (a != null && b != null) ? (a + b) / 2 : (a ?? b); }

export const EXERCISES = {
  pullup: {
    name: 'Pull-Ups',
    icon: '🏋️',
    category: 'Upper Body',
    repBased: true,
    repCounter: {
      getAngle: m => avg2(m.rightElbow, m.leftElbow),
      highThreshold: 150, lowThreshold: 80, invert: false, minRepRange: 55, minRepMs: 500,
    },
    // Keep start gate angle-first so camera framing doesn't block activation.
    isStartPosition: (m, angle) => angle >= 136,
    startCue: 'Reach full hang to start',
    formChecks(m, repCounter) {
      const issues = [];
      const score_base = 100;
      if (repCounter?.lastRep && repCounter.lastRep.startAngle < 150) issues.push('incomplete arm extension');
      if (repCounter?.lastRep && repCounter.lastRep.endAngle > 85) issues.push('not high enough at top');
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
      getAngle: m => avg2(m.rightElbow, m.leftElbow),
      highThreshold: 150, lowThreshold: 80, invert: false, minRepRange: 55, minRepMs: 500,
    },
    // Keep start gate angle-first so camera framing doesn't block activation.
    isStartPosition: (m, angle) => angle >= 136,
    startCue: 'Reach full hang to start',
    formChecks(m, repCounter) {
      const issues = [];
      if (repCounter?.lastRep && repCounter.lastRep.startAngle < 150) issues.push('incomplete arm extension');
      if (repCounter?.lastRep && repCounter.lastRep.endAngle > 85) issues.push('not high enough at top');
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
      getAngle: m => avg2(m.rightElbow, m.leftElbow),
      highThreshold: 155, lowThreshold: 90, invert: false, minRepRange: 50, minRepMs: 450,
    },
    isStartPosition: (m, angle) => angle >= 140 && m.spineTilt > 35,
    startCue: 'Lock out arms in plank',
    formChecks(m, repCounter) {
      const issues = [];
      if (repCounter?.lastRep && repCounter.lastRep.startAngle < 150) issues.push('incomplete lockout');
      if (repCounter?.lastRep && repCounter.lastRep.endAngle > 95) issues.push('not low enough');
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
      getAngle: m => avg2(m.rightElbow, m.leftElbow),
      highThreshold: 155, lowThreshold: 65, invert: false, minRepRange: 70, minRepMs: 420,
    },
    isStartPosition: (m, angle) => angle >= 140,
    startCue: 'Straighten arms to start',
    formChecks(m, repCounter) {
      const issues = [];
      if (repCounter?.lastRep && repCounter.lastRep.startAngle < 150) issues.push('incomplete arm extension');
      if (repCounter?.lastRep && repCounter.lastRep.endAngle > 70) issues.push('not curled high enough');
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
      getAngle: m => m.rightElbow ?? m.leftElbow,
      highThreshold: 155, lowThreshold: 65, invert: false, minRepRange: 70, minRepMs: 420,
    },
    isStartPosition: (m, angle) => angle >= 140,
    startCue: 'Straighten arm to start',
    formChecks(m, repCounter) {
      const issues = [];
      if (repCounter?.lastRep && repCounter.lastRep.startAngle < 150) issues.push('incomplete arm extension');
      if (repCounter?.lastRep && repCounter.lastRep.endAngle > 70) issues.push('not curled high enough');
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
      getAngle: m => avg2(m.rightElbow, m.leftElbow),
      highThreshold: 150, lowThreshold: 80, invert: true,
    },
    isStartPosition: (m, angle) => angle <= 110,
    startCue: 'Bend elbows overhead to start',
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
      getAngle: m => avg2(m.rightKnee, m.leftKnee),
      highThreshold: 150, lowThreshold: 105, invert: false,
    },
    isStartPosition: (m, angle) => angle >= 130,
    startCue: 'Stand tall to start',
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
      getAngle: m => avg2(m.rightHip, m.leftHip),
      highThreshold: 148, lowThreshold: 95, invert: false,
    },
    isStartPosition: (m, angle) => angle >= 128,
    startCue: 'Stand tall to start',
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
      getAngle: m => avg2(m.rightElbow, m.leftElbow),
      highThreshold: 145, lowThreshold: 95, invert: true,
    },
    isStartPosition: (m, angle) => angle <= 120,
    startCue: 'Hold rack position to start',
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
      getAngle: m => avg2(m.rightElbow, m.leftElbow),
      highThreshold: 125, lowThreshold: 95, invert: false,
    },
    isStartPosition: (m, angle) => angle >= 108,
    startCue: 'Lock out to start',
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
