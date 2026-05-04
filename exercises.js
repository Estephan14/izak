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

// Angle at joint B given points A, B, C
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

// Extract useful metrics from the 33 MediaPipe landmarks
export function extractMetrics(lm) {
  const sMid = midpoint(lm[11], lm[12]);
  const hMid = midpoint(lm[23], lm[24]);

  // Spine tilt from vertical (small = upright)
  const spineVec = { x: sMid.x - hMid.x, y: sMid.y - hMid.y };
  const spineTilt = Math.atan2(Math.abs(spineVec.x), -spineVec.y) * (180 / Math.PI);

  const rightElbow   = calcAngle(lm[12], lm[14], lm[16]);
  const leftElbow    = calcAngle(lm[11], lm[13], lm[15]);
  const rightKnee    = calcAngle(lm[24], lm[26], lm[28]);
  const leftKnee     = calcAngle(lm[23], lm[25], lm[27]);
  const rightShoulder = calcAngle(lm[14], lm[12], lm[24]);
  const leftShoulder  = calcAngle(lm[13], lm[11], lm[23]);
  const rightHip     = calcAngle(lm[12], lm[24], lm[26]);
  const leftHip      = calcAngle(lm[11], lm[23], lm[25]);

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
  };
}

// Rep counter — state machine detects full ROM repetitions
export class RepCounter {
  constructor({ getAngle, highThreshold, lowThreshold, invert = false }) {
    this.getAngle = getAngle;
    this.high = highThreshold;
    this.low = lowThreshold;
    this.invert = invert;
    this.count = 0;
    this.phase = 'idle'; // idle | at_start | at_end
    this._repStart = null;
    this.repDurations = [];
    this.repRanges = [];
    this._rangeMin = Infinity;
    this._rangeMax = -Infinity;
  }

  update(metrics, ts) {
    const a = this.getAngle(metrics);
    if (a == null) return;

    this._rangeMin = Math.min(this._rangeMin, a);
    this._rangeMax = Math.max(this._rangeMax, a);

    if (!this.invert) {
      // starts at high angle (arms straight), contracts to low, returns to high
      switch (this.phase) {
        case 'idle':
          if (a >= this.high) this.phase = 'at_start';
          break;
        case 'at_start':
          if (a <= this.low) { this.phase = 'at_end'; this._repStart = ts; }
          break;
        case 'at_end':
          if (a >= this.high) {
            this._finishRep(ts);
            this.phase = 'at_start';
          }
          break;
      }
    } else {
      // starts at low angle (rack position), extends to high, returns to low
      switch (this.phase) {
        case 'idle':
          if (a <= this.low) this.phase = 'at_start';
          break;
        case 'at_start':
          if (a >= this.high) { this.phase = 'at_end'; this._repStart = ts; }
          break;
        case 'at_end':
          if (a <= this.low) {
            this._finishRep(ts);
            this.phase = 'at_start';
          }
          break;
      }
    }
  }

  _finishRep(ts) {
    this.count++;
    if (this._repStart != null) {
      this.repDurations.push(ts - this._repStart);
      if (this.repDurations.length > 12) this.repDurations.shift();
    }
    const range = this._rangeMax - this._rangeMin;
    this.repRanges.push(range);
    if (this.repRanges.length > 12) this.repRanges.shift();
    this._rangeMin = Infinity;
    this._rangeMax = -Infinity;
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
    return 'moving';
  }

  reset() {
    this.count = 0; this.phase = 'idle';
    this._repStart = null; this.repDurations = []; this.repRanges = [];
    this._rangeMin = Infinity; this._rangeMax = -Infinity;
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
      highThreshold: 140, lowThreshold: 90, invert: false,
    },
    formChecks(m) {
      const issues = [];
      const score_base = 100;
      if (m.elbowAsymmetry > 18) issues.push(`arm imbalance (${Math.round(m.elbowAsymmetry)}°)`);
      if (m.hipTilt > 8) issues.push('torso swinging');
      if (m.spineTilt > 20) issues.push('excessive lean');
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
      highThreshold: 140, lowThreshold: 90, invert: false,
    },
    formChecks(m) {
      const issues = [];
      if (m.elbowAsymmetry > 18) issues.push(`arm imbalance (${Math.round(m.elbowAsymmetry)}°)`);
      if (m.hipTilt > 8) issues.push('swinging');
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
      highThreshold: 130, lowThreshold: 95, invert: false,
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 18) issues.push('hips not aligned — plank position');
      if (m.elbowAsymmetry > 15) issues.push('uneven push');
      if (m.hipTilt > 10) issues.push('hip rotation');
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
      highThreshold: 140, lowThreshold: 65, invert: false,
    },
    formChecks(m) {
      const issues = [];
      if (m.hipTilt > 10 || m.spineTilt > 18) issues.push('body swinging — strict form');
      if (m.elbowAsymmetry > 12) issues.push(`uneven curl (${Math.round(m.elbowAsymmetry)}°)`);
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
      highThreshold: 140, lowThreshold: 65, invert: false,
    },
    formChecks(m) {
      const issues = [];
      if (m.hipTilt > 10 || m.spineTilt > 15) issues.push('too much body swing');
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
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 20) issues.push('leaning too far back');
      if (m.elbowAsymmetry > 15) issues.push('uneven extension');
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
      highThreshold: 155, lowThreshold: 100, invert: false,
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 35) issues.push('excessive forward lean');
      if (m.hipTilt > 10) issues.push('hip shift');
      const kAsym = (m.rightKnee != null && m.leftKnee != null) ? Math.abs(m.rightKnee - m.leftKnee) : 0;
      if (kAsym > 15) issues.push('uneven depth');
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
      highThreshold: 155, lowThreshold: 90, invert: false,
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 40) issues.push('back rounding — STOP, reset form');
      else if (m.spineTilt > 28) issues.push('back rounding — lighten load');
      if (m.hipTilt > 10) issues.push('hip shift');
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
      highThreshold: 150, lowThreshold: 90, invert: true,
    },
    formChecks(m) {
      const issues = [];
      if (m.spineTilt > 22) issues.push('excessive back arch — lower weight');
      if (m.elbowAsymmetry > 15) issues.push('uneven press');
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
      highThreshold: 130, lowThreshold: 90, invert: false,
    },
    formChecks(m) {
      const issues = [];
      if (m.elbowAsymmetry > 15) issues.push('uneven press');
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
