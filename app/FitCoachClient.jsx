'use client';

import { useEffect } from 'react';

export default function FitCoachClient() {
  useEffect(() => {
    import('../app.js');
  }, []);

  return (
    <div id="camera-view">
      <video id="video" playsInline autoPlay muted />
      <canvas id="canvas" />

      <div id="stats-bar">
        <div id="rep-block">
          <span id="rep-count">0</span>
          <span className="rep-label">REPS</span>
        </div>
        <span id="exercise-name">-</span>
        <div id="status-area">
          <span id="form-score">-</span>
          <span id="status-dot" />
        </div>
      </div>

      <div id="feedback-area">
        <div id="feedback-text" />
        <div id="phase-indicator" />
      </div>

      <div id="controls">
        <button id="settings-btn" className="ctrl" title="Settings">SET</button>
        <button id="start-btn" className="ctrl" title="Start / Stop">GO</button>
        <button id="flip-btn" className="ctrl" title="Flip camera">FLIP</button>
      </div>

      <div id="settings-panel">
        <div className="panel-header">
          <h2>Settings</h2>
          <button id="close-settings">x</button>
        </div>

        <div className="panel-body">
          <div>
            <p className="section-title">Exercise</p>
            <div className="field">
              <select id="exercise-select" />
            </div>
          </div>

          <div>
            <p className="section-title">API Keys</p>
            <div className="field">
              <label htmlFor="inp-or-key">OpenRouter key</label>
              <input type="password" id="inp-or-key" placeholder="Loaded from .env.local" />
            </div>
            <div className="field" style={{ marginTop: 10 }}>
              <label htmlFor="inp-el-key">ElevenLabs key (voice feedback)</label>
              <input type="password" id="inp-el-key" placeholder="Loaded from .env.local" />
            </div>
            <div className="field" style={{ marginTop: 10 }}>
              <label htmlFor="inp-el-voice">ElevenLabs Voice ID</label>
              <input type="text" id="inp-el-voice" placeholder="Default: Antoni (Arnold-style)" />
              <small>Optional voice override (use any ElevenLabs Voice ID).</small>
            </div>
          </div>

          <div>
            <p className="section-title">Coach Options</p>
            <div className="toggle-row" style={{ marginBottom: 12 }}>
              <label>Voice feedback</label>
              <input type="checkbox" id="voice-toggle" defaultChecked />
            </div>
          </div>

          <button className="btn-primary" id="save-settings">Save &amp; Close</button>
          <button className="btn-secondary" id="reset-reps">Reset Rep Count</button>

          <p style={{ fontSize: '.72rem', color: '#555', textAlign: 'center', lineHeight: 1.5 }}>
            Keys load from server environment when available.<br />
            Camera requires HTTPS or localhost.
          </p>
        </div>
      </div>

      <div id="loading-overlay">
        <div className="spinner" />
        <p id="loading-msg">Loading AI model...</p>
      </div>
    </div>
  );
}
