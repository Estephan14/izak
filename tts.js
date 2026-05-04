// ElevenLabs TTS client — one utterance at a time, skips if busy
const DEFAULT_VOICE = 'pNInz6obpgDQGcFmaJgB'; // Adam (energetic male)

class TTSClient {
  constructor() {
    this.apiKey = '';
    this.voiceId = DEFAULT_VOICE;
    this.enabled = true;
    this._audio = null;
    this._pending = null;
    this._busy = false;
  }

  async speak(text, { urgent = false } = {}) {
    if (!this.enabled || !this.apiKey || !text?.trim()) return;

    if (urgent) {
      this._stop();
      this._pending = text;
      this._flush();
    } else {
      if (this._busy) return; // drop low-priority if already speaking
      this._pending = text;
      this._flush();
    }
  }

  async _flush() {
    if (!this._pending || this._busy) return;
    const text = this._pending;
    this._pending = null;
    this._busy = true;

    try {
      const res = await fetch(
        `https://api.elevenlabs.io/v1/text-to-speech/${this.voiceId || DEFAULT_VOICE}`,
        {
          method: 'POST',
          headers: { 'xi-api-key': this.apiKey, 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            model_id: 'eleven_flash_v2_5',
            voice_settings: { stability: 0.5, similarity_boost: 0.75, speed: 1.1 },
          }),
        }
      );

      if (!res.ok) throw new Error(await res.text());

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      this._audio = new Audio(url);

      await new Promise((resolve) => {
        this._audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
        this._audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };
        this._audio.play().catch(resolve);
      });
    } catch (e) {
      console.warn('[TTS]', e.message || e);
    } finally {
      this._audio = null;
      this._busy = false;
      if (this._pending) this._flush(); // drain one pending utterance
    }
  }

  _stop() {
    if (this._audio) { this._audio.pause(); this._audio = null; }
    this._busy = false;
  }

  stop() { this._stop(); this._pending = null; }
}

export default new TTSClient();
