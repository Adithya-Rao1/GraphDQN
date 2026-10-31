// Shared Timeline: one playhead that drives the 3D stage, why-card and every wave-2 panel.
// Time is in milliseconds of "timeline time" (speed 1x). A keyframe is the seed hold or one edit
// (morph phase followed by a hold phase). Everything else subscribes to the bus.
import { bus } from "./bus.js";

export const DEFAULTS = { seedHoldMs: 2200, morphMs: 2600, holdMs: 2400 };

export class Timeline {
  constructor() {
    this.keyframes = [];
    this.offsets = [];
    this.duration = 0;
    this.t = 0;
    this.playing = false;
    this.speed = 1;
    this.live = false; // a run is still streaming: at the end, wait for more keyframes instead of stopping
    this.loop = false;
    this._last = 0;
    this._raf = 0;
    this._lastIndex = -1;
  }

  /** keyframe = {id, kind: "seed"|"edit", morphMs, holdMs, label, ...anything}. Returns its index. */
  add(kf) {
    const k = { morphMs: 0, holdMs: DEFAULTS.holdMs, ...kf };
    this.offsets.push(this.duration);
    this.duration += k.morphMs + k.holdMs;
    this.keyframes.push(k);
    bus.emit("timeline:keyframes", { timeline: this });
    return this.keyframes.length - 1;
  }

  clear() {
    this.pause();
    this.keyframes = [];
    this.offsets = [];
    this.duration = 0;
    this.t = 0;
    this._lastIndex = -1;
    bus.emit("timeline:keyframes", { timeline: this });
    this._emit(true);
  }

  /** Replace all keyframes (used when the accepted/all filter changes); keeps the playhead on the same keyframe id if possible. */
  replace(kfs, keepId) {
    const prevId = keepId ?? this.keyframes[this.indexAt(this.t)]?.id;
    this.keyframes = [];
    this.offsets = [];
    this.duration = 0;
    for (const kf of kfs) {
      const k = { morphMs: 0, holdMs: DEFAULTS.holdMs, ...kf };
      this.offsets.push(this.duration);
      this.duration += k.morphMs + k.holdMs;
      this.keyframes.push(k);
    }
    const i = this.keyframes.findIndex((k) => k.id === prevId);
    this.t = i >= 0 ? this.offsets[i] : 0;
    this._lastIndex = -1;
    bus.emit("timeline:keyframes", { timeline: this });
    this._emit(true);
  }

  indexAt(t) {
    let lo = 0, hi = this.offsets.length - 1, ans = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.offsets[mid] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return ans;
  }

  /** Current state: {index, keyframe, u (morph progress 0..1), inHold, local (ms into keyframe)}. */
  state(t = this.t) {
    if (!this.keyframes.length) return { index: -1, keyframe: null, u: 0, inHold: false, local: 0, t };
    const index = this.indexAt(t);
    const keyframe = this.keyframes[index];
    const local = Math.min(Math.max(t - this.offsets[index], 0), keyframe.morphMs + keyframe.holdMs);
    const u = keyframe.morphMs > 0 ? Math.min(local / keyframe.morphMs, 1) : 1;
    return { index, keyframe, u, inHold: local >= keyframe.morphMs, local, t };
  }

  play() {
    if (!this.keyframes.length) return;
    if (!this.live && this.t >= this.duration - 1) this.t = 0;
    this.playing = true;
    this._last = performance.now();
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame((n) => this._frame(n));
    bus.emit("timeline:play", this.snapshot());
  }

  pause() {
    this.playing = false;
    cancelAnimationFrame(this._raf);
    bus.emit("timeline:pause", this.snapshot());
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  setSpeed(s) {
    this.speed = s;
    bus.emit("timeline:speed", this.snapshot());
  }

  seek(t) {
    this.t = Math.min(Math.max(t, 0), Math.max(this.duration - 0.001, 0));
    this._emit(true);
  }

  /** Jump to the start of keyframe i's morph (or its settled end state with {end: true}). */
  seekFrame(i, { end = false } = {}) {
    if (!this.keyframes.length) return;
    i = Math.min(Math.max(i, 0), this.keyframes.length - 1);
    const k = this.keyframes[i];
    this.seek(this.offsets[i] + (end ? k.morphMs : 0));
  }

  step(dir) {
    const s = this.state();
    if (s.index < 0) return;
    if (dir < 0 && s.local > 400) return this.seekFrame(s.index);
    this.seekFrame(s.index + dir);
  }

  snapshot() {
    return { ...this.state(), playing: this.playing, speed: this.speed, duration: this.duration, live: this.live };
  }

  _frame(now) {
    if (!this.playing) return;
    const dt = Math.min(now - this._last, 100);
    this._last = now;
    this.t += dt * this.speed;
    if (this.t >= this.duration) {
      if (this.live) {
        this.t = this.duration - 0.001; // waiting for the next edit to arrive
      } else if (this.loop) {
        this.t = 0;
      } else {
        this.t = this.duration - 0.001;
        this._emit(true);
        this.pause();
        return;
      }
    }
    this._emit(false);
    this._raf = requestAnimationFrame((n) => this._frame(n));
  }

  _emit(force) {
    const s = this.state();
    if (s.index !== this._lastIndex || force) {
      const changed = s.index !== this._lastIndex;
      this._lastIndex = s.index;
      if (changed && s.index >= 0) bus.emit("timeline:keyframe", { ...s, timeline: this });
    }
    bus.emit("timeline:tick", { ...s, playing: this.playing, duration: this.duration, timeline: this });
  }
}
