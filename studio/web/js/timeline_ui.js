// DOM for the bottom timeline bar. All logic lives in timeline.js; this only renders and forwards input.
import { bus } from "./bus.js";

export function mountTimelineUI(timeline, { onFilter }) {
  const $ = (id) => document.getElementById(id);
  const playBtn = $("tl-play"), track = $("tl-track"), segs = $("tl-segs"), head = $("tl-head"), info = $("tl-info"), latest = $("tl-latest");

  playBtn.onclick = () => timeline.toggle();
  $("tl-prev").onclick = () => timeline.step(-1);
  $("tl-next").onclick = () => timeline.step(1);
  $("tl-speed").onclick = (e) => {
    const b = e.target.closest("button[data-speed]");
    if (!b) return;
    timeline.setSpeed(parseFloat(b.dataset.speed));
  };
  $("tl-filter").onclick = (e) => {
    const b = e.target.closest("button[data-filter]");
    if (!b) return;
    for (const x of $("tl-filter").children) x.classList.toggle("on", x === b);
    onFilter(b.dataset.filter);
  };
  latest.onclick = () => { timeline.seekFrame(timeline.keyframes.length - 1); timeline.play(); };

  let scrubbing = false;
  const seekFromEvent = (e) => {
    const r = track.getBoundingClientRect();
    timeline.seek(((e.clientX - r.left) / r.width) * timeline.duration);
  };
  track.addEventListener("pointerdown", (e) => {
    if (!timeline.duration) return;
    scrubbing = true;
    track.setPointerCapture(e.pointerId);
    seekFromEvent(e);
  });
  track.addEventListener("pointermove", (e) => { if (scrubbing) seekFromEvent(e); });
  track.addEventListener("pointerup", () => { scrubbing = false; });
  track.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { timeline.step(-1); e.preventDefault(); }
    if (e.key === "ArrowRight") { timeline.step(1); e.preventDefault(); }
  });

  function renderSegments() {
    const total = timeline.duration || 1;
    segs.innerHTML = "";
    timeline.keyframes.forEach((k, i) => {
      const d = document.createElement("div");
      d.className = "tl-seg" + (k.kind === "seed" ? " seed" : "") + (k.rejected ? " rejected" : "");
      d.style.left = (timeline.offsets[i] / total) * 100 + "%";
      d.style.width = ((k.morphMs + k.holdMs) / total) * 100 + "%";
      d.title = k.label;
      const m = document.createElement("div");
      m.className = "morph";
      m.style.width = (k.morphMs / (k.morphMs + k.holdMs)) * 100 + "%";
      const lbl = document.createElement("span");
      lbl.textContent = k.kind === "seed" ? "Seed" : `${i}. ${k.label}`;
      d.append(m, lbl);
      segs.appendChild(d);
    });
    update();
  }

  function update() {
    const s = timeline.state();
    head.style.left = timeline.duration ? (timeline.t / timeline.duration) * 100 + "%" : "0%";
    playBtn.innerHTML = timeline.playing ? "&#10074;&#10074;" : "&#9654;";
    playBtn.setAttribute("aria-label", timeline.playing ? "Pause" : "Play");
    const n = Math.max(timeline.keyframes.length - 1, 0);
    const waiting = timeline.live && timeline.t >= timeline.duration - 5;
    info.innerHTML = (timeline.live ? `<span class="live-dot"></span>` : "") +
      (s.index < 0 ? "No run yet" : s.keyframe.kind === "seed" ? `Seed · ${n} edit${n === 1 ? "" : "s"}${timeline.live ? " so far" : ""}` : `Edit ${s.index} of ${n}${waiting ? " · waiting for next" : ""}`);
    const behind = timeline.live && s.index >= 0 && timeline.keyframes.length - 1 - s.index >= 2;
    latest.hidden = !behind;
    for (const b of $("tl-speed").children) b.classList.toggle("on", parseFloat(b.dataset.speed) === timeline.speed);
  }

  bus.on("timeline:keyframes", renderSegments);
  bus.on("timeline:tick", update);
  bus.on("timeline:play", update);
  bus.on("timeline:pause", update);
  bus.on("timeline:speed", update);
  renderSegments();
}
