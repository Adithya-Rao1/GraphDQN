// Upload box: SMILES / MOL / SDF -> POST /api/molecules/validate -> (lock atoms) -> POST /api/runs -> EventSource.
import { bus } from "./bus.js";
import { store } from "./state.js";
import { firstMolRecord } from "./molparse.js";
import { apiBase, openStream, stopStreams } from "./run.js";

const MAX_FILE_BYTES = 1_000_000;

async function postJson(path, body) {
  let res;
  try {
    res = await fetch(apiBase() + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (err) {
    throw new Error("Cannot reach the studio server. Is it running?");
  }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON error body */ }
  return { res, data };
}

function detailText(data, res) {
  const d = data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  return data?.error || data?.message || `Server returned HTTP ${res.status}`;
}

export function mountUpload({ onPreview, onStatus }) {
  const $ = (id) => document.getElementById(id);
  const smiles = $("smiles"), msg = $("validate-msg"), info = $("valid-info"), runBtn = $("btn-run"), checkBtn = $("btn-validate");
  const drop = $("drop"), file = $("file");
  let validated = null; // {canonical_smiles, molblock, ...}
  let request = null; // what was validated: {smiles} | {molfile}

  const showMsg = (text, kind = "") => {
    msg.hidden = !text;
    msg.textContent = text || "";
    msg.className = "msg" + (kind ? " " + kind : "");
  };

  function invalidate() {
    validated = null;
    runBtn.disabled = true;
    info.hidden = true;
  }
  smiles.addEventListener("input", () => { request = null; invalidate(); showMsg(""); });
  smiles.addEventListener("keydown", (e) => { if (e.key === "Enter") check(); });

  async function check() {
    stopStreams();
    const text = smiles.value.trim();
    if (!request && !text) return showMsg("Enter a SMILES string or drop a MOL / SDF file.");
    const body = request || { smiles: text };
    invalidate();
    showMsg("Checking molecule…", "info");
    checkBtn.disabled = true;
    store.status = "validating";
    try {
      const { res, data } = await postJson("/api/molecules/validate", body);
      if (!res.ok) throw new Error(detailText(data, res));
      if (!data?.ok) throw new Error(data?.error || "Molecule rejected by the server.");
      validated = data;
      showMsg("");
      const q = data.score ? ` · QED <b>${data.score.descriptors.qed.toFixed(2)}</b> · alerts <b>${data.score.descriptors.alerts}</b>` : "";
      info.innerHTML = `Canonical SMILES<br><code>${data.canonical_smiles.replace(/[<>&]/g, "")}</code><br>Heavy atoms <b>${data.heavy_atoms ?? "?"}</b>${q}`;
      info.hidden = false;
      runBtn.disabled = false;
      store.status = "ready";
      onPreview(data, request);
    } catch (err) {
      store.status = "idle";
      showMsg(err.message || String(err));
      onStatus?.("Molecule not accepted");
    } finally {
      checkBtn.disabled = false;
    }
  }
  checkBtn.onclick = check;

  async function run() {
    if (!validated) return;
    const cfg = {
      seed_smiles: validated.canonical_smiles,
      generations: parseInt($("cfg-gens").value, 10) || 10,
      population_size: parseInt($("cfg-pop").value, 10) || 20,
      rng_seed: parseInt($("cfg-seed").value, 10) || 0,
      policy_guidance: parseFloat($("cfg-guid").value),
      locked_atoms: [...store.locked].sort((a, b) => a - b),
    };
    runBtn.disabled = true;
    showMsg("Starting run…", "info");
    try {
      const { res, data } = await postJson("/api/runs", cfg);
      if (res.status === 429) throw new Error("The server is busy (run queue full). Try again in a moment.");
      if (!res.ok) throw new Error(detailText(data, res));
      showMsg("");
      store.status = "running";
      bus.emit("run:created", { run_id: data.run_id, config: cfg });
      openStream(data.run_id);
    } catch (err) {
      showMsg(err.message || String(err));
      runBtn.disabled = false;
    }
  }
  runBtn.onclick = run;
  $("cfg-guid").oninput = (e) => { $("cfg-guid-out").textContent = Number(e.target.value).toFixed(2); };

  // ------------------------------------------------------------------ file drop / pick
  async function takeFile(f) {
    if (!f) return;
    if (f.size > MAX_FILE_BYTES) return showMsg(`File is too large (${(f.size / 1e6).toFixed(1)} MB, limit 1 MB).`);
    if (!/\.(mol|sdf|sd|mdl)$/i.test(f.name)) return showMsg("Only .mol and .sdf files are supported (or paste a SMILES).");
    const text = await f.text();
    const { molfile, multiple } = firstMolRecord(text);
    if (!/M {2}END/.test(molfile)) return showMsg("This file does not look like a MOL/SDF record (no “M  END” line).");
    request = { molfile };
    smiles.value = "";
    smiles.placeholder = `${f.name}${multiple ? " (first record)" : ""}`;
    await check();
    if (multiple) showMsg("The file has several records; only the first one is used.", "info");
  }
  drop.onclick = () => file.click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") file.click(); };
  file.onchange = () => takeFile(file.files[0]);
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => takeFile(e.dataTransfer.files[0]));
  // dropping anywhere on the page works too
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); if (e.target !== drop && e.dataTransfer?.files?.length) takeFile(e.dataTransfer.files[0]); });

  return { check, run, setSmiles: (s) => { smiles.value = s; request = null; invalidate(); }, showMsg };
}
