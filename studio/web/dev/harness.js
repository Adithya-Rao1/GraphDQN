// DEV ONLY: stands in for the wiring ui-structure will add to main.js / index.html.
import { playRecorded } from "../js/run.js";
import * as panels from "../js/panels/index.js";
import * as manual from "../js/manual/index.js";

const q = new URLSearchParams(location.search);
for (const href of ["css/panels.css", "css/manual.css"]) {
  const l = document.createElement("link");
  l.rel = "stylesheet"; l.href = href;
  document.head.appendChild(l);
}
if (q.get("shim") === "1") await import("./manual_fetch_shim.js");

const ctx = window.metis;
window.metisPanels = panels.init(ctx);
window.metisManual = manual.init({ ...ctx, fetch: (...a) => fetch(...a) });

if (q.get("seed") === "1") {
  // load the fixture seed through the real upload box (validate goes through the shim)
  const m = await (await fetch("fixtures/manual_fixture.json")).json();
  document.getElementById("smiles").value = m.seed.smiles;
  document.getElementById("btn-validate").click();
  for (let i = 0; i < 50 && window.metis.store.status !== "ready"; i++) await new Promise((r) => setTimeout(r, 100));
}
const fx = q.get("fixture");
if (fx) {
  try { await playRecorded([`fixtures/${fx}.json`], { spacingMs: +(q.get("spacing") || 40) }); } catch (e) { console.error(e); }
}
if (q.get("manual") === "1") window.metisManual.activate();
document.title = "Metis Lead Studio (dev harness)";
