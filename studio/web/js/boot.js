// Starts the analysis panels (story, lineage, charts, map) and manual CADD mode once main.js has built window.metis.
import { init as initPanels } from "./panels/index.js";
import { init as initManual } from "./manual/index.js";

let tries = 0;
(function start() {
  if (!window.metis) { if (tries++ < 100) setTimeout(start, 100); return; }
  try { initPanels(window.metis); } catch (e) { console.error("panels init failed", e); }
  try {
    initManual(window.metis);
    // the stage switcher may already be on "manual" (deep link / reload)
    if (window.metis.stage && document.getElementById("manual-toggle")?.checked) window.MetisManual?.activate();
  } catch (e) { console.error("manual init failed", e); }
})();
