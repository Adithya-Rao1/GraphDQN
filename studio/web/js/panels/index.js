// Entry point for the analysis panels. `init(ctx)` wires story, lineage, charts and map; each can also be initialised alone.
//   ctx = { bus, store, timeline, stage?, mounts?: {story, lineage, charts, map} }   (defaults to window.metis and #panel-* elements)
import * as story from "./story.js";
import * as lineage from "./lineage.js";
import * as charts from "./charts.js";
import * as map from "./map.js";

export { story, lineage, charts, map };

export function init(ctx = window.metis) {
  if (!ctx) throw new Error("panels.init: window.metis is not ready; call it after main.js has created it, or pass ctx");
  const out = {};
  for (const [name, mod] of Object.entries({ story, lineage, charts, map })) {
    try { out[name] = mod.init(ctx); } catch (err) { console.error(`panels/${name} failed to init`, err); }
  }
  return out;
}
