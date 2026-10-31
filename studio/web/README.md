# Metis Lead Studio: viewer core (`studio/web/`)

Judge-facing front-end. Plain ES modules, no build step. 3Dmol.js **2.5.3** is loaded from cdnjs (pinned, with SRI);
`vendor/3Dmol-min.js` is the identical file and is used automatically when the CDN is unreachable (BSD-3 licensed).

## Running it

| Mode | How |
|---|---|
| With the stub / real server | `uvicorn studio.server.app:app` then open `http://localhost:8000/`. Upload box calls `POST /api/molecules/validate`, `POST /api/runs`, then opens `EventSource /api/runs/{id}/events`. Add `?delay=0.3` to pace the stub's SSE. |
| No backend (`?mock=1`) | Serve the repo root, e.g. `python -m http.server 8000`, open `http://localhost:8000/studio/web/index.html?mock=1`. Loads `studio/mock/mock_run.json` directly (it tries `../mock/mock_run.json`, `mock_run.json`, `/studio/mock/mock_run.json`). `file://` does not work (browsers block `fetch`). |

Query flags: `mock=1`, `autoplay=0`, `spin=0`, `overlay=logp|charge`, `perf=1` (fps HUD), `delay=<s>` (stub SSE pacing), `api=<origin>` (backend on another origin), `smiles=<SMILES>` (prefill + validate).
Keys: Space play/pause, ←/→ step, 1-4 speed (0.5/1/2/4×), S save PNG.

Everything labelled `source: "mock"` shows an amber **MOCK DATA** badge in the header and in exported PNGs.

## Files

| File | Role |
|---|---|
| `index.html`, `css/app.css` | layout (dark default, light via toggle or system; the 3D stage is always dark) |
| `js/main.js` | wiring |
| `js/timeline.js`, `js/timeline_ui.js` | shared Timeline (play/pause, speed, scrub, step, keyframes) and its bar |
| `js/morph.js` | edit morph (union molecule of parent+child via `atom_map`) |
| `js/stage.js` | 3Dmol viewer, per-frame restyling, 2D canvas for glow / alert halos / lock rings, PNG composition |
| `js/overlays.js` | logP / charge atom colouring, structural-alert halo masks |
| `js/whycard.js` | "Why" card HTML |
| `js/upload.js`, `js/run.js` | validate / run / SSE; `ingest(event)` is the single entry point for run events |
| `js/state.js` | `store`: the current run |
| `js/bus.js` | event bus |

## Wave-2 mount points

Empty `<section>`s below the main view: `#panel-lineage`, `#panel-charts`, `#panel-map`, `#panel-story`. They are **hidden while empty**; the first child you append makes the section appear (a grid row under the stage). Each is a plain block of width ≥ 420px; draw whatever you like inside. Import the bus and store from the same modules, or use the globals:

```js
import { bus } from "./bus.js";      // or window.metis.bus
import { store } from "./state.js";  // or window.metis.store
```

`window.metis = { bus, store, timeline, stage, ingest, perf, exportPng }` is available for panels and for tests.

## Event bus (`bus.on(name, fn)` / `bus.emit(name, data)`)

Run events (emitted by `run.js` after the store is updated, so `store` already contains the new data):

| Event | Data |
|---|---|
| `run:created` | `{run_id, config}` after `POST /api/runs` succeeded |
| `run:start` | `{event, payload: RunStartPayload}` (store was reset first) |
| `run:candidate` | `{event, candidate: Candidate, index}` (`index` into `store.candidates`) |
| `run:generation` | `{event, payload: GenerationPayload}` (carries `map_xy`, `population`) |
| `run:policy` | `{event, payload: PolicyMetrics}` |
| `run:end` | `{event, payload: StoryCard}` |
| `run:error` | `{payload: ErrorPayload}` |

Timeline events (the timeline's unit of time is ms at 1× speed; keyframe 0 is the seed, keyframe i ≥ 1 the i-th played edit):

| Event | Data |
|---|---|
| `timeline:keyframes` | `{timeline}`: the keyframe list changed (streaming, filter change, clear) |
| `timeline:keyframe` | `{index, keyframe, u, inHold, local, t, timeline}`: playhead entered a keyframe |
| `timeline:tick` | `{index, keyframe, u, inHold, local, t, playing, duration, timeline}`: every frame while playing, and on every seek. `u` is morph progress 0..1 |
| `timeline:play` / `pause` / `speed` | snapshot |
| `view:keyframe` | `{index, keyframe}` after the stage and why-card switched |
| `lock:change` | `{locked: number[]}` seed atom indices |
| `export:png` | `{url}` |

`keyframe = {id, kind: "seed"|"edit", candidate?, morph, morphMs, holdMs, label, rejected?, smilesOut}`; `keyframe.id` equals `Candidate.id` for edits (use it to link lineage nodes / chart points to the playhead).

Timeline API (`window.metis.timeline`): `play() pause() toggle() setSpeed(x) seek(ms) seekFrame(i, {end}) step(±1) state() snapshot()`, props `keyframes`, `offsets`, `duration`, `t`, `playing`, `live`.
To make a click on a lineage node jump the 3D view: `timeline.seekFrame(timeline.keyframes.findIndex(k => k.id === candidateId))`. Note the default filter is "Accepted": a non-accepted candidate has no keyframe until the user picks "All".

Store (`js/state.js`): `runId, source, status, config, seed{smiles,molblock,score}, candidates[], generations[], policy[], end, bySmiles(Map smiles -> {score, alerts, composite, contributions}), locked(Set), overlay`.

## How the morph works

`Morph` parses `parent_molblock` / `molblock` (V2000), superposes the child on the parent over the `atom_map` pairs (Kabsch/Horn; the viewer does not trust the producer's alignment), then builds one union molecule: mapped atoms interpolate, removed parent atoms shrink + redden + collapse toward the atom they hung off, added child atoms grow out of their attachment atom with a green glow. Per frame the union atoms' positions/colours/radii and bonds are rewritten and 3Dmol rebuilds the mesh (`model.molObj = null`). 3Dmol cannot vary opacity per atom inside one model, so fading is done by blending the colour into the stage background. There is no bloom in 3Dmol: the glow is a 2D canvas on top using additive radial gradients at `viewer.modelToScreen()` positions. Consecutive edits are chained into one camera frame when an edit's parent is the previous edit's child.

## Known limits / assumptions

* Overlays need `Explanation.atom_contributions` (child atoms). A parent's values come from earlier candidates with the same SMILES; the **seed has none** (the `run_start` event carries no per-atom data), so the seed view and the first edit show an explicit "unavailable / approximated" legend rather than invented numbers.
* Seed atom indices for locking are the atom order of the molblock returned by `validate` (or of the dropped MOL file); the engine must use the same order for `locked_atoms`.
* `validate` in the wave-0 stub returns no `molblock`: the viewer then falls back to the dropped file's own coordinates, or shows no preview (and no locking) for a typed SMILES.
* Ring bonds whose order differs only by Kekulé form (1 ↔ 2) are kept at the parent order during the morph to avoid flicker.
