# Analysis panels (`js/panels/`, `css/panels.css`)

Story card, lineage tree, charts and population map. Plain ES modules, no build step, no chart library. They read the shared `store`
and react to the shared event bus; they never fetch anything.

## Wiring (ui-structure)

```html
<link rel="stylesheet" href="css/panels.css">
```
```js
import * as panels from "./panels/index.js";
// after main.js has created window.metis = {bus, store, timeline, stage, ...}
panels.init(window.metis);          // returns {story, lineage, charts, map}; each entry has a redraw method
// optional ctx.mounts = {story, lineage, charts, map} overrides the #panel-* elements
```
Each module also exports its own `init(ctx)` (`panels/story.js` etc.). A failing panel logs and does not stop the others.
Mount points: `#panel-story` (spans the full grid row), `#panel-lineage`, `#panel-charts`, `#panel-map`. All stay empty (hidden) until their first data.

## What each panel subscribes to

| Panel | Bus events | Store fields read | Notes |
|---|---|---|---|
| `story.js` | `run:start` (clears), `run:end` (renders), `timeline:pause` (scrolls the card into view once playback of a finished run ends) | `seed`, `candidates`, `source`, `generations`, `end`, `overlay.autorotate` | Two synced 3Dmol views (persistent viewers, so repeated runs do not leak WebGL contexts). Atoms added along `edit_path` are green on the final, atoms removed are red on the seed; the final is superposed on the seed through the composed `atom_map`s. Adds a floating "Run complete · see the story ↓" chip to `#stage-wrap` while the card is off-screen. |
| `lineage.js` | `run:start`, `run:candidate`, `run:end`, `view:keyframe` | `seed`, `candidates` (accepted only), `end.edit_path` | Live tree while streaming; after `run_end` it shows the winning path, "show all accepted" draws the whole tree. Node click: `timeline.seekFrame(timeline.keyframes.findIndex(k => k.id === id))` then play. A node with no keyframe (filtered out) shows a toast. |
| `charts.js` | `run:start`, `run:generation`, `run:end` (draw-in animation), `view:keyframe` (playhead marker) | `generations`, `candidates`, `bySmiles`, `seed.score` | Composite best/mean, best predicted affinity so far (only when `score.extras` has a binding value, tagged *predicted*), alerts of the best molecule per generation. |
| `map.js` | `run:start`, `run:generation`, `run:end`, `view:keyframe` | `generations[].map_xy/map_method/population`, `end`, `seed`, `source` | Dots coloured by first generation seen; seed star, best ring, edit-path polyline, playhead ring. States `map_method` and that it is a visual aid. The seed sits at the origin for non-mock runs because the engine anchors the map on the seed. |

All panels only use public pieces of the viewer: `store`, `timeline.keyframes/seekFrame/play`, `bus`.

## Honesty rules implemented

* Binding values (`score.extras.plapt_affinity`, or `plapt_pkd`) are always tagged **predicted**; nothing is shown when the keys are absent.
* The `source` of the run is shown on the story card (`mock` additionally watermarks the card); the map footer says so for mock runs.
* The map states its method (`morgan_pca` / `chemberta_pca`) and that axes have no physical meaning.

## Shared helpers

`util.js` (escape, formatting, `bindingOf(score)`, count-up tween, toast, `frameIndex`), `mol2d.js` (`molSvg(molblock)`: the 3D conformer projected face-on to an inline SVG, used for lineage thumbnails and map tooltips).

## Dev / test

`../dev/harness.html?fixture=run_rdkit_local` loads `../index.html` at run time and injects these panels (see `../dev/README` section in `../js/manual/README.md`).
Fixtures in `../fixtures/`: `run_rdkit_local.json` (a real GA run recorded with the local engine, source `live`), `run_binding_dev.json` (same run with **synthetic** `extras.plapt_affinity`, stamped `mock`, to exercise the binding UI), `../../mock/mock_run.json` via `fixture=../../mock/mock_run`.
