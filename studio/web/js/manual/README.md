# Manual CADD mode (`js/manual/`, `css/manual.css`)

Hand-edit a molecule in the same 3D stage: click an atom to pick a site, see the applicable edits, hover one to preview its effect (descriptor
deltas, alerts, sanity verdict, predicted-binding delta, a 3D ghost of the child), click to apply it with the normal morph animation, undo / redo,
replay the steps, export the session as JSONL training data.

## How it fits the shared viewer

A manual session is replayed into the existing viewer as a **pseudo run**: `run_start` (`source: "manual"`, `config.stage: "manual"`), then per applied
edit a `candidate` (`edit.chosen_by: "user"`, `accepted: true`, generation = step number) and a `generation`, and `run_end` when the user presses
"Finish & show story". That is why the stage, morph, why-card, timeline, lineage, charts and story card all work unchanged for a manual path.
Side effect for `main.js`: its status line reads "Running · 0 generations · population 1"; it may special-case `source === "manual"`.

## Wiring (ui-structure)

```html
<link rel="stylesheet" href="css/manual.css">
```
```js
import * as manual from "./manual/index.js";
const m = manual.init({ ...window.metis, getTargetId: () => currentTargetId, fetch: apiFetch /* optional wrapper adding X-Studio-Key */, apiBase: "" });
// m = {activate(), deactivate(), start({smiles, molblock, score, label}), undo(), redo(), exportJsonl(), finish(), session, active}  (also window.MetisManual)
```
Activation (any one of these is enough; `init` does not touch the DOM until activated):
* `window.MetisStages.onStageChange(cb)` is called with a callback taking the stage name; `"manual"` activates, anything else deactivates;
* `bus.emit("stage:change", {stage: "manual"})` (or a bare string);
* `m.activate()` / `m.deactivate()` directly.
`init` returns `null` (and warns) when `#stage-wrap` or the stage is missing. Optional `ctx.mounts.manual` overrides `#panel-manual`.

UI placement: on wide screens a glass dock is floated over the right of `#stage-wrap` (body gets `manual-on manual-docked`; CSS insets `.stage` so the
dock never hides the molecule and the Stage's ResizeObserver re-fits the camera). Under 1000 px the dock moves into `#panel-manual`. `#panel-manual` also holds the
step strip (click a step to seek) and the training-log table with Replay / Finish.

Starting molecule: `store.seed` (a previewed upload or a run's seed), or "last run's final lead" when a non-manual run has ended; programmatically via `start(...)`.
If `store.seed.score` is missing, `POST /api/molecules/validate` supplies it.

## Interaction

| Action | Result |
|---|---|
| click atom | select it as the site (lists edits whose `site` contains it) |
| Ctrl/⌘-click a second atom | narrow to edits whose site contains both (bonds) |
| Shift/Alt-click atom, or `L`, or the Lock button | lock / unlock: locked atoms are passed to `applicable?locked=` and tracked through each edit's `atom_map` |
| hover / focus an edit | `POST /api/manual/preview` (cached) → what-if card + 3D ghost (green = added atoms, red ring = removed, amber = changed element) |
| click an edit | apply (blocked with an explanation when `sane === false` or `ok === false`) |
| Ctrl/⌘+Z, Ctrl/⌘+Shift+Z, Esc | undo, redo, clear selection |

Picking uses only the public Stage surface (`stage.viewer.modelToScreen`, `stage.morph`), so it works on edit keyframes too (the Stage's own click handler is seed-only). Clicking while an earlier step is displayed jumps to the latest step first.

## Server contract used (api-stages; shapes assumed from the brief, tolerant where possible)

| Call | Used for | Notes |
|---|---|---|
| `GET /api/manual/applicable?smiles=&locked=1,2` | edit list | array (or `{edits|items|applicable: [...]}`) of `{edit_id, name, category, site: number[], rationale, tags?}`; `site` indexes the atoms of `smiles` as returned by our molblocks |
| `POST /api/manual/preview` `{smiles, edit_id, site, target_id}` | what-if / ghost / apply | `ManualPreview`; `molblock` must be the child aligned to the parent, `atom_map` (parent idx, child idx); `score.extras` with a binding key enables the binding delta |
| `POST /api/manual/sessions` `{seed_smiles, target_id}` → `{session_id}` | training log | on failure the session is local ("not logged to server") |
| `POST /api/manual/sessions/{id}/steps` `ManualStep` | after each apply; `reward_delta` = composite change | |
| `GET /api/manual/sessions/{id}/export` | Export button (JSONL text) | on failure a client-side JSONL (`session.jsonl()`) is generated and labelled so |
| `POST /api/molecules/validate` | only to score a start molecule that has no score | |

Undone steps stay in the server log and export (they were real transitions); redo does not re-post.

## Files

`index.js` controller · `session.js` pure model (steps, undo/redo, lock mapping, event synthesis, JSONL) · `api.js` client (injectable fetch) · `overlay.js` picking + rings on a canvas above the stage · `ghost.js` 3Dmol ghost model · `ui.js` HTML builders.

## Dev harness and fixtures

```
python3 -m http.server 8765          # from the repo root
http://localhost:8765/studio/web/dev/harness.html?seed=1&shim=1&manual=1&shim_extras=1
```
`dev/harness.html` fetches `../index.html`, injects the panels, manual mode and the CSS (the wiring ui-structure will do in `main.js` / `index.html`).
`dev/manual_fetch_shim.js` (dev only) answers the manual endpoints from `fixtures/manual_fixture.json`: **real** previews generated by the studio components
(`fixtures/build_manual_fixtures.py`; the tree is the seed plus two levels of 4 children, so later molecules may have no list). Switches: `shim_reject=<edit_id,..>`
(simulate a sanity rejection), `shim_extras=1` (SYNTHETIC `plapt_affinity`, labelled, to exercise the binding UI), `shim_nosession=1` (exercise the client-side export), `shim_latency=<ms>`.
Other query flags: `fixture=<name>` plays `fixtures/<name>.json` as a run, `manual=1` activates, `seed=1` loads the fixture seed through the upload box.
