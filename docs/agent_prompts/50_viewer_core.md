# Task: viewer-core (branch demo/viewer-core) — owns studio/web/

Objective: the judge-facing front-end core. The visuals are the most important part of the demo: make it striking, smooth and projector-readable. 3Dmol.js (pin an exact version from cdnjs, at least two weeks old), no build step, no framework required (small vanilla ES modules are fine).

Build against `studio/mock/mock_run.json` (served by the stub server via the real SSE path; also support `?mock=1` that loads the JSON file directly so the page works with no backend).
1. `index.html` + modules: layout (dark theme, large 3D stage, side panel, bottom timeline). Light/dark safe, responsive to a projector.
2. `timeline.js`: a shared Timeline (play/pause, speed, scrub, step, keyframes) driving everything.
3. `morph.js`: edit morph between `parent_molblock` and `molblock` using `atom_map`: mapped atoms interpolate positions; added atoms grow in with a green glow; removed atoms shrink and fade red. Approximate glow with per-frame style changes (radius/opacity/colour) since 3Dmol has no bloom. Keep it smooth (target 30+ fps for ~40 atoms; report what you measured).
4. Overlays (toggleable): atom colouring by `crippen_logp` contributions (hydrophobicity) and by charge; structural-alert halos (red) on `alerts_*` atom indices, disappearing when removed.
5. "Why" card: edit name, rationale, descriptor deltas with up/down arrows, alerts removed/added.
6. Upload box: SMILES text, file drop for MOL/SDF, calls `POST /api/molecules/validate` then `POST /api/runs` then opens `EventSource`. Show validation errors clearly.
7. Atom locking: click an atom in the seed view to toggle it as locked (shown with a lock ring), included as `locked_atoms` in the run request.
8. Export: PNG screenshot button (3Dmol `pngURI`). GIF/video is a stretch.
Do NOT build lineage tree, charts, Pareto/population map, story card, before/after wipe: a wave-2 agent adds those. Leave clear DOM mount points (`#panel-lineage`, `#panel-charts`, `#panel-map`, `#panel-story`) and a documented event bus (`studio/web/README.md`) they can subscribe to.

Acceptance
- With `?mock=1` and with the stub server, the 3-edit mock run plays with the morph, overlays and why-card. Provide screenshots (headless browser or the in-app browser) in the handoff.
- No console errors. State what you could not verify.
