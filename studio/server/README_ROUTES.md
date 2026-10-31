# Studio API routes

Existing (unchanged): `GET /api/health`, `GET /api/presets`, `POST /api/molecules/validate`, `POST /api/runs`,
`DELETE /api/runs/{id}`, `GET /api/runs/{id}/events` (SSE).

Auth: if env `STUDIO_API_KEY` is set, every `/api/*` route except `/api/health` needs header `X-Studio-Key`
(constant-time compare, else 401). CORS stays closed; the Space proxies server-side. Body-size and rate limits unchanged.

| Route | Notes |
|---|---|
| `GET /api/targets`, `GET /api/targets/{id}` | `TargetInfo` from `studio.data.loader` when importable, else the labelled stub in `fixtures/` (health: `components.targets` = `loader`/`fixture`). 404 unknown id. |
| `GET /api/data/{path}` | Files under `studio/data/structures` (env `STUDIO_STRUCTURES_DIR`), only `.pdb/.sdf/.json`; a leading `structures/` is accepted. Traversal, dotfiles, symlink escapes -> 404. |
| `POST /api/runs` | New: `stage` `binding` needs `target_id` and `studio.binding.stage.build_binding_engine` (503 otherwise, also in mock mode); forces `policy_guidance=0`, no reactions, caps `generations<=10`, `population_size<=30` (env `STUDIO_BINDING_MAX_GENERATIONS/POPULATION`), `max_mw<=700`. `manual` -> 422. Unknown `target_id` -> 422. `pharmacology` unchanged. `build_binding_engine` is called with whichever of `target_id`/`config` it accepts. |
| `POST /api/poses` (`PoseRequest`) -> `PoseResult` | Precomputed hit returns `done`/`source="precomputed"` at once; else `queued` and a DynamicBind job (see `POSES.md`). Same (target, molecule) reuses a live job. 10/min/session, 8 active jobs max, concurrency 1, 1 h TTL. |
| `GET /api/poses/{job_id}` | `PoseResult`; 404 unknown/expired. Unreachable service -> `status="failed"` and a note. |
| `GET /api/manual/applicable?smiles=&locked=1,2` | `[{edit_id,name,category,site,rationale}]`, indices refer to the SMILES as sent. |
| `POST /api/manual/preview` (`ManualPreviewRequest`) -> `ManualPreview` | Applies the edit, sanity check (`studio.sanity.check_child` if present; failure gives `ok=true, sane=false, reason`), RDKit score (+ binding `extras` when `target_id` is given and `studio.binding.scorer.BindingScorer` works), explanation, 3D molblock aligned to the parent, `atom_map`. `ok=false` + `reason` when the edit does not apply. |
| `POST /api/manual/sessions` `{seed_smiles, target_id?}` -> `ManualSession`; `POST .../sessions/{id}/steps` (`ManualStep`) -> `ManualSession` | `reward_delta` defaults to the RDKit composite change (server-computed); `t` is set by the server. 500 steps/session. Appended to `$STUDIO_CACHE_DIR/manual/<id>.jsonl`. |
| `GET /api/manual/sessions/{id}/export` | `application/x-ndjson` download, one transition per line: `parent_smiles, edit_id, site, child_smiles, reward` (+ `chosen_by="user"`, `t`, `session_id`, `seed_smiles`, `target_id`). |
