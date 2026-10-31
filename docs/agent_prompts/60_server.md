# Task: server (branch demo/server) — owns studio/server/

Objective: production-minded FastAPI app on ONE port that serves the static front-end and the API, safe for public judges. The existing stub in `studio/server/app.py` defines the routes; keep them and replace the internals.

Build
1. Real run execution: `Engine.run()` consumed in a worker thread/process pool, events pushed to a per-run queue, delivered over SSE (`/api/runs/{id}/events`), with heartbeat comments every ~15 s. Support late-joining clients (replay buffered events from `seq 0`).
2. Injection: construct the real `Engine` from the other streams when importable (try/except), else fall back to the mock replay and mark `source="mock"`. An env var `STUDIO_FORCE_MOCK=1` forces the mock.
3. Upload safety: SMILES / MOL / SDF only, RDKit-sanitize, <= 60 heavy atoms, <= 200 KB body, no pickle or arbitrary file types.
4. Limits: per-session (cookie) and global concurrency limits, queue with 429 + `Retry-After`, per-run wall-clock timeout (default 90 s) and generation/population caps, result cache keyed by (canonical seed SMILES + config hash) that replays prior events with `source="replay"`.
5. `POST /api/molecules/validate` returns the 3D molblock (use `studio.engine.geometry` if present, else a local helper) and the seed `ScoreResult` when a Scorer is available.
6. `GET /api/presets` serving `studio/mock/presets.json` if present (empty list otherwise).
7. Logging without PII, `/api/health` with component availability flags (which real parts loaded).
8. Tests with `fastapi.testclient`: upload validation, limits, SSE ordering, cache replay, timeout.

Acceptance
- Judge flow end to end against the mock engine and (when available) the real one. Report limits you chose and why. Do not add auth or a database.
