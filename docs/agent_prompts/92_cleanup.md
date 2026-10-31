# Task: cleanup (branch demo/cleanup). Owns studio/server/, studio/wiring.py, tests/studio_suite/, studio/README.md.

Problems found by the integrator
- `tests/studio_suite/test_contract.py::test_server_validate_and_stream` now runs the REAL engine (22 s) because the server loads components from `studio/wiring.py`; it should force the mock (use `Settings`/`create_app(...)` with `force_mock`, or `STUDIO_FORCE_MOCK=1`) and a separate fast test should cover the live path with a small config.
- `/api/health` `components` shows `explain: false` and `policy: false` because it checks for public names on the package. Report the truth: registry op count, scorer, explainer, embedder `method`, policy kind (`dqn` with checkpoint name, or `uniform` fallback), engine source (`live`/`mock`), whether `demo_safe` filtering is on. Add `studio/wiring.py: describe()` returning that dict and have the server expose it. Keep the existing keys working for the front-end.
- Importing `studio.server.app` must not build the real engine at import time (slow); make sure the app is created lazily (`create_app()` at startup) and measure import time.
- Add `studio/README.md`: quickstart (venv, `python -m uvicorn studio.server.app:app --port 8000`), env vars (`STUDIO_FORCE_MOCK`, `STUDIO_EMBED`, `STUDIO_ALL_EDITS`, `HF_HOME`), test command, and the directory ownership map.
- Make the whole suite fast and deterministic: report `pytest --durations=10`; target the full suite (`studio` + `tests/studio_suite`) under 60 s excluding model downloads.
- The transformers-dependent embed test is skipped when `transformers` is missing; keep that behaviour but make the skip reason visible.

Acceptance: full suite green, timings reported, health output pasted in your handoff from a real `uvicorn` run (then stop the server).
