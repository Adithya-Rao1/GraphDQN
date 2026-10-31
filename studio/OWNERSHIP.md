# Ownership map (one owner per path; never edit a path you do not own)

| Stream / branch | Owns | Consumes (read-only) |
|---|---|---|
| engine / `demo/engine` | `studio/engine/` (GA loop, operators, provenance, `geometry.py`: 3D coords, constrained embedding, atom_map) | `contracts.py`, `schema.py`; calls Registry, Scorer, Explainer, Embedder, Policy through the Protocols |
| objectives / `demo/objectives` | `studio/scoring/` | `schema.py` |
| registry / `demo/registry` | `studio/registry/` (+ wrappers over `molecular_modifications/`, which stays untouched) | `schema.py` |
| explain / `demo/explain` | `studio/explain/` | `schema.py`, Scorer output |
| viewer-core / `demo/viewer-core` | `studio/web/` (all front-end files) | SSE contract, `studio/mock/mock_run.json` |
| server / `demo/server` | `studio/server/` | all Protocols; may use stand-in fakes until real parts land |
| embed / `demo/embed` | `studio/embed/` | `schema.py` |
| policy / `demo/policy` | `studio/policy/` (+ Lambda scripts under `studio/policy/lambda/`) | `dqn/`, `molecular_modifications/` (read-only) |
| integrator (me) | `studio/schema.py`, `studio/contracts.py`, `studio/mock/`, `docs/`, `tests/studio_suite/`, merges | |

Rules: only the integrator changes `schema.py` / `contracts.py`. If you need a change, say so in your final envelope's `handoff` and work around it locally.
Do not edit `webapp/`, `dqn/`, `ppo/`, `reward/`, `molecular_modifications/` or any non-studio code.
Never push. Commit on your own branch only.
