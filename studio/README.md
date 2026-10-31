# Metis Lead Studio

Upload a molecule, watch a genetic algorithm + RL policy evolve it in 3D, every edit explained.
Scoring is RDKit descriptors only (no ADMET-AI, no PLAPT, no network at run time).

## Quickstart

```bash
python3.12 -m venv .venv --system-site-packages
.venv/bin/pip install -r studio/requirements.txt
.venv/bin/python -m uvicorn studio.server.app:app --port 8000
# open http://localhost:8000  (health: http://localhost:8000/api/health)
```

`studio.server.app:app` is a lazy shim: importing it is cheap (~0.6 s) and the real engine
is built by `create_app()` on server startup via `studio/wiring.py: build_engine()`.

`/api/health` reports the truth in `details`: `engine_source` (`live`/`mock`), `registry_ops`,
`scorer`, `explainer`, `embedder` (`morgan_pca`/`chemberta_pca`), `policy`
(`dqn` + checkpoint name, or `uniform` fallback) and `demo_safe`.

## Environment variables

| Variable | Effect |
|---|---|
| `STUDIO_FORCE_MOCK=1` | Serve the scripted mock replay (`source="mock"`); never build the real engine. |
| `STUDIO_EMBED` | `morgan` (default) or `chemberta`. ChemBERTa falls back to Morgan if it cannot load; health shows the actual `method`. |
| `STUDIO_ALL_EDITS` | Reserved for the registry's edit filter. **Not read by any code on this branch**; `demo_safe` therefore reports `false`. |
| `HF_HOME` | Hugging Face cache dir used by `transformers` when `STUDIO_EMBED=chemberta`. |
| `STUDIO_MAX_CONCURRENT`, `STUDIO_MAX_QUEUE`, `STUDIO_RUN_TIMEOUT`, `STUDIO_MAX_GENERATIONS`, `STUDIO_MAX_POPULATION`, `STUDIO_RUNS_PER_MIN`, `STUDIO_VALIDATES_PER_MIN` | Server limits (see `studio/server/core.py: Settings`). |

## Tests

```bash
.venv/bin/python -m pytest studio tests/studio_suite -q -p no:cacheprovider -rs --durations=10
```

`-rs` prints skip reasons. The ChemBERTa embed test is skipped with
`could not import 'transformers'` when `transformers` is not installed.

## Directory ownership

| Path | Owner |
|---|---|
| `studio/engine/` | engine (GA loop, operators, provenance, 3D geometry) |
| `studio/scoring/` | objectives (RDKit descriptor scorer) |
| `studio/registry/` | registry (edit operations + `edits.yaml`) |
| `studio/explain/` | explain |
| `studio/embed/` | embed (Morgan+PCA, optional ChemBERTa) |
| `studio/policy/` | policy (DQN / uniform) |
| `studio/web/` | viewer-core (front-end) |
| `studio/server/`, `studio/wiring.py`, `studio/README.md`, `tests/studio_suite/` | cleanup / server |
| `studio/schema.py`, `studio/contracts.py`, `studio/mock/`, `docs/` | integrator |

Full map: `studio/OWNERSHIP.md`. Do not edit `webapp/`, `dqn/`, `ppo/`, `reward/`, `molecular_modifications/`.
