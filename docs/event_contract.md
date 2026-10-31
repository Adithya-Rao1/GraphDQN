# Metis Lead Studio: HTTP + event contract (schema 0.1)

Source of truth for the Python models: `studio/schema.py`. This file is the human-readable summary.

## Routes (one FastAPI process, one port, static front-end served at `/`)

| Method | Path | Body -> Response |
|---|---|---|
| GET | `/api/health` | -> `{ok}` |
| GET | `/api/presets` | -> list of presets (`id`, `title`, `smiles`, `goal`, `config`) [owner: presets agent, wave 2] |
| POST | `/api/molecules/validate` | `ValidateRequest` -> `ValidateResponse` (canonical SMILES, 3D molblock, seed score; limit 60 heavy atoms; SMILES / MOL / SDF only) |
| POST | `/api/runs` | `RunConfig` -> `RunCreated {run_id}` (429 when the queue is full) |
| GET | `/api/runs/{run_id}/events` | SSE stream of `Event`s; the SSE `event:` name equals `type` |

## Event envelope

`{type, run_id, seq, payload}`; `seq` starts at 0 and increases by 1. A run is: one `run_start`, then per generation any number of `candidate` events followed by one `generation` and one `policy_metrics`, then exactly one `run_end` (or an `error`).

| type | payload model | meaning |
|---|---|---|
| `run_start` | `RunStartPayload` | config, seed 3D molblock, seed score, `source` = `live` / `replay` / `mock` |
| `candidate` | `Candidate` | a child: SMILES, 3D molblock aligned to its parent, `atom_map` (parent idx, child idx), the `EditRecord`, `ScoreResult`, `Explanation`, `accepted` |
| `generation` | `GenerationPayload` | population summary + optional 2D chemical-space `map_xy` (`morgan_pca` or `chemberta_pca`) |
| `policy_metrics` | `PolicyMetrics` | RL curves: policy choice rate, mean reward of policy vs random edits, loss |
| `run_end` | `StoryCard` | headline, seed vs final table, `edit_path` of candidate ids |
| `error` | `ErrorPayload` | message |

## Invariants every producer must keep

1. `source` is truthful. Mock and replayed data are labelled as such; the UI shows it.
2. Every SMILES is RDKit-sanitizable and canonical. Every molblock is 3D, heavy atoms only.
3. `atom_map` is a partial map from parent atom indices to child atom indices over the maximum common substructure. Unmapped parent atoms are "removed", unmapped child atoms are "added".
4. `Explanation.descriptor_deltas` = child - parent, computed from the same `Descriptors` as the scores. Alerts removed are indexed on the parent, alerts added on the child.
5. Scores are computed from RDKit descriptors only (QED, SA, logP, TPSA, MW, HBD, HBA, rotatable bonds, Fsp3, Lipinski violations, structural alerts). No ADMET-AI, no PLAPT, no network calls in the run loop.
6. Everything is deterministic for a given `RunConfig.rng_seed`.

## Mock data

`studio/mock/mock_run.json` is a scripted 3-edit run on ethyl 4-nitrobenzoate with real RDKit descriptors, every event tagged `source: "mock"`. Regenerate: `python -m studio.mock.build_mock`. The stub server replays it for any run.
