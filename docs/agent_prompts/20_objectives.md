# Task: objectives (branch demo/objectives) — owns studio/scoring/

Objective: fast, cached RDKit-descriptor scoring with user-controllable desirability.

Build
1. `studio/scoring/descriptors.py`: compute `schema.Descriptors` (QED, SA, logP (Crippen), TPSA, MW, HBD, HBA, rotatable bonds, Fsp3, heavy atoms, Lipinski violations, alert count). SA: find a working Ertl SA scorer (RDKit Contrib `sascorer` under `RDConfig.RDContribDir` if present, else the repo's `synthetic_accessibility/sa_score.py` and report which data file it needs). If unavailable, leave `sa=None` and say so.
2. `studio/scoring/alerts.py`: structural alerts via `rdkit.Chem.FilterCatalog` (PAINS, BRENK, NIH); return `AlertHit(name, family, atom_indices)`. See `studio/mock/build_mock.py` for a working snippet; fix its atom-index extraction if it is wrong.
3. `studio/scoring/scorer.py`: implements the `Scorer` Protocol. Per-descriptor desirability in [0,1] from `ObjectiveSpec` (`range`: 1 inside [low,high] with smooth falloff; `maximize`/`minimize`: monotone sigmoid), weighted composite. Sensible defaults when `config.objectives` is empty (drug-like: QED max, logP 1-3, TPSA 40-100, MW 200-450, alerts minimise to 0, Lipinski 0).
4. Cache by (canonical SMILES) for descriptors and by (SMILES, config hash) for scores; thread-safe LRU.
5. `studio/scoring/presets.py`: named objective bundles ("remove toxic alerts", "more drug-like", "more soluble") as `dict[str, ObjectiveSpec]`.

Acceptance
- < 20 ms per molecule uncached median on CPU for 20-40 heavy-atom molecules; report measured median and p95.
- Tests: desirability monotonicity, caching hit, nitro/ester example from the mock gives alerts > 0 and the nitrile/amide products give fewer, config weights change the composite.
