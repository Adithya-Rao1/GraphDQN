# Task: engine (branch demo/engine), owns studio/engine/

Objective: the GA + policy loop that yields schema `Event`s.

Build
1. `studio/engine/ga.py`: DEAP-based loop (`tools.selNSGA2` or tournament + elitism; pick what is simplest and robust). Individuals = SMILES + lineage. Valid RDKit sanitization is mandatory; drop invalid children. Tanimoto-based niching (Morgan) to keep diversity.
2. Operators: mutation = pick an `(EditOp, site)` from `Registry.applicable(smiles, locked_atoms)` and apply it; with probability `config.policy_guidance` choose by `Policy.rank(...)` (softmax over scores), otherwise uniformly. Crossover = RDKit BRICS recombination (`BRICS.BRICSBuild` or fragment swap) when `config.use_crossover`. Locked atoms (seed indices) must survive: verify via substructure/MCS and reject violators.
3. Provenance: every child carries an `EditRecord` (edit id/name/category/site/chosen_by/parents).
4. `studio/engine/geometry.py`: `embed_3d(smiles, ref_mol=None) -> (molblock, mol)` using ETKDG + MMFF, rigid-aligned on the MCS to the parent when `ref_mol` is given (see `studio/mock/build_mock.py:molblock_3d` and `atom_map` for a working reference); `atom_map(parent, child)`.
5. `Engine.run(run_id, config)` per the Protocol: emits `run_start`, `candidate` (with `Explanation` from the Explainer), `generation` (best, mean, population, `map_xy` from the Embedder when given), `policy_metrics`, `run_end` (`StoryCard`: headline built from real descriptor deltas, `edit_path` = chain of accepted candidate ids from seed to best). Deterministic given `rng_seed`.
6. Dependency injection: `Engine(registry, scorer, explainer, embedder, policy)`. Until the real parts land, write small fakes in `studio/engine/fakes.py` (uniform policy, QED-only scorer, template explainer, a registry of ~5 trivial edits) and test with them.
7. Online learning hook: after each generation call `Policy.update(transitions)` with (parent, edit, child, reward delta) of accepted children and report its loss in `policy_metrics`.

Acceptance
- With the fakes: population 20, 10 generations completes in < 60 s on CPU; events validate against `schema.Event`; the same `rng_seed` gives identical output.
- Tests cover: invalid-child rejection, locked atoms honoured, provenance present, crossover produces valid molecules.
- Report the measured run time and children per second.
