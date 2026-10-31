# Task: explain (branch demo/explain) — owns studio/explain/

Objective: exact, deterministic explanations for each edit. No ML model: explanations are RDKit decompositions.

Build
1. `studio/explain/contributions.py`: per-atom `crippen_logp` (`rdMolDescriptors._CalcCrippenContribs`), `tpsa` (`_CalcTPSAContribs`), and `gasteiger` charges for a molecule, as `AtomContributions` (fix the Gasteiger handling in `studio/mock/build_mock.py:contributions`, which is a hack).
2. `studio/explain/explainer.py`: implements the `Explainer` Protocol. Fill `descriptor_deltas` (every `Descriptors` numeric field, child - parent), `alerts_added`/`alerts_removed` (compare by alert name; indices on the right molecule), `atom_contributions` for the child, `tags` derived from thresholds (`lowers_logp`, `raises_tpsa`, `removes_alert`, `adds_alert`, `improves_qed`, ...), and `rationale`.
3. Rationale templates in `studio/explain/templates.py`: compose from the edit's registry rationale (if `edit_op` is given), the largest descriptor changes with signs and units, and alerts resolved. Always cite numbers from the deltas, never invent. Example: "Nitro -> nitrile: removes 2 structural alerts (nitro_group, O-N single bond); logP -0.3, TPSA -43, QED +0.22."
4. `studio/explain/story.py`: `build_story_card(...)` helper: headline text and `descriptor_table` from real seed/final scores and the accepted edit path (the engine agent may call it).
5. A small standalone sanity CLI: `python -m studio.explain.demo "<parent>" "<child>"` prints the explanation.

Acceptance
- < 30 ms per explanation. Tests: deltas equal child minus parent exactly; alert add/remove correct on the nitro -> nitrile example; contributions length equals atom count; rationale contains the true numbers.
