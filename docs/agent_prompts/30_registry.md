# Task: registry (branch demo/registry) — owns studio/registry/

Objective: turn the existing edit classes into a uniform, documented, tested registry.

Context: the edits live in `molecular_modifications/` as classes `ModifyAtom`, `ModifyBond`, `ModifyBioisosteres`, `ModifyFunctionalGroup` (and the DQN action set in `dqn/dqn_env.py:get_all_actions`). Do NOT modify them; wrap them.

Build
1. Inventory: enumerate every distinct edit (the owner says there are 50+) and write `studio/registry/INVENTORY.md` with the real count by category and how each is invoked.
2. `studio/registry/ops.py`: `EditOp` implementations per the Protocol. `sites(smiles)` lists applicable atom-index sites; `apply(smiles, site)` returns valid canonical SMILES (never raises, returns [] on failure). Each op must be deterministic.
3. `studio/registry/registry.py`: `Registry` Protocol (`all`, `get`, `applicable(smiles, locked_atoms)` excluding any op whose site touches a locked atom).
4. `studio/registry/edits.yaml`: for each edit: `id`, `name`, `category`, `tags` (expected effects such as `lowers_logp`, `adds_hbd`, `removes_alert`, `raises_tpsa`), and a one-line med-chem `rationale`. **Mark every rationale `reviewed: false`**; the human owner will review them. Write them carefully and conservatively (state general tendencies, not guarantees).
5. Self-check script `python -m studio.registry.selfcheck`: for ~20 diverse molecules, runs every op on every site and reports products valid / invalid / empty counts per op.

Acceptance
- Count in INVENTORY.md matches reality; every op callable; zero exceptions across the self-check; ops with a 100% failure rate are listed in the handoff (do not hide them).
- Report: number of ops, how many applicable on the mock seed `CCOC(=O)c1ccc(cc1)[N+](=O)[O-]`.
