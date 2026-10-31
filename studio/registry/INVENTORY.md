# Edit inventory (measured, not claimed)

Source of truth: `studio/registry/ops.py::build_ops()` (checked against `edits.yaml` at load). Wrapped classes in `molecular_modifications/` are untouched.

## Counts

| Where | Count | Notes |
|---|---|---|
| `dqn/dqn_env.py::get_all_actions` | **52** action slots | atom 3, bond 3, bioisostere 9, functional group 37 (16 remove + 15 modify + 6 add). Each slot picks the site/target itself (first site, `action=5`/`0..2`, or random). |
| `molecular_modifications/macro_actions.py::build_edit_catalog` | **52** ops | Same 52, one op per slot; site chosen via `enumerate_sites`. |
| **Studio registry (this work)** | **83** ops | Atom/bond slots expanded by target element / bond type, plus 5 extra add-group fragments. |

Registry breakdown:

| Category | Ops | Detail |
|---|---|---|
| atom | 23 | 11 `atom_sub_<El>` + 11 `atom_add_<El>` (C N O S P F Cl Br I B Si, from `ModifyAtom.substitution_atoms`) + `atom_remove` |
| bond | 9 | `bond_add_{single,double,triple,aromatic}`, `bond_remove`, `bond_to_{single,double,triple,aromatic}` |
| bioisostere | 9 | acid->tetrazole, acid->phosphonic, amine->pyridyl, amide->sulfonamide, amide->retroamide, phenyl->{pyridyl,thiophene,furan,pyrrole} |
| functional_group | 42 | 16 `fg_remove_*`, 15 `fg_<src>_to_<dst>`, 11 `fg_add_*` (6 from DQN set + ether(methoxy), ketone(acetyl), amide, sulfonamide, sulfone) |
| **total** | **83** | |

The "50+" figure is the 52 DQN slots. The 83 is a finer expansion of the same primitives (not 83 independent chemistries). The two catalogs are not merged: `macro_actions.EditOp` is a different class from the studio `EditOp` Protocol.

## How each is invoked (all with explicit sites, so no `random` path is reached)

| Ops | Wrapped call | Site |
|---|---|---|
| atom_sub_* | `ModifyAtom.modify_atom(mol, 0, atom_idx, target_element)` | `[atom]`; candidates from `enumerate_site_element_candidates` (non-ring-carbon atoms only) |
| atom_add_* | `ModifyAtom.add_atom(mol, 0, atom_idx, target_element)` | `[atom]`; same list as sub (the class validates *replacement*, not addition, so many sites fail) |
| atom_remove | `ModifyAtom.remove_atom(mol, 0, atom_idx)` | `[atom]` |
| bond_add_* | `ModifyBond.optimize_bond(mol, 0, bond_site, target_bond_type)` | `[i, j]` from `get_valid_bond_add_sites`, **registry policy: only atoms 3..7 bonds apart** |
| bond_remove | `optimize_bond(mol, 1, bond_site)` | `[i, j]` from `get_valid_bond_remove_sites` |
| bond_to_* | `optimize_bond(mol, 2, bond_site, target_bond_type)` | `[i, j]` from `get_valid_bond_modify_sites` |
| bio_* | `ModifyBioisosteres.apply_modification(mol, fg, replacement, match_index)` | SMARTS match atoms (retroamide: + the amide N's external neighbour) |
| fg_remove_* | `ModifyFunctionalGroup.remove_functional_group(mol, fg, instance_idx)` | group atoms + its single attachment atom (only singly attached instances) |
| fg_*_to_* | `modify_functional_group(mol, src, dst, instance_idx)` | same as remove, for the source group |
| fg_add_* | `add_functional_group(mol, fg, site_idx)` | `[atom]` for every heavy atom with >= 1 H |

## Registry guarantees (added on top of the wrapped classes)

`apply` never raises and returns `[]` or one canonical product that parses, is a single fragment, has only supported elements (`find_unsupported_elements`), and differs from the parent. `sites()` is cheap and structural: a listed site is plausible, not guaranteed (see self-check numbers). Symmetry-equivalent sites can yield identical products; deduplication is left to the engine.

## Present in the code but NOT exposed

- Bioisostere tables list swaps with no implementation (`sulfonic_acid`, `imidazole`, `guanidine`, `thioester`, `oxime`, `CF3`, ...): apply_modification returns None for them.
- `FUNCTIONAL_GROUPS` has 25 groups; groups without a fragment (alkene, alkyne, isocyanate, isothiocyanate, methylenedioxy) cannot be added, and several others are not in the DQN remove/modify lists.
- `ModifyAtom/Bond.explore_*`, `suggest_modifications`, `batch_process`: random/compound drivers, not edits.
- `compose_macro_action` (multi-edit chains): the engine composes edits itself.
