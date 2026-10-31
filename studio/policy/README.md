# studio/policy — DQN edit-proposal prior

- `policy.py`: `DQNPolicy` (Policy Protocol), `UniformPolicy` fallback, `load_policy()` (falls back to uniform if no usable checkpoint).
  Design: op-id embedding head over a GINE-style encoder (same 45/10-dim features as `dqn/`, re-implemented in `features.py` because `dqn.utils` imports ADMET/PLAPT/torch_geometric). Q(s, op, site) regressed on one-step delta composite (contextual bandit, gamma=0, replay + Huber).
- `reward.py`: pure-RDKit composite (QED, logP/TPSA ranges, alerts, Lipinski).
- `ops.py`: **stand-in** edit set (14 ops) used for training/eval until the real registry lands. A checkpoint is only valid for the op ids it was trained with (`load` refuses mismatches).
- `train.py`, `eval_rl_vs_random.py`, `search.py` (tiny GA for eval only, not the engine).
- Transition format for `update`: `(parent_smiles, edit_id, site, child_smiles, reward)`.

Run: `python -m studio.policy.train --steps 4000`, then `python -m studio.policy.eval_rl_vs_random --seeds 40`.
