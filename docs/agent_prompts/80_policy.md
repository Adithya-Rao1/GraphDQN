# Task: policy (branch demo/policy), owns studio/policy/. Phase 1 is local; the Lambda phase starts only after the integrator confirms SSH access.

Objective: the RL piece: a DQN edit-proposal prior trained on the fast RDKit-descriptor reward, plus honest evidence of whether it helps.

Phase 1 (local, CPU): `studio/policy/`
1. Read `dqn/` (`dqn_env.py`, `dqn_network.py`, `replay_buffer.py`, `dqn_main.py`, `utils.py`) read-only to understand the graph featurisation and action space; reuse via import where possible.
2. `studio/policy/reward.py`: a pure-RDKit composite reward (QED, logP/TPSA targets, alert count, Lipinski) matching the objectives agent's defaults. Do not import ADMET-AI or PLAPT.
3. `studio/policy/policy.py`: implements the `Policy` Protocol. `rank(smiles, candidates)` scores `(EditOp, site)` pairs with the Q-network (map ops to the DQN action space or learn an op-id embedding head, whichever is simpler; document the choice). `update(transitions)` does one online replay step. Provide `UniformPolicy` as the always-available fallback. Load a checkpoint from `studio/policy/checkpoints/` if present.
4. Training script `python -m studio.policy.train --steps N --start-pool <file>` using starting molecules from a SMILES file; writes logs and a small checkpoint.
5. Evaluation `python -m studio.policy.eval_rl_vs_random`: >= 20 seeds, same GA budget, policy-guided vs uniform mutation, measuring generations (or evaluations) to reach a target composite score. Report mean +/- CI and a paired test. Report the result honestly even if the policy does not help.

Phase 2 (Lambda, only when told): the integrator will provide SSH access. Scripts for it go in `studio/policy/lambda/` (setup, run, fetch checkpoint). Do not run anything remote until instructed.

Acceptance
- Phase 1: unit tests; a short CPU training smoke run; `UniformPolicy` and `Policy` are interchangeable in the engine; the eval script runs on a small config and prints real numbers.
- Never claim an improvement you did not measure.
