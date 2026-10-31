# Task: retrain (branch demo/retrain). Owns studio/policy/ and studio/policy/lambda/.

Problem: the wave-1 policy was trained on 14 stand-in edits; a checkpoint is only valid for the op ids it was trained with, and none is committed, so the demo currently runs with `UniformPolicy`. The real registry has 83 ops (`studio.registry.get_registry()`).

Build
1. Make training and evaluation use the real registry ops (op ids, sites, `apply`) instead of `studio/policy/ops.py` stand-ins. Keep the stand-ins only as a unit-test fixture. Train on the molecules in `studio/policy/data/start_pool.smi` (extend it with ~200 more diverse, drug-like SMILES that you verify with RDKit; no network downloads of datasets).
2. Reward = the repo scorer's composite (`studio.scoring.RDKitScorer`) delta, so the policy learns what the demo actually optimises. If a sanity-check module appears on `demo/integrate` later (`studio/sanity/`), use it when present: skip rejected children.
3. Train on CPU. Target a few thousand steps; report wall time. Save a checkpoint to `studio/policy/checkpoints/` ONLY if it is under ~5 MB, and make `load_policy(op_ids)` find it by default. Record in the checkpoint metadata: the exact op-id list hash, steps, reward definition, seed.
4. Evaluation `python -m studio.policy.eval_rl_vs_random`: use the REAL engine (`studio.engine.Engine` with `UniformPolicy` vs the trained policy at `policy_guidance` 0.5 and 0.8) on >= 6 different seeds x >= 10 rng seeds; measure generations (or evaluations) to reach a target composite and best composite at a fixed budget; mean +/- 95% CI and a paired test. If the trained policy does not beat random, say so plainly in the handoff and in `studio/policy/README.md`; do not tune until it looks good without telling.
5. Lambda scripts in `studio/policy/lambda/` (setup venv on Ubuntu 22.04 with Python 3.10, fetch `demo/integrate`, run train + eval, copy the checkpoint back). Do NOT run anything remotely; the integrator runs them.

Acceptance: checkpoint loads (DQNPolicy, not the fallback) with the real registry; tests pass; evaluation numbers reported honestly with the settings used.
