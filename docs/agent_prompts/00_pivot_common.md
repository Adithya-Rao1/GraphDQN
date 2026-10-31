# Pivot preamble (read after 00_common.md; this overrides it where they differ)

PRODUCT: "Metis Lead Studio" is now a 3-stage pipeline shown in one beautiful 3D UI, deployed publicly. DEADLINE: about 2.5 hours from launch for EVERYTHING (build, integrate, deploy, rehearse). You have ~60-70 minutes. Ship a working slice in 30 minutes, commit early and often, then improve. If something is too hard, cut scope and say so; never fake results.

Pipeline
0. Target repository: ~5 protein targets, each with structure, sequence and known drugs (`studio/data/targets/`).
1. Stage "binding": take a known drug for a target and run the GA (pure GA, `policy_guidance=0`; RL is untrained) with fitness = predicted binding (PLAPT: protein sequence + SMILES -> affinity; ProtBERT for protein, ChemBERTa for ligand) with guards (similarity to the seed drug >= 0.4, MW cap, chemistry sanity). All numbers are PREDICTED and labelled so.
2. Stage "pose": DynamicBind (runs on a Lambda A100) generates ligand pose + protein conformation + its own affinity/confidence for the seed drug and top leads. Seed poses are precomputed for instant display.
3. Stage "pharmacology": the existing interpretable studio (descriptors, alerts, why-cards) optimises the lead, shown in the pocket context, with binding shown alongside.
4. "Manual" mode: the same viewer lets a user pick atoms and edits by hand with live deltas; every step is logged as a training transition (`chosen_by="user"`) and exportable as JSONL.

Deployment: a Hugging Face Space (Gradio SDK, as is) serves the static UI and proxies `/api/*` (including SSE) to a FastAPI on a Lambda GPU box reached through a cloudflared tunnel. A recorded replay is the fallback if the backend is down.

Environment: reuse the shared venv for pure-Python work: `/Users/adithyarao/.superset/worktrees/GraphDQN/demo/main/.venv/bin/python` (rdkit, fastapi, torch, deap, pytest). Do not create a new 1 GB venv unless you must. Read `studio/schema.py` (new models: `TargetInfo`, `PoseRequest/PoseResult`, `ManualPreview*`, `ManualSession`, `RunConfig.stage/target_id`, `ScoreResult.extras`) and `studio/web/README.md`. Do not edit `schema.py` or `contracts.py`; report needed changes in your handoff. Your branch starts from `demo/integrate`. Never push. Honest labelling everywhere (predicted, precomputed, replay).
