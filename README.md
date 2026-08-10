# GraphDQN

Reinforcement-learning-driven molecular optimization: an agent edits a molecule step by
step (atom/bond/bioisostere/functional-group substitutions) to maximize a reward built
from predicted drug-likeness (ADMET), protein-ligand binding affinity, synthetic
accessibility, and optional target selectivity. Two independent RL algorithms are
implemented on the same action space and reward: a DQN that scores candidate
next-molecules with a graph convolutional network, and a PPO agent with an MLP
actor/critic. See [Novelty / Approach](#novelty--approach) for what's actually
distinctive here, evidenced vs. inferred.

## Setup / Install

Requires **Python >= 3.10**.

```bash
git clone https://github.com/Adithya-Rao1/GraphDQN.git
cd GraphDQN
python3 -m venv .venv && source .venv/bin/activate
pip install torch torch_geometric   # install first -- see note below
pip install -e .
```

**PyTorch/PyTorch Geometric install order matters.** `pip install -e .` alone will pull
in whatever default `torch` wheel PyPI resolves, which may not match your CUDA version.
Install `torch` (and `torch_geometric`, which depends on it) yourself first, following
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for your
specific CUDA version, *then* run `pip install -e .` -- it will leave an existing
compatible `torch` install alone.

On a fresh cloud GPU instance, `setup_remote.sh` automates all of the above plus the
external data fetch below. See [Lambda.ai quickstart](#lambdaai-quickstart).

## External data assets

Two subdirectories need data that isn't checked into this repo (see `.gitignore`):

- **`Metis_Data/admet_data/`, `Metis_Data/admet_models/`** -- ADMET-AI's DrugBank
  reference set and trained Chemprop model ensemble. `ADMET/` in this repo vendors
  ADMET-AI's *source code* but not its data. Fetch it by installing the `admet-ai`
  PyPI package once and copying its bundled resources (this is exactly what
  `setup_remote.sh` does):
  ```bash
  pip install admet-ai
  ADMET_AI_DIR=$(python3 -c 'import admet_ai, os; print(os.path.dirname(admet_ai.__file__))')
  mkdir -p Metis_Data/admet_data Metis_Data/admet_models
  cp "$ADMET_AI_DIR/resources/data/admet.csv" Metis_Data/admet_data/admet.csv
  cp "$ADMET_AI_DIR/resources/data/drugbank_approved.csv" Metis_Data/admet_data/drugbank_approved.csv
  cp -r "$ADMET_AI_DIR/resources/models/." Metis_Data/admet_models/
  ```
- **`Metis_Data/sa_score_data/fpscores.pkl.gz`** -- RDKit's standard Contrib SA_Score
  fragment-score file:
  ```bash
  mkdir -p Metis_Data/sa_score_data
  curl -L -o Metis_Data/sa_score_data/fpscores.pkl.gz \
      https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz
  ```

The PLAPT binding-affinity model (`binding_module/binding_affinity/models/affinity_predictor.onnx`)
needs no separate fetch -- it's tracked directly in this repo.

## Quickstart

Single-run smoke tests, once the environment and data above are set up:

```bash
python -m dqn.dqn_main            # DQN, alpha-synuclein target, 1000 episodes
python -m ppo.train_ppo --target-seq "<protein sequence>"   # PPO, single target
```

Both write checkpoints (`checkpoints/dqn/`, `checkpoints/ppo/`) and log via Python's
`logging` module. For a smaller, faster check, use the pilot-sweep entry points instead
(next section) with small `--num-episodes`/`--num-iterations`.

## Reproducing the pilot sweep

```bash
python experiments/run_dqn.py --target-name alpha_synuclein --num-molecules 30 --num-episodes 200
python experiments/run_ppo.py --target-name alpha_synuclein --num-molecules 30 --num-iterations 20
python experiments/sweep.py --algorithms dqn ppo --targets alpha_synuclein egfr_kinase adrb2_gpcr --seeds 0
python experiments/compare_results.py
```

- **Grid**: 2 algorithms x 3 named targets (`experiments/data/targets.py`: alpha-synuclein,
  EGFR kinase, beta-2 adrenergic receptor) x a shared pool of starting molecules sampled
  from the 774-SMILES set in `all_experiments/smiles_800.py`
  (`experiments/data/starting_molecules.py`) x seed.
- **Fixed hyperparameters** for both algorithms in this first sweep (not HPO-tuned) --
  see `dqn/dqn_hyperparams.py` / default args in `experiments/run_ppo.py`. `ppo/experiments.py`
  has a working Ray Tune + Optuna search space for a future tuned PPO sweep, deliberately
  not run yet so the first comparison is apples-to-apples.
- **Output**: per-run checkpoints under `checkpoints/{dqn,ppo}/{run_id}/`, a JSON summary
  per run under `experiments/results/{run_id}.json`, and (with `--wandb`, once
  `WANDB_ENTITY`/`WANDB_PROJECT` env vars or your default wandb login are set) matching
  runs in your wandb project.
- Add `--wandb` to `run_dqn.py`/`run_ppo.py`/`sweep.py` to also log to Weights & Biases.

Runtime/cost for a given Lambda instance type will depend on your `--num-molecules`/
`--num-episodes`/`--num-iterations` choices -- start with the small pilot sizes above
(a handful of molecules, tens of episodes/iterations) to validate the pipeline end to
end before scaling up.

## Novelty / Approach

**Evidenced directly in code:**

- **Action-scoring GCN, not a fixed action head.** Rather than a fixed discrete action
  space, the DQN agent enumerates chemically valid next-molecules via
  `molecular_modifications/` (atom/bond/bioisostere edits), embeds each *candidate* with
  a 3-layer GCN + mean-pool (`dqn/dqn_network.py::DKDQNNetwork`), and picks the
  highest-scoring candidate (`DKDQNAgent.get_action`). This follows the pattern used by
  MolDQN (Zhou et al., 2019) and GCPN (You et al., 2018) -- a deliberate choice of a
  known-good pattern for this reward stack, not a novel architecture in itself.
- **Four-axis combined reward.** `reward/multi_objective.py` combines an 11-property
  ADMET/Chemprop ensemble prediction, PLAPT protein-ligand binding affinity, RDKit
  SA-Score synthesizability, and optional on/off-target selectivity into one weighted
  scalar (weights in `dqn/dqn_hyperparams.py`). Combining drug-likeness + binding
  affinity + synthesizability + selectivity in a single training-time reward is unusual
  relative to typical molecular RL work, which usually optimizes one or two properties
  (e.g. QED, penalized logP).
- **Constrained variants.** `LogPConstrainedEnv`/`QEDConstrainedEnv` in `dqn/all_envs.py`
  implement Tanimoto-similarity-constrained optimization (penalize property gains that
  drift too far from a reference molecule) -- a complete, not minimal, RL formulation.
- **PPO vs. DQN on identical inputs.** Both algorithms now share the same action-space
  library, the same reward formula, and (via `experiments/data/`) the same starting
  molecules and target list, so the sweep in this repo is a genuine value-based vs.
  policy-gradient comparison on one problem, not an apples-to-oranges comparison.

**Plausible but not documented in-repo -- worth confirming, not asserting as fact:**

- `all_experiments/generate_site.py` prepares an AutoDock Vina docking config from a
  receptor/ligand pair using PyMOL, but is a standalone script never called from any
  training code. A "PLAPT for fast in-the-loop reward, Vina for slower validation"
  workflow is a reasonable inference from this separation, but isn't stated anywhere in
  the code, so treat it as a hypothesis about intended usage rather than an established
  pipeline.

**What this repo does *not* currently contain:** no checkpoints, result CSVs, plots, or
wandb run references exist in this repository, and the DQN training loop as originally
committed had a bug that prevented any run from completing more than one episode (fixed
in this pass -- see git history). If you're citing prior results from earlier
experimentation, that evidence lives outside this repository and should be attached
separately.

## License

- Root `LICENSE` (MIT) covers this repository's original code: `dqn/`, `ppo/`,
  `molecular_modifications/`, `reward/`, `experiments/`, `all_experiments/`, and
  `binding_module/selectivity/`.
- `ADMET/` -- MIT, Copyright (c) 2024 Bindwell (vendored [ADMET-AI](https://github.com/swansonk14/admet_ai)).
- `binding_module/binding_affinity/` -- MIT, Copyright (c) 2024 Bindwell (vendored PLAPT).
- `synthetic_accessibility/` -- BSD-3-Clause, Copyright (c) 2013 Novartis Institutes for
  BioMedical Research Inc. (vendored RDKit Contrib SA_Score).

## Lambda.ai quickstart

```bash
git clone https://github.com/Adithya-Rao1/GraphDQN.git
cd GraphDQN
bash setup_remote.sh
```

This creates `.venv/`, installs the package and its dependencies, and fetches the
external data assets above into `Metis_Data/`. One command, fresh instance ready.

**Debugging from VSCode over the actual GPU box:** use VSCode's **Remote-SSH**
extension (Command Palette -> "Remote-SSH: Connect to Host...") to connect directly to
your Lambda instance's IP using the SSH key Lambda already provisions -- no container,
no extra auth flow. Once connected and the folder is open, `.vscode/launch.json`
already has debug configs (`run_dqn (pilot)`, `run_ppo (pilot)`, `sweep (pilot grid)`,
plus the legacy single-run entry points) wired to Python's `debugpy` -- set a breakpoint
and hit F5. Because the whole repo is `pip install -e .`-installed, the debugger's
module resolution matches actual execution exactly regardless of which file you run or
your terminal's cwd.

## Repo structure

```
dqn/                     DQN stack: GCN action-scoring agent, replay buffer, envs
ppo/                      PPO stack: MLP actor/critic, PPO agent, envs
molecular_modifications/  Shared action space: atom/bond/bioisostere/functional-group edits
reward/                   Shared multi-objective reward (used by both dqn/ and ppo/)
ADMET/                    Vendored ADMET-AI: drug-likeness property predictions
binding_module/           Vendored PLAPT (protein-ligand binding affinity) + selectivity
synthetic_accessibility/  Vendored RDKit SA-Score
experiments/              PPO-vs-DQN sweep pipeline: shared molecule/target config,
                          per-algorithm run scripts, sweep orchestrator, comparison table
all_experiments/          Legacy single-run demo scripts (LogP/QED optimization, etc.)
setup_remote.sh           One-command bootstrap for a fresh cloud GPU instance
.vscode/launch.json       VSCode Remote-SSH debug configs for the entry points above
```

See `CLAUDE.md` for a deeper architecture walkthrough (written for AI coding agents,
but equally useful for human contributors).
