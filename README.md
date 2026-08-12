# GraphDQN

Reinforcement-learning-driven molecular optimization: an agent edits a molecule step by
step (atom/bond/bioisostere/functional-group substitutions) to maximize a reward built
from predicted drug-likeness (ADMET), protein-ligand binding affinity, synthetic
accessibility, and optional target selectivity. Two independent RL algorithms are
implemented on the same action space and reward: a DQN that scores candidate
next-molecules with a graph convolutional network, and a PPO agent with an MLP
actor/critic.

## Setup / Install

Requires **Python >= 3.10**.

```bash
git clone <this-repo-url>
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
external data fetch below:

```bash
git clone <this-repo-url>
cd GraphDQN
bash setup_remote.sh
```

This creates `.venv/`, installs the package and its dependencies, and fetches the
external data assets described below into `GraphDQN_Data/`.

## External data assets

Two subdirectories need data that isn't checked into this repo (see `.gitignore`):

- **`GraphDQN_Data/admet_data/`, `GraphDQN_Data/admet_models/`** -- ADMET-AI's DrugBank
  reference set and trained Chemprop model ensemble. `ADMET/` in this repo vendors
  ADMET-AI's *source code* but not its data. Fetch it by installing the `admet-ai`
  PyPI package once and copying its bundled resources (this is exactly what
  `setup_remote.sh` does):
  ```bash
  pip install admet-ai
  ADMET_AI_DIR=$(python3 -c 'import admet_ai, os; print(os.path.dirname(admet_ai.__file__))')
  mkdir -p GraphDQN_Data/admet_data GraphDQN_Data/admet_models
  cp "$ADMET_AI_DIR/resources/data/admet.csv" GraphDQN_Data/admet_data/admet.csv
  cp "$ADMET_AI_DIR/resources/data/drugbank_approved.csv" GraphDQN_Data/admet_data/drugbank_approved.csv
  cp -r "$ADMET_AI_DIR/resources/models/." GraphDQN_Data/admet_models/
  ```
- **`GraphDQN_Data/sa_score_data/fpscores.pkl.gz`** -- RDKit's standard Contrib SA_Score
  fragment-score file:
  ```bash
  mkdir -p GraphDQN_Data/sa_score_data
  curl -L -o GraphDQN_Data/sa_score_data/fpscores.pkl.gz \
      https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz
  ```

The PLAPT binding-affinity model (`binding_module/binding_affinity/models/affinity_predictor.onnx`)
needs no separate fetch -- it's tracked directly in this repo.

Optional: to log runs to Weights & Biases, set the `WANDB_ENTITY` / `WANDB_PROJECT`
environment variables (or just be logged in via `wandb login`; runs default to a
project named `graphdqn` under your account) before passing `--wandb` to any of the
commands below.

## Running things

### Single-run quickstart

Once the environment and data above are set up:

```bash
python -m dqn.dqn_main            # DQN, alpha-synuclein target, 1000 episodes
python -m ppo.train_ppo --target-seq "<protein sequence>"   # PPO, single target
```

Both write checkpoints (`checkpoints/dqn/`, `checkpoints/ppo/`) and log via Python's
`logging` module. For a smaller, faster check, use the sweep entry points below with
small `--num-episodes`/`--num-iterations`.

### PPO-vs-DQN sweep

```bash
python experiments/run_dqn.py --target-name alpha_synuclein --num-molecules 30 --num-episodes 200
python experiments/run_ppo.py --target-name alpha_synuclein --num-molecules 30 --num-iterations 20
python experiments/sweep.py --algorithms dqn ppo --targets alpha_synuclein egfr_kinase adrb2_gpcr --seeds 0
python experiments/compare_results.py
```

- Available named targets are defined in `experiments/data/targets.py` (alpha-synuclein,
  EGFR kinase, beta-2 adrenergic receptor). Starting molecules are sampled from a shared
  pool in `experiments/data/starting_molecules.py`.
- Both algorithms use fixed hyperparameters by default (see `dqn/dqn_hyperparams.py` and
  the default args in `experiments/run_ppo.py`). `ppo/experiments.py` provides a Ray
  Tune + Optuna hyperparameter search for PPO if you want to tune it.
- Output: per-run checkpoints under `checkpoints/{dqn,ppo}/{run_id}/`, a JSON summary per
  run under `experiments/results/{run_id}.json`, and (with `--wandb`) matching runs in
  your wandb project.
- Runtime/cost depends on your `--num-molecules`/`--num-episodes`/`--num-iterations`
  choices -- start with small pilot sizes to validate the pipeline end to end before
  scaling up.

### Legacy single-objective scripts

`all_experiments/` contains standalone DQN scripts for single-property optimization
(LogP, QED, similarity-constrained variants) -- e.g. `python all_experiments/logp_optim.py`.
