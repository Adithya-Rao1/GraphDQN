# GraphDQN

Reinforcement-learning-driven molecular optimization. The agent edits a molecule step by
step (atom/bond/bioisostere/functional-group substitutions) to maximize a multi-objective reward built
from ADMET, protein-ligand binding affinity, synthetic
accessibility, and target selectivity. Can evaluate using the custom DQN or a standard PPO agent with an MLP
actor/critic.

## Setup / Install

Requires **Python >= 3.10**.

```bash
git clone https://github.com/Adithya-Rao1/GraphDQN.git
cd GraphDQN
python3 -m venv .venv && source .venv/bin/activate
pip install torch torch_geometric  
pip install -e .
```

For users working on GPU instances, `setup_remote.sh` automates all of the above plus the
external data fetch below:

```bash
git clone https://github.com/Adithya-Rao1/GraphDQN.git
cd GraphDQN
bash setup_remote.sh
```

## External data assets

- **`GraphDQN_Data/admet_data/`, `GraphDQN_Data/admet_models/`** -- ADMET-AI's DrugBank
  reference set and trained Chemprop model ensemble. Fetch it by installing the `admet-ai`
  PyPI package once and copying its bundled resources:
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

Optional: to log runs to Weights & Biases, set the `WANDB_ENTITY` / `WANDB_PROJECT`
environment variables (or just be logged in via `wandb login`; runs default to a
project named `graphdqn` under your account) before passing `--wandb` to any of the
commands below.

## Experiments

### Single-run quickstart

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

### Legacy single-objective scripts

`all_experiments/` contains standalone DQN scripts for single-property optimization
(LogP, QED, similarity-constrained variants).
