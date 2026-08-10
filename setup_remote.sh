#!/usr/bin/env bash
# Bootstrap a fresh Lambda.ai (or any Linux/macOS) GPU instance to run GraphDQN.
#
# Usage (from an already-cloned repo, or let this script clone it):
#   git clone https://github.com/Adithya-Rao1/GraphDQN.git && cd GraphDQN && bash setup_remote.sh
#
# What this does:
#   1. Creates a Python virtualenv (.venv/)
#   2. pip installs this repo + all dependencies (pyproject.toml)
#   3. Fetches external data assets required by ADMET/ and synthetic_accessibility/
#      (not checked into git -- see .gitignore) into Metis_Data/
#   4. Checks for wandb credentials
#
# After this completes, connect over VSCode's "Remote - SSH" extension and open this
# directory -- .venv/ + pip install -e . means the debugger's module resolution matches
# actual execution exactly, regardless of which file you run or your cwd.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Checking Python version (requires >=3.10)..."
"$PYTHON_BIN" -c "import sys; assert sys.version_info >= (3, 10), f'Python {sys.version} is too old, need >=3.10'"

echo "==> Creating virtualenv at .venv/ ..."
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing PyTorch (CUDA build)..."
# Installs the default CUDA-enabled wheel from PyPI. If this doesn't match your GPU's
# CUDA version, install torch manually first (see https://pytorch.org/get-started/locally/)
# before re-running this script -- `pip install -e .` below will then leave your existing
# torch install alone.
pip install torch torch_geometric

echo "==> Installing graphdqn (pip install -e .)..."
pip install -e .

echo "==> Fetching ADMET-AI reference data + model ensemble into Metis_Data/ ..."
mkdir -p Metis_Data/admet_data Metis_Data/admet_models Metis_Data/sa_score_data

if [ ! -f Metis_Data/admet_data/admet.csv ]; then
    echo "    Installing admet-ai temporarily to extract its bundled resources..."
    pip install --quiet admet-ai
    ADMET_AI_DIR="$(python3 -c 'import admet_ai, os; print(os.path.dirname(admet_ai.__file__))')"
    cp "$ADMET_AI_DIR/resources/data/admet.csv" Metis_Data/admet_data/admet.csv
    cp "$ADMET_AI_DIR/resources/data/drugbank_approved.csv" Metis_Data/admet_data/drugbank_approved.csv
    cp -r "$ADMET_AI_DIR/resources/models/." Metis_Data/admet_models/
    echo "    Done. (admet-ai package itself is not needed after this -- the vendored ADMET/ code is used instead.)"
else
    echo "    Metis_Data/admet_data/admet.csv already present, skipping."
fi

if [ ! -f Metis_Data/sa_score_data/fpscores.pkl.gz ]; then
    echo "    Fetching RDKit's SA_Score fragment data..."
    curl -sL -o Metis_Data/sa_score_data/fpscores.pkl.gz \
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"
else
    echo "    Metis_Data/sa_score_data/fpscores.pkl.gz already present, skipping."
fi

echo "==> binding_module/binding_affinity/models/affinity_predictor.onnx is already tracked in git -- no fetch needed."

echo "==> Checking wandb login..."
if ! wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "    Not logged in to wandb. Run 'wandb login' before starting a run with --wandb,"
    echo "    or set WANDB_API_KEY. Set WANDB_ENTITY/WANDB_PROJECT env vars to route runs"
    echo "    to your own project (defaults to project=graphdqn under your default entity)."
fi

echo "==> Setup complete. Try a smoke test:"
echo "    source .venv/bin/activate"
echo "    python experiments/run_dqn.py --target-name alpha_synuclein --num-molecules 5 --num-episodes 10"
