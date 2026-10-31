# Lambda ops log (box: ubuntu@150.136.42.38, A100-SXM4-40GB)

Every remote command run by the lambda-ops worker. No secrets involved. Times are approximate local order.

## 0. Probe
- `hostname; nvidia-smi ...; python3 --version; df -h; which docker cloudflared nvcc tmux rsync git; ls ~; free -g; nproc`
  -> host 150-136-42-38, A100 40GB idle, Python 3.10.12, 457G free, 216 GB RAM, 30 cores, docker/cloudflared/nvcc/tmux/rsync/git present.

## 1. DynamicBind gate
- `git clone --depth 1 https://github.com/luwei0917/DynamicBind.git ~/DynamicBind`; read README/SETUP_MINIMAL (README: workdir.zip from zenodo 10137507, v2 at 10183369).
- `curl github.com/mamba-org/micromamba-releases/.../micromamba-linux-64 -> ~/bin/micromamba` (v2.10.0). MAMBA_ROOT_PREFIX=~/micromamba.
- Background: `curl zenodo.org/records/10183369/files/workdir.zip` (472 MB) -> /home/ubuntu/metis-v1-storage/dynamicbind/, unzip.
- Background: `micromamba create -n dynamicbind -c conda-forge python=3.10 pip rdkit biopython pyyaml pandas scipy numpy`
- Background: `micromamba create -n relax -c conda-forge python=3.9 openmm pdbfixer libstdcxx-ng openmmforcefields openff-toolkit ambertools biopython pandas` (python 3.9 not 3.8: 3.8 is EOL on conda-forge)
- `micromamba` envs built OK: dynamicbind (py3.10) and relax (py3.9, openmm 8.3.1, pdbfixer, openff-toolkit, ambertools: tleap/antechamber present).
- dynamicbind env, pip: `torch==2.0.1 torchvision==0.15.2 (download.pytorch.org cu117)`, `torch-geometric==2.3.0 pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv (data.pyg.org/whl/torch-2.0.1+cu117.html)`, `e3nn==0.5.1 fair-esm==2.0.0 spyrmsd==0.5.2`, `transformers==4.44.2 numpy<2`, `py-spy`. torch.cuda.is_available()=True on A100.
  - NOTE: env python must be run with `LD_LIBRARY_PATH=$ENV/lib` (rdkit needs the conda libstdc++; system one is too old).
- SUBSTITUTION: upstream ESM step pulls fair-esm weights from dl.fbaipublicfiles.com (not on the allowed list). Wrote `hf_esm_extract.py` (scp'd to ~/): loads `facebook/esm2_t33_650M_UR50D` from the official HF Hub (HF_HOME=/home/ubuntu/metis-v1-storage/hf) and writes the same `{label, representations:{33: [L,1280]}}` .pt format.
- `~/db_example.sh` (manual pipeline of run_single_protein_inference.py: clean_pdb -> esm_embedding_preparation -> hf_esm_extract -> inference.py -> relax_final.py) run in background on the repo example (1qg8 protein, first ligand of 1qg8_input.csv; savings_per_complex 3, inference_steps 20, samples_per_complex 10).
- inference.py first import computes SO(3) tables (utils/so3.py) once (single-thread, slow) and caches `.so3_*2.npy` in ~/DynamicBind/ (py-spy dump confirmed).
## 3 (started in parallel while waiting): studio venv
- `python3 -m venv ~/studio-venv`; background `pip install torch transformers onnxruntime-gpu diskcache rdkit fastapi uvicorn[standard] deap pyyaml scipy scikit-learn httpx pytest numpy pydantic tqdm sentencepiece` (log ~/install_studio_venv.log).
