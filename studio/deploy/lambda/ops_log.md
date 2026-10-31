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
- `~/db_example.sh` result (repo example 1qg8, 10 samples): poses + affinity_prediction.csv produced. Pose step first needs relax env fix: openff-toolkit calls `conda list`; added shim `ln -s ~/bin/micromamba ~/bin/conda` and `CONDA_PREFIX=$HOME/micromamba/envs/relax` -> relax_final.py runs (41 s wall for 10 samples, 4 workers) and writes `*_relaxed.sdf/.pdb`.
- Gate timings (cold, first run, 10 samples, includes one-time ~7 min SO(3) table precompute in utils/so3.py, now cached in ~/DynamicBind/.so3_*2.npy): prep 4.5 s, ESM(HF) 6.8 s, inference.py 537 s total (of which ~400 s one-time SO(3)); warm: ~80 s per batch of 5 samples at 20 steps -> use samples=3.
- Env fix: dynamicbind env had `tqdm`/`numpy<2` only in ~/.local (user site). `PYTHONNOUSERSITE=1 pip install tqdm "numpy<2"` into the env (numpy 2.x from conda broke torch 2.0.1 "Numpy is not available").
- Wrote ~/ /home/ubuntu/poll_job.sh. rsync repo code (demo/main studio, binding_module, molecular_modifications, reward, synthetic_accessibility, dqn) + studio/data overlay from demo/targets export + studio/deploy -> ~/studio-app (9.9 MB).
- `tmux new-session -s dbsvc`: uvicorn dynamicbind_service:app on 127.0.0.1:8100 (studio-venv). Test dock egfr/erlotinib via POST /dock + poll: done in ~55 s wall (protein-prep cache warm), pose_confidence 0.66, affinity 6.49, ligand in pocket (min 3.2 A to CA), CA RMSD apo-vs-predicted 1.18 A in the same frame.
- `tmux new-session -s precompute`: precompute_poses.py (all 5 targets, 23 drugs, samples 3, steps 20); log ~/precompute.log.

## 3. PLAPT on the GPU (venv ~/studio-venv, py3.10)
- `python3 -m venv ~/studio-venv`; pip: torch (CUDA wheel), transformers, onnxruntime-gpu, diskcache, rdkit, fastapi, uvicorn[standard], deap, pyyaml, scipy, scikit-learn, httpx, pytest, numpy, pydantic, tqdm, sentencepiece (log ~/install_studio_venv.log; INSTALL_DONE).
- `HF_HOME=/home/ubuntu/metis-v1-storage/hf STUDIO_CACHE_DIR=/home/ubuntu/metis-v1-storage/cache ~/studio-venv/bin/python studio/deploy/lambda/bench_plapt.py egfr` (downloads Rostlab/prot_bert + seyonec/ChemBERTa-zinc-base-v1 from HF Hub on first load). Results in the envelope. ONNX head runs on the CPU provider (onnxruntime-gpu present but ORT picked CPUExecutionProvider; irrelevant at these sizes).
- `cd ~/studio-app && pytest studio -q` under Python 3.10.12: all pass (no 3.10 incompatibilities found in studio/).

## 4. API deploy (tunnel NOT started)
- rsync (script sync_to_box.sh) -> ~/studio-app; `~/.studio_env` (chmod 600): DYNAMICBIND_URL, HF_HOME, STUDIO_CACHE_DIR, STUDIO_APP, DB_MAX_QUEUE; STUDIO_API_KEY intentionally UNSET per integrator update (commented example line).
- `run_all.sh start` -> tmux dbsvc (:8100) + studio-api (:8000, `uvicorn studio.server.app:app`). `healthcheck.sh`: studio-api OK (engine_source live), GET / returns the studio HTML page, dynamicbind OK.
- Throwaway test instance on :8101 with DB_MAX_QUEUE=0 (killed by `timeout 40`): novel SMILES -> 429, bad SMILES -> 422, unknown target -> 404, precomputed hit still served -> proves cache hits bypass queue/GPU.
- Precomputed hit latency measured on :8100: 22 ms (no GPU).

## 2. DynamicBind precompute (final)
- `precompute_poses.py` finished: 5 targets, 21 drugs, 0 failures (batch wall: abl1 154 s/5, braf 93 s/3, egfr 96 s/4, fxa 98 s/4, hiv1_protease 103 s/5). Output rsynced to studio/data/precomputed_poses/ (6.7 MB) in this branch.
- Final healthcheck: studio-api OK, studio page OK, dynamicbind OK. Tunnel NOT started (a pre-existing system cloudflared, pid 2232, runs with /etc/cloudflared/config.yml and was left alone). No firewall changes, no new instances.
- Box disk: 58 GB used / 439 GB free. micromamba 8.7G, studio-venv 6.4G, HF cache 6.0G, DynamicBind runs 327M.

## 2. DynamicBind service / precompute
- Service verified (see §1/§4 above). `tmux new-session -s precompute`: `precompute_poses.py` finished all 5 targets, 21 drugs, 0 failures (batch wall: abl1 154 s/5 drugs, braf 93 s/3, egfr 96 s/4, fxa 98 s/4, hiv1_protease 103 s/5). Output on box: ~/studio-app/studio/data/precomputed_poses/<target>/<sha1>.json; rsynced back to this branch under studio/data/precomputed_poses/ (6.7 MB).
- Final healthcheck: studio-api OK, studio page OK, dynamicbind OK. Tunnel NOT started; no firewall changes; no new instances.
