# Task: embed (branch demo/embed) — owns studio/embed/

Objective: chemical-space map for the GA population. Simplest honest design: Morgan fingerprints + PCA first; ChemBERTa embeddings as an optional upgrade behind the same interface.

Build
1. `studio/embed/morgan.py`: `Embedder` implementation (`method="morgan_pca"`): 2048-bit Morgan (radius 2) -> PCA(2) fitted on the molecules passed in plus the optional `anchor` (seed), stable across calls within a run (refit on the union seen so far and keep orientation consistent, e.g. Procrustes-align to the previous projection).
2. `studio/embed/chemberta.py`: `method="chemberta_pca"`. Lazy-load `DeepChem/ChemBERTa-5M-MLM` (fallbacks: `DeepChem/ChemBERTa-77M-MLM`, `seyonec/ChemBERTa-zinc-base-v1`) with `transformers`, CPU, mean-pooled last hidden state, batched, cached by canonical SMILES. FIRST verify the checkpoint's licence and file size from the Hub model card and put both in your handoff. Pin an exact revision hash in code. Weights are pickle-format: load only from the official org at the pinned revision. Cache dir from env `HF_HOME` (the Space mounts persistent storage at /metis-data, so deployment sets `HF_HOME=/metis-data/hf-cache`). `STUDIO_EMBED=morgan|chemberta` selects; default morgan.
3. `studio/embed/__init__.py`: `get_embedder()` returning ChemBERTa if enabled and loadable, otherwise Morgan, never raising; set `method` truthfully so the UI can label it.
4. `distance_from_seed(...)` helper for the story card ("distance from seed in embedding space") plus Tanimoto.

Acceptance
- Morgan path: 100 SMILES -> map in < 100 ms. ChemBERTa path: report measured time for 100 SMILES on CPU (cold and warm) and memory. Tests: fallback engages when the model is missing or `transformers` is absent; output is index-aligned with input; deterministic.
- Do not claim ChemBERTa explains anything; it is only used for the map and similarity.
