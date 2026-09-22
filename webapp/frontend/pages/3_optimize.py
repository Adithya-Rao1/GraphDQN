import streamlit as st

from api_client import ApiError, get_client
from components.pareto_progress import render_pareto_sweep_progress
from components.progress import render_run_progress
from style import inject_theme

inject_theme()

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

api = get_client()

st.title("Optimize Molecules")

if st.session_state.get("active_run_id"):
    st.subheader("Training in progress")
    run = render_run_progress(api, st.session_state.active_run_id)
    if run["status"] in ("completed", "failed", "cancelled", "killed"):
        if st.button("Start another run"):
            st.session_state.active_run_id = None
            st.rerun()
    st.stop()

if st.session_state.get("active_sweep_id"):
    st.subheader("Pareto sweep in progress")
    sweep = render_pareto_sweep_progress(api, st.session_state.active_sweep_id)
    if sweep["status"] in ("completed", "failed", "cancelled", "killed"):
        if st.button("Start another sweep"):
            st.session_state.active_sweep_id = None
            st.rerun()
    st.stop()

st.subheader("1. Starting molecule")
molecules = api.list_molecules()
with st.expander("Add a new molecule"):
    new_smiles = st.text_input("SMILES", key="new_mol_smiles")
    if new_smiles:
        preview = api.preview_molecule(new_smiles)
        if preview["valid"]:
            st.image(f"data:image/png;base64,{preview['image_b64']}", width=150)
            if st.button("Save this molecule"):
                api.create_molecule(new_smiles)
                st.rerun()
        else:
            st.error(preview.get("error"))

if not molecules:
    st.info("Add a starting molecule above to continue.")
    st.stop()

mol_options = {f"{m['label'] or m['canonical_smiles']} (#{m['id']})": m["id"] for m in molecules}
mol_choice = st.selectbox("Use molecule", list(mol_options))
starting_molecule_id = mol_options[mol_choice]

st.subheader("2. Target protein")
proteins = api.list_proteins()
example_targets = api.example_targets()

with st.expander("Add a new protein"):
    example_choice = st.selectbox("Start from an example target (optional)",
                                   ["-"] + [t["name"] for t in example_targets])
    default_seq = next((t["sequence"] for t in example_targets if t["name"] == example_choice), "")
    new_prot_name = st.text_input("Name", value=example_choice if example_choice != "-" else "")
    new_prot_seq = st.text_area("Sequence", value=default_seq)
    if st.button("Save this protein") and new_prot_name and new_prot_seq:
        api.create_protein(new_prot_name, new_prot_seq)
        st.rerun()

if not proteins:
    st.info("Add a target protein above to continue.")
    st.stop()

prot_options = {f"{p['name']} (#{p['id']})": p["id"] for p in proteins}
target_choice = st.selectbox("Target protein", list(prot_options))
target_protein_id = prot_options[target_choice]

use_off_target = st.checkbox("Also optimize for selectivity against an off-target protein")
off_target_protein_id = None
if use_off_target:
    off_choice = st.selectbox("Off-target protein", list(prot_options), key="off_target_select")
    off_target_protein_id = prot_options[off_choice]

st.subheader("3. Objectives")
col1, col2 = st.columns(2)
with col1:
    admet_weight = st.slider("ADMET weight", 0.0, 1.0, 0.3, 0.05)
    binding_weight = st.slider("Binding affinity weight", 0.0, 1.0, 0.5, 0.05)
with col2:
    synthetic_weight = st.slider("Synthetic accessibility weight", 0.0, 1.0, 0.2, 0.05)
    selectivity_weight = st.slider("Selectivity weight", 0.0, 1.0, 0.3, 0.05,
                                    disabled=not use_off_target)

admet_props = api.admet_properties()
selected_props = st.multiselect(
    "ADMET properties to optimize",
    [p["name"] for p in admet_props],
    default=[p["name"] for p in admet_props],
)

admet_directions = {}
if selected_props:
    st.markdown('<span class="metis-meta">Direction for each selected property:</span>', unsafe_allow_html=True)
    prop_defaults = {p["name"]: p["default_direction"] for p in admet_props}
    dir_cols = st.columns(min(len(selected_props), 3))
    for i, prop in enumerate(selected_props):
        with dir_cols[i % len(dir_cols)]:
            admet_directions[prop] = st.selectbox(
                prop, ["maximize", "minimize"],
                index=0 if prop_defaults[prop] == "maximize" else 1,
                key=f"dir_{prop}",
            )

st.subheader("4. Training")

tab_single, tab_sweep = st.tabs(["Train a single agent", "Run a Pareto sweep"])

with tab_single:
    st.markdown(
        '<span class="metis-meta">Tip: use ~10-20 episodes for a fast smoke-test run before committing to a full '
        'training run (hundreds of episodes, each involving real ADMET/binding-affinity model inference, can take '
        'a while).</span>',
        unsafe_allow_html=True,
    )
    config_name = st.text_input("Config name", value="My optimization config", key="single_config_name")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_episodes = st.number_input("Episodes", min_value=1, value=20, key="single_num_episodes")
    with col2:
        max_steps = st.number_input("Max steps per episode", min_value=1, value=40, key="single_max_steps")
    with col3:
        seed = st.number_input("Seed", min_value=0, value=0, key="single_seed")
    checkpoint_interval = st.number_input("Checkpoint interval", min_value=1, value=10, key="single_checkpoint_interval")

    if st.button("Train agent", type="primary"):
        try:
            config = api.create_config({
                "name": config_name,
                "starting_molecule_id": starting_molecule_id,
                "target_protein_id": target_protein_id,
                "off_target_protein_id": off_target_protein_id,
                "admet_weight": admet_weight,
                "binding_weight": binding_weight,
                "synthetic_weight": synthetic_weight,
                "selectivity_weight": selectivity_weight if use_off_target else 0.0,
                "admet_properties": selected_props,
                "admet_directions": admet_directions,
            })
            run = api.start_run({
                "config_id": config["id"],
                "seed": int(seed),
                "num_episodes": int(num_episodes),
                "max_steps": int(max_steps),
                "checkpoint_interval": int(checkpoint_interval),
            })
            st.session_state.active_run_id = run["id"]
            st.rerun()
        except ApiError as e:
            st.error(e.detail)

with tab_sweep:
    st.markdown(
        '<span class="metis-meta">Trains a small population of agents, one per point near your objective weights '
        'above, to discover a Pareto-optimal trade-off front instead of a single agent — a research-grade '
        'PGMORL-style sweep (GINEConv + PPO over the macro-edit catalog, optionally fused with a local LLM, '
        'GP-based prediction-guided scheduling). This is substantially more compute than a single training run: '
        'population_size × num_rounds × episodes_per_round total episodes.</span>',
        unsafe_allow_html=True,
    )
    sweep_config_name = st.text_input("Base config name", value="My sweep config", key="sweep_config_name")

    col1, col2, col3 = st.columns(3)
    with col1:
        population_size = st.number_input("Population size", min_value=2, max_value=32, value=6)
    with col2:
        concentration_alpha = st.number_input("Concentration (Dirichlet α)", min_value=0.1, value=20.0,
                                               help="Higher = population clusters tightly around your weight "
                                                    "sliders above; lower = broader spread.")
    with col3:
        sweep_seed = st.number_input("Seed", min_value=0, value=0, key="sweep_seed")

    col1, col2, col3 = st.columns(3)
    with col1:
        num_rounds = st.number_input("Rounds", min_value=1, value=5)
    with col2:
        episodes_per_round = st.number_input("Episodes per round", min_value=1, value=10)
    with col3:
        sweep_max_steps = st.number_input("Max steps per episode", min_value=1, value=40, key="sweep_max_steps")
    eval_episodes_per_round = st.number_input("Eval episodes per round", min_value=1, value=3,
                                               help="Episodes used to measure each member's objective vector "
                                                    "after training, without further updating its policy.")

    st.markdown('<span class="metis-meta">Edit-count mode (how many molecular edits per macro-action step)</span>',
                unsafe_allow_html=True)
    edit_count_mode = st.selectbox("Edit-count mode", ["fixed", "random", "learned"], key="edit_count_mode")
    fixed_edit_count, edit_count_range_min, edit_count_range_max, k_max = 1, None, None, None
    if edit_count_mode == "fixed":
        fixed_edit_count = st.number_input("Edits per step", min_value=1, value=1)
    elif edit_count_mode == "random":
        col1, col2 = st.columns(2)
        with col1:
            edit_count_range_min = st.number_input("Min edits per step", min_value=1, value=1)
        with col2:
            edit_count_range_max = st.number_input("Max edits per step", min_value=1, value=4)
    else:
        k_max = st.number_input("Max learnable edits per step (k_max)", min_value=1, value=4)

    use_sweep_llm = st.checkbox(
        "Fuse with a local LLM (cross-attention)", value=False,
        help="Adds a shared, LoRA-adapted local LLM's context (target protein, this member's weight vector) to "
             "every edit decision. Meaningfully more GPU memory and wall-clock per episode.",
    )
    sweep_llm_model_name = None
    llm_finetune_interval_steps = None
    if use_sweep_llm:
        sweep_llm_model_name = st.text_input("LLM model name", value="Qwen/Qwen2.5-3B-Instruct")
        llm_finetune_interval_steps = st.number_input(
            "Fine-tune the shared LLM adapter every N cumulative macro-steps", min_value=1, value=200,
            help="Periodically fine-tunes the shared LoRA adapter on cached per-step outcomes pooled across "
                 "the whole population, then resets the cache. Lower = adapts faster but more overhead.",
        )

    col1, col2 = st.columns(2)
    with col1:
        use_predictor = st.checkbox(
            "Use GP-based prediction-guided scheduling", value=True,
            help="Off = naive round-robin baseline (every member trained equally each round), useful only for "
                 "comparing against the real scheduler.",
        )
    with col2:
        concurrent = st.checkbox(
            "Train population members concurrently", value=False,
            help="Requires a GPU with enough headroom for multiple members' forward/backward passes "
                 "overlapping in flight.",
        )
    max_concurrent_members = None
    if concurrent:
        max_concurrent_members = st.number_input("Max concurrent members", min_value=1, value=3)

    if st.button("Run Pareto sweep", type="primary"):
        try:
            base_config = api.create_config({
                "name": sweep_config_name,
                "starting_molecule_id": starting_molecule_id,
                "target_protein_id": target_protein_id,
                "off_target_protein_id": off_target_protein_id,
                "admet_weight": admet_weight,
                "binding_weight": binding_weight,
                "synthetic_weight": synthetic_weight,
                "selectivity_weight": selectivity_weight if use_off_target else 0.0,
                "admet_properties": selected_props,
                "admet_directions": admet_directions,
            })
            sweep = api.start_pareto_sweep({
                "base_config_id": base_config["id"],
                "seed": int(sweep_seed),
                "population_size": int(population_size),
                "concentration_alpha": float(concentration_alpha),
                "num_rounds": int(num_rounds),
                "episodes_per_round": int(episodes_per_round),
                "eval_episodes_per_round": int(eval_episodes_per_round),
                "max_steps": int(sweep_max_steps),
                "edit_count_mode": edit_count_mode,
                "fixed_edit_count": int(fixed_edit_count) if edit_count_mode == "fixed" else 1,
                "edit_count_range_min": int(edit_count_range_min) if edit_count_range_min is not None else None,
                "edit_count_range_max": int(edit_count_range_max) if edit_count_range_max is not None else None,
                "k_max": int(k_max) if k_max is not None else None,
                "use_llm": use_sweep_llm,
                "llm_model_name": sweep_llm_model_name,
                "llm_finetune_interval_steps": (
                    int(llm_finetune_interval_steps) if llm_finetune_interval_steps is not None else None
                ),
                "use_predictor": use_predictor,
                "concurrent": concurrent,
                "max_concurrent_members": int(max_concurrent_members) if max_concurrent_members else None,
            })
            st.session_state.active_sweep_id = sweep["id"]
            st.rerun()
        except ApiError as e:
            st.error(e.detail)
