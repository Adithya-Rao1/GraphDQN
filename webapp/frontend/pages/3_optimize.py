import streamlit as st

from api_client import ApiError, get_client
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
    if st.button("Cancel run"):
        api.cancel_run(st.session_state.active_run_id)
    run = render_run_progress(api, st.session_state.active_run_id)
    if run["status"] in ("completed", "failed", "cancelled"):
        if st.button("Start another run"):
            st.session_state.active_run_id = None
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
            st.error(f"Invalid SMILES: {preview.get('error')}")

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
    st.markdown('<span class="gdqn-meta">Direction for each selected property:</span>', unsafe_allow_html=True)
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
st.markdown(
    '<span class="gdqn-meta">Tip: use ~10-20 episodes for a fast smoke-test run before committing to a full '
    'training run (hundreds of episodes, each involving real ADMET/binding-affinity model inference, can take '
    'a while).</span>',
    unsafe_allow_html=True,
)
config_name = st.text_input("Config name", value="My optimization config")
col1, col2, col3 = st.columns(3)
with col1:
    num_episodes = st.number_input("Episodes", min_value=1, value=20)
with col2:
    max_steps = st.number_input("Max steps per episode", min_value=1, value=40)
with col3:
    seed = st.number_input("Seed", min_value=0, value=0)
checkpoint_interval = st.number_input("Checkpoint interval", min_value=1, value=10)

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
