import streamlit as st

from api_client import ApiError, get_client
from components.manual_edit_progress import render_manual_session_progress
from components.mol3d import render_3d_viewer
from style import inject_theme

inject_theme()

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

api = get_client()

st.title("Manual Edit Studio")
st.markdown(
    '<div class="metis-card"><p>Manually apply the same catalog of molecular edits the RL agent uses.</p></div>',
    unsafe_allow_html=True,
)

session_id = st.session_state.get("active_manual_session_id")

if session_id is None:
    st.subheader("Start a new session")
    molecules = api.list_molecules()
    with st.expander("Add a new molecule"):
        new_smiles = st.text_input("SMILES", key="manual_new_mol_smiles")
        if new_smiles:
            preview = api.preview_molecule(new_smiles)
            if preview["valid"]:
                st.image(f"data:image/png;base64,{preview['image_b64']}", width=150)
                if st.button("Save this molecule", key="manual_save_mol"):
                    api.create_molecule(new_smiles)
                    st.rerun()
            else:
                st.error(preview.get("error"))

    if not molecules:
        st.info("Add a starting molecule above to continue.")
        st.stop()

    mol_options = {f"{m['label'] or m['canonical_smiles']} (#{m['id']})": m["id"] for m in molecules}
    mol_choice = st.selectbox("Starting molecule", list(mol_options))
    starting_molecule_id = mol_options[mol_choice]

    proteins = api.list_proteins()
    if not proteins:
        st.info("Add a target protein on the Optimize Molecules page first.")
        st.stop()

    prot_options = {f"{p['name']} (#{p['id']})": p["id"] for p in proteins}
    target_choice = st.selectbox("Target protein", list(prot_options))
    target_protein_id = prot_options[target_choice]

    use_off_target = st.checkbox("Also track selectivity against an off-target protein")
    off_target_protein_id = None
    if use_off_target:
        off_choice = st.selectbox("Off-target protein", list(prot_options), key="manual_off_target_select")
        off_target_protein_id = prot_options[off_choice]

    if st.button("Start session", type="primary"):
        session = api.create_manual_session(starting_molecule_id, target_protein_id, off_target_protein_id)
        st.session_state.active_manual_session_id = session["id"]
        st.rerun()
    st.stop()

session = api.get_manual_session(session_id)

if session["status"] in ("completed", "failed"):
    render_manual_session_progress(api, session_id)
    if st.button("Start a new session"):
        st.session_state.active_manual_session_id = None
        st.rerun()
    st.stop()

if session["status"] == "scoring":
    render_manual_session_progress(api, session_id)
    st.stop()

current = api.get_manual_session_current(session_id)

col1, col2 = st.columns(2)
with col1:
    if current.get("image_b64"):
        st.image(f"data:image/png;base64,{current['image_b64']}", use_container_width=True)
with col2:
    render_3d_viewer(current.get("molblock_3d"), height=280, key=f"manual_viewer_{session_id}")

st.markdown(f'<div class="metis-smiles">{current["smiles"]}</div>', unsafe_allow_html=True)
st.caption(f"{len(session['pending_edits'])} edit(s) applied so far")

catalog = api.get_manual_edit_catalog(session_id)
categories = sorted({op["category"] for op in catalog})
category = st.selectbox("Edit category", categories)
category_ops = [op for op in catalog if op["category"] == category]
op_labels = {op["description"]: op["id"] for op in category_ops}
op_choice = st.selectbox("Edit", list(op_labels))

col1, col2 = st.columns(2)
with col1:
    if st.button("Apply edit", type="primary"):
        try:
            api.apply_manual_edit(session_id, op_labels[op_choice])
            st.rerun()
        except ApiError as e:
            st.error(e.detail)
with col2:
    if st.button("Undo last edit", disabled=not session["pending_edits"]):
        try:
            api.undo_manual_edit(session_id)
            st.rerun()
        except ApiError as e:
            st.error(e.detail)

st.divider()
if st.button("Finish session", type="primary", disabled=not session["pending_edits"]):
    try:
        api.finish_manual_session(session_id)
        st.rerun()
    except ApiError as e:
        st.error(e.detail)
if not session["pending_edits"]:
    st.markdown('<span class="metis-meta">Apply at least one edit before finishing.</span>',
                unsafe_allow_html=True)
