import pandas as pd
import streamlit as st

from api_client import ApiError, get_client
from style import inject_theme

inject_theme()

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

api = get_client()

st.title("My Database")

tab_mol, tab_prot, tab_cfg, tab_runs, tab_candidates = st.tabs(
    ["Molecules", "Proteins", "Configs", "Runs", "Candidates"]
)

with tab_mol:
    with st.form("add_molecule"):
        smiles = st.text_input("SMILES")
        label = st.text_input("Label (optional)")
        submitted = st.form_submit_button("Add molecule")
    if submitted and smiles:
        try:
            preview = api.preview_molecule(smiles)
            if not preview["valid"]:
                st.error(f"Invalid SMILES: {preview.get('error')}")
            else:
                api.create_molecule(smiles, label or None)
                st.success("Molecule added")
                st.rerun()
        except ApiError as e:
            st.error(e.detail)

    molecules = api.list_molecules()
    if molecules:
        df = pd.DataFrame(molecules)[["id", "smiles", "canonical_smiles", "label", "created_at"]]
        st.dataframe(df, use_container_width=True)
        del_id = st.selectbox("Delete molecule id", [None] + [m["id"] for m in molecules])
        if del_id and st.button("Delete", key="del_mol"):
            try:
                api.delete_molecule(del_id)
                st.rerun()
            except ApiError as e:
                st.error(e.detail)
    else:
        st.markdown('<span class="gdqn-meta">No molecules yet.</span>', unsafe_allow_html=True)

with tab_prot:
    with st.form("add_protein"):
        name = st.text_input("Name")
        sequence = st.text_area("Sequence (raw amino-acid sequence, or paste FASTA)")
        submitted = st.form_submit_button("Add protein")
    if submitted and name and sequence:
        clean_seq = "\n".join(
            line for line in sequence.strip().splitlines() if not line.startswith(">")
        ).replace("\n", "").strip()
        try:
            api.create_protein(name, clean_seq)
            st.success("Protein added")
            st.rerun()
        except ApiError as e:
            st.error(e.detail)

    proteins = api.list_proteins()
    if proteins:
        df = pd.DataFrame(proteins)[["id", "name", "created_at"]]
        st.dataframe(df, use_container_width=True)
        del_id = st.selectbox("Delete protein id", [None] + [p["id"] for p in proteins])
        if del_id and st.button("Delete", key="del_prot"):
            try:
                api.delete_protein(del_id)
                st.rerun()
            except ApiError as e:
                st.error(e.detail)
    else:
        st.markdown('<span class="gdqn-meta">No proteins yet.</span>', unsafe_allow_html=True)

with tab_cfg:
    configs = api.list_configs()
    if configs:
        df = pd.DataFrame(configs)[["id", "name", "admet_weight", "binding_weight",
                                     "synthetic_weight", "selectivity_weight", "created_at"]]
        st.dataframe(df, use_container_width=True)
        del_id = st.selectbox("Delete config id", [None] + [c["id"] for c in configs])
        if del_id and st.button("Delete", key="del_cfg"):
            try:
                api.delete_config(del_id)
                st.rerun()
            except ApiError as e:
                st.error(e.detail)
    else:
        st.markdown('<span class="gdqn-meta">No configs yet — create one on the Optimize Molecules page.</span>',
                    unsafe_allow_html=True)

with tab_runs:
    runs = api.list_runs()
    if runs:
        df = pd.DataFrame(runs)[["id", "config_id", "status", "is_finetune", "parent_run_id",
                                  "seed", "num_episodes", "progress_current", "progress_total", "created_at"]]
        st.dataframe(df, use_container_width=True)
    else:
        st.markdown('<span class="gdqn-meta">No training runs yet.</span>', unsafe_allow_html=True)

with tab_candidates:
    st.markdown(
        '<span class="gdqn-meta">Every candidate ever generated, with full provenance: which molecule it '
        'started from, which config/run produced it, which target (and off-target) protein it was optimized '
        'against, and which fine-tune run(s) it fed back into.</span>',
        unsafe_allow_html=True,
    )
    candidates = api.list_candidates()
    if candidates:
        df = pd.DataFrame(candidates)
        df["used_in_finetune_runs"] = df["used_in_finetune_runs"].apply(
            lambda rows: ", ".join(f"#{r['training_run_id']}" for r in rows) if rows else "-"
        )
        cols = [
            "id", "smiles", "target_protein_name", "off_target_protein_name", "starting_smiles",
            "config_name", "training_run_id", "generation_batch_id", "reward", "admet_score",
            "binding_uM", "sa_score", "selectivity", "user_rating", "used_in_finetune_runs", "created_at",
        ]
        st.dataframe(df[cols], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"{len(df)} candidate(s) total")
        with col2:
            scored = df["user_rating"].notna().sum()
            st.caption(f"{scored} scored")
    else:
        st.markdown('<span class="gdqn-meta">No candidates generated yet — visit the Candidates page after '
                    'training an agent.</span>', unsafe_allow_html=True)
