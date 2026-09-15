import streamlit as st

from api_client import ApiError, get_client
from components.mol_card import render_candidate_card
from components.progress import render_batch_progress, render_run_progress
from style import inject_theme

inject_theme()

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

api = get_client()

st.title("Candidates")

if st.session_state.get("active_finetune_run_id"):
    st.subheader("Fine-tuning in progress")
    run = render_run_progress(api, st.session_state.active_finetune_run_id)
    if run["status"] in ("completed", "failed", "cancelled"):
        if st.button("Back to candidates"):
            st.session_state.active_finetune_run_id = None
            st.rerun()
    st.stop()

runs = [r for r in api.list_runs() if r["status"] == "completed"]
if not runs:
    st.info("Train an agent on the Optimize Molecules page first.")
    st.stop()

run_options = {f"Run #{r['id']} (config {r['config_id']}, {'finetune' if r['is_finetune'] else 'base'})": r["id"]
               for r in runs}
run_choice = st.selectbox("Trained run", list(run_options))
selected_run_id = run_options[run_choice]

if st.session_state.get("active_batch_id"):
    batch = render_batch_progress(api, st.session_state.active_batch_id)
    if batch["status"] in ("completed", "failed"):
        if st.button("Done"):
            st.session_state.active_batch_id = None
            st.rerun()
    st.stop()

with st.expander("Generate new candidates", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        num_candidates = st.number_input("Number of candidates", min_value=1, max_value=100, value=10)
    with col2:
        sampling_strategy = st.selectbox("Sampling", ["boltzmann", "greedy"])
    with col3:
        temperature = st.number_input("Temperature", min_value=0.01, value=1.0,
                                       disabled=sampling_strategy != "boltzmann")
    if st.button("Generate"):
        try:
            batch = api.generate_candidates(selected_run_id, {
                "num_candidates": int(num_candidates),
                "sampling_strategy": sampling_strategy,
                "temperature": float(temperature),
            })
            st.session_state.active_batch_id = batch["id"]
            st.rerun()
        except ApiError as e:
            st.error(e.detail)

candidates = api.list_candidates(training_run_id=selected_run_id)

st.subheader("Candidates")
if not candidates:
    st.markdown('<span class="gdqn-meta">No candidates generated yet.</span>', unsafe_allow_html=True)
    st.stop()

selected_ids = []
for candidate in candidates:
    selected = render_candidate_card(candidate, api, allow_scoring=True, allow_select=True)
    if selected:
        selected_ids.append(candidate["id"])

st.divider()
st.subheader("Fine-tune agent on scored candidates")
col1, col2 = st.columns(2)
with col1:
    alpha = st.slider("Blend weight (alpha) — how much to trust your ratings vs. the computed reward",
                       0.0, 1.0, 0.3, 0.05)
with col2:
    num_extra_episodes = st.number_input("Extra training episodes", min_value=1, value=20)

if st.button("Fine-tune agent", type="primary", disabled=not selected_ids):
    try:
        new_run = api.finetune_run(selected_run_id, {
            "candidate_ids": selected_ids,
            "alpha": float(alpha),
            "num_extra_episodes": int(num_extra_episodes),
        })
        st.session_state.active_finetune_run_id = new_run["id"]
        st.rerun()
    except ApiError as e:
        st.error(e.detail)
if not selected_ids:
    st.markdown('<span class="gdqn-meta">Score and select at least one candidate above to fine-tune.</span>',
                unsafe_allow_html=True)
