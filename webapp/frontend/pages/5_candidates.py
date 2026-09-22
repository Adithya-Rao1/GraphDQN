import streamlit as st

from api_client import ApiError, get_client
from components.mol_card import render_candidate_card
from components.progress import render_batch_progress, render_run_progress
from components.trajectory_slideshow import render_trajectory_slideshow
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
    if run["status"] in ("completed", "failed", "cancelled", "killed"):
        if st.button("Back to candidates"):
            st.session_state.active_finetune_run_id = None
            st.rerun()
    st.stop()

source = st.radio("Source", ["Trained run (DQN)", "Pareto sweep member (PGMORL)"], horizontal=True)


def _render_dqn_flow():
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
            if batch["status"] == "completed":
                st.session_state[f"last_batch_{selected_run_id}"] = batch["id"]
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

    batches = api.list_generation_batches(training_run_id=selected_run_id)
    completed_batches = [b for b in batches if b["status"] == "completed"]
    if completed_batches:
        show_steps = st.checkbox("See the molecule at each step of generation (not just the final one)")
        if show_steps:
            batch_options = {
                f"Batch #{b['id']} ({b['num_requested']} rollouts, {b['sampling_strategy']}, {b['created_at']})": b["id"]
                for b in completed_batches
            }
            default_batch_id = st.session_state.get(f"last_batch_{selected_run_id}", completed_batches[0]["id"])
            default_label = next((label for label, bid in batch_options.items() if bid == default_batch_id),
                                  list(batch_options)[0])
            batch_label = st.selectbox("Batch to browse", list(batch_options),
                                        index=list(batch_options).index(default_label))
            traj_batch_id = batch_options[batch_label]
            trajectories = api.get_trajectories(traj_batch_id)
            render_trajectory_slideshow(api, traj_batch_id, trajectories)
            st.divider()

    candidates = api.list_candidates(training_run_id=selected_run_id)

    st.subheader("Candidates")
    if not candidates:
        st.markdown('<span class="metis-meta">No candidates generated yet.</span>', unsafe_allow_html=True)
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
        st.markdown('<span class="metis-meta">Score and select at least one candidate above to fine-tune.</span>',
                    unsafe_allow_html=True)


def _render_pgmorl_flow():
    sweeps = [s for s in api.list_pareto_sweeps() if s["status"] == "completed"]
    if not sweeps:
        st.info("Run a Pareto sweep to completion on the Optimize Molecules page first.")
        st.stop()

    sweep_options = {f"Sweep #{s['id']} (base config {s['base_config_id']}, "
                      f"{s['population_size']} members)": s["id"] for s in sweeps}
    sweep_choice = st.selectbox("Pareto sweep", list(sweep_options), key="pgmorl_sweep_select")
    selected_sweep_id = sweep_options[sweep_choice]

    sweep = api.get_pareto_sweep(selected_sweep_id)
    members = (sweep.get("result_summary_json") or {}).get("members", [])
    if not members:
        st.warning("This sweep has no population members recorded.")
        st.stop()

    def _member_label(m):
        nd = " ✓ Pareto-optimal" if m.get("non_dominated") else ""
        weights = ", ".join(f"{v:.2f}" for v in m["weight_vector"])
        return f"{m['member_id']} — weights [{weights}]{nd}"

    member_options = {_member_label(m): m["config_id"] for m in members if m.get("config_id") is not None}
    member_choice = st.selectbox("Population member", list(member_options), key="pgmorl_member_select")
    selected_member_config_id = member_options[member_choice]

    if st.session_state.get("active_batch_id"):
        batch = render_batch_progress(api, st.session_state.active_batch_id)
        if batch["status"] in ("completed", "failed"):
            if batch["status"] == "completed":
                st.session_state[f"last_batch_{selected_member_config_id}"] = batch["id"]
            if st.button("Done", key="pgmorl_batch_done"):
                st.session_state.active_batch_id = None
                st.rerun()
        st.stop()

    with st.expander("Generate new candidates", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_candidates = st.number_input("Number of candidates", min_value=1, max_value=100, value=10,
                                              key="pgmorl_num_candidates")
        with col2:
            sampling_strategy = st.selectbox("Sampling", ["boltzmann", "greedy"], key="pgmorl_sampling")
        with col3:
            temperature = st.number_input("Temperature", min_value=0.01, value=1.0,
                                           disabled=sampling_strategy != "boltzmann", key="pgmorl_temperature")
        if st.button("Generate", key="pgmorl_generate_btn"):
            try:
                batch = api.generate_sweep_candidates(selected_sweep_id, selected_member_config_id, {
                    "num_candidates": int(num_candidates),
                    "sampling_strategy": sampling_strategy,
                    "temperature": float(temperature),
                })
                st.session_state.active_batch_id = batch["id"]
                st.rerun()
            except ApiError as e:
                st.error(e.detail)

    batches = api.list_generation_batches(pareto_sweep_id=selected_sweep_id,
                                           population_member_config_id=selected_member_config_id)
    completed_batches = [b for b in batches if b["status"] == "completed"]
    if completed_batches:
        show_steps = st.checkbox("See the molecule at each step of generation (not just the final one)",
                                  key="pgmorl_show_steps")
        if show_steps:
            batch_options = {
                f"Batch #{b['id']} ({b['num_requested']} rollouts, {b['sampling_strategy']}, {b['created_at']})": b["id"]
                for b in completed_batches
            }
            default_batch_id = st.session_state.get(f"last_batch_{selected_member_config_id}",
                                                      completed_batches[0]["id"])
            default_label = next((label for label, bid in batch_options.items() if bid == default_batch_id),
                                  list(batch_options)[0])
            batch_label = st.selectbox("Batch to browse", list(batch_options),
                                        index=list(batch_options).index(default_label), key="pgmorl_batch_select")
            traj_batch_id = batch_options[batch_label]
            trajectories = api.get_trajectories(traj_batch_id)
            render_trajectory_slideshow(api, traj_batch_id, trajectories)
            st.divider()

    candidates = api.list_candidates(config_id=selected_member_config_id)

    st.subheader("Candidates")
    if not candidates:
        st.markdown('<span class="metis-meta">No candidates generated yet for this member.</span>',
                    unsafe_allow_html=True)
        st.stop()

    for candidate in candidates:
        render_candidate_card(candidate, api, allow_scoring=True, allow_select=False)


if source == "Trained run (DQN)":
    _render_dqn_flow()
else:
    _render_pgmorl_flow()
