import streamlit as st

from components.mol3d import render_3d_viewer


def _fmt(value, digits=3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_trajectory_slideshow(api, batch_id: int, trajectories: list) -> None:
    state_key = f"traj_state_{batch_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {"traj_idx": 0, "step_idx": 0, "selections": set()}
    state = st.session_state[state_key]

    if not trajectories:
        st.info("No step-by-step data for this batch.")
        return

    num_trajectories = len(trajectories)
    state["traj_idx"] = min(state["traj_idx"], num_trajectories - 1)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Rollout", key=f"prev_traj_{batch_id}", disabled=state["traj_idx"] == 0):
            state["traj_idx"] -= 1
            state["step_idx"] = 0
            st.rerun()
    with col3:
        if st.button("Rollout →", key=f"next_traj_{batch_id}", disabled=state["traj_idx"] >= num_trajectories - 1):
            state["traj_idx"] += 1
            state["step_idx"] = 0
            st.rerun()
    with col2:
        st.markdown(
            f'<div style="text-align:center" class="gdqn-meta">Rollout {state["traj_idx"] + 1} of {num_trajectories}</div>',
            unsafe_allow_html=True,
        )

    steps = trajectories[state["traj_idx"]]["steps"]
    num_steps = len(steps)
    state["step_idx"] = min(state["step_idx"], num_steps - 1)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Step", key=f"prev_step_{batch_id}", disabled=state["step_idx"] == 0):
            state["step_idx"] -= 1
            st.rerun()
    with col3:
        if st.button("Step →", key=f"next_step_{batch_id}", disabled=state["step_idx"] >= num_steps - 1):
            state["step_idx"] += 1
            st.rerun()
    with col2:
        st.markdown(
            f'<div style="text-align:center" class="gdqn-meta">Step {state["step_idx"] + 1} of {num_steps} '
            f'(0 = starting molecule)</div>',
            unsafe_allow_html=True,
        )

    step = steps[state["step_idx"]]
    selectivity_row = (
        f'<tr><td>Selectivity</td><td>{_fmt(step.get("selectivity"))}</td></tr>'
        if step.get("selectivity") is not None else ""
    )

    with st.container(border=True):
        render_3d_viewer(
            step.get("molblock_3d"), height=320,
            key=f"viewer_traj_{batch_id}_{state['traj_idx']}_{state['step_idx']}",
        )
        st.markdown(f"""
          <div class="gdqn-smiles">{step['smiles']}</div>
          <table class="gdqn-metrics">
            <tr><td>Reward</td><td>{_fmt(step['reward'])}</td></tr>
            <tr><td>ADMET score</td><td>{_fmt(step.get('admet_score'))}</td></tr>
            <tr><td>Binding (uM)</td><td>{_fmt(step.get('binding_uM'))}</td></tr>
            <tr><td>SA score</td><td>{_fmt(step.get('sa_score'))}</td></tr>
            {selectivity_row}
          </table>
        """, unsafe_allow_html=True)

        if step.get("applied_edits"):
            mode_label = f" ({step.get('edit_count_mode')} mode)" if step.get("edit_count_mode") else ""
            with st.expander(f"Macro-edit chain{mode_label}: {step.get('k_edits_used', len(step['applied_edits']))} edit(s) applied"):
                for i, edit in enumerate(step["applied_edits"]):
                    st.markdown(
                        f'<div class="gdqn-meta">{i + 1}. <b>{edit["edit_id"]}</b> '
                        f'({edit["category"]}) — {edit["description"]}</div>'
                        f'<div class="gdqn-smiles">→ {edit["resulting_smiles"]}</div>',
                        unsafe_allow_html=True,
                    )

        sel_key = (state["traj_idx"], step["step_index"])
        checked = st.checkbox(
            "Select this step as a candidate",
            value=sel_key in state["selections"],
            key=f"select_step_{batch_id}_{state['traj_idx']}_{step['step_index']}",
        )
        if checked:
            state["selections"].add(sel_key)
        else:
            state["selections"].discard(sel_key)

    st.caption(f"{len(state['selections'])} step(s) selected across all rollouts in this batch")
    if st.button(f"Save {len(state['selections'])} selected step(s) as candidates",
                 disabled=not state["selections"], type="primary", key=f"promote_{batch_id}"):
        selections_payload = [{"trajectory_index": t, "step_index": s} for t, s in state["selections"]]
        created = api.promote_steps(batch_id, selections_payload)
        state["selections"] = set()
        st.success(f"Saved {len(created)} candidate(s) — see them below.")
        st.rerun()
