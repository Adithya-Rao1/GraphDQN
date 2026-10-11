import streamlit as st

from api_client import ApiError, get_client
from components.progress import render_run_progress
from components.llm_finetune_progress import render_llm_finetune_progress
from style import inject_theme

inject_theme()

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

api = get_client()

st.title("Session Library")
st.markdown(
    '<div class="metis-card"><p>Browse completed manual-edit sessions, inspect the score of every edit a '
    'chemist applied, and fine-tune the DQN agent or the shared LLM adapter directly from one or more '
    'selected sessions.</p></div>',
    unsafe_allow_html=True,
)

SELECTION_KEY = "session_library_selected"
if SELECTION_KEY not in st.session_state:
    st.session_state[SELECTION_KEY] = set()
selected = st.session_state[SELECTION_KEY]

show_all = st.checkbox("Show all sessions (including in-progress / failed)", value=False)

sessions = api.list_manual_sessions()
if not show_all:
    sessions = [s for s in sessions if s["status"] == "completed"]

if not sessions:
    st.info("No sessions yet — create one on the Manual Edit Studio page.")
    st.stop()

for session in sessions:
    summary = session.get("result_summary_json") or {}
    delta = summary.get("reward_delta")
    delta_str = f"{'+' if (delta or 0) >= 0 else ''}{delta:.3f}" if delta is not None else "-"
    label = f"{session.get('starting_molecule_label') or '?'} → {session.get('target_protein_name') or '?'}"

    with st.container(border=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f'<b>{label}</b> &mdash; {session["num_edits"]} edit(s), reward Δ {delta_str} '
                f'&mdash; <span class="metis-meta">{session["status"]}</span>',
                unsafe_allow_html=True,
            )
        with col2:
            checked = st.checkbox(
                "Select", value=session["id"] in selected, key=f"sel_session_{session['id']}",
                disabled=session["status"] != "completed",
            )
        if checked:
            selected.add(session["id"])
        else:
            selected.discard(session["id"])

        with st.expander("Per-edit scores"):
            outcomes = api.get_manual_session_outcomes(session["id"])
            if not outcomes:
                st.caption("No scored edits yet.")
            else:
                for o in outcomes:
                    st.markdown(
                        f'<div class="metis-meta">Step {o["step_index"]}: '
                        f'<b>{", ".join(o["applied_edit_ids"])}</b> '
                        f'(site: {o.get("site_key")}) &mdash; reward {o["reward"]:.3f} '
                        f'(ADMET {o.get("admet_score")}, binding {o.get("binding_uM")} uM, '
                        f'SA {o.get("sa_score")}, selectivity {o.get("selectivity")})'
                        f'{" — already used in a fine-tune" if o["consumed_by_finetune"] else ""}</div>'
                        f'<div class="metis-smiles">→ {o["resulting_smiles"]}</div>',
                        unsafe_allow_html=True,
                    )

st.caption(f"{len(selected)} session(s) selected")

st.divider()
st.subheader("Fine-tune DQN agent")
dqn_runs = [r for r in api.list_runs() if r["status"] == "completed" and r.get("final_checkpoint_path")]

active_dqn_run_id = st.session_state.get("session_library_active_dqn_run_id")
if active_dqn_run_id is not None:
    render_run_progress(api, active_dqn_run_id)
elif not dqn_runs:
    st.info("No completed DQN runs with a checkpoint yet — train one on the Optimize Molecules page first.")
else:
    run_options = {f"Run #{r['id']} (config {r['config_id']})": r["id"] for r in dqn_runs}
    run_choice = st.selectbox("Base DQN run to resume", list(run_options))
    num_extra_episodes = st.number_input("Extra episodes", min_value=1, value=50, key="dqn_from_sessions_episodes")
    if st.button("Fine-tune DQN agent from selected sessions", type="primary", disabled=not selected):
        try:
            new_run = api.finetune_run_from_sessions(
                run_options[run_choice],
                {"manual_edit_session_ids": sorted(selected), "num_extra_episodes": num_extra_episodes},
            )
            st.session_state.session_library_active_dqn_run_id = new_run["id"]
            st.rerun()
        except ApiError as e:
            st.error(e.detail)

st.divider()
st.subheader("Fine-tune LLM adapter")
st.caption(
    "Fine-tunes the shared PGMORL LLM LoRA adapter (same pooled-outcome cycle a Pareto sweep triggers "
    "automatically) using the selected sessions' edits. GPU-heavy — expect this to be slow or impractical "
    "without a dedicated GPU box."
)

active_llm_run_id = st.session_state.get("session_library_active_llm_run_id")
if active_llm_run_id is not None:
    render_llm_finetune_progress(api, active_llm_run_id)
else:
    llm_model_name = st.text_input(
        "LLM model name", value="Qwen/Qwen2.5-7B-Instruct", key="llm_from_sessions_model_name",
    )
    warm_start = st.checkbox("Warm-start from the latest saved adapter for this model", value=True)
    if st.button("Fine-tune LLM adapter from selected sessions", type="primary", disabled=not selected):
        try:
            new_run = api.create_llm_finetune_run({
                "manual_edit_session_ids": sorted(selected),
                "llm_model_name": llm_model_name or None,
                "warm_start": warm_start,
            })
            st.session_state.session_library_active_llm_run_id = new_run["id"]
            st.rerun()
        except ApiError as e:
            st.error(e.detail)
