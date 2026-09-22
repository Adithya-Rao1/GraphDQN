import time

import pandas as pd
import streamlit as st

from api_client import ApiError

OBJECTIVE_LABELS = ["ADMET", "Binding", "Synth.", "Selectivity"]


def _members_dataframe(members: list) -> pd.DataFrame:
    rows = []
    for m in members:
        row = {"Member": m["member_id"]}
        for label, w in zip(OBJECTIVE_LABELS, m["weight_vector"]):
            row[f"{label} weight"] = round(w, 3)
        for label, v in zip(OBJECTIVE_LABELS, m["objective_vector"]):
            row[f"{label} score"] = round(v, 3)
        row["Episodes trained"] = m["total_training_steps"]
        if "non_dominated" in m and m["non_dominated"] is not None:
            row["Pareto-optimal"] = "✓" if m["non_dominated"] else ""
        rows.append(row)
    return pd.DataFrame(rows)


def render_pareto_sweep_progress(api, sweep_id: int, allow_kill: bool = True) -> dict:
    try:
        sweep = api.get_pareto_sweep(sweep_id)
    except ApiError as e:
        if e.status_code == 404:
            st.warning("Sweep was killed and discarded — nothing was saved.")
            return {"status": "killed"}
        raise
    status = sweep["status"]

    members = sweep.get("members_snapshot_json") or []

    if status in ("pending", "running"):
        total = max(sweep["progress_total_rounds"], 1)
        current = sweep["progress_current_round"]
        st.progress(min(current / total, 1.0), text=f"{status} — round {current}/{total}")

        if members:
            st.markdown('<span class="metis-meta">Population members (live)</span>', unsafe_allow_html=True)
            st.dataframe(_members_dataframe(members), use_container_width=True)
        else:
            st.markdown('<span class="metis-meta">Initializing population…</span>', unsafe_allow_html=True)

        if allow_kill:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel (keep partial result)", key=f"cancel_sweep_{sweep_id}"):
                    api.cancel_pareto_sweep(sweep_id, discard=False)
                    st.info("Cancelling — this will finish the current round, then stop.")
            with col2:
                if st.button("Kill (don't save)", key=f"kill_sweep_{sweep_id}", type="primary"):
                    api.cancel_pareto_sweep(sweep_id, discard=True)
                    st.info("Killing and discarding — this sweep will not be kept.")

        time.sleep(2)
        st.rerun()
    elif status == "completed":
        st.success("Pareto sweep completed")
        result = sweep.get("result_summary_json") or {}
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Non-dominated members", f"{result.get('num_non_dominated', '?')}/{sweep['population_size']}")
        with col2:
            hv = result.get("final_front_hypervolume")
            st.metric("Front hypervolume", f"{hv:.4f}" if hv is not None else "—")
        with col3:
            wc = result.get("wall_clock_seconds")
            st.metric("Wall clock", f"{wc / 60:.1f} min" if wc else "—")
        if members:
            st.markdown('<span class="metis-meta">Each row is one trade-off point discovered by the sweep — '
                        'its config is saved to your database (tagged as sweep-generated) and can be used to '
                        'generate candidates like any other config (see the Candidates page).</span>',
                        unsafe_allow_html=True)
            st.dataframe(_members_dataframe(members), use_container_width=True)

        if sweep.get("use_llm"):
            with st.expander("LLM adapter fine-tune history"):
                try:
                    checkpoints = api.list_pareto_adapter_checkpoints(sweep_id)
                except ApiError as e:
                    checkpoints = []
                    st.error(e.detail)
                if checkpoints:
                    st.dataframe(pd.DataFrame([
                        {
                            "Adapter checkpoint": c["id"],
                            "Base model": c["base_model_name"],
                            "Training examples": c["num_training_examples"],
                            "Created": c["created_at"],
                        }
                        for c in checkpoints
                    ]), use_container_width=True)
                else:
                    st.markdown(
                        '<span class="metis-meta">No fine-tune cycles have run yet for this sweep '
                        '(needs cumulative_steps_since_finetune to cross llm_finetune_interval_steps).</span>',
                        unsafe_allow_html=True,
                    )
    elif status == "cancelled":
        st.warning("Sweep was cancelled")
    else:
        st.error(f"Sweep failed: {sweep.get('error_message')}")

    return sweep
