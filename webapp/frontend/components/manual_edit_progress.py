import time

import streamlit as st


def _fmt(value, digits=3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_manual_session_progress(api, session_id: int) -> dict:
    session = api.get_manual_session(session_id)
    status = session["status"]

    if status in ("in_progress", "scoring"):
        st.progress(0.5, text=f"scoring session ({status})...")
        time.sleep(2)
        st.rerun()
    elif status == "completed":
        st.success("Session scored")
        summary = session.get("result_summary_json") or {}
        st.markdown(f"""
          <table class="metis-metrics">
            <tr><td>Reward</td><td>{_fmt(summary.get('reward_before'))} &rarr; {_fmt(summary.get('reward_after'))}
                ({'+' if (summary.get('reward_delta') or 0) >= 0 else ''}{_fmt(summary.get('reward_delta'))})</td></tr>
            <tr><td>ADMET score</td><td>{_fmt(summary.get('admet_before'))} &rarr; {_fmt(summary.get('admet_after'))}</td></tr>
            <tr><td>Binding (uM)</td><td>{_fmt(summary.get('binding_uM_before'))} &rarr; {_fmt(summary.get('binding_uM_after'))}</td></tr>
            <tr><td>SA score</td><td>{_fmt(summary.get('sa_score_before'))} &rarr; {_fmt(summary.get('sa_score_after'))}</td></tr>
            <tr><td>Selectivity</td><td>{_fmt(summary.get('selectivity_before'))} &rarr; {_fmt(summary.get('selectivity_after'))}</td></tr>
          </table>
        """, unsafe_allow_html=True)
    elif status == "failed":
        st.error(f"Scoring failed: {session.get('error_message')}")

    return session
