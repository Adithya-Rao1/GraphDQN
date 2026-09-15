import time
from datetime import datetime, timezone

import streamlit as st

from api_client import ApiError

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "killed"}


def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def render_run_progress(api, run_id: int, allow_kill: bool = True) -> dict:
    try:
        run = api.get_run(run_id)
    except ApiError as e:
        if e.status_code == 404:
            st.warning("Run was killed and discarded — nothing was saved.")
            return {"status": "killed"}
        raise
    status = run["status"]

    if status in ("pending", "running"):
        total = max(run["progress_total"], 1)
        current = run["progress_current"]
        progress_text = f"{status} — episode {current}/{total}"

        started_at = _parse_dt(run.get("started_at"))
        if started_at and current > 0:
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            remaining = (elapsed / current) * (total - current)
            progress_text += f" · ~{_format_duration(remaining)} remaining"

        st.progress(min(current / total, 1.0), text=progress_text)

        if allow_kill:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel (keep partial result)", key=f"cancel_{run_id}"):
                    api.cancel_run(run_id, discard=False)
                    st.info("Cancelling — this will finish the current episode, then stop.")
            with col2:
                if st.button("Kill (don't save)", key=f"kill_{run_id}", type="primary"):
                    api.cancel_run(run_id, discard=True)
                    st.info("Killing and discarding — this run will not be kept.")

        time.sleep(2)
        st.rerun()
    elif status == "completed":
        st.success("Run completed")
        st.json(run.get("result_summary_json") or {})
    elif status == "cancelled":
        st.warning("Run was cancelled")
    else:
        st.error(f"Run failed: {run.get('error_message')}")

    return run


def render_batch_progress(api, batch_id: int) -> dict:
    batch = api.get_generation_batch(batch_id)
    status = batch["status"]

    if status in ("pending", "running"):
        st.progress(0.5, text=f"generating candidates ({status})...")
        time.sleep(2)
        st.rerun()
    elif status == "completed":
        st.success("Generation completed")
    elif status == "failed":
        st.error(f"Generation failed: {batch.get('error_message')}")

    return batch
