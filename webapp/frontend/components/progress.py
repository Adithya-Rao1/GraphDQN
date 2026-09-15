import time

import streamlit as st

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def render_run_progress(api, run_id: int) -> dict:
    """Fetches and renders a training run's status/progress.

    Polls via a plain fetch + sleep + st.rerun loop (no server push exists) --
    if the run is still in progress, this function never returns normally
    (the rerun replaces the whole script). Returns the run dict once terminal.
    """
    run = api.get_run(run_id)
    status = run["status"]

    if status in ("pending", "running"):
        total = max(run["progress_total"], 1)
        current = run["progress_current"]
        st.progress(min(current / total, 1.0), text=f"{status} — episode {current}/{total}")
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
