import time

import streamlit as st


def render_llm_finetune_progress(api, run_id: int) -> dict:
    run = api.get_llm_finetune_run(run_id)
    status = run["status"]

    if status in ("pending", "running"):
        st.progress(0.5, text=f"fine-tuning LLM adapter ({status})...")
        time.sleep(2)
        st.rerun()
    elif status == "completed":
        st.success("LLM adapter fine-tune cycle completed")
        st.json(run.get("result_summary_json") or {})
    else:
        st.error(f"LLM fine-tune failed: {run.get('error_message')}")

    return run
