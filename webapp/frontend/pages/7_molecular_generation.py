import streamlit as st

from components.coming_soon import render_coming_soon
from style import inject_theme

inject_theme()

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

st.title("Molecular Generation")
render_coming_soon(
    "🧬", "Molecular Generation",
    "Generate novel drug-like scaffolds from scratch using generative models.",
)
