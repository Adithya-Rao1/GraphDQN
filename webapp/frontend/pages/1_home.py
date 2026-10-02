import streamlit as st

from style import inject_theme

inject_theme()

st.title("Metis")
st.markdown(
    """
    <div class="metis-card">
      <p>Metis is an AI drug-discovery and computational chemistry platform for molecular
      optimization, generation, dynamics, and property prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("token") is None:
    st.info("Sign up or log in to get started.")
else:
    st.success(f"Signed in as {st.session_state.get('user_email')}")
    st.page_link("pages/3_optimize.py", label="Go to Optimize Molecules →")
