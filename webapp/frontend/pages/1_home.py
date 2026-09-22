import streamlit as st

from style import inject_theme

inject_theme()

st.title("Metis")
st.markdown(
    """
    <div class="metis-card">
      <p>Metis is an AI drug-discovery platform bringing together molecular
      optimization, generation, dynamics, and property prediction in one
      place.</p>
      <p class="metis-meta">
        Optimize Molecules (below) is fully built today &mdash; it optimizes a
        molecule for ADMET properties, binding affinity, synthetic
        accessibility, and selectivity using a reinforcement-learning agent
        trained specifically on your target: upload a starting molecule and
        (optionally) a target protein &middot; choose which properties to
        optimize and in which direction &middot; train your own agent &middot;
        generate and score optimized candidates &middot; fine-tune the agent
        on what you liked. The other tools in the sidebar are on the way.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("token") is None:
    st.info("Sign up or log in to get started.")
else:
    st.success(f"Signed in as {st.session_state.get('user_email')}")
    st.page_link("pages/3_optimize.py", label="Go to Optimize Molecules →")
