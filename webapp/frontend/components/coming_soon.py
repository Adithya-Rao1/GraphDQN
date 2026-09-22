import streamlit as st


def render_coming_soon(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="metis-card">
          <h3>{icon} {title}</h3>
          <p>{description}</p>
          <span class="metis-badge">Coming soon</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
