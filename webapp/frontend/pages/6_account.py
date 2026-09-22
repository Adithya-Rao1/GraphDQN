import streamlit as st

from api_client import get_client
from style import inject_theme

inject_theme()

if not st.session_state.get("token"):
    st.warning("Please log in first.")
    st.stop()

api = get_client()

st.title("Account")
me = api.me()
st.markdown(f"""
<div class="metis-card">
  <div>Email: {me['email']}</div>
  <div class="metis-meta">Member since {me['created_at']}</div>
</div>
""", unsafe_allow_html=True)

if st.button("Log Out"):
    st.session_state.token = None
    st.session_state.user_email = None
    st.rerun()
