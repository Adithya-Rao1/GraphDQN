import streamlit as st

from style import inject_theme

st.set_page_config(page_title="GraphDQN", layout="wide")
inject_theme()

if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.user_email = None

home = st.Page("pages/1_home.py", title="Home", default=True)

if st.session_state.token is None:
    pages = [home, st.Page("pages/2_signup_login.py", title="Sign Up / Login")]
else:
    pages = [
        home,
        st.Page("pages/3_optimize.py", title="Optimize Molecules"),
        st.Page("pages/4_database.py", title="My Database"),
        st.Page("pages/5_candidates.py", title="Candidates"),
        st.Page("pages/6_account.py", title="Account"),
    ]

st.navigation(pages).run()
