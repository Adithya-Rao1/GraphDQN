import streamlit as st

from api_client import ApiError, get_client
from style import inject_theme

inject_theme()
api = get_client()

st.title("Sign Up / Login")

tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

with tab_login:
    with st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In")
    if submitted:
        try:
            result = api.login(email, password)
            st.session_state.token = result["token"]
            st.session_state.user_email = result["email"]
            st.rerun()
        except ApiError as e:
            st.error(e.detail)

with tab_signup:
    with st.form("signup_form"):
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password (min 8 characters)", type="password", key="signup_password")
        submitted = st.form_submit_button("Sign Up")
    if submitted:
        try:
            result = api.signup(email, password)
            st.session_state.token = result["token"]
            st.session_state.user_email = result["email"]
            st.rerun()
        except ApiError as e:
            st.error(e.detail)
