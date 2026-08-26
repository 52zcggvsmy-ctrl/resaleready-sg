"""Simple demo authentication based on Streamlit session state."""

import os

import streamlit as st


DEMO_USERNAME = os.getenv("RESALEREADY_USERNAME", "admin")
DEMO_PASSWORD = os.getenv("RESALEREADY_PASSWORD", "r3ady4r3sale=")


def init_auth_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", "")


def login_form() -> None:
    st.title("🏠 ResaleReady SG")
    st.subheader("Understand the HDB resale journey. Explore the resale market.")
    st.info("Demo only — this login is not suitable for a production system.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", width="stretch")

    with st.expander("Demo credentials"):
        st.code("Username: admin\nPassword: r3ady4r3sale=")

    if submitted:
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.pop("resaleready_chat_history", None)
    st.rerun()
