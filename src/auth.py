"""Prototype authentication backed by Streamlit Secrets and session state."""

import hmac

import streamlit as st

from src.rag.uploads import SESSION_UPLOAD_RESULTS_KEY, SESSION_UPLOAD_STORE_KEY


AUTH_CONFIGURATION_ERROR = (
    "Authentication is not configured. Add an [auth] section containing username "
    "and password to Streamlit Secrets, then restart the app."
)


def get_auth_credentials() -> tuple[str, str] | None:
    """Return configured prototype credentials, failing closed if unavailable."""

    try:
        auth_secrets = st.secrets["auth"]
        raw_username = auth_secrets.get("username", "")
        raw_password = auth_secrets.get("password", "")
    except (FileNotFoundError, KeyError, TypeError):
        return None

    if not isinstance(raw_username, str) or not isinstance(raw_password, str):
        return None
    username = raw_username.strip()
    password = raw_password
    if not username or not password:
        return None
    return username, password


def credentials_match(
    submitted_username: str,
    submitted_password: str,
    configured_username: str,
    configured_password: str,
) -> bool:
    """Compare credentials without ordinary equality checks."""

    username_matches = hmac.compare_digest(
        submitted_username.encode("utf-8"), configured_username.encode("utf-8")
    )
    password_matches = hmac.compare_digest(
        submitted_password.encode("utf-8"), configured_password.encode("utf-8")
    )
    return username_matches and password_matches


def init_auth_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", "")


def login_form() -> None:
    st.title("🏠 ResaleReady SG")
    st.subheader("Understand the HDB resale journey. Explore the resale market.")
    st.info("Demo only — this login is not suitable for a production system.")

    configured_credentials = get_auth_credentials()
    if configured_credentials is None:
        st.error(AUTH_CONFIGURATION_ERROR)
        return

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", width="stretch")

    if submitted:
        configured_username, configured_password = configured_credentials
        if credentials_match(
            username,
            password,
            configured_username,
            configured_password,
        ):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.pop("resaleready_chat_history", None)
    st.session_state.pop(SESSION_UPLOAD_STORE_KEY, None)
    st.session_state.pop(SESSION_UPLOAD_RESULTS_KEY, None)
    st.rerun()
