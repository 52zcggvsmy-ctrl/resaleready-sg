"""ResaleReady SG Streamlit entry point."""

import streamlit as st

from src.auth import init_auth_state, login_form, logout
from src.pages import about, home, market_explorer, methodology, resale_qa


st.set_page_config(
    page_title="ResaleReady SG",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_auth_state()

if not st.session_state.authenticated:
    login_form()
    st.stop()

with st.sidebar:
    st.title("🏠 ResaleReady SG")
    st.caption("Buyer-side HDB resale navigator")
    page_name = st.radio(
        "Navigate",
        ["Home", "Market Explorer", "Ask ResaleReady", "About Us", "Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Signed in as **{st.session_state.username}**")
    if st.button("Log out", width="stretch"):
        logout()

PAGES = {
    "Home": home.render,
    "Market Explorer": market_explorer.render,
    "Ask ResaleReady": resale_qa.render,
    "About Us": about.render,
    "Methodology": methodology.render,
}

PAGES[page_name]()
