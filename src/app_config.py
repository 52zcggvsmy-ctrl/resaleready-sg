"""Application configuration helpers shared by Streamlit pages."""

from __future__ import annotations

import os

import streamlit as st


def secret_or_environment(name: str, default: str = "") -> str:
    """Read a Streamlit secret first, then an environment variable."""

    try:
        secret = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        secret = None
    return str(secret or os.getenv(name, default)).strip()
