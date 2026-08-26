"""Citizen-facing RAG question-and-answer page."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

import streamlit as st

from src.rag.qa import validate_question
from src.rag.runtime import create_qa_service

CHAT_HISTORY_KEY = "resaleready_chat_history"
DEFAULT_CHAT_MODEL = "gpt-5.6-luna"
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _load_qa_service(chat_model: str, _api_key: str):
    return create_qa_service(api_key=_api_key, chat_model=chat_model)


def _secret_or_environment(name: str, default: str = "") -> str:
    try:
        secret = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        secret = None
    return str(secret or os.getenv(name, default)).strip()


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.caption("Supporting official HDB sources")
    for source in sources:
        location = ""
        if source.get("section"):
            location = f" — {source['section']}"
        elif source.get("page"):
            location = f" — page {source['page']}"
        title = source.get("title", "Official HDB source")
        url = source.get("url", "")
        if url:
            st.markdown(f"- [{title}]({url}){location}")
        else:
            st.markdown(f"- {title}{location}")


def _render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(str(message.get("content", "")))
        if role == "assistant":
            _render_sources(list(message.get("sources", [])))


def _append_assistant_message(
    history: list[dict[str, Any]],
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    retrieval_query: str = "",
    blocked: bool = False,
) -> None:
    history.append(
        {
            "role": "assistant",
            "content": content,
            "sources": sources or [],
            "retrieval_query": retrieval_query,
            "blocked": blocked,
        }
    )


def render() -> None:
    st.title("💬 Ask ResaleReady")
    st.write(
        "Ask about the buyer-side HDB resale process. Answers are generated from "
        "the curated official HDB documents in the ResaleReady knowledge base."
    )
    st.warning(
        "ResaleReady provides general information, not eligibility decisions, property "
        "valuations, predictions, or financial or legal advice. Verify important matters "
        "through the linked official HDB sources or HDB directly."
    )

    st.session_state.setdefault(CHAT_HISTORY_KEY, [])
    history: list[dict[str, Any]] = st.session_state[CHAT_HISTORY_KEY]

    action_col, guidance_col = st.columns([1, 4])
    with action_col:
        if st.button("Clear conversation", disabled=not history, width="stretch"):
            st.session_state[CHAT_HISTORY_KEY] = []
            st.rerun()
    with guidance_col:
        st.caption(
            "Follow-up questions are supported. Avoid entering NRIC numbers, contact "
            "details, or other personal information."
        )

    for message in history:
        _render_message(message)

    api_key = _secret_or_environment("OPENAI_API_KEY")
    chat_model = _secret_or_environment("RESALEREADY_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    if not api_key:
        st.info(
            "Ask ResaleReady is not configured yet. Add `OPENAI_API_KEY` to Streamlit "
            "Secrets or your local environment to enable the Q&A service."
        )
        st.chat_input("Ask about the HDB resale buying process", disabled=True)
        return

    question = st.chat_input(
        "Ask about the HDB resale buying process",
        max_chars=1_000,
    )
    if not question:
        return

    prior_history = list(history)
    user_message = {"role": "user", "content": question}
    history.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        # This check must run before service creation, which can build/embed the index.
        safeguard = validate_question(question)
        if not safeguard.allowed:
            blocked_answer = safeguard.message or "I cannot help with that request."
            st.markdown(blocked_answer)
            _append_assistant_message(
                history,
                blocked_answer,
                blocked=True,
            )
            st.rerun()

        try:
            with st.spinner("Checking official HDB sources..."):
                service = _load_qa_service(chat_model, api_key)
                result = service.answer(question, history=prior_history)
        except Exception:
            LOGGER.exception("Ask ResaleReady failed to answer a question.")
            error_message = (
                "I could not access the ResaleReady knowledge base just now. Please try "
                "again later or verify your question through official HDB channels."
            )
            st.error(error_message)
            _append_assistant_message(history, error_message)
            st.rerun()

        st.markdown(result.answer)
        sources = [asdict(source) for source in result.sources]
        _render_sources(sources)
        _append_assistant_message(
            history,
            result.answer,
            sources=sources,
            retrieval_query=result.retrieval_query,
            blocked=result.blocked,
        )
        st.rerun()
