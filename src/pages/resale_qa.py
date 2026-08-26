"""Citizen-facing RAG question-and-answer page."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import streamlit as st

from src.app_config import secret_or_environment
from src.rag.qa import validate_question
from src.rag.runtime import attach_uploaded_store, create_qa_service
from src.rag.uploads import SESSION_UPLOAD_STORE_KEY, UploadedVectorStore

CHAT_HISTORY_KEY = "resaleready_chat_history"
DEFAULT_CHAT_MODEL = "gpt-5.6-luna"
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _load_qa_service(chat_model: str, _api_key: str):
    return create_qa_service(api_key=_api_key, chat_model=chat_model)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.caption("Supporting sources")
    for source in sources:
        location = ""
        if source.get("section"):
            location = f" — {source['section']}"
        elif source.get("page"):
            location = f" — page {source['page']}"
        title = source.get("title", "Reference source")
        if source.get("source_kind") == "uploaded_demo":
            filename = source.get("local_filename", "uploaded file")
            st.markdown(
                f"- **{title}** — uploaded demo document, unverified ({filename}){location}"
            )
            continue
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
        "Ask about the buyer-side HDB resale process. Answers prioritise the curated "
        "official HDB documents in the ResaleReady knowledge base."
    )
    st.warning(
        "ResaleReady provides general information, not eligibility decisions, property "
        "valuations, predictions, or financial or legal advice. It cannot help with "
        "unrelated topics or reveal secrets and internal instructions. Verify important "
        "matters through the linked official HDB sources or HDB directly."
    )

    uploaded_store = st.session_state.get(SESSION_UPLOAD_STORE_KEY)
    if isinstance(uploaded_store, UploadedVectorStore) and uploaded_store.document_count:
        st.caption(
            f"This session also contains {uploaded_store.document_count} unverified "
            "uploaded demo document(s). Official HDB sources remain primary."
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
            "details, credentials, or other personal information."
        )

    for message in history:
        _render_message(message)

    api_key = secret_or_environment("OPENAI_API_KEY")
    chat_model = secret_or_environment("RESALEREADY_CHAT_MODEL", DEFAULT_CHAT_MODEL)
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
    user_message = {"role": "user", "content": question, "blocked": False}
    history.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        safeguard = validate_question(question, history=prior_history)
        if not safeguard.allowed:
            user_message["blocked"] = True
            blocked_answer = safeguard.message or "I cannot help with that request."
            st.markdown(blocked_answer)
            _append_assistant_message(history, blocked_answer, blocked=True)
            st.rerun()

        try:
            with st.spinner("Checking official HDB sources..."):
                service = _load_qa_service(chat_model, api_key)
                if isinstance(uploaded_store, UploadedVectorStore):
                    service = attach_uploaded_store(
                        service,
                        uploaded_store,
                        api_key=api_key,
                    )
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
