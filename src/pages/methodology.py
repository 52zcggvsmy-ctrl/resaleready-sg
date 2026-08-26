import streamlit as st


def render() -> None:
    st.title("⚙️ Methodology")
    st.subheader("Market Explorer")
    st.write(
        "Streamlit provides the interface and session-state login. Pandas validates, "
        "normalises, filters, aggregates, and exports resale transaction data. "
        "Streamlit renders the metrics, table, and charts."
    )
    st.subheader("Ask ResaleReady")
    st.write(
        "Curated official HDB PDF and TXT documents are cleaned, divided into "
        "token-aware chunks, embedded, and stored in a FAISS index with source "
        "metadata. For a follow-up question, OpenAI first creates a standalone "
        "retrieval query. The query embedding is compared with the FAISS index, and "
        "the most relevant extracts are supplied to a separate grounded-answer prompt."
    )
    st.caption(
        "Question → safeguards → follow-up rewrite → embedding → FAISS retrieval → "
        "grounded answer → source display"
    )
    st.subheader("Knowledge Base uploads")
    st.write(
        "The administrative demo accepts PDF and UTF-8 TXT files up to 10 MB. Each "
        "file is validated, extracted, cleaned, chunked, embedded, and added to a "
        "session-scoped in-memory FAISS index. Curated HDB sources keep the majority "
        "of retrieval slots and appear first in the answer context. Uploads are "
        "labelled unverified and are discarded on logout, session expiry, or app restart."
    )
    st.caption("Upload → validate → extract → clean → chunk → embed → active session knowledge base")
    st.subheader("Safeguards")
    st.markdown(
        "- Input length, privacy, prompt-injection, valuation, and advice checks\n"
        "- Curated official HDB sources remain primary\n"
        "- Uploaded text is treated as untrusted reference data, never as instructions\n"
        "- Clear insufficiency response when the knowledge base does not support an answer\n"
        "- No definitive eligibility decisions, valuations, predictions, or financial or legal advice\n"
        "- Supporting source titles and trust labels shown after answers"
    )
