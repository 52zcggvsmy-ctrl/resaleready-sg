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
        "metadata. For a permitted follow-up question, OpenAI first creates a "
        "standalone retrieval query. Relevant extracts are then supplied to a separate "
        "grounded-answer prompt."
    )
    st.caption(
        "Question → deterministic safeguards → follow-up rewrite → embedding → "
        "FAISS retrieval → grounded answer → output validation → source display"
    )
    st.subheader("Knowledge Base uploads")
    st.write(
        "The administrative demo accepts PDF and UTF-8 TXT files up to 10 MB. Each "
        "file is validated, extracted, cleaned, chunked, embedded, and added to a "
        "session-scoped in-memory FAISS index. Curated HDB sources keep the majority "
        "of retrieval slots and appear first. Uploads are labelled unverified and are "
        "discarded on logout, session expiry, or app restart."
    )
    st.caption(
        "Upload → validate → extract → clean → chunk → embed → active session knowledge base"
    )
    st.subheader("Layered safeguards")
    st.markdown(
        "- Buyer-side HDB resale domain restriction with conversation-aware follow-ups\n"
        "- Unicode-normalised length, structure, privacy, injection, disclosure, valuation, and advice checks\n"
        "- Blocked inputs stop before query rewriting, retrieval, or OpenAI generation\n"
        "- Stable instructions are separated from user text and retrieved content\n"
        "- Questions, metadata, and extracts are serialized as untrusted JSON reference data\n"
        "- API keys and credentials are never included in model input\n"
        "- Answers require valid citations to retrieved sources\n"
        "- Generated output is screened for secret-like values and policy leakage\n"
        "- Empty, irrelevant, uncited, or insufficient evidence fails closed\n"
        "- No definitive eligibility decisions, valuations, predictions, or financial or legal advice"
    )
    st.caption(
        "Safeguards reduce risk but cannot guarantee perfect model behaviour. Important "
        "information should still be verified with official HDB channels."
    )
