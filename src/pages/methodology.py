import streamlit as st


def render() -> None:
    st.title("⚙️ Methodology")
    st.subheader("Market Explorer")
    st.write("Streamlit provides the interface and session-state login. Pandas validates, normalises, filters, aggregates, and exports resale transaction data. Streamlit renders the metrics, table, and charts.")
    st.subheader("Ask ResaleReady")
    st.write("Curated official HDB PDF and TXT documents are cleaned, divided into token-aware chunks, embedded, and stored in a FAISS index with source metadata. For a follow-up question, OpenAI first creates a standalone retrieval query. The query embedding is compared with the FAISS index, and the most relevant HDB extracts are supplied to a separate grounded-answer prompt.")
    st.caption("Question → safeguards → follow-up rewrite → embedding → FAISS retrieval → grounded answer → official source links")
    st.subheader("Safeguards")
    st.markdown("- Input length, privacy, prompt-injection, valuation, and advice checks\n- Answers restricted to retrieved official HDB context\n- Clear insufficiency response when the knowledge base does not support an answer\n- No definitive eligibility decisions, valuations, predictions, or financial or legal advice\n- Supporting official HDB source titles and links shown after answers")
