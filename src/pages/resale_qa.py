import streamlit as st


def render() -> None:
    st.title("💬 Ask ResaleReady")
    st.info("Coming next: retrieval-augmented answers grounded in official HDB sources.")
    st.text_input("Ask about the buyer-side HDB resale journey", disabled=True, placeholder="RAG integration will be added in a later phase")
    st.caption("Planned service boundary: `src/rag/` will own document ingestion, retrieval, prompting, citations, and answer generation.")
