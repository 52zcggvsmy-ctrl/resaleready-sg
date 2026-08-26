import streamlit as st


def render() -> None:
    st.title("⚙️ Methodology")
    st.subheader("Implemented now")
    st.write("Streamlit provides the interface and session-state login. Pandas validates, normalises, filters, aggregates, and exports resale transaction data. Streamlit renders the metrics, table, and charts.")
    st.subheader("Planned RAG interface")
    st.write("A later `src/rag/` package will ingest approved official documents, create searchable chunks, retrieve relevant evidence, and generate cited answers. It will remain separate from the structured market-data workflow.")
    st.subheader("Safeguards")
    st.markdown("- General-information disclaimer\n- No collection of NRIC, financial account details, or other sensitive data\n- Historical data clearly distinguished from valuation or prediction\n- Future answers will require source citations and uncertainty handling")
