import streamlit as st


def render() -> None:
    st.title("Welcome to ResaleReady SG")
    st.write(
        "A buyer-side navigator for understanding Singapore's HDB resale journey "
        "and exploring historical resale transactions."
    )
    st.warning(
        "For general information only. It is not financial, legal, eligibility, "
        "valuation, or price-prediction advice."
    )

    market_col, ask_col, knowledge_col = st.columns(3)
    with market_col:
        st.subheader("📊 Market Explorer")
        st.write(
            "Filter historical transactions by town, flat type, period, and budget; "
            "then compare summary statistics and trends."
        )
    with ask_col:
        st.subheader("💬 Ask ResaleReady")
        st.write(
            "Ask buyer-side resale questions and receive concise answers grounded in "
            "curated official HDB documents, with supporting source links."
        )
    with knowledge_col:
        st.subheader("📄 Knowledge Base")
        st.write(
            "Demonstrate how PDF and TXT uploads are validated, processed, and added "
            "to a session-only retrieval index."
        )

    st.subheader("Get started")
    st.write(
        "Choose **Market Explorer** for historical transaction data, "
        "**Ask ResaleReady** for process information, or **Knowledge Base** for the "
        "administrative upload demonstration."
    )
