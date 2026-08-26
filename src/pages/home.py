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

    left, right = st.columns(2)
    with left:
        st.subheader("📊 Market Explorer")
        st.write(
            "Filter historical transactions by town, flat type, period, and budget; "
            "then compare summary statistics and trends."
        )
    with right:
        st.subheader("💬 Ask ResaleReady")
        st.write(
            "Ask buyer-side resale questions and receive concise answers grounded in "
            "curated official HDB documents, with supporting source links."
        )

    st.subheader("Get started")
    st.write(
        "Choose **Market Explorer** for historical transaction data or "
        "**Ask ResaleReady** for official process information."
    )
