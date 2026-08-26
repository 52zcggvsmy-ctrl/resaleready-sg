import streamlit as st


def render() -> None:
    st.title("Welcome to ResaleReady SG")
    st.write(
        "A buyer-side navigator for understanding Singapore's HDB resale journey "
        "and exploring historical resale transactions."
    )
    st.warning("For general information only. It is not financial, legal, or eligibility advice.")

    left, right = st.columns(2)
    with left:
        st.subheader("📊 Market Explorer")
        st.write("Filter historical transactions by town, flat type, period, and budget; then compare summary statistics and trends.")
    with right:
        st.subheader("💬 Ask ResaleReady")
        st.write("A future evidence-grounded Q&A assistant for official HDB resale guidance. The RAG component is intentionally not enabled yet.")

    st.subheader("Suggested next step")
    st.write("Open **Market Explorer** from the sidebar. It runs immediately with the included demo dataset.")
