import streamlit as st


def render() -> None:
    st.title("ℹ️ About Us")
    st.write(
        "ResaleReady SG is an AI Bootcamp capstone prototype focused on prospective "
        "buyers of HDB resale flats in Singapore."
    )
    st.subheader("Current scope")
    st.markdown(
        "- Historical resale transaction exploration\n"
        "- Source-grounded buyer-side HDB resale Q&A\n"
        "- Follow-up questions with conversation-aware retrieval\n"
        "- Administrative PDF/TXT upload demonstration with session-only retrieval\n"
        "- Public, non-personally identifiable inputs only"
    )
    st.subheader("Important limitations")
    st.write(
        "This prototype does not determine eligibility, grants, loan amounts, "
        "valuation, affordability, or whether a buyer should proceed with a purchase. "
        "Its Q&A coverage is primarily based on the curated HDB documents in its "
        "knowledge base. Uploaded demo documents are unverified, session-only "
        "references. Users should verify current rules with HDB and other relevant "
        "agencies."
    )
