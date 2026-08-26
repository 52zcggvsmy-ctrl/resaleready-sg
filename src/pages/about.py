import streamlit as st


def render() -> None:
    st.title("ℹ️ About Us")
    st.write("ResaleReady SG is an AI Bootcamp capstone prototype focused on prospective buyers of HDB resale flats in Singapore.")
    st.subheader("Current scope")
    st.markdown("- Historical resale transaction exploration\n- Plain-language buyer journey support (planned)\n- Public, non-personally identifiable inputs only")
    st.subheader("Important limitations")
    st.write("This prototype does not determine eligibility, grants, loan amounts, valuation, affordability, or whether a buyer should proceed with a purchase. Users should verify current rules with HDB and other relevant agencies.")
