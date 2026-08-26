import streamlit as st


def render() -> None:
    st.title("Welcome to ResaleReady SG")
    st.write(
        "ResaleReady SG is a web-based application designed to help prospective HDB "
        "resale flat buyers better understand and navigate the resale flat buying "
        "journey in Singapore."
    )

    with st.expander("⚠️ IMPORTANT NOTICE"):
        st.markdown(
            "**IMPORTANT NOTICE:** This web application is a prototype developed for "
            "educational purposes only. The information provided here is NOT intended "
            "for real-world usage and should not be relied upon for making any decisions, "
            "especially those related to financial, legal, or healthcare matters.\n\n"
            "Furthermore, please be aware that the LLM may generate inaccurate or "
            "incorrect information. You assume full responsibility for how you use any "
            "generated output.\n\n"
            "Always consult with qualified professionals for accurate and personalised advice."
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
            "Upload PDF or TXT files to bring in external / own sources of info for use "
            "in Ask ResaleReady."
        )

    st.subheader("Get started")
    st.write(
        "Choose **Market Explorer** to learn about historical transaction data, "
        "**Ask ResaleReady** to answer questions pertaining to HDB resale flat buying "
        "journey in Singapore, or use the **Knowledge Base** to bring in external/own "
        "sources of info for use in Ask ResaleReady."
    )
