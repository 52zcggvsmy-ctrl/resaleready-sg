"""Submission-ready project overview for ResaleReady SG."""

from __future__ import annotations

import streamlit as st

from src.rag.ingestion import load_document_manifest
from src.rag.runtime import SOURCE_MANIFEST


def _render_curated_sources() -> None:
    """List the curated HDB references directly from the ingestion manifest."""

    try:
        source_specs = load_document_manifest(SOURCE_MANIFEST)
    except (OSError, ValueError):
        st.info(
            "The curated-source list is temporarily unavailable. The Q&A knowledge "
            "base uses official HDB buyer-side resale documents."
        )
        return

    for spec in source_specs.values():
        updated = f" · source updated {spec.last_updated}" if spec.last_updated else ""
        st.markdown(
            f"- [{spec.document_title}]({spec.source_url}) — "
            f"{spec.source_organization}{updated}"
        )


def render() -> None:
    st.title("ℹ️ About ResaleReady SG")
    st.write(
        "**ResaleReady SG** is an AI Bootcamp capstone prototype that helps prospective "
        "buyers understand Singapore's HDB resale journey and explore official "
        "historical resale transactions in one accessible application."
    )
    st.info(
        "The prototype combines deterministic data analysis with retrieval-augmented "
        "generation (RAG). Pandas performs every market calculation, while the AI "
        "assistant answers process questions from retrieved reference material."
    )

    st.subheader("Problem addressed")
    st.write(
        "Buying an HDB resale flat involves several stages, specialised terms, official "
        "documents, and market information spread across different pages and datasets. "
        "A first-time buyer may find it difficult to understand the sequence of steps, "
        "locate the relevant guidance, and place historical transaction prices in "
        "context. ResaleReady SG brings these tasks into a single guided prototype "
        "without attempting to replace HDB's official services."
    )

    scope_col, objective_col = st.columns(2)
    with scope_col:
        st.subheader("Project scope")
        st.markdown(
            "- Buyer-side HDB resale information in Singapore\n"
            "- Historical transaction exploration from the official resale dataset\n"
            "- Source-grounded questions about the resale buying process\n"
            "- Administrative demonstration of PDF/TXT knowledge-base uploads\n"
            "- Public, non-personally identifiable user inputs"
        )
    with objective_col:
        st.subheader("Objectives")
        st.markdown(
            "- Explain the resale process in clear, everyday language\n"
            "- Make historical transactions easier to filter and interpret\n"
            "- Ground factual answers in traceable reference documents\n"
            "- Demonstrate a maintainable Streamlit, Pandas, OpenAI, and FAISS solution\n"
            "- Apply safeguards appropriate to a citizen-facing AI prototype"
        )

    st.subheader("Intended users")
    st.write(
        "The main users are prospective and first-time HDB resale flat buyers who want "
        "an initial overview before consulting official HDB channels. The Knowledge Base "
        "page is an administrative/demo feature for assessors or presenters to show how "
        "additional reference documents are processed during a Streamlit session."
    )

    st.subheader("Two main use cases")
    market_col, qa_col = st.columns(2)
    with market_col:
        with st.container(border=True):
            st.markdown("#### 1. HDB Resale Market Explorer")
            st.write(
                "A buyer selects a town, flat type, approximate budget, and transaction "
                "period. The app uses Pandas to calculate matching transaction counts, "
                "median resale price, median floor area, and the percentage of sales at "
                "or below the selected budget. It also presents a monthly trend, price "
                "distribution, detailed table, and CSV download."
            )
    with qa_col:
        with st.container(border=True):
            st.markdown("#### 2. Ask ResaleReady")
            st.write(
                "A buyer asks a question about the HDB resale process. The app validates "
                "the question, retrieves relevant chunks from the FAISS knowledge base, "
                "and asks an OpenAI model for a concise answer grounded in those extracts. "
                "Follow-up questions are rewritten into standalone retrieval queries, and "
                "supporting source titles are shown with the answer."
            )

    st.subheader("Prototype features")
    feature_col_one, feature_col_two = st.columns(2)
    with feature_col_one:
        st.markdown(
            "- Single-user demonstration login using Streamlit session state\n"
            "- Cached loading and robust preprocessing of the transaction CSV\n"
            "- Dynamic town, flat-type, budget, and date controls\n"
            "- Deterministic summary statistics and interactive charts\n"
            "- Downloadable filtered transaction table"
        )
    with feature_col_two:
        st.markdown(
            "- Curated PDF/TXT ingestion with source metadata\n"
            "- Token-aware chunking, OpenAI embeddings, and FAISS retrieval\n"
            "- Conversation-aware, source-cited RAG answers\n"
            "- Session-only administrative document uploads\n"
            "- Layered privacy, domain, injection, grounding, and output safeguards"
        )

    st.subheader("Official and trustworthy data sources")
    st.markdown(
        "- **Historical transactions:** HDB's *Resale Flat Prices (Based on "
        "Registration Date), From Jan 2017 onwards*, published through "
        "[data.gov.sg](https://data.gov.sg/collections/189/view). The included demo CSV "
        "is used only if the official local dataset cannot be loaded.\n"
        "- **Resale-process knowledge base:** Curated buyer-side pages published by the "
        "Housing & Development Board (HDB). Each local source is matched to a manifest "
        "record containing its title, organisation, official URL, filename, and update "
        "information."
    )
    with st.expander("View the curated official HDB documents"):
        _render_curated_sources()

    st.subheader("Technology overview")
    st.write(
        "The user interface and navigation are built with **Python and Streamlit**. "
        "**Pandas** performs CSV validation, filtering, aggregation, and export. "
        "**PyPDF** extracts PDF text, **tiktoken** supports token-aware chunking, and "
        "the **OpenAI Embeddings API** converts text into vectors. **FAISS** stores and "
        "searches those vectors by cosine similarity. The **OpenAI Responses API** is "
        "used for follow-up query rewriting and grounded answer generation."
    )

    st.subheader("Limitations")
    st.markdown(
        "- The Market Explorer describes recorded historical transactions; it does not "
        "predict prices or establish the market value of a flat.\n"
        "- ResaleReady does not calculate affordability, loans, grants, or stamp duty, "
        "and it does not determine whether a person is eligible.\n"
        "- The Q&A coverage is limited to the documents currently indexed and may not "
        "contain every policy detail or the latest change.\n"
        "- Uploaded documents are unverified, session-only references and never replace "
        "the curated official HDB sources.\n"
        "- AI responses and prompt-injection safeguards can reduce risk but cannot "
        "guarantee perfect accuracy or behaviour."
    )

    st.warning(
        "**Prototype and non-official disclaimer:** ResaleReady SG is an educational AI "
        "Bootcamp project. It is not an official HDB service and does not provide "
        "financial, legal, eligibility, or property-valuation advice. Verify important "
        "information with HDB or the relevant official agency before acting on it."
    )
