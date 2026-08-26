"""Submission-ready technical methodology for ResaleReady SG."""

from __future__ import annotations

import streamlit as st


RAG_FLOWCHART = r"""
digraph rag_workflow {
    graph [rankdir=TB, bgcolor="transparent", pad="0.25", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10,
          color="#315B7D", fillcolor="#EAF3F8", margin="0.16,0.10"];
    edge [fontname="Arial", fontsize=9, color="#536878", arrowsize=0.75];

    question [label="User asks an HDB\nresale question", fillcolor="#DDEBF7"];
    guard [label="Validate input\nScope · privacy · injection · disclosure", fillcolor="#FFF2CC"];
    blocked [label="Safe refusal or\nout-of-scope guidance", fillcolor="#FCE4D6"];
    followup [shape=diamond, label="Context-dependent\nfollow-up?", fillcolor="#E2F0D9"];
    rewrite [label="OpenAI prompt 1\nRewrite as a standalone\nretrieval query", fillcolor="#E2F0D9"];
    embed [label="Create query embedding\nOpenAI Embeddings API"];
    faiss [label="FAISS cosine-similarity search\nTop relevant chunks + metadata"];
    evidence [shape=diamond, label="Relevant evidence\navailable?", fillcolor="#FFF2CC"];
    fallback [label="Deterministic\ninsufficient-evidence response", fillcolor="#FCE4D6"];
    prompt [label="Build grounded prompt\nStable instructions + untrusted JSON context"];
    answer [label="OpenAI prompt 2\nGenerate concise cited answer", fillcolor="#E2F0D9"];
    output [label="Validate output\nProtocol · citations · secret leakage", fillcolor="#FFF2CC"];
    display [label="Display answer and\nsupporting source titles", fillcolor="#DDEBF7"];

    question -> guard;
    guard -> blocked [label=" blocked "];
    guard -> followup [label=" allowed "];
    followup -> rewrite [label=" yes "];
    followup -> embed [label=" no "];
    rewrite -> embed;
    embed -> faiss;
    faiss -> evidence;
    evidence -> fallback [label=" no "];
    evidence -> prompt [label=" yes "];
    prompt -> answer;
    answer -> output;
    output -> fallback [label=" invalid / insufficient "];
    output -> display [label=" supported "];
}
"""


MARKET_FLOWCHART = r"""
digraph market_workflow {
    graph [rankdir=TB, bgcolor="transparent", pad="0.25", nodesep="0.35", ranksep="0.45"];
    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10,
          color="#315B7D", fillcolor="#EAF3F8", margin="0.16,0.10"];
    edge [fontname="Arial", fontsize=9, color="#536878", arrowsize=0.75];

    csv [label="Official HDB resale CSV\nDevelopment fallback if unavailable", fillcolor="#DDEBF7"];
    cache [label="Streamlit cached loader"];
    clean [label="Pandas preprocessing\nNormalise columns · parse month\nconvert numbers · discard malformed rows"];
    options [label="Derive valid towns, flat types,\nbudget range, and available months"];
    filters [label="User selects town · flat type\napproximate budget · period", fillcolor="#E2F0D9"];
    filter [label="Pandas filters rows by\ntown · flat type · period"];
    matches [shape=diamond, label="Matching\ntransactions?", fillcolor="#FFF2CC"];
    empty [label="Helpful empty-state message\nand filter guidance", fillcolor="#FCE4D6"];
    metrics [label="Deterministic calculations\ncount · medians · % at/below budget"];
    outputs [label="Monthly median trend\nprice distribution · clean table"];
    display [label="Streamlit metrics and charts\nSingapore-dollar formatting · CSV download", fillcolor="#DDEBF7"];

    csv -> cache -> clean -> options -> filters -> filter -> matches;
    matches -> empty [label=" no "];
    matches -> metrics [label=" yes "];
    metrics -> outputs -> display;
}
"""


def render() -> None:
    st.title("⚙️ Methodology")
    st.write(
        "This page explains how the working ResaleReady SG prototype transforms "
        "structured transaction data and unstructured HDB documents into its two main "
        "user experiences. The design keeps market calculations deterministic and uses "
        "the language model only where natural-language interpretation is needed."
    )

    st.subheader("Overall system architecture")
    st.write(
        "The prototype uses a small layered architecture so that the Streamlit pages "
        "remain focused on presentation rather than data or AI logic."
    )
    ui_col, service_col, data_col = st.columns(3)
    with ui_col:
        with st.container(border=True):
            st.markdown("#### Presentation layer")
            st.write(
                "Streamlit entry point, multipage navigation, session-state login, "
                "filters, chat history, charts, tables, source display, and feedback."
            )
    with service_col:
        with st.container(border=True):
            st.markdown("#### Application layer")
            st.write(
                "Reusable Python modules for CSV preprocessing, Pandas calculations, "
                "document ingestion, retrieval, prompt construction, OpenAI calls, and "
                "input/output safeguards."
            )
    with data_col:
        with st.container(border=True):
            st.markdown("#### Data and model layer")
            st.write(
                "Official transaction CSV, curated HDB PDF/TXT sources and provenance "
                "manifest, FAISS vector store, OpenAI embeddings, and the OpenAI "
                "Responses API."
            )
    st.caption(
        "Authentication is a hardcoded single-user demonstration implemented with "
        "Streamlit session state; it is not a production identity system."
    )

    st.divider()
    st.header("Use Case 1 — RAG HDB Resale Q&A")

    st.subheader("Document ingestion")
    st.write(
        "Curated official HDB PDF and TXT files are stored under `data/rag_sources/`. "
        "A JSON manifest records each document's title, source organisation, official "
        "URL, local filename, update information, and optional cleaning boundaries. "
        "The reproducible `scripts/build_vector_store.py` command validates this "
        "manifest and processes the files in a stable filename order."
    )
    st.markdown(
        "1. **Extract:** PyPDF reads text page by page from PDF files; UTF-8 text files "
        "are read directly and can retain heading-based sections.\n"
        "2. **Clean:** Unicode and whitespace are normalised, known page-export chrome "
        "is removed, and unusable content is rejected.\n"
        "3. **Preserve provenance:** Page or section, title, organisation, official URL, "
        "filename, and update information remain attached to every chunk.\n"
        "4. **Build deterministically:** Stable content-based chunk identifiers and "
        "source checksums make rebuilds reviewable."
    )

    st.subheader("Chunking, embeddings, and FAISS retrieval")
    st.write(
        "Clean text is encoded with `cl100k_base` and divided into windows of "
        "approximately **850 tokens with 120-token overlap**. The overlap helps retain "
        "meaning when a relevant passage crosses a chunk boundary. OpenAI's "
        "`text-embedding-3-small` model converts each chunk into a numeric vector. "
        "The vectors are L2-normalised and stored in a FAISS `IndexFlatIP`; inner-product "
        "search on normalised vectors is used as cosine similarity. The aligned chunk "
        "text and metadata are stored separately so every retrieved result remains "
        "traceable to its source."
    )
    st.write(
        "At question time, the same embedding model converts the retrieval query into a "
        "vector. FAISS returns the top matching chunks. The default Q&A service requests "
        "four results. When a session upload is active, curated HDB content retains the "
        "majority of those positions and the upload receives at most one position."
    )

    st.subheader("RAG question-answering, prompt engineering, and prompt chaining")
    st.write(
        "RAG means that the model receives selected reference extracts together with the "
        "question instead of being asked to rely on general model memory. ResaleReady uses "
        "two focused prompts:"
    )
    st.markdown(
        "- **Prompt 1 — retrieval-query rewrite:** used only when conversation history "
        "exists. It resolves phrases such as ‘it’, ‘that’, or ‘how much’ and returns one "
        "standalone search query. It does not answer the question.\n"
        "- **Prompt 2 — grounded answer:** receives the original question, standalone "
        "query, and retrieved chunks. It must return a supported answer with valid "
        "`[Source n]` citations, or declare that the evidence is insufficient."
    )
    st.write(
        "This sequence is **prompt chaining**: the rewrite output improves retrieval, "
        "and the retrieved evidence becomes the controlled input to the answer prompt. "
        "Stable instructions are kept separate from user and document content, which is "
        "serialized as an explicitly untrusted JSON payload."
    )

    st.subheader("OpenAI integration")
    st.write(
        "A reusable client module calls the **OpenAI Responses API** for rewriting and "
        "answer generation, and a separate embedding-provider interface calls the "
        "**OpenAI Embeddings API** for indexing and retrieval. The selected chat model "
        "is configurable through Streamlit Secrets. The API key is read at the application "
        "boundary and is never inserted into prompts. Response storage is disabled, and a "
        "stable safety identifier is attached to model requests."
    )

    st.subheader("Flowchart 1 — RAG HDB Resale Q&A workflow")
    st.graphviz_chart(RAG_FLOWCHART, width="stretch")
    st.caption(
        "Blocked requests stop before retrieval or generation. Unsupported or invalidly "
        "cited answers use a fixed safe fallback instead of presenting a guess."
    )

    st.subheader("Safeguard architecture and prompt-injection defences")
    st.write(
        "Safeguards are applied at several boundaries rather than relying only on a list "
        "of blocked phrases."
    )
    st.markdown(
        "- **Before model access:** Unicode-normalised length and structure limits, "
        "personal-information checks, buyer-side HDB resale scope control, and detection "
        "of common instruction overrides, role changes, secret requests, valuation "
        "requests, and financial/legal advice requests.\n"
        "- **At the prompt boundary:** system instructions remain separate from user "
        "content; retrieved text and metadata are labelled untrusted JSON reference data. "
        "The model is explicitly told never to execute document instructions or reveal "
        "prompts, credentials, secrets, or configuration.\n"
        "- **At the evidence boundary:** curated official HDB documents remain primary; "
        "uploads are labelled unverified; absent evidence returns a deterministic fallback.\n"
        "- **After generation:** the app checks the supported/insufficient protocol, "
        "requires citations to valid retrieved positions, displays only cited sources, "
        "and rejects secret-shaped output or internal-policy leakage.\n"
        "- **Testing:** an offline adversarial test set covers injection, prompt and API-key "
        "disclosure, role override, out-of-scope and unsupported questions, normal questions, "
        "and malicious instructions embedded in retrieved content."
    )

    st.divider()
    st.header("Use Case 2 — HDB Resale Market Explorer")

    st.subheader("Structured-data pipeline")
    st.write(
        "The application loads `data/structured/hdb_resale_transactions.csv`, which "
        "contains the official HDB resale transaction data used by the prototype. "
        "Streamlit's data cache avoids repeating the same CSV work on every interface "
        "rerun and invalidates the cached result when the local file modification time changes."
    )
    st.markdown(
        "1. Column names and text values are stripped and normalised.\n"
        "2. `month` is parsed with the required `YYYY-MM` format.\n"
        "3. `resale_price`, `floor_area_sqm`, and lease commencement year are converted "
        "to numeric values.\n"
        "4. Rows with missing required fields, blank text, malformed dates, or non-positive "
        "numeric values are discarded and counted.\n"
        "5. Address and price-per-square-metre fields are derived for display.\n"
        "6. Valid towns, town-specific flat types, budget steps, and available months are "
        "derived dynamically from the cleaned dataset."
    )
    st.write(
        "If the official local file is missing or unusable, a small demo CSV allows the "
        "prototype to remain testable and is clearly labelled as a development fallback."
    )

    st.subheader("Market Explorer calculations")
    st.write(
        "The selected town, flat type, and date period determine the matching rows. The "
        "budget is deliberately **not** used to hide higher-priced transactions; it is a "
        "comparison input. Pandas then calculates:"
    )
    st.markdown(
        "- **Number of transactions:** count of matching rows.\n"
        "- **Median resale price:** median of `resale_price`.\n"
        "- **Median floor area:** median of `floor_area_sqm`.\n"
        "- **Percentage at or below budget:** matching prices less than or equal to the "
        "budget, divided by all matching transactions, multiplied by 100.\n"
        "- **Monthly trend:** median resale price grouped by transaction month.\n"
        "- **Price distribution:** transaction counts in S$50,000 bands."
    )
    st.write(
        "Streamlit presents the results as Singapore-dollar metrics, line and bar charts, "
        "and a newest-first table. Users can download the filtered records as CSV. If no "
        "rows match, the page shows guidance instead of attempting calculations. No LLM "
        "is used for any Market Explorer number, filter, table, or chart."
    )

    st.subheader("Flowchart 2 — HDB Resale Market Explorer workflow")
    st.graphviz_chart(MARKET_FLOWCHART, width="stretch")
    st.caption(
        "All figures shown in the Market Explorer are reproducible Pandas calculations "
        "from the filtered historical transaction rows."
    )

    st.divider()
    st.subheader("Methodology limitations")
    st.markdown(
        "- Historical transactions describe past recorded sales and are not valuations "
        "or predictions of future prices.\n"
        "- Median statistics do not account for every flat-specific feature, condition, "
        "renovation, floor level, remaining lease, or negotiation factor.\n"
        "- RAG coverage depends on the quality, scope, and freshness of the curated source "
        "documents and on text that can be extracted from them.\n"
        "- Vector similarity may retrieve incomplete or only partly relevant passages; "
        "the grounded-answer and citation checks reduce but do not eliminate this risk.\n"
        "- Uploaded reference material is unverified and lasts only for the active session.\n"
        "- The login, upload management, and local vector-store lifecycle are designed for "
        "a classroom prototype, not a production public service.\n"
        "- LLM safeguards reduce prompt-injection and disclosure risk but cannot guarantee "
        "perfect protection or factual accuracy. Important matters must be verified with HDB."
    )

    st.warning(
        "ResaleReady SG is an educational, non-official prototype. It does not determine "
        "eligibility, calculate affordability, provide financial or legal advice, value a "
        "property, or replace guidance from HDB and other official agencies."
    )
