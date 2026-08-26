# ResaleReady SG

A Streamlit prototype for prospective HDB resale flat buyers in Singapore.

## Current features

- Prototype single-user login with session-state authentication
- Official HDB resale transaction filtering by town, flat type, period, and approximate budget
- Summary metrics, monthly median-price trend, price distribution, transaction table, and CSV export
- Robust CSV preprocessing with an included demo-data fallback
- Curated HDB document ingestion, token-aware chunking, and FAISS retrieval foundation
- About Us and Methodology pages

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Sign in with:

- Username: `admin`
- Password: `r3ady4r3sale=`

This authentication is for demonstration of prototype only.

## Data

The app selects data in this order:

1. The official dataset at `data/structured/hdb_resale_transactions.csv`.
2. The included `data/demo_resale_transactions.csv` fallback if the official file cannot be loaded.

Expected columns are `month`, `town`, `flat_type`, `block`, `street_name`, `storey_range`, `floor_area_sqm`, `flat_model`, `lease_commence_date`, and `resale_price`. Extra columns, including `remaining_lease`, are allowed. Rows with missing, malformed, or non-positive core values are skipped and reported in the interface.

## RAG document ingestion

Curated official HDB PDF and TXT files live in `data/rag_sources/`. Each file must have a matching entry in `data/rag_sources/manifest.json` so its title, source organisation, official URL, and optional cleaning boundaries remain explicit and reviewable.

First validate text extraction and chunking without making an API request:

```bash
python scripts/build_vector_store.py --dry-run
```

To build the FAISS index, provide an OpenAI API key in your shell and run the deterministic build script:

```bash
export OPENAI_API_KEY="your-key"
python scripts/build_vector_store.py
```

The key must remain outside the repository. The generated index, aligned chunk metadata, and build manifest are written to `data/vector_store/` and are ignored by Git. The default configuration uses `text-embedding-3-small`, 850-token chunks, and 120-token overlap.

Retrieve the most relevant chunks from Python after building the index:

```python
from src.rag import retrieve

results = retrieve("When should I request a valuation?", top_k=3)
for result in results:
    print(result.score, result.metadata["document_title"])
```

## Project structure

```text
app.py                                      # Entry point, navigation, authentication gate
src/auth.py                                 # Demo login and session state
src/data.py                                 # Loading, validation, filtering, and aggregation
src/pages/                                  # Page-level UI modules
src/rag/                                    # Ingestion, embeddings, FAISS store, and retrieval
scripts/build_vector_store.py               # Reproducible vector-store build command
tests/test_data.py                          # Deterministic data-pipeline tests
tests/test_rag_ingestion.py                 # Offline ingestion and retrieval tests
data/structured/hdb_resale_transactions.csv # Official HDB resale transactions
data/demo_resale_transactions.csv           # Development/error fallback
data/rag_sources/                           # Curated official HDB PDF/TXT documents
data/vector_store/                          # Generated retrieval index and metadata
.streamlit/config.toml
requirements.txt
```

## Validation

Run the deterministic tests with:

```bash
python -m unittest discover -s tests
```

## Scope and limitations

The Market Explorer is based on and only describes historical transactions. It does not predict prices, establish market value, calculate affordability, determine HDB eligibility, or provide financial or legal advice. If in doubt, please verify any information with official sources.
