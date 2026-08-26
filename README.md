# ResaleReady SG

A Streamlit prototype for prospective HDB resale flat buyers in Singapore.

## Current features

- Prototype single-user login with session-state authentication
- Official HDB resale transaction filtering by town, flat type, period, and approximate budget
- Summary metrics, monthly median-price trend, price distribution, transaction table, and CSV export
- Optional AI explanation using only selected filters and Pandas-computed statistics
- Robust CSV preprocessing with an included demo-data fallback
- Curated HDB document ingestion, token-aware chunking, and FAISS retrieval foundation
- Conversation-aware, source-grounded HDB resale Q&A
- Layered domain, privacy, prompt-injection, disclosure, grounding, and output safeguards
- Administrative PDF/TXT upload demonstration with a session-only FAISS index
- About Us and Methodology pages

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Sign in using the prototype credentials shared separately with the assessor. This
authentication is for demonstration of prototype only.

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

## Ask ResaleReady configuration

The Q&A service uses the OpenAI Responses API for follow-up rewriting and grounded answers. Set the API key outside the repository. For local use:

```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

For Streamlit Community Cloud, add the following under the deployed app's **Settings → Secrets**:

```toml
OPENAI_API_KEY = "your-key"
RESALEREADY_CHAT_MODEL = "gpt-5.6-luna"
```

The model setting is optional. If the generated FAISS files are absent, the app builds the small curated knowledge base on the first Q&A or upload request and reuses it for later requests in the same deployment instance.

## Streamlit Community Cloud deployment

1. Push the current repository to GitHub and confirm that `app.py`, `requirements.txt`, the official transaction CSV, and all files under `data/rag_sources/` are present.
2. In [Streamlit Community Cloud](https://share.streamlit.io), choose **Create app** and select the repository, the deployment branch, and `app.py` as the entrypoint.
3. Under **Advanced settings**, select Python 3.11 and paste the secrets block shown above. Do not add `.streamlit/secrets.toml` to Git; it is already covered by `.gitignore`.
4. Deploy the app and wait for the Python dependencies in `requirements.txt` to install. No `packages.txt` system dependencies are required.
5. Sign in with the demo credentials and open every page. The first Q&A or document-upload request may take longer because the small FAISS knowledge base is built when generated index files are absent.
6. Confirm that Market Explorer metrics and charts load, Ask ResaleReady answers a normal HDB question with supporting sources, safeguard examples are refused, and the About Us and Methodology diagrams render.

Generated FAISS files and session uploads are intentionally ephemeral on Community Cloud. Curated PDF/TXT sources remain in Git, so the index can be rebuilt after an app restart.

## Knowledge Base upload demo

After signing in, open **Knowledge Base** to upload PDF or UTF-8 TXT files. Each file is limited to 10 MB and passes through validation, text extraction, cleaning, 850-token chunking with 120-token overlap, OpenAI embedding, and addition to an in-memory FAISS index.

Uploaded documents are unverified, session-only reference material. Their original files are not written to the repository, they are removed on logout or when the session/app ends, and they are never treated as executable instructions or official HDB policy. During Q&A, curated official HDB sources keep the majority of retrieval slots and appear before uploaded references.

## LLM safeguards

Ask ResaleReady uses layered safeguards rather than relying on keyword blocking alone:

- deterministic domain, input-structure, privacy, injection, and disclosure checks before model access;
- conversation-aware validation for legitimate follow-up questions;
- stable model instructions separated from an untrusted JSON payload containing the question and retrieved sources;
- explicit treatment of document text and metadata as reference data, never instructions;
- no API key, credential, environment, or configuration values included in model input;
- valid retrieved-source citations required for supported answers;
- deterministic fallback for absent, insufficient, uncited, or invalidly cited evidence; and
- generated-output screening for secret-like values and internal-policy leakage.

The complete requirement mapping, expected adversarial behaviour, limitations, and manual procedure are documented in `docs/SAFEGUARDS.md`.

Run the documented safeguard set without an API key:

```bash
python scripts/run_safeguard_checks.py
```

## Project structure

```text
app.py                                      # Entry point, navigation, authentication gate
src/app_config.py                           # Streamlit secret/environment configuration
src/auth.py                                 # Demo login and session state
src/data.py                                 # Loading, validation, filtering, and aggregation
src/market_explanation.py                   # Statistics-only optional market explanation
src/openai_client.py                        # OpenAI Responses API and safety identifier boundary
src/prompts.py                              # Isolated rewrite and grounded-answer instructions
src/pages/knowledge_base.py                 # Administrative upload demonstration
src/pages/                                  # Other page-level UI modules
src/rag/safeguards.py                       # Deterministic input and output safeguards
src/rag/uploads.py                          # Upload validation and in-memory FAISS index
src/rag/                                    # Ingestion, retrieval, safeguards, and Q&A chain
scripts/build_vector_store.py               # Reproducible curated vector-store build command
scripts/run_safeguard_checks.py             # Offline documented adversarial checks
tests/safeguard_cases.json                  # Machine-readable safeguard examples
tests/test_data.py                          # Deterministic data-pipeline tests
tests/test_rag_ingestion.py                 # Offline ingestion and retrieval tests
tests/test_rag_qa.py                        # Offline prompt-chain tests
tests/test_rag_safeguards.py                # Offline adversarial and fallback tests
tests/test_rag_uploads.py                   # Offline upload validation and retrieval tests
docs/SAFEGUARDS.md                          # Safeguard design and expected behaviour
data/structured/hdb_resale_transactions.csv # Official HDB resale transactions
data/demo_resale_transactions.csv           # Development/error fallback
data/rag_sources/                           # Curated official HDB PDF/TXT documents
data/vector_store/                          # Generated curated retrieval index and metadata
.streamlit/config.toml
requirements.txt
```

## Validation

Run the deterministic tests with:

```bash
python -m unittest discover -s tests
```

## Scope and limitations

The Market Explorer is based on and only describes historical transactions. It does not predict prices, establish market value, calculate affordability, determine HDB eligibility, or provide financial or legal advice. Ask ResaleReady is limited to general buyer-side HDB resale information grounded primarily in its curated HDB knowledge base and does not make eligibility decisions or replace official HDB guidance. Uploaded demo documents are unverified references and must be checked against official sources. Safeguards reduce but cannot eliminate LLM error or prompt-injection risk. If in doubt, please verify any information with official sources.
