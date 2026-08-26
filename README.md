# ResaleReady SG

A Streamlit prototype for prospective HDB resale flat buyers in Singapore.

## Current features

- Prototype single-user login with session-state authentication
- Official HDB resale transaction filtering by town, flat type, period, and approximate budget
- Summary metrics, monthly median-price trend, price distribution, transaction table, and CSV export
- Robust CSV preprocessing with an included demo-data fallback
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

1. A CSV supplied through the development override on the Market Explorer page.
2. The official dataset at `data/structured/hdb_resale_transactions.csv`.
3. The included `data/demo_resale_transactions.csv` fallback if the official file cannot be loaded.

Expected columns are `month`, `town`, `flat_type`, `block`, `street_name`, `storey_range`, `floor_area_sqm`, `flat_model`, `lease_commence_date`, and `resale_price`. Extra columns, including `remaining_lease`, are allowed. Rows with missing, malformed, or non-positive core values are skipped and reported in the interface.

## Project structure

```text
app.py                                      # Entry point, navigation, authentication gate
src/auth.py                                 # Demo login and session state
src/data.py                                 # Loading, validation, filtering, and aggregation
src/pages/                                  # Page-level UI modules
src/rag/                                    # Reserved boundary for later RAG integration
tests/test_data.py                          # Deterministic data-pipeline tests
data/structured/hdb_resale_transactions.csv # Official HDB resale transactions
data/demo_resale_transactions.csv           # Development/error fallback
.streamlit/config.toml
requirements.txt
```

## Validation

Run the deterministic data tests with:

```bash
python -m unittest discover -s tests
```

## Scope and limitations

The Market Explorer is based on and only describes historical transactions. It does not predict prices, establish market value, calculate affordability, determine HDB eligibility, or provide financial or legal advice. If in doubt, please verify any information with official sources.
