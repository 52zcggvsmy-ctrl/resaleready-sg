# ResaleReady SG

A Streamlit prototype for prospective HDB resale flat buyers in Singapore.

## Current features

- Prototype single-user login with session-state authentication
- Historical transaction filtering by town, flat type, period, and price
- Summary metrics, price visualisations, transaction table, and CSV export
- CSV upload with an included demo-data fallback
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

1. A CSV uploaded on the Market Explorer page.
2. `data/resale_transactions.csv`, if present.
3. The included `data/demo_resale_transactions.csv` fallback.

Download the official HDB resale flat prices CSV from data.gov.sg and save it as `data/resale_transactions.csv`. It is intentionally ignored by Git because the full dataset can be large and may be refreshed independently.

Expected columns are `month`, `town`, `flat_type`, `block`, `street_name`, `storey_range`, `floor_area_sqm`, `flat_model`, `lease_commence_date`, and `resale_price`. Extra columns, including `remaining_lease`, are allowed.

## Project structure

```text
app.py                      # Entry point, navigation, authentication gate
src/auth.py                 # Demo login and session state
src/data.py                 # CSV loading, validation, normalisation, filtering
src/pages/                  # Page-level UI modules
src/rag/                    # Reserved boundary for later RAG integration
data/demo_resale_transactions.csv
.streamlit/config.toml
requirements.txt
```

## Scope and limitations

The Market Explorer is based on and only describes historical transactions. It does not predict prices, establish market value, calculate affordability, determine HDB eligibility, or provide financial or legal advice. If in doubt, please verify any information with official sources.
