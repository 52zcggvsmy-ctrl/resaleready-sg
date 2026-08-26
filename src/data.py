"""Loading, normalising, and filtering HDB resale transaction data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "resale_transactions.csv"
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "demo_resale_transactions.csv"
REQUIRED_COLUMNS = {"month", "town", "flat_type", "block", "street_name", "storey_range", "floor_area_sqm", "flat_model", "lease_commence_date", "resale_price"}


@dataclass(frozen=True)
class LoadResult:
    data: pd.DataFrame
    source_name: str
    is_demo: bool


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    frame["month"] = pd.to_datetime(frame["month"], format="%Y-%m", errors="coerce")
    frame["resale_price"] = pd.to_numeric(frame["resale_price"], errors="coerce")
    frame["floor_area_sqm"] = pd.to_numeric(frame["floor_area_sqm"], errors="coerce")
    frame = frame.dropna(subset=["month", "town", "flat_type", "resale_price", "floor_area_sqm"])
    frame["price_per_sqm"] = frame["resale_price"] / frame["floor_area_sqm"]
    frame["address"] = frame["block"].astype(str) + " " + frame["street_name"].astype(str)
    return frame.sort_values("month", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return _normalise(pd.read_csv(path))


def load_transactions(uploaded_file=None) -> LoadResult:
    if uploaded_file is not None:
        return LoadResult(_normalise(pd.read_csv(uploaded_file)), uploaded_file.name, False)
    if DEFAULT_DATA_PATH.exists():
        return LoadResult(load_csv(str(DEFAULT_DATA_PATH)), DEFAULT_DATA_PATH.name, False)
    return LoadResult(load_csv(str(DEMO_DATA_PATH)), DEMO_DATA_PATH.name, True)


def filter_transactions(
    frame: pd.DataFrame,
    towns: list[str],
    flat_types: list[str],
    start_date,
    end_date,
    min_price: float,
    max_price: float,
) -> pd.DataFrame:
    mask = (
        frame["town"].isin(towns)
        & frame["flat_type"].isin(flat_types)
        & frame["month"].dt.date.between(start_date, end_date)
        & frame["resale_price"].between(min_price, max_price)
    )
    return frame.loc[mask].copy()
