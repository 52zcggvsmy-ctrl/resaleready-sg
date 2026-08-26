"""Loading, validating, and analysing HDB resale transaction data."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DATA_PATH = PROJECT_ROOT / "data" / "structured" / "hdb_resale_transactions.csv"
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "demo_resale_transactions.csv"
REQUIRED_COLUMNS = {
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "resale_price",
}
TEXT_COLUMNS = {"town", "flat_type", "block", "street_name", "storey_range", "flat_model"}


@dataclass(frozen=True)
class LoadResult:
    data: pd.DataFrame
    source_name: str
    is_demo: bool
    rows_dropped: int = 0
    fallback_reason: str | None = None


@dataclass(frozen=True)
class SummaryMetrics:
    transaction_count: int
    median_resale_price: float
    median_floor_area_sqm: float
    within_budget_percentage: float


def preprocess_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalise a transaction frame and discard unusable rows."""
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("CSV contains duplicate column names.")

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    input_rows = len(frame)
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()

    frame["town"] = frame["town"].str.upper()
    frame["flat_type"] = frame["flat_type"].str.upper()
    frame["month"] = pd.to_datetime(frame["month"], format="%Y-%m", errors="coerce")
    frame["resale_price"] = pd.to_numeric(frame["resale_price"], errors="coerce")
    frame["floor_area_sqm"] = pd.to_numeric(frame["floor_area_sqm"], errors="coerce")
    frame["lease_commence_date"] = pd.to_numeric(frame["lease_commence_date"], errors="coerce")

    missing_values = frame[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    blank_text = frame[list(TEXT_COLUMNS)].eq("").any(axis=1)
    invalid_numbers = (
        frame["resale_price"].le(0)
        | frame["floor_area_sqm"].le(0)
        | frame["lease_commence_date"].le(0)
    )
    frame = frame.loc[~(missing_values | blank_text | invalid_numbers)].copy()

    if frame.empty:
        raise ValueError("CSV does not contain any valid HDB resale transactions.")

    frame["lease_commence_date"] = frame["lease_commence_date"].astype("int64")
    if "remaining_lease" in frame.columns:
        frame["remaining_lease"] = frame["remaining_lease"].astype("string").str.strip()

    frame["address"] = frame["block"].str.cat(frame["street_name"], sep=" ")
    frame["price_per_sqm"] = frame["resale_price"] / frame["floor_area_sqm"]
    frame = frame.sort_values("month", ascending=False).reset_index(drop=True)
    return frame, input_rows - len(frame)


@st.cache_data(show_spinner=False)
def load_csv(path: str, modified_time_ns: int | None = None) -> tuple[pd.DataFrame, int]:
    """Load and preprocess a file, invalidating cache when it changes."""
    del modified_time_ns
    return preprocess_transactions(pd.read_csv(path, low_memory=False))


@st.cache_data(show_spinner=False)
def load_uploaded_csv(contents: bytes) -> tuple[pd.DataFrame, int]:
    return preprocess_transactions(pd.read_csv(BytesIO(contents), low_memory=False))


def _load_path(path: Path) -> tuple[pd.DataFrame, int]:
    return load_csv(str(path), path.stat().st_mtime_ns)


def load_transactions(uploaded_file=None) -> LoadResult:
    """Use an upload when supplied, otherwise official data with demo fallback."""
    if uploaded_file is not None:
        data, rows_dropped = load_uploaded_csv(uploaded_file.getvalue())
        return LoadResult(data, uploaded_file.name, False, rows_dropped)

    try:
        data, rows_dropped = _load_path(OFFICIAL_DATA_PATH)
        source_name = str(OFFICIAL_DATA_PATH.relative_to(PROJECT_ROOT))
        return LoadResult(data, source_name, False, rows_dropped)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        try:
            data, rows_dropped = _load_path(DEMO_DATA_PATH)
        except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as fallback_error:
            raise RuntimeError(
                "Neither the official transaction dataset nor the demo fallback could be loaded."
            ) from fallback_error
        return LoadResult(
            data,
            str(DEMO_DATA_PATH.relative_to(PROJECT_ROOT)),
            True,
            rows_dropped,
            str(error),
        )


def valid_towns(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["town"].dropna().unique().tolist())


def valid_flat_types(frame: pd.DataFrame, town: str | None = None) -> list[str]:
    relevant = frame if town is None else frame.loc[frame["town"].eq(town)]
    return sorted(relevant["flat_type"].dropna().unique().tolist())


def filter_transactions(
    frame: pd.DataFrame,
    town: str,
    flat_type: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (
        frame["town"].eq(town)
        & frame["flat_type"].eq(flat_type)
        & frame["month"].between(start, end)
    )
    return frame.loc[mask].copy()


def calculate_summary(frame: pd.DataFrame, budget: float) -> SummaryMetrics:
    if frame.empty:
        raise ValueError("Cannot calculate summary metrics for an empty dataset.")
    return SummaryMetrics(
        transaction_count=len(frame),
        median_resale_price=float(frame["resale_price"].median()),
        median_floor_area_sqm=float(frame["floor_area_sqm"].median()),
        within_budget_percentage=float(frame["resale_price"].le(budget).mean() * 100),
    )


def monthly_median_trend(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(period=frame["month"].dt.to_period("M").dt.to_timestamp())
        .groupby("period", as_index=False)["resale_price"]
        .median()
        .sort_values("period")
    )


def resale_price_distribution(frame: pd.DataFrame, bin_width: int = 50_000) -> pd.DataFrame:
    minimum = int(frame["resale_price"].min() // bin_width * bin_width)
    maximum = int(frame["resale_price"].max() // bin_width * bin_width + bin_width)
    edges = list(range(minimum, maximum + bin_width, bin_width))
    labels = [f"S${left // 1_000}k–{right // 1_000}k" for left, right in zip(edges, edges[1:])]
    bands = pd.cut(
        frame["resale_price"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    distribution = bands.value_counts(sort=False).rename_axis("price_band").reset_index(name="transactions")
    return distribution.loc[distribution["transactions"].gt(0)].reset_index(drop=True)
