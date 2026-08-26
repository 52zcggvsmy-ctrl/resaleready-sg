import pandas as pd
import streamlit as st

from src.data import filter_transactions, load_transactions


def _money(value: float) -> str:
    return f"S${value:,.0f}"


def render() -> None:
    st.title("📊 HDB Resale Market Explorer")
    st.caption("Explore historical transactions. Past prices do not predict future prices or establish a flat's valuation.")

    uploaded = st.file_uploader("Optional: use an HDB resale CSV", type="csv", help="If omitted, data/resale_transactions.csv is used when present; otherwise the demo file is loaded.")
    try:
        result = load_transactions(uploaded)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"The CSV could not be loaded. {error}")
        return

    if result.is_demo:
        st.info(f"Demo mode: using **{result.source_name}**. Add `data/resale_transactions.csv` to use the full dataset.")
    else:
        st.success(f"Loaded **{result.source_name}** ({len(result.data):,} transactions).")

    frame = result.data
    min_date, max_date = frame["month"].min().date(), frame["month"].max().date()
    price_floor = int(frame["resale_price"].min() // 10_000 * 10_000)
    price_ceiling = int((frame["resale_price"].max() // 10_000 + 1) * 10_000)

    with st.sidebar:
        st.subheader("Explorer filters")
        towns = st.multiselect("Town", sorted(frame["town"].unique()), default=sorted(frame["town"].unique()))
        flat_types = st.multiselect("Flat type", sorted(frame["flat_type"].unique()), default=sorted(frame["flat_type"].unique()))
        period = st.date_input("Transaction period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        budget = st.slider("Resale price (S$)", price_floor, price_ceiling, (price_floor, price_ceiling), step=10_000)

    if len(period) != 2:
        st.info("Choose both a start and end date.")
        return
    if not towns or not flat_types:
        st.info("Select at least one town and one flat type.")
        return

    filtered = filter_transactions(frame, towns, flat_types, period[0], period[1], budget[0], budget[1])
    if filtered.empty:
        st.warning("No transactions match these filters. Try widening the period or budget.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{len(filtered):,}")
    b.metric("Median price", _money(filtered["resale_price"].median()))
    c.metric("Average price", _money(filtered["resale_price"].mean()))
    d.metric("Median floor area", f"{filtered['floor_area_sqm'].median():,.0f} m²")

    left, right = st.columns(2)
    with left:
        st.subheader("Median resale price over time")
        trend = filtered.assign(period=filtered["month"].dt.to_period("M").dt.to_timestamp()).groupby("period", as_index=False)["resale_price"].median()
        st.line_chart(trend, x="period", y="resale_price", x_label="Month", y_label="Median price (S$)")
    with right:
        st.subheader("Median resale price by town")
        by_town = filtered.groupby("town", as_index=False)["resale_price"].median().sort_values("resale_price")
        st.bar_chart(by_town, x="town", y="resale_price", x_label="Town", y_label="Median price (S$)", horizontal=True)

    st.subheader("Matching transactions")
    table = filtered[["month", "town", "flat_type", "address", "storey_range", "floor_area_sqm", "flat_model", "lease_commence_date", "resale_price", "price_per_sqm"]].copy()
    table["month"] = table["month"].dt.strftime("%Y-%m")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "month": "Month", "town": "Town", "flat_type": "Flat type", "address": "Address",
            "storey_range": "Storey", "floor_area_sqm": st.column_config.NumberColumn("Area (m²)", format="%.0f"),
            "flat_model": "Flat model", "lease_commence_date": "Lease commenced",
            "resale_price": st.column_config.NumberColumn("Resale price", format="S$ %,.0f"),
            "price_per_sqm": st.column_config.NumberColumn("Price / m²", format="S$ %,.0f"),
        },
    )
    st.download_button("Download filtered CSV", table.to_csv(index=False).encode("utf-8"), "resaleready_filtered_transactions.csv", "text/csv")
