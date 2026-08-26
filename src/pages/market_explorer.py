import pandas as pd
import streamlit as st

from src.data import (
    calculate_summary,
    filter_transactions,
    load_transactions,
    monthly_median_trend,
    resale_price_distribution,
    valid_flat_types,
    valid_towns,
)


def _money(value: float) -> str:
    return f"S${value:,.0f}"


def _default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def render() -> None:
    st.title("📊 HDB Resale Market Explorer")
    st.caption(
        "Explore official historical transactions. Past prices do not predict future prices "
        "or establish a flat's valuation."
    )

    with st.expander("Development data override"):
        uploaded = st.file_uploader(
            "Optional HDB resale CSV",
            type="csv",
            help="The official structured dataset is used by default. Uploading a file overrides it for this session.",
        )

    try:
        result = load_transactions(uploaded)
    except (ValueError, RuntimeError, OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Transaction data could not be loaded. {error}")
        return

    if result.is_demo:
        st.warning(
            f"Development fallback: using **{result.source_name}** because the official dataset "
            f"could not be loaded ({result.fallback_reason})."
        )
    else:
        st.success(f"Loaded **{result.source_name}** ({len(result.data):,} valid transactions).")
    if result.rows_dropped:
        st.warning(f"Skipped {result.rows_dropped:,} malformed or incomplete rows during preprocessing.")

    frame = result.data
    min_date = frame["month"].min().date()
    max_date = frame["month"].max().date()
    price_step = 10_000
    price_floor = int(frame["resale_price"].min() // price_step * price_step)
    price_ceiling = int(frame["resale_price"].max() // price_step * price_step + price_step)
    budget_options = list(range(price_floor, price_ceiling + price_step, price_step))
    median_budget = int(round(frame["resale_price"].median() / price_step) * price_step)

    with st.sidebar:
        st.subheader("Explorer filters")
        town_options = valid_towns(frame)
        town = st.selectbox("Town", town_options)
        flat_type_options = valid_flat_types(frame, town)
        flat_type = st.selectbox(
            "Flat type",
            flat_type_options,
            index=_default_index(flat_type_options, "4 ROOM"),
        )
        budget = st.select_slider(
            "Approximate budget",
            options=budget_options,
            value=min(max(median_budget, price_floor), price_ceiling),
            format_func=_money,
        )
        period = st.date_input(
            "Period of interest",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    if len(period) != 2:
        st.info("Choose both a start and end date for the period of interest.")
        return

    filtered = filter_transactions(frame, town, flat_type, period[0], period[1])
    if filtered.empty:
        st.warning(
            "No historical transactions match this town, flat type, and period. "
            "Try widening the period or selecting a different combination."
        )
        return

    summary = calculate_summary(filtered, budget)
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{summary.transaction_count:,}")
    b.metric("Median resale price", _money(summary.median_resale_price))
    c.metric("Median floor area", f"{summary.median_floor_area_sqm:,.0f} m²")
    d.metric(
        "At or below budget",
        f"{summary.within_budget_percentage:.1f}%",
        help=f"Share of matching transactions priced at or below {_money(budget)}.",
    )
    st.caption(
        f"Results match **{town} · {flat_type} · {period[0]:%b %Y} to {period[1]:%b %Y}**. "
        f"The selected budget of **{_money(budget)}** is used for the affordability comparison."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly median resale-price trend")
        trend = monthly_median_trend(filtered)
        st.line_chart(
            trend,
            x="period",
            y="resale_price",
            x_label="Month",
            y_label="Median resale price (S$)",
        )
    with right:
        st.subheader("Resale-price distribution")
        distribution = resale_price_distribution(filtered)
        st.bar_chart(
            distribution,
            x="price_band",
            y="transactions",
            x_label="Resale-price band",
            y_label="Transactions",
        )

    st.subheader("Filtered transactions")
    filtered["within_budget"] = filtered["resale_price"].le(budget)
    columns = [
        "month",
        "town",
        "flat_type",
        "address",
        "storey_range",
        "floor_area_sqm",
        "flat_model",
        "lease_commence_date",
    ]
    if "remaining_lease" in filtered.columns:
        columns.append("remaining_lease")
    columns.extend(["resale_price", "price_per_sqm", "within_budget"])
    table = filtered[columns].copy()
    table["month"] = table["month"].dt.strftime("%Y-%m")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "month": "Month",
            "town": "Town",
            "flat_type": "Flat type",
            "address": "Address",
            "storey_range": "Storey",
            "floor_area_sqm": st.column_config.NumberColumn("Area (m²)", format="%.0f"),
            "flat_model": "Flat model",
            "lease_commence_date": "Lease commenced",
            "remaining_lease": "Remaining lease",
            "resale_price": st.column_config.NumberColumn("Resale price", format="S$ %,.0f"),
            "price_per_sqm": st.column_config.NumberColumn("Price / m²", format="S$ %,.0f"),
            "within_budget": st.column_config.CheckboxColumn("Within budget"),
        },
    )
    st.download_button(
        "Download filtered CSV",
        table.to_csv(index=False).encode("utf-8"),
        "resaleready_filtered_transactions.csv",
        "text/csv",
    )
