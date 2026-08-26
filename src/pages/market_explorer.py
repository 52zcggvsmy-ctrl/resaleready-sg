import logging

import pandas as pd
import streamlit as st

from src.app_config import secret_or_environment
from src.data import (
    calculate_summary,
    filter_transactions,
    load_transactions,
    monthly_median_trend,
    resale_price_distribution,
    valid_flat_types,
    valid_towns,
)
from src.market_explanation import (
    MarketExplanationContext,
    MarketExplanationError,
    generate_market_explanation,
)
from src.openai_client import OpenAIResponsesClient


FILTER_KEYS = (
    "market_town",
    "market_flat_type",
    "market_budget",
    "market_period",
)
MARKET_EXPLANATION_STATE_KEY = "market_ai_explanation"
DEFAULT_CHAT_MODEL = "gpt-5.6-luna"
LOGGER = logging.getLogger(__name__)


def _money(value: float) -> str:
    return f"S${value:,.0f}"


def _default_option(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else options[0]


def _reset_filters(defaults: dict) -> None:
    for key in FILTER_KEYS:
        st.session_state[key] = defaults[key]
    st.session_state.pop(MARKET_EXPLANATION_STATE_KEY, None)


@st.cache_resource(show_spinner=False)
def _load_explanation_client(chat_model: str, _api_key: str) -> OpenAIResponsesClient:
    return OpenAIResponsesClient(api_key=_api_key, model=chat_model)


def _render_optional_explanation(context: MarketExplanationContext) -> None:
    st.subheader("Optional AI explanation")
    st.write(
        "Ask AI to explain the figures already calculated above. The model receives "
        "only your selected filters and the displayed summary statistics—not individual "
        "transaction records—and it does not recalculate them."
    )
    st.caption(
        "Historical transaction data is indicative only. This explanation is not a "
        "property valuation, price forecast, purchase recommendation, or financial advice."
    )

    api_key = secret_or_environment("OPENAI_API_KEY")
    chat_model = secret_or_environment("RESALEREADY_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    if not api_key:
        st.info(
            "Configure `OPENAI_API_KEY` in Streamlit Secrets or the local environment "
            "to enable this optional explanation."
        )

    if st.button(
        "✨ Explain these results",
        disabled=not api_key,
        help="Generates a short explanation from the displayed filters and statistics.",
    ):
        st.session_state.pop(MARKET_EXPLANATION_STATE_KEY, None)
        try:
            with st.spinner("Explaining the historical results..."):
                client = _load_explanation_client(chat_model, api_key)
                explanation = generate_market_explanation(client, context)
        except (MarketExplanationError, RuntimeError, ValueError):
            LOGGER.exception("Unable to generate a validated market explanation.")
            st.error(
                "A safe explanation could not be generated just now. The calculated "
                "metrics and charts above remain available and unchanged."
            )
        except Exception:
            LOGGER.exception("OpenAI market explanation request failed.")
            st.error(
                "The optional AI explanation is temporarily unavailable. The calculated "
                "metrics and charts above remain available and unchanged."
            )
        else:
            st.session_state[MARKET_EXPLANATION_STATE_KEY] = {
                "signature": context.signature,
                "content": explanation,
            }

    stored = st.session_state.get(MARKET_EXPLANATION_STATE_KEY)
    if isinstance(stored, dict) and stored.get("signature") == context.signature:
        with st.container(border=True):
            st.markdown("**AI explanation of indicative historical transactions**")
            st.write(str(stored.get("content", "")))
            st.caption(
                "The explanation interprets fixed Pandas results; it does not alter any "
                "calculation shown on this page."
            )


def _initialise_filter_state(frame: pd.DataFrame) -> dict:
    town_options = valid_towns(frame)
    default_town = _default_option(town_options, "ANG MO KIO")
    default_flat_type = _default_option(valid_flat_types(frame, default_town), "4 ROOM")

    price_step = 10_000
    price_floor = int(frame["resale_price"].min() // price_step * price_step)
    price_ceiling = int(frame["resale_price"].max() // price_step * price_step + price_step)
    budget_options = list(range(price_floor, price_ceiling + price_step, price_step))
    default_budget = int(round(frame["resale_price"].median() / price_step) * price_step)
    default_budget = min(max(default_budget, price_floor), price_ceiling)

    month_options = sorted(frame["month"].drop_duplicates().tolist())
    defaults = {
        "market_town": default_town,
        "market_flat_type": default_flat_type,
        "market_budget": default_budget,
        "market_period": (month_options[0], month_options[-1]),
    }

    if st.session_state.get("market_town") not in town_options:
        st.session_state.market_town = default_town
    current_flat_types = valid_flat_types(frame, st.session_state.market_town)
    if st.session_state.get("market_flat_type") not in current_flat_types:
        st.session_state.market_flat_type = _default_option(current_flat_types, "4 ROOM")
    if st.session_state.get("market_budget") not in budget_options:
        st.session_state.market_budget = default_budget
    current_period = st.session_state.get("market_period")
    if (
        not isinstance(current_period, (tuple, list))
        or len(current_period) != 2
        or any(value not in month_options for value in current_period)
    ):
        st.session_state.market_period = defaults["market_period"]

    return {
        "town_options": town_options,
        "budget_options": budget_options,
        "month_options": month_options,
        "defaults": defaults,
    }


def render() -> None:
    st.title("📊 HDB Resale Market Explorer")
    st.write(
        "See what similar HDB resale flats have sold for using official historical "
        "transaction records. Choose a town, flat type, budget, and period to explore the market."
    )
    st.warning(
        "Historical resale transactions are indicative only. They are not property valuations "
        "or price predictions and should not be the sole basis for a purchase decision."
    )

    try:
        with st.spinner("Loading HDB resale transactions…", show_time=True):
            result = load_transactions()
    except (ValueError, RuntimeError, OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Transaction data could not be loaded. {error}")
        st.info("Confirm that the official transaction CSV is available, then reload the page.")
        return

    if result.is_demo:
        st.warning(
            f"Development fallback: using **{result.source_name}** because the official dataset "
            f"could not be loaded ({result.fallback_reason})."
        )
    if result.rows_dropped:
        st.warning(f"Skipped {result.rows_dropped:,} malformed or incomplete rows during preprocessing.")

    frame = result.data
    dataset_start = frame["month"].min()
    dataset_end = frame["month"].max()
    source_label = "demo fallback" if result.is_demo else "official HDB resale transactions"
    st.caption(
        f"Data source: {source_label} · {len(frame):,} valid records · "
        f"{dataset_start:%B %Y} to {dataset_end:%B %Y}"
    )

    filter_state = _initialise_filter_state(frame)
    header, reset = st.columns([5, 1])
    with header:
        st.subheader("Find relevant past transactions")
        st.caption("Filters update the results below. Budget is used for comparison, not to hide higher-priced sales.")
    with reset:
        st.button(
            "↺ Reset filters",
            on_click=_reset_filters,
            args=(filter_state["defaults"],),
            width="stretch",
        )

    with st.container(border=True):
        first, second, third = st.columns(3)
        with first:
            town = st.selectbox(
                "Town",
                filter_state["town_options"],
                key="market_town",
                help="Select the HDB town you are considering.",
            )
        flat_type_options = valid_flat_types(frame, town)
        if st.session_state.market_flat_type not in flat_type_options:
            st.session_state.market_flat_type = _default_option(flat_type_options, "4 ROOM")
        with second:
            flat_type = st.selectbox(
                "Flat type",
                flat_type_options,
                key="market_flat_type",
                help="Only flat types with transactions in the selected town are shown.",
            )
        with third:
            budget = st.select_slider(
                "Approximate budget",
                options=filter_state["budget_options"],
                format_func=_money,
                key="market_budget",
                help="Used to calculate the share of matching sales at or below your budget.",
            )
        period = st.select_slider(
            "Transaction period",
            options=filter_state["month_options"],
            format_func=lambda value: value.strftime("%b %Y"),
            key="market_period",
            help="Choose the first and last transaction months to include.",
        )
        st.caption(f"Selected period: **{period[0]:%B %Y} to {period[1]:%B %Y}**")

    filtered = filter_transactions(frame, town, flat_type, period[0], period[1])
    if filtered.empty:
        st.info(
            "No recorded transactions match this town, flat type, and period. "
            "Try a wider period, another flat type, or reset the filters."
        )
        return

    summary = calculate_summary(filtered, budget)
    within_budget_count = int(filtered["resale_price"].le(budget).sum())

    st.subheader("At a glance")
    a, b, c, d = st.columns(4)
    a.metric("Matching transactions", f"{summary.transaction_count:,}")
    b.metric("Median resale price", _money(summary.median_resale_price))
    c.metric("Median floor area", f"{summary.median_floor_area_sqm:,.0f} m²")
    d.metric(
        "Sales at or below budget",
        f"{summary.within_budget_percentage:.1f}%",
        help=f"{within_budget_count:,} of {summary.transaction_count:,} matching transactions were at or below {_money(budget)}.",
    )
    st.caption(
        f"{within_budget_count:,} of {summary.transaction_count:,} matching transactions sold at or below "
        f"your selected budget of **{_money(budget)}**."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly median resale price")
        st.caption("The median transacted price for each month with at least one matching sale.")
        trend = monthly_median_trend(filtered)
        st.line_chart(
            trend,
            x="period",
            y="resale_price",
            x_label="Transaction month",
            y_label="Median resale price (S$)",
        )
    with right:
        st.subheader("Resale-price distribution")
        st.caption("Number of matching transactions in each S$50,000 price band.")
        distribution = resale_price_distribution(filtered)
        st.bar_chart(
            distribution,
            x="price_band",
            y="transactions",
            x_label="Resale-price band",
            y_label="Transactions",
        )

    explanation_context = MarketExplanationContext(
        town=town,
        flat_type=flat_type,
        period_start=period[0].strftime("%B %Y"),
        period_end=period[1].strftime("%B %Y"),
        budget=float(budget),
        transaction_count=summary.transaction_count,
        median_resale_price=summary.median_resale_price,
        median_floor_area_sqm=summary.median_floor_area_sqm,
        within_budget_count=within_budget_count,
        within_budget_percentage=summary.within_budget_percentage,
    )
    _render_optional_explanation(explanation_context)

    st.subheader("Transaction details")
    st.caption(
        f"Showing {len(filtered):,} matching records, newest first. "
        "Use the table controls to search, sort, or choose visible columns."
    )
    filtered["budget_status"] = filtered["resale_price"].le(budget).map(
        {True: "Within budget", False: "Above budget"}
    )
    columns = [
        "month",
        "address",
        "storey_range",
        "floor_area_sqm",
        "flat_model",
        "lease_commence_date",
    ]
    if "remaining_lease" in filtered.columns:
        columns.append("remaining_lease")
    columns.extend(["resale_price", "price_per_sqm", "budget_status"])
    table = filtered[columns].copy()
    table["month"] = table["month"].dt.strftime("%b %Y")
    st.dataframe(
        table,
        width="stretch",
        height=520,
        hide_index=True,
        column_config={
            "month": "Transaction month",
            "address": "Address",
            "storey_range": "Storey",
            "floor_area_sqm": st.column_config.NumberColumn("Floor area (m²)", format="%.0f"),
            "flat_model": "Flat model",
            "lease_commence_date": "Lease commenced",
            "remaining_lease": "Remaining lease",
            "resale_price": st.column_config.NumberColumn("Resale price", format="S$ %,.0f"),
            "price_per_sqm": st.column_config.NumberColumn("Price per m²", format="S$ %,.0f"),
            "budget_status": "Budget comparison",
        },
    )
    st.download_button(
        "Download these transactions (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        "resaleready_filtered_transactions.csv",
        "text/csv",
    )

    st.divider()
    st.caption(
        "Reminder: These are historical transaction records, not official valuations, "
        "asking prices, or forecasts of future resale prices."
    )
